# Findings — the road to working all-modality serving

The engineering arc behind this project, recorded so the reasoning is legible and the
eliminated hypotheses don't get re-chased. The short version: getting every modality of
LongCat-Next working on one GB10 took **two** distinct debugging wins that presented as the
*same* symptom — incoherent generation — but had unrelated root causes. The first was a
**precision floor**; the second was a **structural omission** that only surfaced once
precision was no longer the problem.

---

## Act I — generation looked broken, and precision was the lever

For weeks, generation produced output that was color- and texture-plausible but
structurally wrong: **images "tiled" into abstract fur/texture with no global composition;
audio collapsed to a "drone."** Both were running on the 4-bit NVFP4 backbone.

We ruled hypotheses out in order — this elimination is the load-bearing part, because each
dead end is a thing the next person doesn't need to re-investigate:

1. **Decode stack** — an oracle round-trip (real image → encode → decode) reconstructed a
   clean image; audio likewise. Decoder / refiner / VQ codebooks are sound.
   (`oracle/decode_roundtrip.py`, `oracle/audio_decode_roundtrip.py`)
2. **Positions / KV / newline cadence** — instrumented the gen loop: positions advance
   monotonically, hidden states vary (cosine 0.85–0.98), newline handling correct.
3. **Feedback embedding** — numerically identical to canonical.
4. **Heads** — `visual_head` / `audio_head` are full-precision in the checkpoint (0 quant
   markers). Not a quantization casualty.
5. **Calibration** — built both format-matched and content-matched NVFP4 calibrations.
   Document-page calibration → grayscale output; food/colorful calibration → beige output.
   Calibration demonstrably controls **palette**, but it never fixed **structure**.
   Calibration is *not* the generation lever.
6. **Sampling knobs** — tight top-k, per-level top-k (L0-greedy), CFG sweeps, negative
   prompts. Each changed the *flavor* of the wrongness, never the correctness.

### The teacher-forcing probe (the turn)

`oracle/teacher_force_image.py` fed a real image's `[324, 8]` RVQ codes as history and
measured the depth head's argmax against the true next token (vocab 16384/level, chance
0.006%):

- L0 top-1 ≈ 20.7%, **correct token median rank ~4 of 16384**.
- Decoding the teacher-forced argmax **recovered the source subject** (plane→plane,
  toad→toad) even at 8–20% exact top-1, because RVQ summation + the flow-matching decoder
  tolerate neighborhood-level error.

⇒ The conditional and the head are **sound given correct history**. The free-run failure
was **autoregressive drift / exposure bias**: slightly-off tokens fed back, compounding
over 324×8 steps. A *sharpness* deficit in the 4-bit conditional, not content-blindness.

### The fix

Load the original BF16 weights as **bitsandbytes int8** and run the model's own
`generate()`. At 8-bit the *same* pipeline produced faithful images and intelligible
voice-cloned speech (operator-judged). Image *tiling* and audio *drone* were the **same**
defect — RVQ level-0 (coarse-layer) collapse — and 8-bit crossed the floor for both at
once. This matches the shared-RVQ-summation tokenizer behavior described in arXiv
2603.27538, *Lexicalizing Modalities as Discrete Tokens*.

`oracle/q8_unified.py` is the capability proof: one 8-bit load serving all five task types.
**But it is not a server** — no batching, concurrency, per-request sampling, or prefix
cache. It proved the model *can*; it didn't make the model *serve*.

---

## Act II — at 8-bit, on a real server, images still tiled — and precision wasn't it

Moving the validated 8-bit precision into a real SGLang serving stack (continuous batching,
RadixAttention, OpenAI API), the backbone now ran at `w8a8_int8` — **8-bit, the precision
question already settled.** And the images *still tiled.*

The reflex was to suspect precision again. It wasn't. The serving leg ran entirely at
8-bit; the precision floor had been crossed in Act I. Chasing it again — a SmoothQuant
detour on the gate/up projections — had **zero effect**, which only re-confirmed precision
was not the serving lever.

The real cause was **structural**. The HF oracle's `prepare_inputs_for_generation`
auto-inserts a spatial anchor between the prompt and `image_start`:

```
<longcat_img_token_size>37 37</longcat_img_token_size>
```

(the anyres prefix, declaring the 37×37 token grid). **Our serving gen loop dropped it.**
Without that anchor the model emitted locally-plausible texture with no global 37×37
composition — the exact "tabby blob" seen across every serving attempt. Restoring the
anyres prefix produced a coherent image (operator: *"that's a cat"*). The same fix made the
classifier-free-guidance unconditional path correct too: its suffix-preservation length is
computed from that same anchor string, so anchor and uncond-mask stay in lockstep.

**Lesson:** a symptom that's identical across two legs of a system (incoherent generation)
can have two unrelated root causes. The precision finding from Act I was real *and* did not
carry to the serving leg — that leg's bug was orthogonal and structural. Don't let a closed
finding pre-explain a new failure.

---

## Act III — adversarial review caught a bug black-box testing couldn't

With all modalities generating coherently through the server, a **multi-agent adversarial
review** (per-subsystem reviewers diffing our implementation against the canonical model,
each finding then adversarially verified) surfaced something testing never would have: in
the MoE forward, the **identity/zero-expert contribution was added *after* the
`routed_scaling_factor` multiply instead of before**, leaving identity experts ~6×
under-weighted relative to the routed experts.

The model still produced coherent output with the bug present — which is exactly why it's
the kind of defect end-to-end tests miss. It was fixed as a correctness-only change
(no operator-visible quality delta), because *matching the canonical computation* is the
bar, not *looking fine*.

**Lesson:** end-to-end "it works" is necessary but not sufficient. Differential review
against a reference catches silent correctness drift that output inspection can't.

---

## What shipped

The repository root is the result: a single SGLang process on one GB10 serving **every
modality** — text, image/audio/video understanding, image generation, voice-clone audio
generation, and tool calling — behind an **OpenAI-compatible API**, quantized to
`w8a8_int8`, security-hardened for distribution, validated by a 7/7 self-test.

## Hardware reality (constant throughout)

- The 8-bit footprint sits **right under the GB10 unified-memory ceiling (~115 GB
  headless)**, beyond which the box powers fully off (not an OOM-kill — a power-down). The
  consumers are the BF16-kept multimodal modules (tokenizers, decoders, the 282k-row
  over-embedding), **not** KV cache.
- **MLA makes KV cache nearly free** (~16 KB/token), so context length is never the
  constraint here — the levers that make serving fit are `--mem-fraction-static` and the
  paged KV pool, not cache eviction.
- **Full BF16 from disk is blocked** on GB10: accelerate `device_map` offload leaves
  thousands of expert weights on `meta` (disk offload is broken for this custom MoE), and
  it won't fit RAM either. 8-bit on-GPU is the working substitute.

## Standing methodological notes

- **The operator judges generative output.** Self-grading led to over-claiming twice (an
  audio clip and an image batch both read as "working" before the operator's ears/eyes
  corrected it). Report objective stats + deliver the artifact; let the human call it.
- **A precision dead end is not a 6-bit invitation.** bitsandbytes does 4/8-bit only; true
  6-bit (modelopt MXFP6) is unproven on sm_121 and the size win is small because the
  precision-sensitive mass is the expert bulk. 8-bit stands.

## Where the earlier exploration lives

The original 4-bit NVFP4 SGLang port (the `overlay/` modules) and the streaming per-expert
calibration tooling (`calibration/`) are preserved in this repo's **git history** at commit
`be21cc8` — the snapshot before this restructure. They're the ancestors of the shipped
`new_files/` overlay; kept in history rather than the working tree because the *story* of
that path (above) is the asset, not the superseded code.
