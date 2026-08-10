# Findings — the road to working all-modality serving

The engineering arc behind this project. The short version: getting every modality of
LongCat-Next working on one GB10 took two debugging wins that looked like the same bug —
incoherent generation. We moved to 8-bit, then found and fixed a structural omission in the
serving generation loop.

One note on precision: 8-bit (`w8a8_int8`) is what we validated. An earlier 4-bit (NVFP4)
attempt generated incoherently, but it ran on the still-buggy serving pipeline and was never
retried after the fixes below — so 8-bit is the validated setting, not a proven floor.

---

## Act I — generation looked broken; moving to 8-bit produced coherent output

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

Loading the BF16 weights as **bitsandbytes int8** and running the model's own `generate()`
produced faithful images and intelligible voice-cloned speech. The image tiling and audio
drone looked like one defect — RVQ level-0 (coarse-layer) collapse — consistent with the
shared-RVQ-summation tokenizer behavior in arXiv 2603.27538, *Lexicalizing Modalities as
Discrete Tokens*.

`oracle/q8_unified.py` is the capability proof: one 8-bit load serving all five task types.
**But it is not a server** — no batching, concurrency, per-request sampling, or prefix
cache. It proved the model *can*; it didn't make the model *serve*.

---

## Act II — at 8-bit, on a real server, images still tiled — and precision wasn't it

Moving the validated 8-bit precision into a real SGLang serving stack (continuous batching,
RadixAttention, OpenAI API), the backbone now ran at `w8a8_int8` — **already at 8-bit, the
precision we'd moved to in Act I.** And the images *still tiled.*

The reflex was to suspect precision again. But the serving leg ran entirely at 8-bit — the
same precision that had produced coherent generation in Act I — so precision wasn't the
variable that changed here. (SmoothQuant on the MoE gate/up
projections is part of the int8 quant recipe — it migrates activation outliers into the
weights — but it is a quantization-quality measure, not the fix for this bug; see the model
card for what it is and does.)

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

- The 8-bit footprint runs headless on one GB10 with comfortable margin. Worth knowing when
  budgeting: the memory is consumed by the BF16-kept multimodal modules (tokenizers, decoders,
  the 282k-row over-embedding), **not** KV cache — so leave headroom there, not for context.
  (Over-allocating GPU memory on this box is unusually unforgiving; serve headless.)
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

## The stability + agent campaign (2026-07-30)

Triggered by field feedback from a DGX-class user: long image-bearing conversations froze
their node (or earlyoom killed the container), root-caused by them to PyTorch CUDA
allocator fragmentation in the vision encoder. Reproduced, fixed, and extended here:

- **`/dev/shm` SIGBUS (new find):** the first multi-image request killed SGLang outright —
  it ships pixel tensors between processes via `/dev/shm`, and every launcher ran with
  Docker's 64 MB default. `--shm-size=32g` everywhere; tmpfs allocates lazily.
- **Fragmentation repro + fix:** 80-turn multi-image soak, host `MemAvailable` sampled per
  turn. One image per turn: flat (no leak). Images accumulating in history (the real-world
  shape): −7.6 GB and still declining. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
  −2.9 GB converging to flat. Now the entrypoint default.
- **Generation heads cost ~25 GB, lazily, outside `--mem-fraction-static`:** measured
  25.9 → 1.5 GB free on first image generation. This is why the all-modality profile keeps
  0.72 — and why `LCN_AGENT=1` (generation endpoints refused) can afford 0.75 = the full
  131072-token KV pool.
- **Radix cache re-enabled** (viable only with the allocator fix): 7/7 modality selftest,
  and a 15.6k-token shared prefix drops from 5.9 s to 0.36 s (16×) — the enabler for
  agentic clients that resend a big system prompt every turn.
- **Anthropic Messages route** (`anthropic_route.py`): Claude Code drives the model
  end-to-end (multi-turn tool loops, streaming). Two parser findings along the way: the
  model emits TWO tool-call syntaxes (XML arg pairs AND TS-style `functions.name({...})`
  with unquoted keys and no closing tag — both now parsed), and the chat template's
  `arguments.items()` breaks on OpenAI-style JSON-string arguments, so tool-use history is
  pre-rendered into message content as canonical XML instead.
- **Speed verdicts (bench: 3 workloads × 3 runs, temp 0):** bf16 KV ~21 tok/s decode.
  fp8 KV (`LCN_KV_DTYPE=fp8_e4m3`) works — 124,720-token pool (2.3×) at the same
  fraction — but decodes at ~12.5 tok/s (−41%): capacity-over-speed opt-in, not default.
  CUDA graph capture fails on this port (even bs 8). NGRAM speculative decoding
  CUDA-faults inside the n-gram input embedding (draft positions violate its history-hash
  indexing) — would need a draft-aware embedding layer.

## Where the earlier exploration lives

The original 4-bit NVFP4 SGLang port (the `overlay/` modules) and the streaming per-expert
calibration tooling (`calibration/`) are preserved in this repo's **git history** at commit
`be21cc8` — the snapshot before this restructure. They're the ancestors of the shipped
`new_files/` overlay; kept in history rather than the working tree because the *story* of
that path (above) is the asset, not the superseded code.

---

## Spec-decode + generation: why they don't compose, and the shape of a fix

`LCN_NGRAM=1` currently forces `LCN_AGENT=1` (generation endpoints 403'd). That
coupling is not a design choice — it is a guard around a fatal crash, and the
crash is worth fixing on its own merits.

**Mechanism.** `ScheduleBatch.prepare_for_decode` (schedule_batch.py:2785):

```python
if not self.spec_algorithm.is_none():
    # Spec decoding owns decode preparation (allocation, seq-lens bookkeeping).
    spec_prepare_for_decode(self)
    return
```

When speculation is configured, decode preparation is delegated wholesale and
returns early, skipping the plain path — whose own comment notes "input_ids is
set at end of previous run_batch". Nothing on the spec path assigns `input_ids`,
so a batch that needs plain decode (because generation is active and must run
the model's Python state machines) arrives with `input_ids=None`. That yields
`raw_num_tokens=0` and the eager buffer-registry fill dies:

```
RuntimeError: The size of tensor a (0) must match the size of tensor b (19)
  in cuda_graph_buffer_registry._foreach_copy   (slot=positions dst=(0,) src=(1,))
```

**Why a fix is tractable.** The coarse granularity this needs already exists:
`lcn_cuda_graph_veto()` is global model state, not per-request — if generation is
active, the WHOLE batch runs eagerly. The same all-or-nothing rule applies here.
The missing pieces are a batch-level `force_plain_decode` flag threaded through
prepare -> spec worker -> forward, and assigning `input_ids` from the accepted
tokens of the previous verify step.

**Why it is worth doing.** Not to accelerate generation — n-gram drafting has
nothing to predict from on visual/audio codec token streams. The win is that ONE
server could run NGRAM for text while still serving image and voice generation,
with generation batches falling back quietly. That removes the forced choice
between speed and versatility, and turns a fatal crash into graceful
degradation for anyone who reaches the path by editing the entrypoint, running
the engine directly, or porting this overlay forward.

**Known remaining risk.** The model's generation state machines process one
token per decode step (watching for image_start/audio_start, counting toward the
1369 visual tokens of a 37x37 image). If a batch ever mixes accepted-multi-token
spec output with an active generation, those counters need ordered multi-token
handling. The all-or-nothing veto avoids this by construction, but it is the
first thing to check if a fix misbehaves.

### Implementing it: what the source actually required

Four things the sketch above got wrong or missed, each found by reading the
v0.5.16 scheduler rather than by reasoning from the crash:

1. **`batch.spec_algorithm` must NOT be flipped to NONE.** That looks like the
   elegant fix (every downstream `is_none()` branch would then agree), but
   `self.model_worker` is bound ONCE at init — with speculation configured it is
   permanently the draft worker — while the FutureMap stays ngram-configured.
   Flipping the batch flag desynchronizes the batch from the relay bookkeeping.
   A dedicated `batch.lcn_force_plain_decode` flag, consulted in exactly two
   places, keeps every other invariant untouched.

2. **The single-token fallback path already exists.** `NGRAMWorker.
   forward_batch_generation` has a non-verify `else` branch (used by prefill)
   that runs a plain target forward and reports it as `accept_tokens[:,0]=predict,
   accept_lens=1`, then builds the next `NgramVerifyInput`. Making
   `_prepare_for_speculative_decoding` return early for a flagged batch drops
   decode into that same branch — no new forward path, and the result shape the
   next draft prep expects is produced for free.

3. **`input_ids` cannot come from the FutureMap.** `FutureMap.stash()`
   early-returns for ngram ("FIXME: remove once precomputed draft is supported"),
   so `output_tokens_buf` is never written; `resolve_forward_inputs` gates its
   gather on `future_map.spec_algo.is_none()`, which is false forever. Both
   normal sources are empty. The token has to be rebuilt exactly the way
   `NGRAMWorker._prepare_draft_tokens` reconstructs history: under overlap the
   previous round's accepted tokens have NOT yet reached `req.output_ids`, so the
   newest token is `accept_tokens[i*stride + accept_lens[i] - 1]`, falling back to
   `output_ids[-1]` / `origin_input_ids[-1]`.

4. **The gen-entry latch never arms on the spec path.** `lcn_trigger_scan` is
   called from `update_after_decode`, which lives inside `model_runner.sample()` —
   bypassed by spec verify. Generation entered at PREFILL still works (prefill
   takes the plain `else` branch, so `sample()` runs), but a gen-entry token
   ACCEPTED during a verify round would be missed entirely and the fallback would
   never engage. Fixed by scanning accepted tokens in `lcn_write_spec_tokens`
   (`scan_triggers=True` at that call site only — drafts must not arm it, since a
   rejected draft would arm a generation that never happened).

### Measured result (2026-08-09, `longcat-next-gb10:v0516-specgen`)

Launched all-modality with speculation on — `LCN_NGRAM=1 LCN_AGENT=0`,
mem-fraction 0.72, overlap schedule ON, decode graphs bs<=32 — i.e. exactly the
combination the old guard existed to prevent.

**The crash is gone and the fallback demonstrably engages.** Log evidence:

```
[lcn] spec->plain decode fallback engaged for generation (steps=1)
[lcn] spec->plain decode fallback engaged for generation (steps=1000)
Decode batch, #running-req: 1, accept len: 1.00, accept rate: 0.00, cuda graph: False
[ImageGen] req=2: image complete at token 1405, forcing image_end
[ImageGen] req=2: image generation ended, 1369 visual tokens accumulated
Image saved to /workspace/outputs/longcat_img_..._refined.png
```

A full 37x37 image generated to completion under a spec-configured scheduler —
`accept len 1.00 / accept rate 0.00 / cuda graph False` is precisely the intended
fallback signature. Text passed too. The explicit counter matters: without it a
green selftest could not distinguish "fallback worked" from "generation never
reached the spec path".

**The generated image was CONTENT-BROKEN — owner verdict, 2026-08-09: "the image
doesn't even have a hint of an apple"** (prompt: "A photograph of a red apple on a
wooden table"). A valid 1040x1040 PNG with the full 1369 visual tokens, and the
wrong picture. `selftest.py` checks only the PNG magic bytes and byte count, so it
scored PASS; the fallback counter proved only that the PATH ran, not that its
output was correct. This is the third time in this campaign that a green battery
plus a plausible log signature was reported as validation of a generation change —
the human eyes/ears gate is the ONLY thing that has ever caught this class.

The paired comparison (same binary, same prompt and sampling params, only
`LCN_NGRAM` differing) settled it — owner verdict: **NGRAM on = "a white smudge";
NGRAM off = "an apple"**. The fallback corrupts generation.

### ROOT CAUSE: the future_map relay freezes seq_lens

`FutureMap.resolve_seq_lens_cpu` (overlap_utils.py:434) overwrites the batch's
sequence lengths at forward entry:

```python
draft_input = batch.spec_info
if draft_input is None: return
fi = draft_input.future_indices
if fi is None: return
...
batch.seq_lens = self.new_seq_lens_buf[fi]
```

It is gated only on `future_indices`, which `run_batch` sets for EVERY spec batch —
including a fallback batch. So each fallback step runs:

1. plain `prepare_for_decode` advances `seq_lens` S -> S+1 and allocates KV at S;
2. `run_batch` then calls `resolve_seq_lens_cpu`, which **discards that increment**
   and restores the value published by the previous forward.

Each fallback step publishes what it SAW (`new_seq_lens = batch.seq_lens.clone()`
in the worker's non-verify branch), so the resolved value is a fixed point:
`resolved_N = published_{N-1} = resolved_{N-1}`. **seq_lens freezes** for the whole
generation.

One mechanism, both symptoms, and both quantitatively:

- **The white smudge.** The n-gram embedding reads its context column at
  `seq_lens - 1` (`_init_ngram_embedding_info`, DECODE branch). Frozen seq_lens ->
  the column never advances -> the model generates all 1369 visual tokens against a
  stale context, never seeing its own output. That is low-information output, not
  noise, which is exactly what a smudge looks like. It also explains why the
  `[ImageGen] row 30/37` progress logs looked healthy: the model's own state machine
  counts tokens independently of seq_lens.
- **The 1405-slot leak.** `alloc_for_decode` writes `req_to_token` at the seq_lens
  position. Frozen seq_lens -> every step allocates at the SAME position and
  overwrites that entry, orphaning the previous slot. Exactly one leaked slot per
  decode step — which is why the leak equalled the generated-token count to the
  digit, twice.

**Fix:** skip the future_map seq_lens resolution for a batch flagged
`lcn_force_plain_decode` — plain prep owns seq_lens for that step. Publishing still
happens each step, so the hand-back to real verify rounds carries the correct value.

Lesson worth keeping: under overlap, `batch.seq_lens` is NOT owned by
`prepare_for_decode` when speculation is configured — it is relayed. Any path that
mixes plain preparation into a spec-configured scheduler has to decide who owns the
relayed state, not just who owns the forward.

**A second, independent defect: a KV pool accounting leak.** The scheduler
died at the first idle AFTER the image was saved:

```
ValueError: pool memory leak detected!
  [full] total=82482, available=79639, evictable=1438, protected=0, session_held=0
```

`82482 - 79639 - 1438 = 1405` — exactly the image's final token count, i.e. ONE
leaked slot per fallback decode step. The check (`invariant_checker._check_all_pools`
from `Scheduler.on_idle`) runs unconditionally, not only under speculation, and the
shipped non-NGRAM all-modality config passes 7/7 selftest repeatedly, so the leak is
introduced by the fallback rather than pre-existing.

**Isolated to the fallback by a same-binary control (2026-08-09).** Image
`:v0516-specgen`, identical prompt and sampling params, only `LCN_NGRAM` differing:

| cell | NGRAM | fallback engagements | leaked slots |
|---|---|---|---|
| 1 | on  | yes | 1405 (twice, exactly) |
| 2 | off | 0   | **0** |

Same binary, so the non-spec overlay edits (the probe registration and the
`scan_triggers` parameter, both inert without speculation) are held constant. The
leak is introduced by the fallback path itself.

Not yet root-caused. What is already ruled out: `alloc_for_decode` is NOT missing
the counter bump — allocation.py:591 does `req.kv.kv_allocated_len += token_per_req`,
so the plain path is internally self-consistent. The live hypothesis is a mismatch
between plain-path allocation and the spec-configured RELEASE bookkeeping (the spec
path pre-allocates a reserve via `alloc_for_spec_decode` and tracks
`kv_allocated_len` vs `kv_committed_len` differently; the seq-lens publish also
differs — the spec branch relays `on_publish(new_seq_lens)` where the non-spec
branch publishes `seq_lens + 1`).

**Status: NOT SHIPPABLE — two open defects.** What IS established: the fatal crash
is gone, and a generation now runs to completion under a spec-configured scheduler.
What is NOT established: that the output is correct (it is not — see the owner's
verdict above), or that KV is accounted correctly. Do not enable `LCN_NGRAM=1`
without `LCN_AGENT=1`; the entrypoint keeps that coupling as the default and
`LCN_NGRAM_ALLOW_GEN=1` is a dev-only opt-in.

Do not repeat the reporting error either: for this path, "generated a valid PNG"
and "the fallback counter incremented" are progress indicators, NOT validation.
Only the owner's eyes/ears close a generation change.
