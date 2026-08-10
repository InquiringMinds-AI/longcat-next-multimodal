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

**RESULT (2026-08-09, `:v0516-specgen2`) — the quality bug is FIXED.** Owner
verdict on the same prompt with the guard in place: **"It's an apple."** So the
seq_lens freeze was the generation-quality defect, confirmed by the same human gate
that caught it. Noted alongside, in the owner's own framing: it is the first apple
he has seen generated with no apparent stem, "a little odd, but isn't technically a
defect, since apples can have their stems removed." Recorded as an observation to
watch on future samples, NOT as a known defect.

**The KV leak is unchanged — still exactly 1405.** So "one mechanism explains both"
was half wrong: the seq_lens freeze was the quality bug and is NOT the leak's cause.

### The leak: two more hypotheses killed

The number is EXACTLY 1405 in all three runs (82482/79639/1438, 80882/78047/1430,
70117/67286/1426 -> 1405 every time), and 1405 is the image generation's full
sequence length. Note `evictable` is ALSO ~1405-1438 — i.e. roughly TWO copies of
the request's KV exist: one properly cached and one orphaned. Since the request
generates one token per decode step, "one full extra copy" and "one extra slot per
decode step" are the same number here; the live hypothesis is a double allocation
per fallback step, with the second slot never reaching `req_to_token`.

- **Uncond/CFG KV never freed** — dead. The model does allocate a second KV
  sequence for the CFG unconditional path (`_alloc_uncond_kv`, then `alloc.alloc(1)`
  per step, freed in one shot by `_free_uncond_kv`), and an unfreed copy would be
  exactly this size. But the logs show ZERO "Freed uncond KV" AND zero uncond
  activity of any kind.
- **...because CFG is not running at all.** `IMAGE_GEN_CFG_SCALE` defaults to 3.0,
  so the gate that failed is `self._model_runner is not None` — the reference
  installed by `_setup_kv_pool_refs`. The success path logs
  `[ImageGen] CFG initialized`, and that line never appears while other
  `[ImageGen]` lines from the same logger do.

### ROOT CAUSE #2 (the leak): kv_committed_len is double-counted

Found by measurement, not inference, after three source-reading hypotheses had
already died. Instrumenting the fallback counter to dump the per-request KV
counters gave it immediately:

| fallback step | seq_len | kv_allocated_len | kv_committed_len | output_ids |
|---|---|---|---|---|
| 1    | 192 | 192 | 192  | 0   |
| 500  | 478 | 478 | **936**  | 459 |
| 1000 | 978 | 978 | **1936** | 959 |

`kv_allocated_len` tracks `seq_len` exactly. `kv_committed_len` runs at ~2x, and
the excess equals `len(output_ids)` — one extra increment per decode step. Exactly
two writers exist on this path, and BOTH fire for a fallback batch:

- `schedule_batch.py:2909` — `req.kv_committed_len += 1` in plain `prepare_for_decode`
- `batch_result_processor.py:591` — `req.kv_committed_len += num_accept_tokens`
  (= 1 for a fallback step), which runs for any batch carrying spec results, and a
  fallback batch does carry them (accept_lens + speculative_num_draft_tokens)

Commit then desynchronizes from `kv_allocated_len`, the request's KV is released
against the wrong committed prefix, and one slot per decode step is orphaned.

**Fix:** skip the plain path's increment when `lcn_force_plain_decode` is set. This
is the SAME ownership rule as the seq_lens fix, and the second instance of the same
underlying mistake: **when speculation is configured, the spec bookkeeping owns the
relayed per-request state — plain prep must not also write it.** Two counters
(`seq_lens`, `kv_committed_len`) needed the same treatment; if a third relayed
counter surfaces, look here first.

Independent confirmation that the leak scales with generation length, not per
request: a short TTS generation leaked 39 slots where the 1405-token image leaked
1405.

**Open question worth its own investigation:** whether CFG is also inactive in the
SHIPPED build. If it is, production image generation has been running without
classifier-free guidance — a quality issue independent of any of this work, and one
the automated battery cannot see. The control is cheap (run one image with NGRAM
off and grep for `CFG initialized`) and has NOT been run yet; do not assume either
answer.

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

### Validation with both fixes (`:v0516-specgen4`, strict leak check ARMED)

Run with `LCN_NGRAM=1 LCN_AGENT=0 LCN_NGRAM_ALLOW_GEN=1`, decode graphs bs<=32,
overlap on, and `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` left at its default so
any residual leak would CRASH the server rather than warn:

| check | result |
|---|---|
| selftest modalities | **7/7** (both generation paths included) |
| anthropic route | 5/5 incl. tool_call + tool_roundtrip |
| degeneracy probe | 6/6 |
| KV leak reports | **0**, server up 25+ min |
| KV counters at fallback step 1000 | seq_len=1018 kv_alloc=1018 kv_commit=1017 (agree) |
| agent workload | **76.45 tok/s @ 100% fidelity** (vs ~22 baseline) |

The agent number is the point of the whole exercise: the ~3.3x NGRAM speedup is
intact IN A SERVER THAT ALSO GENERATES IMAGES AND VOICE. The speed-vs-versatility
choice the old guard forced is gone.

**Owner verdict on this build's artifacts (2026-08-09):** image — "apples good."
Audio — correct speech, then "a few seconds of silence" and a trailing "um?", with
his own hedge: "I think that could just be a normal variation for the voice gen."

Taken seriously rather than accepted, because a late end-of-audio detection would
produce exactly that, and the fallback changes how the audio state machine steps.
Duration is an objective proxy (a spurious tail lengthens the WAV), so it is
measurable without either of us judging by ear.

NGRAM ON, same sentence, n=13, sorted seconds:

```
3.78 3.92 3.92 3.96 4.32 4.36 4.39 4.42 4.76 5.38 6.42 7.44 7.63
```

A tight ~4s cluster plus a clear long tail — 4/13 run well over, and the clip he
heard (7.44s) is nearly double the median. So the tail is NOT typical even for this
build; it is an occasional artifact, not the normal rendering.

Note the documented TTS history for this port is entirely about the ONSET (first-word
garble, fixed via `LCN_TTS_SILENCE_FRAMES`/`LCN_TTS_TRIM_LEAD_MS`). A trailing
artifact is NOT documented anywhere, which is why the matched NGRAM-off control is
worth running rather than assuming known variation.

⚠ Power caveat, stated up front so the result is not over-read: n=13 per arm cannot
separate a moderate rate difference from noise. A large gap (e.g. 4/13 vs 0/13)
would be suggestive; anything smaller is not a result. Do not report a single
favorable arm as a finding.

**CONTROL RESULT — the tail is BASELINE TTS behavior, not caused by the fallback.**
Same build, same sentence, NGRAM off, n=13:

```
3.34 3.49 3.56 3.62 3.63 3.82 4.54 5.07 5.29 5.36 5.84 6.02 7.11
```

| | NGRAM ON | NGRAM OFF |
|---|---|---|
| median | 4.39 s | 4.54 s |
| mean | 4.98 s | 4.67 s |
| >5 s | 4/13 | **6/13** |
| >6 s | 3/13 | 2/13 |
| max | 7.63 s | 7.11 s |

The long tail appears in BOTH arms, and the control has MORE samples over 5s with a
slightly longer median. There is no signal that speculation or the plain-decode
fallback causes it. The owner's own hedge ("could just be a normal variation for the
voice gen") was correct and the Claude-side suspicion was wrong — recorded that way
deliberately, because the suspicion was reasonable (a late end-of-audio detection
produces exactly this shape) and the only thing that separated it from the truth was
running the control.

What this does NOT establish: that the tail is harmless or unimprovable. It is a
pre-existing TTS quality issue on this port, independent of all the spec-decode
work, and a candidate for its own investigation (the ONSET got one; the tail never
has). Rate is roughly 2-6 in 13 depending on threshold — frequent enough to be
worth a look if voice output quality matters.

**One anomaly, not dismissed:** the first selftest run of this build scored
`tool_calling` FAIL ("no tool_calls"); a standalone repeat passed 3/3 identically
and a full selftest re-run in the SAME sequence (right after both generations)
scored 7/7. Unexplained, and suspicious because the request runs at temperature 0
where a one-off should not happen. The n-gram table is server-global and persists
across requests, so post-generation table state is the natural suspect. Watch for
it; if it recurs, that is the thread to pull.

**Status: candidate, pending the human gate.** The entrypoint still defaults to the
old coupling (`LCN_NGRAM_ALLOW_GEN=1` opts in) until the owner's verdict lands and
a soak has run — a single 25-minute window is not evidence about long-session KV
behavior.

Do not repeat the reporting error made mid-campaign either: for this path,
"generated a valid PNG" and "the fallback counter incremented" are progress
indicators, NOT validation. Only the owner's eyes/ears close a generation change.

---

## CFG has never run: `_setup_kv_pool_refs` is defined but never called

Surfaced while chasing the KV leak (2026-08-09), and independent of all the
spec-decode work.

`LongcatNextForCausalLM._setup_kv_pool_refs(model_runner)` sets
`self._model_runner`, which every classifier-free-guidance path depends on. Its
docstring says "Called by model_runner to provide KV pool access for CFG
dual-path" — but **nothing calls it**: not the overlay, not any patch, not
upstream sglang. The only other assignment is `self._model_runner = None` in
`__init__`.

So `self._model_runner` is permanently None and the whole CFG feature is dead:

- the gate `if IMAGE_GEN_CFG_SCALE != 1.0 and self._model_runner is not None` is
  always False (the scale itself defaults to 3.0, so the scale is not what
  disables it),
- `_alloc_uncond_kv` early-returns,
- `_free_uncond_kv` no-ops.

**Verified by log absence in BOTH configs**: zero `[ImageGen] CFG initialized`,
zero `uncond` lines of any kind, with NGRAM on AND with NGRAM off (the
shipped-equivalent path) — while other `[ImageGen]` lines from the same logger
appear normally.

Every image this project has ever generated has therefore been UNGUIDED. This is
a live quality question on the flagship's headline capability, and it is a wiring
bug, not a tuning choice.

**Cost of enabling it, so the tradeoff is judged on evidence:** the unconditional
path is a full second KV sequence (~1405 slots for a 37x37 image — the pool
demonstrably absorbs that, since the leak consumed exactly that much without
exhausting it) plus a second backbone forward per decode step, so expect image
generation to run roughly 2x slower. Memory headroom must be re-measured, and the
owner's eyes decide whether the quality difference is worth it — paired samples,
same prompt, CFG on vs off.

### CFG wired up and verified live (2026-08-10)

`ModelRunner.lcn_setup_model_kv_pool_refs()`, called at the end of
`init_ngram_embedding_manager`, now invokes the model's `_setup_kv_pool_refs(self)`.
Confirmed active in the log for the first time in this project's history:

```
[ImageGen] CFG initialized: uncond_req=2, seq_len=19
[ImageGen] Freed uncond KV: 1423 tokens
```

The unconditional sequence is allocated AND freed cleanly (1423 tokens), so CFG
introduces no KV leak of its own.

**Measured cost** (same build, NGRAM off both arms, only `IMAGE_GEN_CFG_SCALE`
differing — so the two arms are one build and one env var, no rebuild needed):

| | CFG on (3.0) | CFG off (1.0) |
|---|---|---|
| apple, end-to-end | 411 s | 331 s |

~1.24x slower end-to-end. The ~95s refiner is a fixed cost, so the decode portion
roughly doubled (316s vs 236s) — consistent with a second backbone forward per step.

Paired samples delivered for the owner's eyes on two prompts: the easy apple
(sanity — should stay good) and a compositional prompt with two objects, colors and
a spatial relation ("a red apple to the left of a yellow banana"), which is the
discriminating case because prompt adherence is what CFG buys.

⚠ Note for interpreting the project's history: FINDINGS Act I lists "CFG sweeps"
among the sampling knobs tried against the tiling problem, and Act II says the
anyres fix "made the classifier-free-guidance unconditional path correct too".
Since `_setup_kv_pool_refs` has never been called, **CFG cannot have been running
during any of that work** — those sweeps were almost certainly no-ops on an
unguided model. Do not treat the earlier CFG conclusions as evidence about CFG.

**First owner signal on the CFG pair (2026-08-10):** on the compositional prompt he
flagged the CFG-OFF image — "the fourth image has a really bizarre shape for its
'banana'" — and did not flag the CFG-ON one. That points the expected direction
(guidance improves prompt adherence), but it is ONE sample per arm and is NOT
treated as a result: the TTS-tail investigation earlier the same day died exactly
that way, where a suspicion that looked obvious dissolved once a matched control ran.

Follow-up designed for generality rather than repeat-variance: FOUR distinct
compositional prompts per arm, each probing a different failure mode that guidance
is supposed to fix —

| prompt | probes |
|---|---|
| red apple to the left of a yellow banana | two objects, colors, spatial relation |
| blue mug next to a green book | color binding across objects |
| three oranges in a white bowl | COUNT |
| black cat sitting under a wooden chair | containment / occlusion |

Count and color-binding are the classic guidance-sensitive cases, so a real effect
should be most visible there. Same build, same env-var-only arm switch.

### CFG VERDICT: worse at the configured scale — kept wired, default OFF

Owner judgment on the four paired prompts (2026-08-10), CFG on vs off, same build,
arms differing only by `IMAGE_GEN_CFG_SCALE` 3.0 vs 1.0:

| prompt | probes | preferred |
|---|---|---|
| blue mug next to a green book | colour binding | **CFG OFF** |
| three oranges in a white bowl | count | **CFG OFF** |
| black cat under a wooden chair | containment | **CFG OFF** |
| red apple left of a yellow banana | spatial relation | ambiguous — he flagged the OFF banana's shape as "really bizarre" but did not call the ON image better overall |

Verdict: **"I like the second image of each pair better"** — the OFF arm, on all
three of the cleanly-judged prompts.

So classifier-free guidance at the checkpoint's own configured scale makes output
WORSE by the only instrument that counts here. The accidental behaviour this
project has always shipped (CFG never wired, therefore never active) turns out to
have been the better setting.

**Resolution:** keep the wiring — it fixes a genuine bug, and a never-called method
whose docstring claims it is called is a trap for the next reader — but gate it
behind `LCN_CFG=1`, DEFAULT OFF. That reproduces exactly the behaviour every
validated image in this project was generated with, while making the feature
reachable for experiments. Cost when enabled: ~1.24x slower image generation plus a
full second KV sequence per in-flight image.

**Not concluded:** that CFG is useless here. Only scale 3.0 was tested, and the
early evidence was mixed rather than uniform (the one prompt where the OFF sample
drew a complaint was the spatial-relation one). A gentler scale (1.5-2.0) is
untested and is the obvious next experiment if image quality is ever revisited —
the harness now exists, and an arm is one env var.

⚠ Also untested: memory headroom with CFG active under CONCURRENCY. It allocates a
second KV sequence per in-flight image and was only ever exercised single-request.
Anyone enabling it on this box (which hard-powers-off past ~110-115GB) must
re-measure first.

### TTS tail: transcript hypothesis REFUTED; the acoustic phase is the cause

Chasing the trailing "(silence) um?" artifact (owner-heard, ~2-6 in 13 depending on
threshold, present with AND without speculation so it is baseline behaviour).

The obvious suspect was the transcript phase: the model recites the requested text
before the acoustic codebooks render it, and that recitation is SAMPLED, not greedy
(`LCN_TRANSCRIPT_TEMPERATURE` 0.5, top-k 5, top-p 0.85 — deliberately, because the
original detects on the sampled token, not argmax). A wandering transcript that
appended "um" before emitting the pad token would render as speech and explain the
artifact exactly.

**Refuted at zero cost by correlating existing logs**, no test cycle needed. The
engine already logs why each transcript ended, and the soak logs audio byte counts
per cycle:

```
transcript ended (natural (audiotext_pad)) after 6 steps   x4, every sample
soak cycle 1: audio_bytes 79244
soak cycle 2: audio_bytes 135884      <- 1.7x the audio, SAME 6-step transcript
```

Identical transcript length, wildly different audio length => the extra duration is
produced AFTER the transcript, in the acoustic phase. The transcript sampler is not
the cause and `LCN_TRANSCRIPT_GREEDY=1` is not the fix.

**Where to look next:** the acoustic phase runs until `_is_audio_end_token` fires,
which requires level-0 to sample exactly `codebook_sizes[0]` (8192); the only other
stop is a 1000-frame safety cap (~40s), which is not being hit (no "hit safety cap"
warnings). So the model simply takes a variable number of frames to emit end-of-audio,
and sometimes fills that time with silence and a filler utterance. The lever is the
acoustic stopping criterion / its sampling, not the transcript.

Not yet investigated: whether the acoustic codebook sampling params can be tightened
without hurting prosody, and whether an energy-based trailing-silence trim (the
analogue of the existing `LCN_TTS_TRIM_LEAD_MS` onset trim, which already solved the
mirror-image problem at the START) would remove the tail more cheaply than changing
generation. The onset got exactly this treatment; the tail never has.

### SOAK PASSED — the NGRAM/generation coupling is removed (2026-08-10)

12-cycle generation soak (`test/soak_specgen.py`) with the strict idle KV-leak
check ARMED, so a residual leak would kill the server rather than warn. Each cycle:
3 speculative text decodes + 1 audio generation, plus a full image generation every
4th. 4000+ plain-decode fallback steps engaged across the run.

MemAvailable per cycle (GB):

```
25.78  25.81  25.90 | 6.06  6.00  5.97  5.95  5.89  5.89  5.90  5.89  5.94
       (no image yet) ^ first image generation
```

The step at cycle 4 is the DOCUMENTED lazy allocation of the generation heads
(~20GB, outside `--mem-fraction-static`) and lands exactly on the previously
measured ~6GB floor. After it the series is FLAT — cycle 12 (5.94) sits above
cycles 8-11 — so nothing accumulates across three full image generations. **Zero
leak reports. Server alive throughout.**

That was the gate, so the entrypoint default is flipped: **`LCN_NGRAM=1` no longer
implies `LCN_AGENT=1`.** One server now serves NGRAM-accelerated text AND image /
voice generation. `LCN_NGRAM_AGENT_COUPLE=1` restores the old pairing.

Verified with MINIMAL env (only `LCN_NGRAM=1` set, everything else defaulted):
agent not forced, `mem_fraction_static=0.72` (all-modality), decode CUDA graphs on
at max-bs 32, `speculative_algorithm='NGRAM'`.

Also fixed in the same build: `--cuda-graph-max-bs` is deprecated upstream in
favour of `--cuda-graph-max-bs-decode` (warned on every start); switched.

Remaining deprecation noise is pre-existing and unrelated (diffusers
`LoRACompatibleLinear`, `pynvml`), EXCEPT three that are ours and worth a future
cleanup: `LongcatNextProcessor` declares `image_processor_class` /
`video_processor_class` / `audio_processor_class` directly, which transformers now
wants registered via `AutoImageProcessor` / `AutoVideoProcessor` /
`AutoFeatureExtractor`.

### Intermittent tool_calling failure under NGRAM — REAL, not a flake

`selftest` scored `tool_calling: no tool_calls` in **2 of 3 runs** under NGRAM
(2026-08-09/10). It was initially written off as a one-off when a standalone repeat
passed 3/3; the second occurrence, on the final validated build, disproves that.
Recorded as an open defect, because tool calling IS the agent path and NGRAM exists
to serve that path.

What makes it interesting: the same battery run that failed selftest's check scored
5/5 on the ANTHROPIC route including `tool_call` and a full `tool_roundtrip`, minutes
later. So the capability is intact; something about the position of that request in
the sequence breaks it.

In selftest, `tool_calling` runs last, after: image generation -> image understanding
-> audio generation -> audio understanding -> video understanding.

**Elimination so far** (`test/probe_toolcall.py`, alternating cold vs
after-predecessor, temperature 0, selftest's exact tools and prompt):

| predecessor | cold | after |
|---|---|---|
| audio generation | 5/5 | **5/5** |
| image understanding | 5/5 | **5/5** |
| image generation | 3/3 | **3/3** |

**No single predecessor triggers it** — not a short generation, not a multimodal
understanding request, not even the long image generation that drives ~1400
plain-decode fallback steps. So the trigger is CUMULATIVE sequence state, which
matches the observation that the failure has only ever appeared inside selftest's
full run and never in isolation.

Next: `test/probe_toolcall_sequence.py` replays selftest's exact modality order and
then issues selftest's exact tool-call request, dumping `content`, `finish_reason`
and the full message on failure. That capture is the decision point, because the
gateway leaves unparsed model output in `content` untouched:

```python
normal, calls = parse_tool_calls(msg.get("content") or "", tools)
if calls:                      # content is only rewritten when parsing SUCCEEDS
```

Two outcomes with opposite fixes, and selftest's pass/fail cannot distinguish them:
* an unhandled tool-call DIALECT (the parser already normalizes three: the
  `<longcat_tool_call>` XML arg-pairs, the TS-style object-literal call, and the
  Claude-imitation `<function_calls>` JSON) -> a few lines in `parse_tool_calls`
  plus a regression fixture built from the captured text;
* ordinary prose with no call attempted -> a parser shim cannot help, and the
  question becomes why the tools system block stops conditioning the model after a
  long multimodal context.

Standing caution for whoever picks this up: do NOT conclude "flaky" from a passing
repeat. This defect has already survived one such dismissal.

**Sequence replay did NOT reproduce it (0/3) — but the replay was incomplete.**
`probe_toolcall_sequence.py` covered text -> image gen -> image understanding ->
audio gen -> audio understanding and stopped, OMITTING selftest's step 6, VIDEO
understanding, which is the step immediately BEFORE the failing tool call. So that
run eliminates nothing; the one untested predecessor is the nearest one.

Rather than patch the replay, the reproduction is now the improved `selftest.py`
itself: it is the exact known trigger (~2 in 3 historically) and, as of this
session, dumps `content` / `finish_reason` / the full message on failure. Running it
repeatedly is both the reproducer and the instrument.

Reference: video understanding is also the heaviest multimodal input in the battery
(an mp4 built from the generated image), which makes it a plausible trigger for
context-conditioning loss on the following request.

**Rate corrected, and still uncaptured.** Three runs of the improved selftest scored
7/7 including `tool_calling`. Running total across the session: **2 failures in 6
selftest runs (~1 in 3)**, not the "2 of 3" inferred earlier from a three-run sample
— that was over-read, and the correction matters because it changes how many runs a
capture attempt needs. The defect is still real (two confirmed, independent
occurrences on different builds) and is NOT to be closed as flaky; it is simply less
frequent than first claimed. The instrumentation is now permanent, so the next
occurrence documents itself with no bespoke probe.

**The transparency work paid off immediately on a different problem.** The same three
runs render the SAME fixed sentence ("Self test, all systems nominal.") at:

| run | seconds | peak | rms |
|---|---|---|---|
| 1 | 4.77 | 0.124 | 0.0147 |
| 2 | **2.52** | 0.381 | 0.0333 |
| 3 | 5.35 | 0.232 | 0.0147 |

A 2.1x duration spread on identical input, now visible in the ROUTINE battery instead
of requiring the bespoke 13-sample study that first established the tail. Note also
that the short render has ~3x the peak and ~2x the RMS of the long ones, which is
consistent with the long renders padding quiet material (silence / trailing filler)
onto the same speech — a useful, cheap discriminator for future work on the tail.

### CORRECTION (2026-08-10): duration was never a valid measure of the tail

Owner, on the duration study above: *"length of audio is not entirely decided by a
tail, cadence has a significant impact on how long it takes to say a thing."*

He is right, and it retires the instrument. Total duration is `lead + speech_span +
trail`, and a slower read inflates `speech_span` by exactly the same arithmetic a
tail inflates `trail`. Duration therefore cannot distinguish them at any sample size
— the 13-sample distributions and the 2.1x routine-battery spread measured *something
varying*, but never established *what*.

The rms/peak corroboration was worse than uninformative, it was circular. I read "low
rms in long renders" as quiet padding. But rms is energy per sample, so inter-word
pauses in an unhurried read depress it identically. That observation is consistent
with both hypotheses and discriminates neither; treating it as a "cheap discriminator"
above was an error.

**What survives:**
- The owner directly HEARD a tail — "a few seconds of silence" then "um?". That is
  perceptual evidence a trailing artifact exists in at least that render, and it does
  not depend on any duration measurement.
- The NGRAM-on vs NGRAM-off control still supports "not caused by the fallback,"
  because it was PAIRED — whatever duration mixes together, it mixed the same way in
  both arms, and neither arm stood out. Weaker than stated, but not overturned.

**What does NOT survive:** that the tail is "occasional" at ~4/13, that a long render
indicates a tail at all, and the rms/peak inference.

**Instrument replaced** (`test/selftest.py`, `audio_stats`). A 20ms energy envelope
now reports the components separately instead of their sum:

| field | meaning |
|---|---|
| `trail_ms` | silence after the last voiced frame — the tail, and nothing else |
| `lead_ms` | silence before the first voiced frame (guards the fixed onset trim) |
| `speech_sec` | first-to-last voiced span — the part cadence owns |
| `cps` | input characters per `speech_sec` — a cadence proxy |

Reading: stable `cps` + rising `trail_ms` = tail. Falling `cps` + flat `trail_ms` =
the model simply read it slower, which is not a defect. The acoustic-phase finding
above (transcript ruled out) still stands — it rested on transcript step counts and
byte sizes, not on duration — but "how big is the tail" has NO trustworthy prior
measurement, and must be re-established with `trail_ms` before any trim is designed.

### TTS: the variance is INTERIOR PAUSE LENGTH (2026-08-10, n=10)

Re-measuring with the replacement instrument (`test/media_stats.py`, extracted from
selftest so saved artifacts can be re-analyzed offline). Ten renders of the fixed
sentence "Self test, all systems nominal.", none owner-adjudicated — so this is the
shape of ORDINARY output, not of the reported defect.

| seconds | speech_sec | cps | lead_ms | trail_ms | max_gap_ms |
|---|---|---|---|---|---|
| 5.35 | 5.12 | 6.1 | 140 | 80 | 1300 |
| 4.77 | 4.46 | 7.0 | 140 | 160 | 600 |
| 4.74 | 4.54 | 6.8 | 140 | 60 | 900 |
| 4.70 | 4.56 | 6.8 | 140 | 0 | 460 |
| 4.46 | 4.16 | 7.5 | 140 | 160 | 900 |
| 4.15 | 3.38 | 9.2 | 140 | 620 | 500 |
| 3.88 | 3.52 | 8.8 | 160 | 200 | 480 |
| 3.14 | 2.68 | 11.6 | 160 | 300 | 260 |
| 3.02 | 2.44 | 12.7 | 140 | 440 | 140 |
| 2.52 | 2.14 | 14.5 | 140 | 240 | 80 |

**1. Duration is anti-correlated with the trail.** The longest renders have the
smallest trails (80, 60, 0 ms); the shortest have larger ones (440, 240). The old
duration proxy did not merely fail to isolate the tail — it pointed the wrong way, so
every conclusion drawn from it was worse than uninformed.

**2. The cadence swing is interior silence, not word rate.** `max_gap_ms` tracks `cps`
almost monotonically: 1300 ms of pause at cps 6.1 down to 80 ms at cps 14.5. A 1.3 s
pause inside a 5.1 s reading of a one-comma sentence is the single largest component
of the variance.

**Consequence for the reported defect.** The owner heard silence and then "um?". A
voiced filler ENDS the trail, so that artifact would register as a large INTERIOR gap,
not as `trail_ms` — meaning the thing to look for is the extreme upper tail of the
pause distribution above, not a trailing-silence trim. The energy-based trim proposed
earlier (as the analogue of `LCN_TTS_TRIM_LEAD_MS`) would not have touched it. That
proposal is withdrawn pending evidence.

**Deliberately not built:** a "has the defect" flag. A first heuristic (>=400 ms gap
with <=600 ms of audio after) fired on 2 of these 10 ordinary renders. With zero
adjudicated defective samples there is nothing to calibrate against, and a threshold
guessed from the good side only is a verdict wearing a measurement's clothes. The
numbers are reported; a human adjudicates. (`image_stats`' palette advisory is
different in kind — it has an adjudicated bad sample on one side, three good on the
other, and wide margin between.)

**Status:** no defective sample has been captured with the new instrument. The next
useful step is not a fix but a CAPTURE — the owner flagging a render that sounds wrong
so its `max_gap_ms` / `after_gap_ms` can be compared against this baseline.

### tool_calling: the ~1-in-3 rate is refuted for this build (2026-08-10)

With the transparent selftest and the corrected sequence probe (video step restored),
the defect did not reproduce:

| attempt | result |
|---|---|
| selftest x3 (earlier) | 3/3 clean |
| probe_toolcall_sequence x4, full chain incl. video | 4/4 clean |
| selftest x6 | 6/6 clean, 7/7 modalities each |

That is 9 consecutive clean selftest runs on `v0516-specgen-final`. Under the
previously inferred ~1-in-3 failure rate, 9 clean runs is a ~2.6% outcome, so that
rate does not describe this build.

What this does NOT establish: that the defect is fixed. Nothing targeted it, no
mechanism was identified, and the two original failures were real (2 of 6, across two
builds). The honest statement is that the rate is LOWER than believed and unmeasured,
which also means reproducing it on demand is now the expensive part.

Capture readiness is in place instead: selftest dumps the raw emission on failure, so
the next occurrence distinguishes the two mutually-exclusive causes (unhandled dialect
-> shimmable in `parse_tool_calls`; no call attempted -> not shimmable) without
needing a reproduction harness.

### TTS: OWNER ADJUDICATION — the defect is CONTENT LOSS, not a tail (2026-08-10)

Two renders of "Self test, all systems nominal." were sent for adjudication, chosen as
the extremes of the pause distribution. Owner verdict:

| render | cps | speech_sec | max_gap_ms | trail_ms | owner |
|---|---|---|---|---|---|
| a_110903 | 6.1 | 5.12 | 1300 | 80 | "very slow cadence, and ends with 'all systems' skipping **nominal**" |
| a_114958 | 11.6 | 2.68 | 260 | 300 | "faster cadence, and no missing words or extra tail" |

**The slow render TRUNCATED.** That retires the framing this was investigated under all
day. It is not a trailing artifact and not a cadence preference — the model dropped the
final word. No silence-geometry measurement can see that: a truncated render has a
perfectly ordinary `trail_ms` (80 ms here, among the smallest observed).

**`cps` was confounded, in the direction that misled.** It is computed as
`len(input_text) / speech_sec`, which assumes the whole input was spoken. a_110903 spoke
~23 of 31 characters, so its true rate is ~4.5 cps, not the 6.1 reported. Low cps was
read as a stylistic slow read when it may simply BE the truncation signature — the
metric hid the defect inside a plausible explanation.

**Unification with the earlier owner-heard sample.** That one had silence and a stray
"um?" — content GAIN. Truncation is content LOSS. Both are the acoustic phase ending at
the wrong time (early vs late), consistent with the already-established finding that the
TRANSCRIPT phase is clean: the model knows the right words, the acoustic rendering of
them terminates unreliably. One mechanism, two signs — not two defects.

**Instrument replaced again** (`test/tts_roundtrip.py`): generate, transcribe through
this same server's audio-understanding path, diff word-by-word against the input.
`missing` / `extra` are the signal; silence stats ride along so truncation can be
correlated against them. Two limits kept explicit in the file: the transcriber is itself
a model (a missing word is evidence, not proof — every render is saved for listening),
and `cps` is only a cadence measure on a COMPLETE render.

**Standing correction to method:** three instruments in a row failed the same way —
`seconds`, the deleted `tail_gap_ms` flag, and `cps` — each producing a plausible reading
for a state it could not actually distinguish. Every one was validated against
self-consistency rather than against an adjudicated sample. The adjudication took one
message and overturned a day of measurement.

### The round-trip transcription check FAILS against ground truth — do not use it

Run against the 11 saved renders, `tts_roundtrip.py` reported 6/11 missing a word, with
a suspiciously perfect split by `cps` (complete 6.1-7.5, truncated 8.8-14.5, no overlap).
It is wrong. On the only two renders with owner adjudication it disagrees BOTH times,
in both directions:

| file | owner | ASR transcript |
|---|---|---|
| `..110903` (5.35s, cps 6.1) | "skipping **nominal**" | `'Self test, all systems nominal.'` — complete |
| `..114958` (3.14s, cps 11.6) | "no missing words or extra tail" | `'Self test all systems'` — missing nominal |

File identity was checked rather than assumed: the durations match the owner's cadence
descriptions ("very slow" = 5.35s, "faster" = 3.14s), so nothing was mislabeled in
transit. The path is also visibly unreliable — one render transcribed as meta-commentary
(`'The audio contains the following words spoken in sequence:\n\nSelf test\nAll systems'`)
rather than a transcript.

Most likely mechanism: the transcriber reconstructs the expected sentence from priors,
and does so more readily on longer audio — which would manufacture exactly the clean
duration-correlated split observed. Self-consistent, and false.

**A caveat that cuts the other way, and is NOT a rescue of the instrument:** if a render
CONTAINS "nominal" but renders it unintelligibly, ASR could recover it acoustically while
a listener correctly reports it absent. If so the audio is still defective — for a TTS
product "present but unintelligible" and "missing" are one failure — but it means the
round-trip cannot separate TRUNCATED from GARBLED either.

**Fourth instrument in one session to fail the same way** (`seconds`, `tail_gap_ms`,
`cps`, now the round-trip): each was checked for self-consistency and none against an
adjudicated sample. Standing rule for this defect going forward: OWNER LABELS ARE THE
ONLY GROUND TRUTH, and any automatic TTS metric must be validated against a labeled set
BEFORE it is used to draw a conclusion — not after.

The 6/11 number and the cps separation are recorded here only as the discredited output
of a broken check. **Do not cite them.**

### Segment structure — an ASR-free view, and a reading that reconciles the conflict

Owner, on the round-trip contradiction: *"the asr heard a word that wasnt there? and
missed a word that was there? thats odd, odd that they both flip"*. Correct, and it kills
the prior-reconstruction mechanism proposed above: a BIASED transcriber hallucinates the
expected word in both cases; it cannot drop a word that is present. Symmetric errors mean
noise — except noise does not produce a perfect 11-sample split either. Retract the
mechanism, keep the tension.

A 20ms envelope segmentation of the two adjudicated clips, no transcription involved:

```
a_110903 (owner: "skipping nominal")
  sil140 SPEECH540 sil280 SPEECH500 sil40 SPEECH80 sil1300 SPEECH500 sil1220 SPEECH360 sil20 SPEECH280 sil80
a_114958 (owner: complete, no tail)
  sil160 SPEECH220 sil80 SPEECH360 sil80 SPEECH40 sil260 SPEECH600 sil60 SPEECH980 sil300
```

`110903` carries TWO internal silences over 1.2s and ends in 360ms/280ms fragments;
`114958` has ordinary 60-260ms gaps and one clean 980ms final run. The multi-second
internal silence is the owner's ORIGINAL report ("a few seconds of silence") appearing as
structure rather than as a duration number.

**A reading in which both the owner and the ASR are right** (hypothesis, not established):
in `110903` "nominal" is PRESENT BUT SHATTERED across fragments after a 1.2s gap — a
listener correctly reports it as not delivered while ASR reassembles it acoustically; in
`114958` the fast 980ms final run slurs "systems nominal" together, clear to a human and
exactly the shape ASR clips. If so, the round-trip's word-completeness output is a
function of CADENCE AND INTELLIGIBILITY, not of content — which would explain the perfect
cps separation as the check simply re-measuring cps.

This also relocates the defect: fragmentation and multi-second internal silence, i.e. the
acoustic phase losing coherence mid-utterance, not a boundary decision at the end.

**Protocol change:** 8 fresh renders (durations 2.56-5.46s, cps 5.9-14.0) were sent for
BLIND labelling — stats deliberately withheld so the labels are not anchored by the
numbers hoped to be predictive. Stats are recorded and will be compared after. This is
the labelled set that every candidate metric must be tested against BEFORE use, per the
four-instrument lesson above.

### ROOT-CAUSE CANDIDATE: the end-of-audio token is SAMPLED, and rep-penalty targets held sounds

Owner, on the truncated render (2026-08-10): *"systems ends abruptly at the end of the
audiofile, truncating the sibilance of the ending s, there is no space for it so shatter
or have jack shit"*.

That refutes the "present but shattered" hypothesis outright — the WAVEFORM ends
mid-fricative, so there is no audio for "nominal" to be shattered into. It also INVERTS
the reading of `trail_ms`: a complete render ends with some trailing silence, while one
cut off mid-word has no room for any. Low `trail_ms` is the TRUNCATION signature, not the
clean result it was recorded as. Both adjudicated clips fit (truncated 80ms, complete
300ms), and it explains the earlier duration anti-correlation: `a_110903` is long AND
truncated — bloated by two 1.2s internal silences, then cut before finishing.

**Not the safety cap.** `longcat_next_mm.py:53` sets `max_audio_steps = 1000` (~40s);
these renders use ~60-140 steps. Ruled out.

**The mechanism, from the code.** Level-0 acoustic tokens are drawn with
`torch.multinomial` (`_generate_audio_codebook_step`, ~line 912-936) under
`AUDIO_GEN_TEMPERATURE=0.5`, `TOP_K=5`, `TOP_P=0.85`, `REPETITION_PENALTY=1.3`. The
end-of-audio token is `codebook_sizes[0]` (8192) and sits in that SAME distribution, with
no minimum-length guard. If it enters the top-5 at any step it can be drawn mid-word.
This explains the abrupt cut, the absent trailing silence, and the variability across
identical input (it is stochastic), and it covers BOTH signs of the defect: sampled early
-> truncation; not sampled when due -> run-on silence and a stray "um?".

**Why a fricative specifically.** `AUDIO_GEN_REPETITION_PENALTY=1.3` is applied to level-0
codebook tokens. A sustained sound — a held /s/, a long vowel, silence — IS a repeated
token, so the penalty suppresses exactly the tokens that CONTINUE the sound, raising
everything else relative to it including the end token. Repetition penalty is a text-
decoding heuristic where repeats signal degeneracy; in an acoustic codebook a repeat is
just a held sound. Predicts truncation should concentrate on sustained sounds, which is
where the owner heard it.

**Both testable with NO code change** — all four are env vars read at module import, so a
container restart suffices (no rebuild):
  * `AUDIO_GEN_REPETITION_PENALTY=1.0` — does truncation stop when held sounds are not penalized?
  * `AUDIO_GEN_TOP_K=1` — does it stop when the end token must be the argmax?

**Sequencing (owner's paired-comparison rule):** 8 baseline renders are already with the
owner for BLIND labelling. Those labels are the control arm; the variant arm is generated
only after, and judged the same blind way. A predicted split was recorded BEFORE the
labels arrive, from `trail_ms` alone: truncated = adj_04 (60), adj_05 (60), adj_03 (100);
complete = adj_08 (160), adj_01 (200), adj_07 (200), adj_06 (540), adj_02 (600). The
100/160 boundary is thin; the extremes are the real test.

## ⚠ PRODUCTION BUG: multimodal prefix-cache collision serves another request's media

Found 2026-08-10, chasing the owner's question about the round-trip contradiction:
*"none of it explains why the asr is claiming a word that isnt there, and ignoring a word
that is"*. Neither of the two mechanisms proposed for that could. The answer was that the
model **was not listening to those files at all**.

**Mechanism.** Multimodal content is not in the token IDs — the processor writes
placeholder pads and the content arrives as embeddings. The radix prefix cache keys on
token IDs. sglang's guard is per-item content hashing; `MultimodalDataItem`'s docstring:
*"Each item has its own hash and pad_value, enabling per-image RadixAttention caching"*,
and `set_pad_value()` opens with:

```python
if self.pad_value is not None:
    return
```

Our processor assigns a CONSTANT pad_value before sglang can derive one, on all three
modalities, so `hash` is never computed and every item of a modality shares one pad value:

```
processors/longcat_next.py:318   img_item.pad_value   = self.image_pad_token_id
processors/longcat_next.py:375   vid_item.pad_value   = self.image_pad_token_id
processors/longcat_next.py:437   audio_item.pad_value = self._audio_safe_pad
```

Same prompt + same media token count => identical input_ids => the later request is served
the earlier one's KV, i.e. ANOTHER REQUEST'S MEDIA.

**Proof** (`test/probe_mm_cache.py`, two byte-identical-length 3s cuts of the reference
recording, different content):

```
phase 1, IDENTICAL prompt      A/B/B/A -> all four: 'Self test all systems'
phase 2, UNIQUE prompt prefix  A -> 'It was impossible to tell.'
                               B -> '"was of short duration. My fears were."'  (x2, stable)
                               A -> '"Me, it was impossible to tell when."'
```

Phase 1 returns content that is in NEITHER clip — the reference is a woman reading a
story; "Self test all systems" is the TTS text from EARLIER requests in the same session.
Stale cross-request content, not merely a mixup between A and B. Phase 2 breaks the shared
prefix and immediately yields distinct, clip-appropriate transcripts.

**Consequences.**
1. The entire round-trip analysis is void — every transcript may have been served from a
   previous request. The owner's 2/2 "contradiction" was the instrument reading a
   different file, not disagreeing about the same one.
2. This is a LIVE correctness bug for image, video, and audio understanding, not a test
   artifact. Repeated identical prompts over different media — an ordinary agent pattern —
   return stale answers. Cross-request content bleed also has a privacy dimension.
3. selftest could never catch it: every image it generates is an apple, so a stale answer
   and a correct one are indistinguishable. A test whose inputs do not VARY cannot detect
   a bug that returns the wrong input.

**Not yet fixed.** The fix is to stop pre-setting `pad_value` and let `set_pad_value()`
hash the feature — but `pad_input_ids` rewrites the placeholder tokens to `pad_value`, and
the model's placeholder detection currently assumes the constant, so both sides must move
together. Rebuild + full battery + human validation required. Radix cache can be disabled
(`LCN_RADIX=0`) as an immediate mitigation at the cost of the warm-prefix win that makes
agentic clients responsive.

### Both fixes VALIDATED on hardware (2026-08-10, image `v0516-mmhash-tools`)

Owner's ruling on the mitigation option — *"refusing prompt caching is not a solution, it
is a cop out"* — so the cache was fixed rather than disabled. Radix stays ON.

**1. Multimodal cache collision — FIXED.** Image probe, IDENTICAL prompt, the exact case
that previously returned clip A's description four times:

```
A -> 'A white circle centered on a red background.'
B -> 'A black square is centered on a solid green background.'   (was: white circle on red)
B -> 'A black square is centered on a solid green background.'
A -> 'A white circle is centered on a solid red background.'
```

Audio likewise: phase 1 (shared prefix) now matches phase 2 (broken prefix), each clip
returning its own content instead of stale text from earlier requests. The offsets backstop
logged ZERO warnings across the whole battery, so every hashed pad was covered by its
item's offsets and none needed the fallback. Identical media still hashes identically and
still hits the cache — only collisions between DIFFERENT media stop.

**2. Tool-call syntax 4 — FIXED.** `tool_calling` passed 4/4 on the build where it had just
failed, each parsing to `get_weather{"city": "Tokyo"}`. The offline parser regression
(`test/test_tool_parsing.py`) covers all four dialects plus two negative cases and runs
without a GPU, so this specific regression can never again require a server to detect.

**Battery:** parser 8/8, selftest 7/7 x4, anthropic 5/5, degeneracy 6/6. MemAvailable
30.4 GB after load.

**Two lessons, both structural rather than incidental:**
* *A test whose inputs never vary cannot detect a bug that returns the wrong input.*
  selftest asks about an apple it just generated, every run — so it would have passed 7/7
  indefinitely while the server described images nobody sent. Test inputs must VARY across
  runs to have any power against a wrong-content bug.
* *Report raw material, not verdicts.* Both defects were found by dumping what the model
  actually emitted or received. The tool-call dialect was invisible behind "no tool_calls
  parsed" for two builds; the cache bug was invisible behind an argument over which of two
  contradictory transcripts was correct — the answer being that neither clip was read.

**Incidental TTS signal** (unlabelled, consistent-with only): 3 of 4 renders in this battery
had `trail_ms=0` at cps 6.4/6.5/7.9 — slow renders with no trailing silence, the predicted
truncation signature. The owner's blind labels on the 8 adjudication clips remain the gate.

### ⚠ Retroactive caveat: past conclusions drawn from server-side transcription are suspect

The cache collision was present for the whole campaign, not just today. Any earlier finding
that used THIS SERVER to transcribe or describe media — especially batches issued
back-to-back with an identical prompt, which is the worst case — may have been reading a
different file than the one submitted.

Known to be affected in kind (flagged, not re-litigated here):
* The 2026-07-31 v0.5.12-vs-v0.5.16 TTS comparison recorded in ROADMAP §1 — but ONLY its
  transcript-derived line: *"their transcripts STOP AT THE FIRST CLAUSE"*. That came from
  this path and is no longer sound evidence.

  **CORRECTION (owner, 2026-08-10): _"My ears heard these clips, not asr."_** The
  OWNER EAR VERDICT in that same section — *"every v512 clip SKIPS the word 'Self'
  entirely"* — is human adjudication and is NOT affected; an earlier draft of this caveat
  wrongly swept it in. The clip durations (2.2-3.4s vs 5.1-7.3s) are direct measurements
  and stand too. So the v512 content-loss finding rests on ears and a stopwatch, both
  sound, and it independently corroborates the acoustic-sampler truncation re-derived
  today. Only the ASR transcript line needs re-taking.
* Today's `tts_roundtrip` run (already marked discredited above, for what turns out to be
  the wrong reason — the transcriber was not biased, it was reading other files).

What is NOT affected: anything adjudicated by the owner's own eyes and ears, anything
measured from the waveform or pixels directly (`media_stats`, segment structure), and the
code-level findings. This is precisely why the standing rule puts owner labels above
model-generated judgments — the rule held here even though the reason it held was one
nobody had guessed.

Re-running any affected comparison is cheap now that the fix is in; none is currently
load-bearing for an open decision, so this is recorded rather than actioned.

## Batched prefill corrupts concurrent multimodal requests (2026-08-10) — PARTIALLY FIXED

Investigated at the owner's direction after a deferred note about offsets being used as
direct indices into the batch-flattened tensor.

**It is a real, active bug, not a latent risk.** `_get_mm_items` flattened every request's
items into one list, discarding which request each came from; `_replace_mm_embeddings`
then used request-relative offsets as direct indices into the batch-flattened EXTEND
region, reading `extend_prefix_lens_cpu[0]` — the FIRST request's prefix — for every item
and adding no per-request base at all.

With ONE request in flight the base is 0 and index 0 is the right request, so the buggy
and correct code agree exactly. **Every test in this project issues one request at a time,
so the bug is invisible to the suite by construction.** A server whose entire value is
serving many clients had only ever been tested serving one.

**Measured, identical images and prompts, only batching differing:** sequential 3/3,
concurrent 1/3, with the damaged requests collapsing to a one-token reply.

**Fix applied** (`f05d5e9`): items carry their request index; the scatter subtracts that
request's own prefix and adds its start within the flattened batch. The hashed-pad
remapping was also switched from offset arithmetic to matching on `pad_value`, which is
globally unique now that it is content-derived and so needs no position arithmetic at all.

**Result: improved but NOT resolved.** Post-fix, sequential 6/6 and concurrent 4/6 — the
failure went from systematic to intermittent (round 1 concurrent 3/3, round 2 concurrent
1/3). Batch composition varies run to run (`#new-seq: 2 ... #running-req: 1` — three
requests fired, two batched), which fits an intermittent presentation.

**Do not read the improvement as a fix.** The residual failure text is the lead:
`'Aplain the design principles behind this image and why it might be effective.'` — a
coherent continuation of a DIFFERENT prompt, i.e. the model holds an image and answers a
question nobody asked. That is context bleeding between batched sequences, not scrambled
embeddings, which points away from the offsets and toward per-request state elsewhere.
`LCN_NGRAM=1` is on and the n-gram embedding manager carries per-request history; an
`LCN_NGRAM=0` arm is in flight to isolate it.

**Method note — a probe that exonerated the bug it was built to find.** The first
concurrency probe ran CONCURRENT then SEQUENTIAL over the same images and prompt. The
concurrent round corrupted the answers and CACHED them; the sequential "control" replayed
the cache. Both arms scored 2/6, which by the probe's own stated criterion means
"concurrency is innocent" — and that reading was reported before the confound was caught.
Reversing the order only moves the confound (a correct cache makes the concurrent arm a
pure cache hit that never prefills). `test/probe_mm_concurrency.py` now gives each arm its
own cold image variant per round; variants resize the shape rather than recolour it, since
a re-tinted "red" is arguably orange and would fail the colour check, making a false BAD
indistinguishable from real corruption.

## Five verified fixes land; concurrent multimodal corruption resolved (2026-08-10)

Build `v0516-mmfix5`. Four defects came from a Codex audit and were each independently
verified here before any code was written; the fifth (`pad_mask`/chunk interaction) was
found while verifying the others.

**Validation, with power this time:**

| check | result |
|---|---|
| concurrency probe, 10 rounds | sequential 30/30, **concurrent 30/30** |
| selftest x2 | 7/7 each |
| tool parser (offline) | 8/8 |
| audio cache probe | each clip returns its own content in BOTH phases |
| anthropic / degeneracy | 5/5, 6/6 |
| MemAvailable after load | 30.3 GB (unchanged) |

Pre-fix concurrent failure rate was 2 in 18 (~11%), so 30/30 clean is a ~3% outcome if
unfixed. This is the first concurrency result in this campaign with enough samples to mean
anything — a 12/12 clean run earlier was read as evidence and was not.

**The root cause, and why it hid.** `accept_tokens` is `predict[accept_index].flatten()`
over an `(bs, draft_token_num)` index — STRIDED WITH PADDING — while `update_token_table`
reads CONTIGUOUSLY by cumulative `req_lens`. Request i was read from `sum(accept_lens[:i])`
while its data sat at `i*stride`. Those agree only while every earlier request accepted the
full stride, so **request 0 is always correct** and later requests corrupt as soon as one
accepts short. Single-request-safe, intermittent under concurrency, and invisible to a
serial test suite.

**The pattern across all four audit findings is one assumption, four times:** correct for
the first or only request, wrong for the rest — half-open offsets, `extend_prefix_lens_cpu[0]`,
the flattened item list that discarded request identity, and the strided token write. Not
four coincidences; a codebase built and tested one request at a time, in which "the batch"
was never real to the code being written.

**NOT validated by this run, stated plainly:**
* The chunk-truncation and orphaned-generation-state fixes are UNEXERCISED. Their warnings
  logged zero times because those conditions never arose — no aborted generation, no chunk
  boundary inside a media item. Reasoned and code-verified, but untested. Reading zero
  warnings as success would repeat the exact error made earlier today with the hashed-pad
  backstop, whose silence meant it never ran at all.
* selftest passing does NOT prove the inclusive-offset fix is correct, only that it did not
  regress. A clobbered end marker degrades quality subtly rather than failing a check;
  confirming it directly needs instrumentation on the actual input_ids.
* A proper cross-chunk media fix is still owed — the current change makes truncation
  deterministic and loud, not absent.
