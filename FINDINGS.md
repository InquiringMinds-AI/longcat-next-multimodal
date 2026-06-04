# Findings — the road to working generation

The debugging arc that led to "precision is the lever," recorded so the next phase
doesn't re-chase eliminated hypotheses.

## What was wrong, and the order we ruled things out

Free-run generation at 4-bit NVFP4 produced color/coarse-correct but structurally-wrong
output: images "tiled" into abstract texture; audio collapsed to a "drone." Investigation
eliminated, in turn:

1. **Decode stack** — an oracle round-trip (real image → encode → decode) reconstructed a
   clean image. Decoder/refiner/VQ codebooks sound. (`decode_roundtrip.py`)
2. **Positions / KV / newline cadence** — instrumented the gen loop: positions advance
   monotonically, hidden states vary (cosine 0.85–0.98), newline handling correct.
3. **Feedback embedding** — numerically identical to canonical (verified).
4. **Sampling** — canonical params; rep-penalty 1.0 is a no-op.
5. **Heads** — `visual_head`/`audio_head` are full-precision in the checkpoint (0 quant
   markers). Not quantized.
6. **Calibration** — built format-matched **and** content-matched NVFP4 calibrations
   (document pages → grayscale output; food/colorful images → beige output: calibration
   demonstrably controls **palette**). It did **not** fix structure. Calibration is *not*
   the generation lever. (Also discovered the old `imgcalib` checkpoint was a no-op —
   byte-identical to base scales.)
7. **Sampling knobs** — tight top-k, per-level top-k (L0-greedy), CFG sweeps, negative
   prompts. Each changed the *flavor* of the wrongness (less duplication, different
   texture), never the correctness. Operator verdict on the best of these: "trippy; none
   any less off."

## The teacher-forcing probe (the turn)

`teacher_force_image.py`: fed a real image's `[324,8]` codes as history and measured the
depth head's argmax vs the real next token (vocab 16384/level, chance 0.006%):

- Prompt-matched red circle: **L0 top-1 20.7%, correct token median rank 4**.
- Decoding the teacher-forced argmax **recovered the source subject** (plane→plane,
  toad→toad) — even at 8–20% exact top-1 — because the RVQ summation + flow-matching
  decoder tolerate neighborhood-level (rank ~4) error.

⇒ The conditional + head are **sound given correct history**. The free-run failure is
**autoregressive drift / exposure bias**: own slightly-off tokens fed back compound over
324×8 steps. Not content-blindness, not a peripheral bug — a **sharpness** deficit in the
4-bit conditional.

## The fix and its confirmation

Loading the original BF16 as **bnb int8** and running canonical `model.generate()`:

- **Image** (operator-judged): toad, cat, apple, mountain landscape all faithful and
  recognizable — "much better, very impressive." The red-circle abstract case still fails
  (separate composition/OOD problem).
- **Audio** (operator-judged): three voice-clone syntheses, "all three sound like the same
  female speaker, perfect enunciation, naturalistic prosody, less accented than the 4-bit
  attempt." The text stream recited each target exactly.

Same precision lever fixed *both* — confirming the shared-RVQ-base theory: image tiling and
audio drone were one defect (coarse-layer collapse), crossed by 8-bit.

## Dead ends worth not repeating

- **Full BF16 from disk**: accelerate `device_map="auto"` + offload on GB10 leaves ~7,870
  expert weights on `meta` (no data) — disk offload is broken for this custom MoE. Also
  won't fit RAM (156 GB > 121 GB). 8-bit on-GPU is the substitute.
- **6-bit "middle ground"**: bitsandbytes only does 4-bit and 8-bit (no 5/6). True 6-bit
  (modelopt MXFP6) is unproven on sm_121 and the size win is limited because the
  precision-sensitive part is the *expert bulk*. Worth it only to pull off the crash
  ceiling, not for cache (MLA makes cache free).
- **Calibration as a quality lever** for generation: it isn't. It controls palette, not
  structure. Precision does.

## Standing methodological note

The operator judges correctness on generative output. Self-grading led to over-claiming
twice (an audio clip and an image batch both read as "working" before the operator's eyes
corrected them). Report stats + the artifact; let the human call it.
