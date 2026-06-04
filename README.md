# LongCat-Next on DGX Spark — full-modality multimodal generation

Running Meituan's **LongCat-Next** (75B-total / ~A3B-active any-to-any multimodal MoE,
LongCat-Flash-Lite backbone + MLA attention + N-gram over-embedding, native discrete
RVQ tokenizers for vision & audio) **end-to-end on a single NVIDIA DGX Spark (GB10,
128 GB unified memory)** — every understanding *and* generation modality.

> **Milestone (2026-06-04).** This repository marks the point at which the *full feature
> set* was validated working on this hardware. Before this, it was an assumption and a
> hope. Now, on one GB10:
>
> | Capability | Status |
> |---|---|
> | Understanding — text / image / audio / video → text | ✅ working |
> | Text generation | ✅ working |
> | **Image generation** (text → image) | ✅ faithful for natural prompts (operator-judged) |
> | **Audio generation** (voice-clone speech synthesis) | ✅ intelligible, voice-consistent (operator-judged) |

## The key finding: precision was the lever

Generation looked *broken* for months — image gen produced "trippy abstract art,"
audio gen produced a "drone." We chased calibration, sampling (CFG, top-k, per-level
top-k, negative prompts), prompt format, and decode — all of which only changed the
*flavor* of the wrongness. A teacher-forcing probe then showed the conditional was
**sound** (fed correct history, the head's argmax reconstructed the source subject); the
free-run failure was the **rank-4 imprecision of 4-bit NVFP4 compounding into
autoregressive drift**.

**The fix:** load the original BF16 weights as **bitsandbytes int8** and run the model's
own `generate()`. At 8-bit the *same* pipeline yields faithful images and intelligible
voice-cloned speech. The entire "broken generation" era was a 4-bit precision artifact —
the image *tiling* and the audio *drone* were the **same** RVQ level-0 (coarse-layer)
collapse, exactly as the paper (arXiv 2603.27538, *Lexicalizing Modalities as Discrete
Tokens*) predicts for its shared RVQ-summation tokenizers.

## How to run (8-bit canonical path — the validated recipe)

Inside `lmsysorg/sglang:v0.5.12.post1-cu130` (aarch64), with the BF16 model at
`~/models/LongCat-Next`:

```bash
pip install --break-system-packages transformers==4.57.1 "huggingface-hub<1.0" \
            accelerate bitsandbytes==0.49.2 librosa soundfile torchcodec
pip uninstall -y --break-system-packages kernels    # 4.57.1 needs hub<1.0, which breaks kernels 0.14.1
```

- **Load:** `BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=[...heads,tokenizers,lm_head,router...])`,
  `device_map={"":0}` (fits on-GPU, no offload).
- **Quirks the scripts handle:** a `Qwen2RMSNorm`→`Qwen2_5_VLRMSNorm` shim; an SDPA
  varlen fallback patched into every `modular_longcat*` module (the container's
  `flash_attn` is `None`), including the lazily-loaded image refiner; `model.text_tokenizer = tok`.
- **Generate:** build the prompt (image-gen ends with `<longcat_img_start>`; audio-gen
  uses the spk_syn chat template ending in `<longcat_audiogen_start>`), then
  `model.generate(...)`. Decode via the model's built-in `decode_visual_ids_and_save` /
  `decode_audio_ids_and_save`.

See `scripts/q8_unified.py` for one load serving all five task types, and
`scripts/q8_imagegen.py` / `scripts/q8_audio.py` for the single-modality drivers.

## Hardware reality

- 8-bit footprint is **~110 GB allocated / ~114.6 GB reserved** — *right at* the GB10
  crash-to-off ceiling (~110–115 GB). Near-zero margin; safe for a single process, tight
  for sustained serving. (The BF16-kept multimodal modules — tokenizers, decoders, the
  282k-row embedding — are the consumers, not KV cache.)
- **MLA makes KV cache nearly free** (~16 KB/token: 14 layers × (kv_lora_rank 512 +
  rope 64) × 2B), so context is never the constraint here.
- Full **BF16 from disk is blocked** on GB10: accelerate `device_map` offload leaves
  ~7,870 expert weights on `meta` (disk-offload broken for this custom MoE). 8-bit
  on-GPU is the working substitute. See `scripts/bf16_imagegen.py` for the attempt.

## Layout

- `scripts/` — the **8-bit canonical path** (`q8_*.py`, the breakthrough) + the
  generation/decode machinery (`gen_*`, `persistent_*`, `decode_*`, `teacher_force_image.py`).
- `overlay/` — the earlier **sglang 4-bit NVFP4 port** modules + launchers (the original
  effort; superseded for *generation* by the 8-bit canonical path, but it carries the
  serving infrastructure — see "next").
- `calibration/` — streaming per-expert NF4 calibration on Spark (`stream_calibrate.py`)
  + sequence builders. (Calibration turned out **not** to be the generation lever;
  precision was. Kept for completeness.)

## Known residuals (not precision-related)

- Abstract / empty-background / single-object-on-blank prompts (e.g. "a red circle on
  white") still degrade — a composition/OOD issue, not precision.
- Background **text-bleed** (garbled signs/watermarks) survives 8-bit — a model/training trait.

## Next: real serving

`q8_unified.py` is a **capability proof-of-concept** (one load, route by modality), not a
server — no batching, concurrency, per-call sampling, or prefix-cache reuse. Those
features *are* a serving stack, and the `overlay/` sglang port already implements them
(continuous batching, RadixAttention prefix cache, async concurrency, per-request
`SamplingParams`, OpenAI API). The original port's generation looked dead because it ran
at 4-bit; now that precision is known to be the fix, the production path is **the sglang
port at 8-bit** — leveraging its serving infrastructure with the now-validated precision.

## Credit

LongCat-Next and its tokenizers © Meituan (MIT). sglang © the SGLang team (Apache-2.0).
This repo is the Spark bring-up, debugging, and the 8-bit generation recipe.
