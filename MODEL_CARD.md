---
license: mit
base_model: meituan-longcat/LongCat-Next
tags:
  - multimodal
  - any-to-any
  - image-generation
  - text-to-speech
  - w8a8_int8
  - dgx-spark
  - gb10
  - sglang
language:
  - en
  - zh
---

# LongCat-Next — w8a8_int8 for DGX Spark (GB10)

An 8-bit (`w8a8_int8`) quantization of Meituan **LongCat-Next**, packaged to serve **every
modality on a single NVIDIA GB10 system (`sm_121`)** through one SGLang process: text
generation, image understanding and generation, audio understanding and voice-clone
generation, and video understanding. Developed and validated on a DGX Spark.

Serving code, Dockerfile, and an OpenAI-compatible gateway:
**https://github.com/InquiringMinds-AI/longcat-next-multimodal**

## What's in this repo

Self-contained (~90 GB) — nothing else to download:

- `model-*-of-00015.safetensors` (15 shards) — the backbone quantized to **`w8a8_int8`**:
  per-channel symmetric int8 weights with per-token dynamic int8 activations on the MoE
  experts and the n-gram over-embedding. Attention, router, generation heads, tokenizers,
  and decoders are kept at higher precision.
- `model-smoothscale.safetensors` — the per-channel **SmoothQuant** scales (α=0.5) for the
  MoE gate/up projections. These are **load-bearing**: the int8 expert gate/up weights were
  re-quantized with the scales absorbed (`W·s`) and the runtime divides the matching
  activations by `smooth_scale`, so `(X/s)·(W·s) = X·W` holds at int8. `down_proj` untouched.
- The **image decoder** (`image_decoder/`) and **audio vocoder** (`cosy24k_vocoder/`)
  required for generation, plus tokenizers, configs, and the trust-remote-code modeling
  files. Config weight-paths resolve in-directory.

## Why 8-bit

Switching to 8-bit (`w8a8_int8`) is what made image and audio generation coherent: at 4-bit
(NVFP4) both collapsed — images tiled into abstract texture, audio into a drone — and at
8-bit they came out faithful. That's why this checkpoint is 8-bit.

We did not re-test 4-bit after the later generation-path fixes and optimizations, so 8-bit
is the *validated* setting, not a proven minimum — it's possible the corrected pipeline would
now work at 4-bit too. Understanding was robust at both precisions; generation was the
sensitive path.

## Quantization & provenance

Quantized from the BF16 source (`meituan-longcat/LongCat-Next`) with per-channel symmetric
int8 weights on the MoE experts and the n-gram over-embedders, and per-token dynamic int8
activations at runtime via SGLang's CUTLASS `w8a8_int8` path. The MoE gate/up projections
additionally use **SmoothQuant** (α=0.5) to migrate activation outliers into the weights
before int8 — see `model-smoothscale.safetensors` above. The full, reproducible export
scripts are in the serving repo under `quantize/` (`smoothquant_export.py` for the SmoothQuant pass).

## Hardware & serving

- **Any NVIDIA GB10 system.** Built for the GB10 superchip (`sm_121`). Validated on a DGX
  Spark, and **expected to run on any GB10-based machine** — the dependency is the chip, not
  the specific product. The cu130 SGLang base image is required (it's the one whose Triton
  compiles for `sm_121`); not expected to run on other GPUs.
- **Optimized for headless GB10 operation** at time of publishing (serve with the screen
  off / remote-only for maximum memory headroom).
- **Context: native 128k**, or **256k via YaRN** (`LCN_YARN=1`, opt-in). MLA keeps the KV cache
  cheap, so long context costs little memory.
- Quickstart, the OpenAI-compatible API, and per-modality examples are in the serving repo
  README.

## License

MIT (© Meituan). The bundled English demo voice is public-domain (LibriVox); the Chinese
demo voice is Meituan's LongCat example clip (MIT). See the serving repo `LICENSE`.
