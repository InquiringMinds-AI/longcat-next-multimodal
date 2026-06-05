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

An 8-bit (`w8a8_int8`) quantization of Meituan **LongCat-Next**, packaged to serve **all
modalities on a single NVIDIA DGX Spark (GB10, `sm_121`)** through one SGLang process:
text generation, image understanding & generation, audio understanding & voice-clone generation,
and video understanding. Serving code, Dockerfile, and an OpenAI-compatible gateway:
**<GITHUB_REPO>**.

## What's in this repo
Self-contained (~90 GB) — nothing else to download:
- `model-*.safetensors` — backbone quantized to **w8a8_int8** (per-channel int8 weights + per-token
  int8 activations) for the MoE experts and the n-gram over-embedding; attention/router/heads/
  tokenizers/decoders kept higher-precision. SmoothQuant applied to the gate/up projections.
- tokenizers + configs, the **image decoder** (`image_decoder/`) and **audio vocoder**
  (`cosy24k_vocoder/`) needed for generation. Config weight-paths resolve in-directory.

## Why 8-bit
8-bit is the floor for coherent generation on this model: 4-bit (NVFP4) collapses both image
(tiling/abstract) and audio (drone). 8-bit produces faithful images and intelligible voice-clone
audio. Operator-judged.

## Quantization & provenance
Quantized from the BF16 source with per-channel symmetric int8 (MoE experts + OE embedders),
SmoothQuant on gate/up. The scripts are in the serving repo under `quantize/` (reproducible).

## Hardware / serving
- **NVIDIA DGX Spark (GB10, `sm_121`)**; the cu130 SGLang base is required (only it compiles Triton
  for `sm_121`). Not expected to run unchanged on other GPUs.
- **Memory ceiling:** GB10 unified memory has a hard ~115 GB headless ceiling (crash-to-off beyond
  it). This config runs well under it; serve headless.
- Quickstart, OpenAI-compatible API, and per-modality examples are in the serving repo README.

## License
MIT (© Meituan). Bundled English demo voice is public-domain (LibriVox); Chinese demo voice is
Meituan's LongCat example clip (MIT). See the serving repo `LICENSE`.
