# Reproducing the w8a8_int8 weights

These scripts rebuild the served weights from the **BF16 source** of Meituan LongCat-Next, so the
distributed checkpoint isn't an opaque blob. Run them inside the cu130 container (transformers etc.
already present). Order:

1. **`quantize_w8a8_int8.py`** — per-channel symmetric int8 of the MoE expert weights
   (gate/up/down), writing `weight` (int8) + `weight_scale` per output channel. Attention, router,
   heads, tokenizers, decoders, lm_head, n-gram embedding are left out of int8 (kept higher
   precision) via the `quantization_config.ignore` list in `config.json`.
2. **`int8_oe_embedders.py`** — int8 the n-gram over-embedding tables (`ngram_embeddings.embedders.*`),
   emitting a **1-D** `weight_scale` per row (the OE loader expects 1-D; the MoE loader expects
   `[N,1]` — keep the two conventions distinct).
3. **`reshape_scales.py`** — ensure MoE `weight_scale` tensors are `[N,1]` (2-D) for the fused-MoE
   loader's `copy_`.
4. **`smoothquant_export.py`** — SmoothQuant the gate/up projections: migrate per-channel activation
   outliers into the weights (`s = act_max^0.5 / weight_max^0.5`), re-quantize, and store
   `model.layers.L.mlp.smooth_scale` buffers (the runtime divides the expert input by `s`).
   Needs per-layer activation maxima (`/tmp/sq_actmax.pt`) from a calibration pass.

The image decoder + cosy24k vocoder are copied from the BF16 source unchanged (kept full precision).

> Note: SmoothQuant was applied but is **not** what makes generation coherent here (that was the
> serving gen-loop fixes); it's retained as harmless and is part of how the shipped weights were made.
