"""INT8 per-slot GEMV path for the depth-head FFN (LCN_INT8_HEADS=1).

Scope, set by measurement (research/int8_heads/bench_depth_head.py, cold reads,
GB10 2026-08-14): the depth head is the frame-time majority on both generation
paths (visual ~52ms of 174ms/frame, audio ~135ms of 177ms/frame at batch 1),
and int8 GEMV beats bf16 ~2.3x at 1-4 rows but LOSES at 8+ rows (x0.82). The
attention projections run at rows = batch x depth(8) — quantizing them is a
loss — while the FFN einsum is per-depth-slot matmuls at rows = batch, and
holds ~89% of the transformer-layer bytes. So ONLY the FFN weights (linear1/
linear2 of each CasualDepthTransformerLayer) are quantized; attention and the
per-level output heads stay BF16, which is also where the quality risk was.

Quantization is per-output-channel (symmetric, runtime, no calibration), the
scheme proven top-k-preserving on this box by the community INT8 LM Head v2
patch for Qwen3.5-122B, whose batched-GEMV kernel this adapts: the weight tile
is read ONCE per launch and reused for up to 4 batch rows. The bf16 FFN
weights are freed after quantization (~1.7GB back across both heads).

Faithfulness note: the einsum path never adds linear1/linear2 .bias (the
checkpoint ships biases the reference forward ignores) — this path ignores
them identically.
"""
import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({'BLOCK_M': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=['M', 'K', 'NUM_BATCH'])
@triton.jit
def _int8_gemv(out_ptr, w_ptr, x_ptr, s_ptr, M, K, stride_ob, stride_xb,
               NUM_BATCH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for ks in range(0, K, BLOCK_K):
        co = ks + tl.arange(0, BLOCK_K)
        km = co < K
        w = tl.load(w_ptr + rows[:, None] * K + co[None, :],
                    mask=rmask[:, None] & km[None, :], other=0).to(tl.float32)
        x0 = tl.load(x_ptr + 0 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
        acc0 += tl.sum(w * x0[None, :], axis=1)
        if NUM_BATCH > 1:
            x1 = tl.load(x_ptr + 1 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
            acc1 += tl.sum(w * x1[None, :], axis=1)
        if NUM_BATCH > 2:
            x2 = tl.load(x_ptr + 2 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
            acc2 += tl.sum(w * x2[None, :], axis=1)
        if NUM_BATCH > 3:
            x3 = tl.load(x_ptr + 3 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
            acc3 += tl.sum(w * x3[None, :], axis=1)
    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
    tl.store(out_ptr + 0 * stride_ob + rows, (acc0 * s).to(tl.float32), mask=rmask)
    if NUM_BATCH > 1:
        tl.store(out_ptr + 1 * stride_ob + rows, (acc1 * s).to(tl.float32), mask=rmask)
    if NUM_BATCH > 2:
        tl.store(out_ptr + 2 * stride_ob + rows, (acc2 * s).to(tl.float32), mask=rmask)
    if NUM_BATCH > 3:
        tl.store(out_ptr + 3 * stride_ob + rows, (acc3 * s).to(tl.float32), mask=rmask)


class Int8SlotFFN:
    """Holds per-depth-slot int8 weights for one layer's linear1+linear2."""

    def __init__(self, w1, w2, depth, ffn_scale, dim):
        # w1 [ffn_scale*dim, dim] -> slots [depth, t, dim], t = ffn_scale*dim/depth
        # (the einsum's reshape: (t, depth, dim) with slot as the MIDDLE axis)
        t = ffn_scale * dim // depth
        w1r = w1.reshape(t, depth, dim)
        w2r = w2.reshape(dim, depth, t)
        self.depth, self.t, self.dim = depth, t, dim
        self.w1_int8, self.s1 = [], []
        self.w2_int8, self.s2 = [], []
        for l in range(depth):
            for (src, wl, sl) in ((w1r[:, l, :], self.w1_int8, self.s1),
                                  (w2r[:, l, :], self.w2_int8, self.s2)):
                w = src.contiguous().float()
                s = w.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                wl.append((w / s.unsqueeze(1)).round().clamp(-127, 127)
                          .to(torch.int8).contiguous())
                sl.append(s.to(torch.float16).contiguous())

    def _gemv(self, w_int8, scales, x):
        """x [B,K] fp32 -> [B,M] fp32."""
        M, K = w_int8.shape
        B = x.shape[0]
        x16 = x.to(torch.float16)
        out = torch.empty(B, M, dtype=torch.float32, device=x.device)
        grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
        for b0 in range(0, B, 4):
            nb = min(4, B - b0)
            _int8_gemv[grid](out[b0:], w_int8, x16[b0:], scales, M, K,
                             out.stride(0), x16.stride(0), NUM_BATCH=nb)
        return out

    def forward(self, res):
        """res [B, depth, dim] (post-layernorm2) -> FFN output [B, depth, dim].

        Equivalent math to the einsum pair in CasualDepthTransformerLayer.forward:
          x = einsum('bld,tld->blt', res, W1r); gelu; einsum('blt,dlt->bld', x, W2r)
        computed slot-by-slot in int8 with fp32 activations.
        """
        B = res.shape[0]
        out = torch.empty(B, self.depth, self.dim,
                          dtype=torch.float32, device=res.device)
        resf = res.to(torch.float32)
        for l in range(self.depth):
            h = self._gemv(self.w1_int8[l], self.s1[l], resf[:, l, :])
            h = torch.nn.functional.gelu(h)
            out[:, l, :] = self._gemv(self.w2_int8[l], self.s2[l], h)
        return out.to(res.dtype)


def attach_int8_ffn(head, depth):
    """Quantize every transformer layer's FFN in-place and FREE the bf16 weights.

    Idempotent; call after weights are loaded. Returns bytes freed."""
    freed = 0
    for layer in head.transformer_layers:
        if getattr(layer, "_int8_ffn", None) is not None:
            continue
        w1, w2 = layer.linear1.weight.data, layer.linear2.weight.data
        layer._int8_ffn = Int8SlotFFN(w1, w2, depth,
                                      layer.transformer_ffn_scale,
                                      layer.transformer_dim)
        freed += w1.numel() * w1.element_size() + w2.numel() * w2.element_size()
        # int8 copies are half the bf16 bytes; freeing the originals nets the gain
        layer.linear1.weight.data = torch.empty(0, device=w1.device, dtype=w1.dtype)
        layer.linear2.weight.data = torch.empty(0, device=w2.device, dtype=w2.dtype)
    return freed
