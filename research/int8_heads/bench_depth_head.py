#!/usr/bin/env python3
"""Depth-head micro-benchmark: batch (1/2/4/8) x dtype (bf16/int8), model UNLOADED.

The ROADMAP gate for the int8-heads item: size the win before building it.
Two measurements, both on random weights (timing depends on bytes, not values):

  1. REAL head forward (CasualDepthTransformerHead from the serving image) at
     batch 1/2/4/8, visual and audio configs — the bf16 anchor, including the
     attention/norm/einsum overhead a shape-level bench misses.
  2. Per-shape GEMV sweep: bf16 F.linear vs the int8 batched-GEMV Triton kernel
     (adapted from the DGX-Spark community INT8 LM Head v2 patch — per-channel
     runtime quantization, weight tile read ONCE per batch) on the head's three
     constituent shapes. Projected per-call gain = byte-weighted ratio.

Numerics sanity per shape: max |int8 - bf16| relative to the bf16 row scale —
NOT a quality verdict (that is the owner's paired A/B on real output), just a
check that the kernel computes the matmul it claims.

Run inside the serving image with the server DOWN (needs its own CUDA memory):
  docker run --rm --gpus all -v ~/longcat-outputs:/workspace/outputs \
      --entrypoint python3 longcat-next-gb10:v0516-multitts5 \
      /workspace/outputs/bench_depth_head.py
"""
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
WARMUP, ITERS = 10, 50

# --- int8 batched GEMV (adapted from patch_int8_lmhead.py v2, autotune trimmed) ---
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
    tl.store(out_ptr + 0 * stride_ob + rows, (acc0 * s).to(tl.float16), mask=rmask)
    if NUM_BATCH > 1:
        tl.store(out_ptr + 1 * stride_ob + rows, (acc1 * s).to(tl.float16), mask=rmask)
    if NUM_BATCH > 2:
        tl.store(out_ptr + 2 * stride_ob + rows, (acc2 * s).to(tl.float16), mask=rmask)
    if NUM_BATCH > 3:
        tl.store(out_ptr + 3 * stride_ob + rows, (acc3 * s).to(tl.float16), mask=rmask)


def quantize(w):
    scales = w.float().abs().amax(dim=1).clamp(min=1e-12) / 127.0
    w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return w_int8.contiguous(), scales.to(torch.float16).contiguous()


def int8_forward(w_int8, scales, x):
    """x [B,K] fp16 -> [B,M] fp16; batches of >4 run in chunks of 4."""
    M, K = w_int8.shape
    B = x.shape[0]
    out = torch.empty(B, M, dtype=torch.float16, device=DEV)
    grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
    for b0 in range(0, B, 4):
        nb = min(4, B - b0)
        _int8_gemv[grid](out[b0:], w_int8, x[b0:], scales, M, K,
                         out.stride(0), x.stride(0), NUM_BATCH=nb)
    return out


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1000  # ms


def timeit_cycle(fns):
    """Time fns cycled round-robin — each call sees a weight evicted from cache
    by the intervening reads. Returns ms per call."""
    n = len(fns)
    for i in range(min(WARMUP, n) * 2):
        fns[i % n]()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(ITERS):
        fns[i % n]()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1000


def bench_shapes(tag, shapes):
    """shapes: list of (label, M, K, calls_per_head_call).

    COLD-READ timing: the first version of this sweep timed one weight tensor in
    a tight loop, and small weights (8-17MB) stayed resident in cache — several
    bf16 numbers implied >800 GB/s, impossible for LPDDR5x. Production streams
    the WHOLE head (1.3-2.9GB) per call, so every weight read is cold. Each
    shape now cycles through enough independent weight copies (>=512MB per
    dtype arm) that consecutive calls cannot hit cache."""
    print(f"\n=== {tag}: per-shape GEMV sweep, COLD reads (ms per head-call worth) ===")
    total = {}
    for label, M, K, ncalls in shapes:
        bytes_bf = M * K * 2
        ncopies = max(2, (512 << 20) // bytes_bf + 1)
        ws = [torch.randn(M, K, dtype=torch.bfloat16, device=DEV) for _ in range(ncopies)]
        qs = [quantize(w) for w in ws]
        for B in (1, 2, 4, 8):
            x16 = torch.randn(B, K, dtype=torch.float16, device=DEV)
            xbf = x16.to(torch.bfloat16)
            t_bf = timeit_cycle([
                (lambda w=w: torch.nn.functional.linear(xbf, w)) for w in ws]) * ncalls
            t_i8 = timeit_cycle([
                (lambda wi=wi, s=s: int8_forward(wi, s, x16)) for wi, s in qs]) * ncalls
            # numerics sanity (not a quality verdict)
            ref = torch.nn.functional.linear(x16.float(), ws[0].float())
            got = int8_forward(qs[0][0], qs[0][1], x16).float()
            err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
            key = B
            a, b = total.get(key, (0.0, 0.0))
            total[key] = (a + t_bf, b + t_i8)
            gbps = bytes_bf * ncalls / (t_bf / 1000) / 1e9
            print(f"  {label:22s} B={B}  bf16 {t_bf:7.3f} ({gbps:5.0f}GB/s)  "
                  f"int8 {t_i8:7.3f}  x{t_bf/max(t_i8,1e-9):4.2f}  relerr {err:.3e}")
        del ws, qs
        torch.cuda.empty_cache()
    for B in (1, 2, 4, 8):
        a, b = total[B]
        print(f"  {tag} TOTAL B={B}: bf16 {a:.3f} ms  int8 {b:.3f} ms  x{a/b:.2f}")


def bench_real_head(tag, hidden_size, dim, ffn_scale, layers, codebook_sizes):
    from sglang.srt.models.longcat_next_heads import CasualDepthTransformerHead
    head = CasualDepthTransformerHead(hidden_size, codebook_sizes, layers, dim,
                                      ffn_scale).to(DEV, torch.bfloat16).eval()
    depth = len(codebook_sizes)
    emb = torch.randn(max(codebook_sizes) + 1, hidden_size,
                      dtype=torch.bfloat16, device=DEV)
    emb_fn = lambda ids: emb[ids]
    print(f"\n=== {tag}: REAL head forward (bf16 anchor), ms/call ===")
    for B in (1, 2, 4, 8):
        x = torch.randn(B, hidden_size, dtype=torch.bfloat16, device=DEV)
        toks = torch.randint(0, min(codebook_sizes), (B, depth - 1), device=DEV)
        with torch.no_grad():
            t = timeit(lambda: head(x, toks, emb_fn, depth - 1))
        print(f"  B={B}: {t:.3f} ms/call  ({t*depth:.1f} ms/frame at {depth} levels)")
    del head, emb
    torch.cuda.empty_cache()


def main():
    print(torch.cuda.get_device_name(0))

    # Visual: dim 2048, ffn x16, 4 layers, depth 8, vq 16384
    # Per head-call: 4 layers x (4 attn projs + 8+8 FFN slot GEMVs) + 1 head slice
    bench_real_head("visual", 3072, 2048, 16, 4, [16384] * 8)
    bench_shapes("visual", [
        ("attn 2048x2048 x16", 2048, 2048, 16),      # 4 projs x 4 layers
        ("ffn-up 4096x2048 x32", 4096, 2048, 32),    # 8 slots x 4 layers
        ("ffn-dn 2048x4096 x32", 2048, 4096, 32),
        ("head 16385x2048 x1", 16385, 2048, 1),
    ])

    # Audio: dim 3072, ffn x16, 4 layers, depth 8, mixed vq
    bench_real_head("audio", 3072, 3072, 16, 4,
                    [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024])
    bench_shapes("audio", [
        ("attn 3072x3072 x16", 3072, 3072, 16),
        ("ffn-up 6144x3072 x32", 6144, 3072, 32),
        ("ffn-dn 3072x6144 x32", 3072, 6144, 32),
        ("head 8193x3072 x1", 8193, 3072, 1),
    ])


if __name__ == "__main__":
    main()
