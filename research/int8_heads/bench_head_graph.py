#!/usr/bin/env python3
"""Launch-tax remedy bench: eager vs CUDA-graphed vs torch.compiled head forward.

The 2026-08-14 timeline analysis showed the generation step is launch-latency
bound (~4.6k launches, 46.5ms/step idle, mostly inside the depth head's
8-level x 4-layer loop). The head is shape-static per level, so the launch tax
should be erasable. This measures the ceiling OFFLINE before any serving work,
at B=1 (the production shape), int8 FFN attached (production config):

  eager    — as shipped
  graph    — manual torch.cuda.CUDAGraph capture of one level call, replayed
             with inputs copied into static buffers (deployment-faithful)
  compile  — torch.compile(mode="reduce-overhead") as the low-code alternative

Known capture hazard, measured not assumed: FlashVarLenAttention builds its
seqlens tensor host-side EVERY call (an H2D pageable copy — also one of the
562 memcpys/step in the trace); if capture fails on it, that line is the
first thing a real implementation must hoist.

Run inside the serving image, server DOWN (~5GB):
  docker run --rm --gpus all -v ~/longcat-outputs:/workspace/outputs \
      --entrypoint python3 <image> /workspace/outputs/bench_head_graph.py
"""
import time

import torch

DEV = "cuda"
WARMUP, ITERS = 20, 100


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1000


def check_attn_equivalence():
    """Dense-SDPA fast path vs the varlen path on the same module + data."""
    from sglang.srt.models.longcat_next_heads import FlashVarLenAttention
    torch.manual_seed(0)
    for (dim, heads, depth, nb) in ((2048, 16, 8, 1), (3072, 24, 8, 4)):
        m = FlashVarLenAttention(dim, heads, causal=True).to(DEV, torch.bfloat16).eval()
        h = torch.randn(nb * depth, dim, dtype=torch.bfloat16, device=DEV)
        seq = torch.full((nb,), depth, dtype=torch.int32, device=DEV)
        cu = torch.arange(0, nb + 1, device=DEV, dtype=torch.int32) * depth
        with torch.no_grad():
            ref = m(h, seq_len=seq)
            fast = m(h, cu_len=cu, max_seqlen=depth)
        err = (fast.float() - ref.float()).abs().max().item() / \
            max(ref.float().abs().max().item(), 1e-9)
        print(f"attn equivalence dim={dim} nb={nb}: relerr {err:.3e} "
              f"{'PASS' if err < 2e-2 else 'FAIL'}")
        assert err < 2e-2, "dense fast path diverges from varlen"


def bench(tag, hidden_size, dim, ffn_scale, layers, codebook_sizes):
    from sglang.srt.models.longcat_next_heads import CasualDepthTransformerHead
    from sglang.srt.models.int8_head_ffn import attach_int8_ffn
    depth = len(codebook_sizes)
    head = CasualDepthTransformerHead(hidden_size, codebook_sizes, layers, dim,
                                      ffn_scale).to(DEV, torch.bfloat16).eval()
    attach_int8_ffn(head, depth)
    emb = torch.randn(max(codebook_sizes) + 1, hidden_size,
                      dtype=torch.bfloat16, device=DEV)
    emb_fn = lambda ids: emb[ids]
    x = torch.randn(1, hidden_size, dtype=torch.bfloat16, device=DEV)
    toks = torch.randint(0, min(codebook_sizes), (1, depth - 1), device=DEV)
    level = depth - 1

    print(f"\n=== {tag} (B=1, int8 FFN attached) ===")
    with torch.no_grad():
        t_eager = timeit(lambda: head(x, toks, emb_fn, level))
    print(f"  eager   : {t_eager:7.3f} ms/call  ({t_eager*depth:6.1f} ms/frame)")

    # --- manual CUDA graph ---
    try:
        with torch.no_grad():
            sx = x.clone()
            stoks = toks.clone()
            for _ in range(3):  # autotune + allocator warmup on a side stream
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    head(sx, stoks, emb_fn, level)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = head(sx, stoks, emb_fn, level)

            def run_graph():
                sx.copy_(x, non_blocking=True)
                stoks.copy_(toks, non_blocking=True)
                g.replay()
                return out
            t_graph = timeit(run_graph)
        print(f"  graph   : {t_graph:7.3f} ms/call  ({t_graph*depth:6.1f} ms/frame)"
              f"  x{t_eager/t_graph:4.2f}")
    except Exception as e:
        print(f"  graph   : CAPTURE FAILED — {type(e).__name__}: {str(e)[:160]}")

    # --- torch.compile reduce-overhead ---
    try:
        compiled = torch.compile(head, mode="reduce-overhead", dynamic=False)
        with torch.no_grad():
            t_comp = timeit(lambda: compiled(x, toks, emb_fn, level))
        print(f"  compile : {t_comp:7.3f} ms/call  ({t_comp*depth:6.1f} ms/frame)"
              f"  x{t_eager/t_comp:4.2f}")
    except Exception as e:
        print(f"  compile : FAILED — {type(e).__name__}: {str(e)[:160]}")

    del head, emb
    torch.cuda.empty_cache()


def main():
    print(torch.cuda.get_device_name(0))
    check_attn_equivalence()
    bench("visual", 3072, 2048, 16, 4, [16384] * 8)
    bench("audio", 3072, 3072, 16, 4,
          [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024])


if __name__ == "__main__":
    main()
