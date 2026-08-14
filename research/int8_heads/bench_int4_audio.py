#!/usr/bin/env python3
"""int4 audio-FFN gate: speed (cold reads + real head) and numerics vs int8/bf16.

The question this answers (owner, 2026-08-14): can int4 on the audio head push
TTS generation under realtime (80ms/frame at 12.5fps)? Frame budget going in:
audio head ~68ms/frame at int8 (FFN ~44ms of it), backbone+host ~47ms.
int4 halves the FFN bytes again -> projected head ~46ms, frame ~93ms — CLOSE
but not under; this bench measures what actually materializes on GB10.

Numerics: relerr vs the bf16 einsum for int8 (per-channel) and int4 (g128).
Expect int4 ~2-4x the int8 error; whether that is AUDIBLE is the owner's
paired-clip call, not this script's.

Run inside the serving image, server DOWN (~4GB):
  docker run --rm --gpus all -v ~/longcat-outputs:/workspace/outputs \
      --entrypoint python3 <image> /workspace/outputs/bench_int4_audio.py
"""
import time

import torch

DEV = "cuda"
WARMUP, ITERS = 15, 60
DIM, FFN, DEPTH = 3072, 16, 8
CODEBOOKS = [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024]


def timeit_cycle(fns):
    n = len(fns)
    for i in range(min(WARMUP, n) * 2):
        fns[i % n]()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(ITERS):
        fns[i % n]()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1000


def ref_ffn(res, w1, w2):
    x = torch.einsum('bld,tld->blt', res,
                     torch.reshape(w1, (FFN * DIM // DEPTH, DEPTH, DIM)))
    x = torch.nn.functional.gelu(x)
    return torch.einsum('blt,dlt->bld', x,
                        torch.reshape(w2, (DIM, DEPTH, FFN * DIM // DEPTH)))


def main():
    from sglang.srt.models.int8_head_ffn import Int8SlotFFN, Int4SlotFFN
    print(torch.cuda.get_device_name(0))
    torch.manual_seed(0)

    # --- numerics on one layer ---
    w1 = torch.randn(FFN * DIM, DIM, dtype=torch.bfloat16, device=DEV) * 0.02
    w2 = torch.randn(DIM, FFN * DIM, dtype=torch.bfloat16, device=DEV) * 0.02
    res = torch.randn(1, DEPTH, DIM, dtype=torch.bfloat16, device=DEV)
    ref = ref_ffn(res.float(), w1.float(), w2.float())
    for name, cls in (("int8", Int8SlotFFN), ("int4-g128", Int4SlotFFN)):
        ffn = cls(w1, w2, DEPTH, FFN, DIM)
        got = ffn.forward(res).float()
        err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        print(f"FFN numerics {name:9s}: relerr {err:.3e}")

    # --- cold-read per-shape speed, B=1, ffn shapes only ---
    print("\ncold-read GEMV (ms per head-call worth, B=1):")
    for label, M, K, ncalls in (("ffn-up 6144x3072 x32", 6144, 3072, 32),
                                ("ffn-dn 3072x6144 x32", 3072, 6144, 32)):
        ncopies = max(2, (512 << 20) // (M * K * 2) + 1)
        ws = [torch.randn(M, K, dtype=torch.bfloat16, device=DEV) for _ in range(ncopies)]
        x = torch.randn(1, K, dtype=torch.float16, device=DEV)
        xb = x.to(torch.bfloat16)
        from sglang.srt.models.int8_head_ffn import (_int8_gemv,
                                                     _quantize_int4_grouped,
                                                     _int4_gemv)
        def q8(w):
            s = w.float().abs().amax(dim=1).clamp(min=1e-12) / 127.0
            return ((w.float() / s.unsqueeze(1)).round().clamp(-127, 127)
                    .to(torch.int8).contiguous(), s.to(torch.float16).contiguous())
        q8s = [q8(w) for w in ws]
        q4s = [_quantize_int4_grouped(w) for w in ws]
        grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
        o16 = torch.empty(1, M, dtype=torch.float16, device=DEV)
        o32 = torch.empty(1, M, dtype=torch.float32, device=DEV)
        t_bf = timeit_cycle([(lambda w=w: torch.nn.functional.linear(xb, w)) for w in ws]) * ncalls
        t_i8 = timeit_cycle([(lambda wq=wq, s=s: _int8_gemv[grid](
            o16, wq, x, s, M, K, o16.stride(0), x.stride(0), NUM_BATCH=1))
            for wq, s in q8s]) * ncalls
        t_i4 = timeit_cycle([(lambda wq=wq, s=s: _int4_gemv[grid](
            o32, wq, x, s, M, K, o32.stride(0), x.stride(0),
            NUM_BATCH=1, GROUP=128)) for wq, s in q4s]) * ncalls
        print(f"  {label}: bf16 {t_bf:6.2f}  int8 {t_i8:6.2f}  int4 {t_i4:6.2f}  "
              f"(int4 x{t_i8/max(t_i4,1e-9):.2f} vs int8)")
        del ws, q8s, q4s
        torch.cuda.empty_cache()

    # --- real audio head, ms/call at B=1: int8 vs int4 attached ---
    from sglang.srt.models.longcat_next_heads import CasualDepthTransformerHead
    from sglang.srt.models.int8_head_ffn import attach_int8_ffn, attach_int4_ffn
    for name, attach in (("int8", attach_int8_ffn), ("int4", attach_int4_ffn)):
        head = CasualDepthTransformerHead(3072, CODEBOOKS, 4, DIM, FFN
                                          ).to(DEV, torch.bfloat16).eval()
        attach(head, DEPTH)
        emb = torch.randn(max(CODEBOOKS) + 1, 3072, dtype=torch.bfloat16, device=DEV)
        emb_fn = lambda ids: emb[ids]
        x = torch.randn(1, 3072, dtype=torch.bfloat16, device=DEV)
        toks = torch.randint(0, min(CODEBOOKS), (1, DEPTH - 1), device=DEV)
        with torch.no_grad():
            t = timeit_cycle([lambda: head(x, toks, emb_fn, DEPTH - 1)])
        print(f"real audio head {name}: {t:.3f} ms/call  ({t*DEPTH:.1f} ms/frame)")
        del head, emb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
