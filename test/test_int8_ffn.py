#!/usr/bin/env python3
"""Int8SlotFFN equivalence vs the reference einsum pair (random weights, GPU).

Not bit-identical by design — int8 quantization changes the math. This checks
the int8 path computes THE SAME FFN (correct slot mapping, correct transpose,
gelu in the right place) to within quantization error: relative error vs the
bf16 einsum should sit near the per-op quantization noise (~1e-2), and the
SLOT-SHUFFLED control must be much worse (proves the test can fail).

Run inside the serving image (no server, ~1GB):
  docker run --rm --gpus all --entrypoint python3 <image> \
      /workspace/scripts/test_int8_ffn.py
"""
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")


def ref_ffn(res, w1, w2, depth, ffn_scale, dim):
    x = torch.einsum('bld,tld->blt', res,
                     torch.reshape(w1, (ffn_scale * dim // depth, depth, dim)))
    x = torch.nn.functional.gelu(x)
    return torch.einsum('blt,dlt->bld', x,
                        torch.reshape(w2, (dim, depth, ffn_scale * dim // depth)))


def main():
    from sglang.srt.models.int8_head_ffn import Int8SlotFFN
    torch.manual_seed(0)
    fails = 0
    for (dim, ffn_scale, depth, B) in ((2048, 16, 8, 1), (2048, 16, 8, 4),
                                       (3072, 16, 8, 2), (3072, 16, 8, 8)):
        w1 = torch.randn(ffn_scale * dim, dim, dtype=torch.bfloat16, device="cuda") * 0.02
        w2 = torch.randn(dim, ffn_scale * dim, dtype=torch.bfloat16, device="cuda") * 0.02
        res = torch.randn(B, depth, dim, dtype=torch.bfloat16, device="cuda")
        ref = ref_ffn(res.float(), w1.float(), w2.float(), depth, ffn_scale, dim)
        ffn = Int8SlotFFN(w1, w2, depth, ffn_scale, dim)
        got = ffn.forward(res).float()
        err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        # negative control: shuffle slot order — must break badly
        ffn.w1_int8 = ffn.w1_int8[1:] + ffn.w1_int8[:1]
        ffn.s1 = ffn.s1[1:] + ffn.s1[:1]
        bad = ffn.forward(res).float()
        err_bad = (bad - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        ok = err < 0.05 and err_bad > 10 * err
        fails += 0 if ok else 1
        print(f"dim={dim} B={B}: relerr {err:.3e} (shuffled control {err_bad:.3e}) "
              f"{'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
