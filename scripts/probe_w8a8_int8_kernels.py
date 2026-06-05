#!/usr/bin/env python3
"""Phase-0 kernel-viability gate for the w8a8_int8 serving path on DGX Spark (GB10, sm_121).

Does NOT load the model. Exercises ONLY the two int8 kernels the w8a8_int8 path depends on,
with tiny tensors, to answer the single question that can kill the approach:
  do these kernels EXECUTE on sm_121, or do they raise "no kernel image for device"?

  (1) sgl_kernel.int8_scaled_mm        -> CUTLASS int8 GEMM (the Linear path, w8a8_int8.py:apply)
  (2) a trivial Triton kernel          -> proves Triton+ptxas targets sm_121 in this container
                                          (proxy for the Triton int8 FusedMoE runner, which is
                                           MoeRunnerBackend.TRITON per W8A8Int8MoEMethod)
Run inside the cu130 sglang container. Seconds, near-zero memory.
"""
import torch, traceback

print(f"[env] torch={torch.__version__} cap={torch.cuda.get_device_capability()} "
      f"dev={torch.cuda.get_device_name()}", flush=True)

# (1) CUTLASS int8 GEMM — same call shape as W8A8Int8LinearMethod.apply
try:
    from sgl_kernel import int8_scaled_mm
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
    M, K, N = 8, 1024, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    x_q, x_scale = per_token_quant_int8(x)             # dynamic per-token activation quant
    w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda").t()  # [K,N] COLUMN-major (matches process_weights_after_loading's .t())
    w_scale = torch.rand(N, 1, dtype=torch.float32, device="cuda")          # per-channel
    out = int8_scaled_mm(x_q.view(-1, K), w, x_scale.view(-1, 1), w_scale, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"[PASS] int8_scaled_mm (CUTLASS int8 GEMM): out={tuple(out.shape)} {out.dtype}", flush=True)
except Exception as e:
    print(f"[FAIL] int8_scaled_mm: {e!r}", flush=True); traceback.print_exc()

# (2) Triton compile+run on sm_121 (proxy for the int8 Triton MoE runner)
try:
    import triton, triton.language as tl
    @triton.jit
    def _add(xp, yp, op, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0); off = pid * BLOCK + tl.arange(0, BLOCK); m = off < n
        tl.store(op + off, tl.load(xp + off, mask=m) + tl.load(yp + off, mask=m), mask=m)
    n = 8192
    a = torch.randn(n, device="cuda"); b = torch.randn(n, device="cuda"); o = torch.empty_like(a)
    _add[(triton.cdiv(n, 256),)](a, b, o, n, BLOCK=256); torch.cuda.synchronize()
    err = float((o - (a + b)).abs().max())
    print(f"[PASS] triton compile+run on sm_121: max_err={err:.2e}", flush=True)
except Exception as e:
    print(f"[FAIL] triton: {e!r}", flush=True); traceback.print_exc()

# (3) Informational: confirm the int8 MoE method + Triton runner import cleanly
try:
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8MoEMethod, W8A8Int8Config
    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
    print(f"[INFO] W8A8Int8MoEMethod present; MoeRunnerBackend.TRITON={MoeRunnerBackend.TRITON}", flush=True)
except Exception as e:
    print(f"[INFO] moe import: {e!r}", flush=True)

print("PROBE DONE", flush=True)
