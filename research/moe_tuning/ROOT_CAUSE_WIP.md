# Root-causing the unsafe MoE configs — WORK IN PROGRESS

Goal: find why the M<=256 tuned configs produce NaN, fix it, and re-enable them
(recovering the decode gain: +2.9% with the full set vs +1.8% shipped). Owner's
call 2026-08-09: do this ourselves rather than wait for upstream.

Read `README.md` in this directory first for the symptom and the paired evidence.

## Hypotheses tested and REFUTED — do not re-walk these

1. **A single bad entry (M=128).** Removing it made things worse: 1/7 vs 4/7.
2. **Pipeline depth (`num_stages=5`).** Every failing entry except M=24 used
   `num_stages=5` and nothing in the stable set exceeded 4 — a clean
   correlation. Capping all entries at 4 still failed (4/7, then 0/7), and
   produced corrupted-but-passing text.
3. **up/down `BLOCK_SIZE_M` mismatch.** `moe_align_block_size` pads using the
   UP config's `BLOCK_SIZE_M` while the down GEMM runs with `down_config`'s,
   so a larger down tile would over-run the alignment. Checked all 18 entries:
   up and down `BLOCK_SIZE_M` are **identical everywhere**. No mismatch exists.

## What the evidence actually constrains

- It is the CONFIG, not the runtime M. The 512 config used at M=170 works;
  the 256 config used at M=193 fails. Same shapes, different config.
- It is NONDETERMINISTIC. The same 193-token audiogen prefill through the same
  M=256 config both passed and produced NaN on different runs.
- No single parameter separates safe from unsafe. The boundary case is stark:
  M=256 (unsafe) is `BM=64, BK=256, stages=5` and M=512 (safe) is
  `BM=64, BK=256, stages=2` — but M=2048 runs `stages=4, BK=256` safely and
  capping 256 to `stages=4` did NOT fix it.
- Failures always appeared on mid-size prefills (170-193 tokens) in the audio
  path. Large prefills and text decode never failed. This is probably about
  which config those shapes select, not about audio per se.

## Code leads worth pursuing

In `${SG}/layers/moe/moe_runner/triton_utils/fused_moe.py`:

```python
padded_tokens = (
    min(num_tokens * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
    if down_moe_use_tma
    else 0
)
total_tokens = num_tokens * topk + padded_tokens
intermediate_cache1 = torch.empty((total_tokens, N), ...)   # UNINITIALISED
```

`moe_align_block_size` always pads expert runs to a multiple of
`BLOCK_SIZE_M`, but the extra allocation is only made when `down_moe_use_tma`.
The buffer is `torch.empty`, so any row the kernel does not write keeps garbage,
and garbage that happens to decode as NaN propagates. That fits a
nondeterministic fault whose outcome depends on what was previously in the
allocator pool — worth testing directly by poisoning free memory with NaN
before the call and seeing whether failures become deterministic.

Note `USE_TMA` is true only for M=1 and M=2 in the down configs, so the padding
branch is inactive for nearly every entry.

## Next step: a standalone correctness harness (the missing tool)

Reproducing currently costs a container launch plus a full battery (~20 min) and
is intermittent, so a negative result proves nothing. That loop cannot find a
race. Build instead a script that:

1. constructs synthetic w8a8_int8 MoE inputs (the tuner's `benchmark_config` in
   `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` already
   builds weights, per-channel scales and real routing — start from it);
2. forces a specific config — `try_get_optimal_moe_config` consults
   `get_config()` from `sglang.srt.layers.moe.moe_runner.triton_utils` as an
   override, which is the hook to use;
3. runs each config across many M values and many repetitions, comparing output
   against a reference config and checking for NaN/inf;
4. optionally poisons the allocator pool with NaN first, to convert an
   uninitialised-memory fault into a deterministic one.

That turns a 20-minute coin flip into a fast, high-power per-config verdict, and
it doubles as the output-correctness gate the tuner never had — which is what
allowed a fast-but-wrong config to be selected in the first place.

## Iteration tooling already in place

`SGLANG_MOE_CONFIG_DIR` points the runtime at
`<dir>/configs/triton_3_6_0/`, so config variants are testable with a container
restart instead of an image rebuild. Variants from this investigation live on
Spark in `~/longcat-outputs/moe_override/` and `~/longcat-outputs/moe_capped/`;
crash logs are `~/longcat-outputs/server_*.log`.
