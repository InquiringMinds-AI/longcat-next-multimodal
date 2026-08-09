# Full MoE tuning results (all 18 batch sizes) — ARCHIVE, NOT SHIPPED

These are the complete outputs of the fused-MoE Triton tuning ladder run on GB10
(2026-07-31 .. 2026-08-09, ~9 days of GPU time), covering every batch size from
M=1 to M=4096 for both the up and down projections.

**Only the M >= 512 entries ship.** `new_files/moe_configs/` contains that
subset; the Dockerfile copies it into the runtime's
`configs/triton_3_6_0/` directory. The files here are the unabridged results,
kept for future work.

## Why the small/mid-M entries are excluded

The M <= 256 entries are numerically unsound on this hardware. With the full
18-entry set installed, the server dies with:

```
_assert_async_cuda_kernel: Assertion `probability tensor contains either
`inf`, `nan` or element < 0` failed
torch.AcceleratorError: CUDA error: device-side assert triggered
```

the scheduler is SIGKILLed, and the gateway survives so every subsequent request
returns connection-refused.

Paired evidence, same image and mounts and test sequence, all-modality defaults:

| build | selftest | asserts |
|---|---|---|
| untuned (`:v0516-spec`) | 7/7 | 0 |
| full 18-entry tuned | 3 runs, 3 failures (4/7, 1/7, 4/7) | 1 each |
| M >= 512 only | 5 consecutive batteries, 7/7 each | 0 |

It is **not** a memory problem (`OOMKilled=false`, normal headroom, memory frees
only after the process dies) and **not** one bad entry.

Two hypotheses were tested and refuted:

1. **A single bad entry (M=128).** Removing it made things worse (1/7).
2. **Pipeline depth.** Every failing-set entry except M=24 used `num_stages=5`
   and nothing in the stable set exceeded 4 — a clean correlation. Capping all
   entries at 4 still failed (4/7, then 0/7).

The decisive observation is that two runs executed the *same* 193-token audiogen
prefill through the *same* M=256 config with opposite outcomes — one passing, one
producing NaN. Identical config plus identical shape yielding both results means
the fault is **nondeterministic**, which rules out a wrong-but-deterministic tile
choice and points at a race (most likely in the async-copy pipeline) that these
configs expose on sm_121.

All three failures were mid-size prefills (170-193 tokens) in the audio path.
Large prefills and text decode never failed.

## The part that matters for any future tuning

`tuning_fused_moe_triton_sep.py` selects configs **purely by latency**. It never
compares a candidate's output against a reference. A fast-but-racy config is
exactly what it is built to choose, so tuning output on new hardware must be
validated for correctness separately — a green benchmark says nothing about
numerics.

Worse than the crashes: with `num_stages` capped, `image_understanding` returned

> "The\nTherl's in the image is a\nA red"

and still counted as a PASS, because the test only checks for a substring. Silent
corruption of this kind would ship unnoticed. Any future tuning pass needs an
output-correctness gate, not just a battery of pass/fail modality checks.

## Cost of excluding them

Measured with `test/bench_decode.py` and `test/bench_prefill.py` (same scripts,
same machine, builds differing only in installed configs):

| variant | decode bs=1 | prefill 6.7k | stable |
|---|---|---|---|
| untuned | 21.02 tok/s | 2684 tok/s | yes |
| full 18-entry | 21.62 (+2.9%) | 3184 (+18.6%) | **no** |
| M >= 512 (shipped) | 21.40 (+1.8%) | 3064 (+14.2%) | yes |

So excluding twelve entries costs about 1.1 points of decode and 4 points of
prefill relative to the unsafe full set, and keeps most of the gain.

## If you want to recover the rest

The runtime honours `SGLANG_MOE_CONFIG_DIR`, so config variants can be tested by
pointing it at `<dir>/configs/triton_3_6_0/` on the output mount — a container
restart instead of an image rebuild. Variants used during this investigation are
in `~/longcat-outputs/moe_override/` and `~/longcat-outputs/moe_capped/` on Spark.

Bisecting the twelve excluded entries could recover part of the decode gain, but
budget several batteries per candidate: the fault is intermittent, so a single
green run proves nothing.
