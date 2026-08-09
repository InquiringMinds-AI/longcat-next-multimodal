# Full MoE tuning results (all 18 batch sizes) — ARCHIVE, NOT SHIPPED

These are the complete outputs of the fused-MoE Triton tuning ladder run on GB10
(2026-07-31 .. 2026-08-09, ~9 days of GPU time), covering every batch size from
M=1 to M=4096 for both the up and down projections.

**Only the M=1 and M=2 entries are excluded** (16 of 18 ship). `new_files/moe_configs/` contains that
subset; the Dockerfile copies it into the runtime's
`configs/triton_3_6_0/` directory. The files here are the unabridged results,
kept for future work.

## ROOT CAUSE FOUND (2026-08-09): USE_TMA on the down projection

`USE_TMA: True` appears on exactly two entries — M=1 and M=2 — and it is
the fault. Excluding those two entries makes the full ladder stable.

How it was localised: an env-gated NaN check (`LCN_NAN_CHECK=1`) on both
sides of every MoE call, reporting whether NaN arrives or is created. It
fired immediately and unambiguously, twice:

```
[NAN-CHECK] layer=0 tokens=1 input_bad=False output_bad=True -> CREATED HERE
[NAN-CHECK] layer=1 tokens=1 input_bad=True  output_bad=True -> arrived from upstream
```

**`tokens=1` — a DECODE step, not a prefill.** Every earlier hypothesis was
aimed at the wrong shape: the 170-193 token prefills in the logs were
bystanders that happened to precede the assert, which fires at sampling.
M=1 selects the M=1 config, the TMA one. That is also why the standalone
harness never reproduced it — the captures were of the 150-220 token range,
so an M=1 decode call was never replayed.

Two fixes were measured. Disabling TMA in place is stable but LOSES the
decode gain (20.93 tok/s, below the 21.02 untuned baseline) — the tuner
chose that tile geometry *because* TMA made it fast, so without TMA it is
simply a bad config. DROPPING the two entries is better: M=1 decode then
resolves by nearest-M to the tuned M=4 config, which has TMA off natively
and was selected on its own merits.

## Why M=1 and M=2 are excluded

With the full 18-entry set installed, the server dies with:

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
| all 18, TMA disabled | 3 batteries, 7/7, 0 NaN-check hits | 0 |
| **16 entries (M=1,2 dropped)** | **4 batteries, 7/7, 0 NaN-check hits** | **0** |

It is **not** a memory problem (`OOMKilled=false`, normal headroom, memory frees
only after the process dies).

### Hypotheses refuted before the NaN check found it — do not re-walk these

1. **A single bad mid-M entry (M=128).** Removing it made things worse (1/7).
2. **Pipeline depth.** Every M<=256 entry except M=24 used `num_stages=5` and
   nothing above M=256 exceeded 4 — a clean correlation, and a coincidence.
   Capping all entries at 4 still failed (4/7, then 0/7).
3. **up/down `BLOCK_SIZE_M` mismatch.** Identical on all 18 entries.
4. **Reproducing it in an isolated MoE call.** Clean with synthetic inputs, with
   real captured activations and routing, with a NaN-poisoned allocator pool,
   and with the real checkpoint weights for all 14 layers. The captures were of
   the 150-220 token range — the wrong shape, since the fault is at M=1.

The trap that cost the most time: every assert was preceded in the log by a
170-193 token audio prefill, so all four hypotheses above chased mid-size
prefills. Those prefills were bystanders. The assert fires at SAMPLING, after
the decode steps that follow, and the NaN check showed the corruption is created
at `tokens=1`. Log adjacency is not causation.

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
| M >= 512 | 21.37 (+1.7%) | 3055 (+13.8%) | yes |
| all 18, TMA off | 20.93 (-0.4%) | 2980 | yes, but slower than untuned |
| **16 entries (SHIPPED)** | **21.57 (+2.6%)** | **3039 (+13.2%)** | **yes** |

⚠ Do NOT benchmark with `LCN_NAN_CHECK=1` set. It calls `.any()` on both sides
of every MoE call — 28 device syncs per decoded token — and silently drags
decode to ~20.9 regardless of config. Two runs were wasted on that before the
uniformity of the numbers gave it away.

So excluding just the two TMA entries costs ~0.3 points of decode and ~5 of
prefill against the unsafe full set, and keeps nearly all of the gain.

## Tooling

- `quantize/moe_config_check.py` — the correctness gate the tuner lacks. Runs a
  config hundreds of times in seconds against NaN/inf and divergence checks, and
  can replay real captured MoE calls (`--replay`, `--real-weights`) or poison the
  allocator pool (`--poison`). It did NOT find this bug, because an isolated call
  does not reproduce it — but it cheaply eliminated four hypotheses.
- `LCN_NAN_CHECK=1` — the thing that actually found it. Checks both sides of every
  MoE call and reports whether NaN arrived or was created. Never benchmark with
  it on.
- `LCN_DUMP_MOE_DIR` + `LCN_DUMP_MOE_RANGE` — capture real MoE inputs at a chosen
  token-count range, for replay.
- `SGLANG_MOE_CONFIG_DIR` — test config variants with a container restart instead
  of an image rebuild. Variants on Spark: `~/longcat-outputs/moe_{override,capped,
  full,notma,drop12}/`.

## Worth reporting upstream

`USE_TMA` on the down projection producing NaN at M=1 on sm_121 is a genuine
SGLang bug, not a configuration mistake on our side. The tuner will select it
again on any Blackwell-class device, because TMA really is faster when it works
— the M=1 config was the fastest candidate measured.
