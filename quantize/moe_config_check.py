#!/usr/bin/env python3
"""Correctness gate for tuned fused-MoE Triton configs.

The sep tuner ranks candidates purely by latency and never compares their
output to a reference, so a fast-but-wrong config is exactly what it selects —
which is how the M<=256 entries shipped a NaN-producing fault (see
research/moe_tuning/README.md). This is the missing check: run each config
against a reference and look for NaN/inf and numerical divergence.

Reproducing the fault through the server costs ~20 min and is intermittent.
This runs a config hundreds of times in seconds, so a negative result actually
means something.

Run inside a container built from this repo, with the model mounted:

  docker run --rm --gpus all --shm-size=32g \
    -v $HOME/models/LongCat-Next-w8a8int8:/workspace/model:ro \
    -v $HOME/longcat-outputs:/workspace/outputs \
    --entrypoint python3 longcat-next-gb10:latest \
    /workspace/outputs/moe_config_check.py --configs /workspace/outputs/moe_configs

Options:
  --poison   fill freed GPU memory with NaN before each call, to turn an
             uninitialised-buffer fault into a deterministic one
"""
import argparse, itertools, json, os, sys

import torch

# fused_experts reads get_server_args(); establish the context standalone.
from sglang.srt import runtime_context as _rc
from sglang.srt.server_args import ServerArgs

_rc._CONTEXT._server_args = ServerArgs(model_path=os.environ.get("LCN_MODEL", "/workspace/model"))

# fused_experts allocates its output through the TP group, so a single-process
# parallel environment has to exist even though we never communicate.
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel

init_distributed_environment(
    world_size=1, rank=0, local_rank=0,
    distributed_init_method="tcp://127.0.0.1:29577", backend="gloo",
)
initialize_model_parallel(tensor_model_parallel_size=1)

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_experts, override_config
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import get_moe_configs
from sglang.srt.layers.moe.topk import TopKConfig, select_experts

# LongCat-Next MoE shape (verified): 256 routed experts, topk 12,
# moe_intermediate_size 1024 -> w1 is 2*1024 (gate+up), hidden 3072.
E, TOPK, INTER, HIDDEN = 256, 12, 1024, 3072
DTYPE = torch.bfloat16


def build_inputs(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(M, HIDDEN, dtype=DTYPE, device="cuda", generator=g)
    w1 = torch.randint(-127, 127, (E, 2 * INTER, HIDDEN), dtype=torch.int8, device="cuda", generator=g)
    w2 = torch.randint(-127, 127, (E, HIDDEN, INTER), dtype=torch.int8, device="cuda", generator=g)
    # per-channel int8: one scale per output channel per expert
    w1_scale = torch.rand((E, 2 * INTER), dtype=torch.float32, device="cuda", generator=g) * 0.01
    w2_scale = torch.rand((E, HIDDEN), dtype=torch.float32, device="cuda", generator=g) * 0.01
    gating = torch.randn(M, E, dtype=torch.float32, device="cuda", generator=g)
    topk_out = select_experts(hidden, gating, TopKConfig(top_k=TOPK, renormalize=True))
    return hidden, w1, w2, w1_scale, w2_scale, topk_out


def install_config(up_cfg, down_cfg, tmpdir):
    """Inject a config the way production does — through the file lookup — so
    BOTH projections come from the tuned data.

    override_config() only forces the UP config: try_get_optimal_moe_config
    takes its early-exit branch and leaves down_config as None, so the down
    GEMM silently runs on defaults. Since the down configs differ from the up
    ones (num_stages 5 vs 2 at the boundary entries), testing through the
    override would have exonerated configs it never actually exercised.
    """
    d = os.path.join(tmpdir, "configs", "triton_3_6_0")
    os.makedirs(d, exist_ok=True)
    base = "E=256,N=1024,device_name=NVIDIA_GB10,dtype=int8_w8a8,per_channel_quant=True"
    # single-entry maps: nearest-M then resolves to this config at every shape
    json.dump({"1": up_cfg}, open(os.path.join(d, base + ".json"), "w"))
    json.dump({"1": down_cfg}, open(os.path.join(d, base + "_down.json"), "w"))
    os.environ["SGLANG_MOE_CONFIG_DIR"] = tmpdir
    get_moe_configs.cache_clear()   # lru_cached, would otherwise serve the old map


def run(cfg, inputs):
    hidden, w1, w2, w1_scale, w2_scale, topk_out = inputs
    runner_cfg = MoeRunnerConfig(inplace=False)
    ctx = override_config(cfg) if cfg else _null_ctx()
    with ctx:
        return fused_experts(
            hidden.clone(), w1, w2, topk_out, runner_cfg,
            use_int8_w8a8=True, per_channel_quant=True,
            w1_scale=w1_scale, w2_scale=w2_scale,
        )


class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def poison_pool():
    """Dirty the caching allocator with NaN so an unwritten output row shows up
    as NaN deterministically instead of depending on what ran before."""
    blocks = [torch.full((64, 1024, 1024), float("nan"), dtype=DTYPE, device="cuda")
              for _ in range(4)]
    del blocks  # freed back into the pool, still NaN-filled


def check(name, cfg, M_values, reps, poison, ref_cfg=None, down_cfg=None, tmpdir=None):
    bad = []
    if cfg is not None and down_cfg is not None:
        install_config(cfg, down_cfg, tmpdir)
        cfg = None  # configs now come from files, for BOTH projections
    for M in M_values:
        for r in range(reps):
            if poison:
                poison_pool()
            inputs = build_inputs(M, seed=1000 + r)
            out = run(cfg, inputs)
            torch.cuda.synchronize()
            n_nan = int(torch.isnan(out).sum())
            n_inf = int(torch.isinf(out).sum())
            note = ""
            if ref_cfg is not None and not (n_nan or n_inf):
                ref = run(ref_cfg, inputs)
                torch.cuda.synchronize()
                if not torch.isnan(ref).any():
                    denom = ref.abs().max().clamp(min=1e-6)
                    rel = ((out - ref).abs().max() / denom).item()
                    if rel > 0.05:
                        note = f" rel_err={rel:.3f}"
                        bad.append((M, r, "diverged" + note))
            if n_nan or n_inf:
                bad.append((M, r, f"nan={n_nan} inf={n_inf}"))
    status = "BAD " if bad else "ok  "
    detail = ""
    if bad:
        Ms = sorted({b[0] for b in bad})
        detail = f"  {len(bad)} failures across M={Ms}; first: {bad[0]}"
    print(f"{status}{name}{detail}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="/workspace/outputs/moe_configs",
                    help="dir holding the tuned E=256,... JSONs")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--poison", action="store_true")
    ap.add_argument("--m-values", default="1,7,64,170,193,256,512,1024",
                    help="runtime M values to exercise each config at")
    ap.add_argument("--only", default="", help="comma-separated config keys to test")
    args = ap.parse_args()

    M_values = [int(x) for x in args.m_values.split(",")]
    up_path = None
    for f in os.listdir(args.configs):
        if f.startswith("E=256") and not f.endswith("_down.json"):
            up_path = os.path.join(args.configs, f)
    if not up_path:
        sys.exit(f"no up-projection config found in {args.configs}")
    configs = json.load(open(up_path))
    down_configs = json.load(open(up_path.replace(".json", "_down.json")))
    tmpdir = os.path.join(args.configs, "_harness_tmp")
    keys = sorted(configs, key=int)
    if args.only:
        want = set(args.only.split(","))
        keys = [k for k in keys if k in want]

    print(f"checking {len(keys)} configs from {os.path.basename(up_path)}")
    print(f"M values: {M_values}   reps: {args.reps}   poison: {args.poison}")
    print("baseline first (no override = runtime picks / defaults):")
    check("DEFAULT (no override)", None, M_values, args.reps, args.poison)

    print("per-config:")
    verdicts = {}
    for k in keys:
        verdicts[k] = check(f"M={k:<5} up={configs[k]}", configs[k], M_values,
                            args.reps, args.poison,
                            down_cfg=down_configs[k], tmpdir=tmpdir)

    bad = [k for k, v in verdicts.items() if v]
    print()
    print(f"SUMMARY: {len(bad)}/{len(keys)} configs produced NaN/inf or diverged")
    if bad:
        print("BAD:", ", ".join(f"M={k}" for k in bad))
    else:
        print("all configs clean under this harness — the fault may need the "
              "real model's weights/activations, not synthetic ones")


if __name__ == "__main__":
    main()
