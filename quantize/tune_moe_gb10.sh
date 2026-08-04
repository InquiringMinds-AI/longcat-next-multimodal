#!/bin/bash
# Tune the fused-MoE Triton kernels for LongCat-Next on GB10 (ROADMAP #4).
#
# Run INSIDE a container from the longcat-next-gb10 image, with the model dir
# and an output dir mounted, AFTER capturing real routing distributions:
#
#   1. Serve once with LCN_DUMP_TOPK_DIR=/workspace/outputs/topk_ids and send
#      two >=4096-token prefills (the hook in longcat_flash.py saves
#      topk_ids_layer<L>_idx{0,1}.pt per MoE layer).
#   2. Stop serving (the tuner needs the GPU to itself), then:
#        docker run -d --name lcn-moe-tune --gpus all --shm-size=32g \
#          -v $HOME/models/LongCat-Next-w8a8int8:/workspace/model:ro \
#          -v $HOME/longcat-outputs:/workspace/outputs \
#          --entrypoint bash longcat-next-gb10:latest \
#          /workspace/outputs/tune_moe_gb10.sh
#      (never --rm: the config JSONs land in /workspace/outputs/moe_configs)
#
# The stock sep tuner needs three fixes for this checkpoint, applied here at
# runtime (build-independent):
#   - common_utils.get_model_config has no LongcatNext branch
#     (E=n_routed_experts=256, topk=moe_topk=12,
#      intermediate=expert_ffn_hidden_size=1024 -> filename N=1024)
#   - tuning_fused_moe_triton_sep.py hardcodes per_channel_quant=False at
#     both kernel wrappers AND in the output filename; this w8a8_int8
#     checkpoint is per-channel (the int8 scale tensors the tuner allocates
#     are already per-channel shaped — only the flags are wrong).
#
# CHECKPOINTING (added after the first full run): the stock tuner holds every
# batch size's winning config in the Ray driver's memory and writes the two
# JSONs exactly once, after the WHOLE ladder finishes (~5 days here). It never
# prints the configs, so a crash at hour 100 loses everything with nothing on
# disk or in the log to rebuild from. The patch below dumps each batch size's
# result to $CKPT as soon as that size completes, and the recovery block after
# the run reassembles the final JSONs from those checkpoints if the tuner died
# before its own write. Checkpoints live on the output mount, so they survive
# the container.
set -euo pipefail

BENCH=/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton
OUT=/workspace/outputs/moe_configs
TOPK=/workspace/outputs/topk_ids
CKPT=$OUT/checkpoints
mkdir -p "$OUT" "$CKPT"
export LCN_CKPT_DIR="$CKPT"

# The 2026-08-04 run died at ~hour 95 (during the last small-M sizes) when Ray's
# memory monitor OOM-killed the BenchmarkWorker at 116.84GB/121.69GB node usage,
# taking every completed config with it. This container is launched with
# --entrypoint bash, which BYPASSES entrypoint.sh, so the project-standard
# allocator setting never got applied — on GB10's unified memory the caching
# allocator's per-shape blocks accumulate across thousands of tuned configs and
# count against system RAM. Set it here so it holds regardless of entrypoint.
# (Do NOT raise RAY_memory_usage_threshold instead: 116GB is already inside the
# ~110-115GB band where Spark hard-powers-off, so the monitor firing is the
# safety net, not the bug.)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# the serving image ships without ray (the tuner's worker framework)
python3 -c "import ray" 2>/dev/null || pip install --no-cache-dir -q ray

python3 - <<'PYEOF'
import re

p = "/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/common_utils.py"
src = open(p).read()
if "LongcatNextForCausalLM" not in src:
    anchor = """    else:
        # Default: Mixtral
        E = config.num_local_experts // ep_size"""
    branch = """    elif architecture in ("LongcatNextForCausalLM", "LongcatFlashForCausalLM"):
        # sglang resolves this model to its own LongcatFlashConfig, which maps
        # the checkpoint's expert_ffn_hidden_size to moe_intermediate_size.
        E = config.n_routed_experts // ep_size
        topk = config.moe_topk
        intermediate_size = config.moe_intermediate_size
    else:
        # Default: Mixtral
        E = config.num_local_experts // ep_size"""
    assert src.count(anchor) == 1
    open(p, "w").write(src.replace(anchor, branch))
    print("common_utils: Longcat branch added")

p = "/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py"
src = open(p).read()
# these edits are not idempotent (the asserts below expect pristine source), so
# a re-exec inside a container that already ran once must skip them
if "LCN_CKPT_DIR" in src:
    print("sep tuner: already patched, skipping")
    raise SystemExit(0)
n = src.count("per_channel_quant=False,")
assert n == 2, n
src = src.replace("per_channel_quant=False,", "per_channel_quant=True,")
# the announced-filename call passes a bare False for per_channel_quant
anchor = """        use_int4_w4a16,
        False,"""
assert src.count(anchor) == 1
src = src.replace(anchor, """        use_int4_w4a16,
        True,""")
# save_configs_sep builds the WRITTEN filenames without per_channel_quant at
# all (get_config_file_name defaults it False) -> the runtime would never
# find them. Pass it explicitly.
anchor = """    filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        block_shape,
        down_moe=down_moe,
    )"""
assert src.count(anchor) == 1
src = src.replace(anchor, """    filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        block_shape,
        per_channel_quant=True,
        down_moe=down_moe,
    )""")
# load_topk_ids hardcodes DeepSeek-V3's layout (61 layers, 3 dense = 116
# capture files); LongCat-Next has 14 MoE modules captured as layer0..13
# x idx{0,1} = 28 files. The tuner asks for 100 samples -> cycle ours.
anchor = """    num_layers = 61
    dense_layers = 3"""
assert src.count(anchor) == 1
src = src.replace(anchor, """    num_layers = 14
    dense_layers = 0""")
anchor = '''        f"{topk_ids_dir}/topk_ids_layer{i % moe_layers + dense_layers}_idx{i // moe_layers}.pt"'''
assert src.count(anchor) == 1
src = src.replace(anchor, '''        f"{topk_ids_dir}/topk_ids_layer{(i % (moe_layers * 2)) % moe_layers + dense_layers}_idx{(i % (moe_layers * 2)) // moe_layers}.pt"''')
# Resume: drop batch sizes that already have a checkpoint, so a rerun after a
# crash costs only the size that was in flight instead of the whole ladder.
# (The stock tuner has no skip logic — restarting it re-tunes everything.)
anchor = """    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
        batch_sizes.reverse()
    else:
        batch_sizes = [args.batch_size]"""
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + """

    _ckpt_dir = os.environ.get("LCN_CKPT_DIR")
    if _ckpt_dir and os.path.isdir(_ckpt_dir):
        _done = set()
        for _f in os.listdir(_ckpt_dir):
            if _f.startswith("ckpt_M") and _f.endswith(".json"):
                try:
                    _done.add(int(_f[len("ckpt_M"):-len(".json")]))
                except ValueError:
                    pass
        _skipped = sorted(b for b in batch_sizes if b in _done)
        if _skipped:
            batch_sizes = [b for b in batch_sizes if b not in _done]
            print(f"resume: skipping already-checkpointed batch sizes {_skipped}")
            if not batch_sizes:
                print("resume: all batch sizes checkpointed, nothing left to tune")
                return""")

# BenchmarkWorker.tune returns its winning configs to the Ray driver, which
# buffers all of them until the final write. Dump each one as it is produced.
anchor = """        return (
            trace0.config_dict(best_block_m),
            trace1.config_dict(best_block_m),
            trace0.time_cost(best_block_m),
            trace1.time_cost(best_block_m),
        )"""
assert src.count(anchor) == 1
src = src.replace(anchor, """        _c0 = trace0.config_dict(best_block_m)
        _c1 = trace1.config_dict(best_block_m)
        _ckpt_dir = os.environ.get("LCN_CKPT_DIR")
        if _ckpt_dir:
            os.makedirs(_ckpt_dir, exist_ok=True)
            _dst = os.path.join(_ckpt_dir, f"ckpt_M{num_tokens}.json")
            with open(_dst + ".tmp", "w") as _fh:
                json.dump(
                    {"batch_size": num_tokens, "config0": _c0, "config1": _c1},
                    _fh,
                    indent=4,
                )
            os.replace(_dst + ".tmp", _dst)
            print(f"checkpointed batch_size={num_tokens} -> {_dst}")
        # drop the caching allocator's per-shape blocks between batch sizes;
        # their accumulation is what walked the node into Ray's OOM kill
        torch.cuda.empty_cache()
        return (
            _c0,
            _c1,
            trace0.time_cost(best_block_m),
            trace1.time_cost(best_block_m),
        )""")
open(p, "w").write(src)
print("sep tuner: per-channel enabled + LongCat layer layout (14 MoE, 0-based) + per-batch checkpointing")
PYEOF

MODEL=/workspace/model
DTYPE=int8_w8a8

cd "$BENCH"
set +e
python3 tuning_fused_moe_triton_sep.py \
    --model "$MODEL" \
    --tp-size 1 \
    --dtype "$DTYPE" \
    --topk-ids-dir "$TOPK" \
    --tune 2>&1 | tee "$OUT/tune_log.txt"
rc=${PIPESTATUS[0]}
set -e
echo "tuner exit code: $rc"

# collect whatever the tuner wrote next to itself
cp -v "$BENCH"/*.json "$OUT"/ 2>/dev/null || true

# Recovery: if the tuner never reached its own write, rebuild the two config
# JSONs from the per-batch checkpoints. Mirrors save_configs_sep exactly
# (same filename derivation, same sort_config, ascending-M key order) so the
# result is byte-comparable to a clean run over the same batch sizes. A
# partial ladder is still usable — the runtime interpolates from the M values
# present — so this writes whatever completed.
BENCH="$BENCH" CKPT="$CKPT" OUT="$OUT" MODEL="$MODEL" DTYPE="$DTYPE" TOPK="$TOPK" \
python3 - <<'PYEOF' || echo "recovery: skipped (see error above)"
import glob, json, os, sys

bench, ckpt_dir = os.environ["BENCH"], os.environ["CKPT"]
out, model = os.environ["OUT"], os.environ["MODEL"]
dtype_arg, topk_dir = os.environ["DTYPE"], os.environ["TOPK"]

files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_M*.json")))
if not files:
    print("recovery: no checkpoints found, nothing to rebuild")
    raise SystemExit(0)

sys.path.insert(0, bench)
from common_utils import get_model_config, sort_config
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_file_name,
)
try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
    )
except ImportError:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        get_config_dtype_str,
    )

mc = get_model_config(model, 1, 1, False, topk_dir)
dtype_str = get_config_dtype_str(
    mc["dtype"],
    use_fp8_w8a8=dtype_arg == "fp8_w8a8",
    use_int8_w8a8=dtype_arg == "int8_w8a8",
    use_int8_w8a16=dtype_arg == "int8_w8a16",
    use_int4_w4a16=dtype_arg == "int4_w4a16",
)

up, down = {}, {}
for f in files:
    d = json.load(open(f))
    m = int(d["batch_size"])
    up[m] = sort_config(d["config0"])
    down[m] = sort_config(d["config1"])

for cfgs, down_moe in ((up, False), (down, True)):
    name = os.path.basename(
        get_config_file_name(
            mc["num_experts"],
            mc["shard_intermediate_size"] // 2,
            dtype_str,
            mc["block_shape"],
            per_channel_quant=True,
            down_moe=down_moe,
        )
    )
    dst = os.path.join(out, name)
    if os.path.exists(dst):
        print(f"recovery: {name} already written by the tuner, leaving it")
        continue
    with open(dst, "w") as fh:
        json.dump({str(m): cfgs[m] for m in sorted(cfgs)}, fh, indent=4)
        fh.write("\n")
    print(f"recovery: wrote {dst} from {len(cfgs)} checkpoints "
          f"(M={sorted(cfgs)})")
PYEOF

echo "DONE — configs in $OUT (checkpoints in $CKPT)"
exit "$rc"
