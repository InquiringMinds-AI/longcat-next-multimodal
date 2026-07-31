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
set -euo pipefail

BENCH=/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton
OUT=/workspace/outputs/moe_configs
TOPK=/workspace/outputs/topk_ids
mkdir -p "$OUT"

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
open(p, "w").write(src)
print("sep tuner: per-channel enabled + LongCat layer layout (14 MoE, 0-based)")
PYEOF

cd "$BENCH"
python3 tuning_fused_moe_triton_sep.py \
    --model /workspace/model \
    --tp-size 1 \
    --dtype int8_w8a8 \
    --topk-ids-dir "$TOPK" \
    --tune 2>&1 | tee "$OUT/tune_log.txt"

# collect whatever the tuner wrote next to itself
cp -v "$BENCH"/*.json "$OUT"/ 2>/dev/null || true
find / -maxdepth 4 -name "E=256,N=1024*" -newer "$OUT/tune_log.txt" -exec cp -v {} "$OUT"/ \; 2>/dev/null || true
echo "DONE — configs in $OUT"
