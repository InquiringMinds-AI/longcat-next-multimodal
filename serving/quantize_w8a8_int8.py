#!/usr/bin/env python3
"""Stream-quantize LongCat-Next BF16 -> w8a8_int8 checkpoint (MoE experts ONLY).

- Quantizes only  model.layers.N.mlp.experts.N.{gate,up,down}_proj.weight
  to per-output-channel symmetric int8 (+ .weight_scale fp32 [out]).
- Everything else copied verbatim (BF16) -> kept FLOAT via the `ignore` list,
  because sgl-kernel's CUTLASS int8_scaled_mm has no sm_121 impl (dense int8 would crash).
- Streams one safetensors shard at a time (CPU, ~10-15GB peak). Run in the cu130 container.

Mounts:  -v ~/models/LongCat-Next:/src:ro  -v <out>:/dst
"""
import os, re, json, shutil, torch
from collections import defaultdict
from safetensors.torch import safe_open, save_file

SRC, DST = "/src", "/dst"
EXPERT_RE = re.compile(r".*\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")

# Keep these BF16 (every non-expert Linear). Patterns are `re:`-style fullmatch targets,
# matching what should_ignore_layer / W8A8Int8Config will use at load time.
IGNORE = [
    "re:.*\\.self_attn\\..*",     # MLA q_a/q_b/kv_a_proj_with_mqa/kv_b/o_proj (+ fused_qkv_a_proj)
    "re:.*\\.mlps\\..*",          # dense/shortcut MLPs (NOTE: plural; != mlp.experts)
    "re:.*\\.router\\..*",        # MoE router classifier (routing precision is critical)
    "re:.*visual_tokenizer.*",
    "re:.*audio_tokenizer.*",
    "re:.*visual_head.*",
    "re:.*audio_head.*",
    "re:.*embed_tokens.*",
    "re:.*ngram_embed.*",
    "re:.*lm_head.*",
]

def quant_per_channel_int8(w):           # w: [out, in] (bf16/fp16)
    wf = w.to(torch.float32)
    scale = (wf.abs().amax(dim=1, keepdim=True) / 127.0).clamp_(min=1e-8)   # [out,1]
    q = (wf / scale).round_().clamp_(-127, 127).to(torch.int8)             # [out,in]
    return q, scale.to(torch.float32)                                     # int8 [out,in], fp32 [out,1] (sglang FusedMoE copy_ needs 2-D)

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wmap = idx["weight_map"]
byshard = defaultdict(list)
for name, shard in wmap.items():
    byshard[shard].append(name)
shards = sorted(byshard)

new_wmap, n_quant, n_copy = {}, 0, 0
for si, shard in enumerate(shards):
    out = {}
    with safe_open(f"{SRC}/{shard}", framework="pt", device="cpu") as f:
        for name in byshard[shard]:
            t = f.get_tensor(name)
            if EXPERT_RE.match(name):
                q, s = quant_per_channel_int8(t)
                out[name] = q
                out[name.replace(".weight", ".weight_scale")] = s
                n_quant += 1
            else:
                out[name] = t
                n_copy += 1
    save_file(out, f"{DST}/{shard}", metadata={"format": "pt"})
    for k in out:
        new_wmap[k] = shard
    print(f"[{si+1}/{len(shards)}] {shard}: {len(out)} tensors  (int8 experts so far: {n_quant})", flush=True)

# index
json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_wmap},
          open(f"{DST}/model.safetensors.index.json", "w"), indent=2)

# config.json + quantization_config
cfg = json.load(open(f"{SRC}/config.json"))
cfg["quantization_config"] = {"quant_method": "w8a8_int8", "is_dynamic": True, "ignore": IGNORE}
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)

# aux files (tokenizer, generation_config, preprocessor, etc.) — everything not a shard / not rewritten
for fn in os.listdir(SRC):
    if fn.endswith(".safetensors") or fn in ("config.json", "model.safetensors.index.json"):
        continue
    src = f"{SRC}/{fn}"
    if os.path.isfile(src):
        shutil.copy2(src, f"{DST}/{fn}")

print(f"DONE  int8_experts={n_quant}  copied={n_copy}  shards={len(shards)}", flush=True)
