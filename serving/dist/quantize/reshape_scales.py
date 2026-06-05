#!/usr/bin/env python3
"""In-place fix: reshape per-expert *.weight_scale from [N] -> [N,1] so sglang's
FusedMoE int8 loader (_load_w13: expert_data[inter,1].copy_(loaded_weight)) matches.
Rewrites ONLY shards that contain 1-D weight_scale tensors; leaves config.json + the
big multimodal/embed shards untouched (preserves the config fixes already applied)."""
import json, torch
from collections import defaultdict
from safetensors.torch import safe_open, save_file

D = "/dst"
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
byshard = defaultdict(list)
for n, s in idx.items():
    byshard[s].append(n)

# Only the fused-MoE expert-proj loader needs [N,1] scales. The n-gram OE embedder loader
# expects 1-D [N] scales (it assigns into a 1-D slice). So reshape ONLY MoE expert scales and
# EXPLICITLY skip OE embedder scales — reshaping those would hard-fail at model load.
def _is_moe_expert_scale(n):
    return n.endswith(".weight_scale") and ".mlp.experts." in n
def _is_oe_scale(n):
    return ".ngram_embeddings.embedders." in n

fixed = 0
for shard in sorted(byshard):
    names = byshard[shard]
    # quick scan: does this shard hold any 1-D MoE-expert weight_scale?
    with safe_open(f"{D}/{shard}", framework="pt", device="cpu") as f:
        need = any(_is_moe_expert_scale(n) and len(f.get_slice(n).get_shape()) == 1 for n in names)
        if not need:
            print(f"skip {shard} (no 1-D MoE expert scales)", flush=True)
            continue
        out = {}
        for n in names:
            t = f.get_tensor(n)
            if _is_moe_expert_scale(n) and not _is_oe_scale(n) and t.ndim == 1:
                t = t.unsqueeze(-1)          # [N] -> [N,1] (MoE expert proj only; NOT OE embedders)
                fixed += 1
            out[n] = t
    save_file(out, f"{D}/{shard}", metadata={"format": "pt"})
    print(f"rewrote {shard}: {len(out)} tensors", flush=True)

print(f"DONE reshaped {fixed} weight_scale tensors -> [N,1]", flush=True)
