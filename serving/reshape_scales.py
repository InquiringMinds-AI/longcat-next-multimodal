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

fixed = 0
for shard in sorted(byshard):
    names = byshard[shard]
    # quick scan: does this shard hold any 1-D weight_scale?
    with safe_open(f"{D}/{shard}", framework="pt", device="cpu") as f:
        need = any(n.endswith(".weight_scale") and len(f.get_slice(n).get_shape()) == 1 for n in names)
        if not need:
            print(f"skip {shard} (no 1-D scales)", flush=True)
            continue
        out = {}
        for n in names:
            t = f.get_tensor(n)
            if n.endswith(".weight_scale") and t.ndim == 1:
                t = t.unsqueeze(-1)          # [N] -> [N,1]
                fixed += 1
            out[n] = t
    save_file(out, f"{D}/{shard}", metadata={"format": "pt"})
    print(f"rewrote {shard}: {len(out)} tensors", flush=True)

print(f"DONE reshaped {fixed} weight_scale tensors -> [N,1]", flush=True)
