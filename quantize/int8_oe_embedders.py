#!/usr/bin/env python3
"""In-place: per-row int8-quantize the N-gram OE embedders so the port's NgramEmbedding
int8 load path fires (it expects int8 embedders + .weight_scale; BF16 hits the broken
size-1-placeholder legacy path -> OOB). Keeps OE at ~16GB int8 (vs 62GB BF16) -> under ceiling.
Emits embedders.N.weight (int8 [rows,dim]) + embedders.N.weight_scale (bf16 [rows]); updates index."""
import json, torch
from collections import defaultdict
from safetensors.torch import safe_open, save_file

D = "/dst"
EMB = ".ngram_embeddings.embedders."
idx = json.load(open(f"{D}/model.safetensors.index.json"))
wm = idx["weight_map"]
byshard = defaultdict(list)
for n, s in wm.items():
    byshard[s].append(n)

new_scale_entries, done = {}, 0
for shard in sorted(byshard):
    names = byshard[shard]
    with safe_open(f"{D}/{shard}", framework="pt", device="cpu") as f:
        has = any(EMB in n and n.endswith(".weight") and len(f.get_slice(n).get_shape()) == 2
                  and f.get_slice(n).get_dtype() not in ("I8", "int8") for n in names)
        if not has:
            print(f"skip {shard}", flush=True); continue
        out = {}
        for n in names:
            t = f.get_tensor(n)
            if EMB in n and n.endswith(".weight") and t.dtype != torch.int8 and t.ndim == 2:
                wf = t.float()
                scale = (wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0)  # [rows,1]
                out[n] = (wf / scale).round().clamp(-127, 127).to(torch.int8)            # [rows,dim] int8
                sn = n.replace(".weight", ".weight_scale")
                out[sn] = scale.squeeze(-1).to(torch.bfloat16)                           # [rows] per-row scale
                new_scale_entries[sn] = shard
                done += 1
            else:
                out[n] = t
    save_file(out, f"{D}/{shard}", metadata={"format": "pt"})
    print(f"int8'd OE embedders in {shard}: {len(out)} tensors", flush=True)

wm.update(new_scale_entries)
idx["weight_map"] = wm
json.dump(idx, open(f"{D}/model.safetensors.index.json", "w"), indent=2)
print(f"DONE int8'd {done} OE embedders, added {len(new_scale_entries)} scale tensors", flush=True)
