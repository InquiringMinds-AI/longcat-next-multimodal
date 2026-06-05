#!/usr/bin/env python3
"""SmoothQuant the LongCat MoE experts (gate/up only) in-place in the int8 checkpoint.
 s[L] = act_max[L]^a / weight_max[L]^(1-a)   (per hidden/input channel, a=0.5)
 - weight_max from the BF16 source experts (per input channel, over outputs+experts, gate&up)
 - re-quantize gate/up = (W_bf16 * s) -> per-output-channel int8 + .weight_scale, write to /dst
 - store s as model.layers.L.mlp.smooth_scale buffer; down_proj untouched
Runtime (separate patch): expert input is divided by s before the experts."""
import os, json, re, torch
from collections import defaultdict
from safetensors.torch import safe_open, save_file

SRC="/src"; DST="/dst"; ALPHA=0.5
act_max = torch.load("/tmp/sq_actmax.pt", map_location="cpu")  # {layer_id: [hidden]}
idx = json.load(open(f"{DST}/model.safetensors.index.json")); wm = idx["weight_map"]
src_idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
GU = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up)_proj\.weight$")

# group source gate/up weights by layer
layer_gu = defaultdict(list)
for n in src_idx:
    m = GU.match(n)
    if m: layer_gu[int(m.group(1))].append(n)

# Pass 1: weight_max per layer (per input channel = dim 1 of [out,in])
print("[sq] computing weight_max per layer...", flush=True)
wmax = {}
src_byshard = defaultdict(list)
for n in src_idx:
    if GU.match(n): src_byshard[src_idx[n]].append(n)
for shard in sorted(src_byshard):
    with safe_open(f"{SRC}/{shard}", framework="pt", device="cpu") as f:
        for n in src_byshard[shard]:
            L = int(GU.match(n).group(1))
            w = f.get_tensor(n).float().abs().amax(0)  # [in]
            wmax[L] = w if L not in wmax else torch.maximum(wmax[L], w)
    print(f"[sq] wmax scanned {shard}", flush=True)

# smoothing scale per layer
s = {}
for L in sorted(wmax):
    a = act_max[L].float().clamp(min=1e-5); wv = wmax[L].clamp(min=1e-5)
    s[L] = (a.pow(ALPHA) / wv.pow(1-ALPHA)).clamp(min=1e-4, max=1e4)
    print(f"[sq] layer {L}: s range [{float(s[L].min()):.3f}, {float(s[L].max()):.3f}]", flush=True)

def qc(w):  # per-output-channel symmetric int8; scale kept [out,1] (loader needs 2D)
    sc=(w.abs().amax(1,keepdim=True)/127.0).clamp(min=1e-8)
    return (w/sc).round().clamp(-127,127).to(torch.int8), sc.to(torch.float32)

# Pass 2: re-quantize smoothed gate/up into the DST checkpoint shards; add smooth_scale buffers
dst_byshard=defaultdict(list)
for n in wm: dst_byshard[wm[n]].append(n)
new_entries={}
for shard in sorted(dst_byshard):
    names=dst_byshard[shard]
    gu_here=[n for n in names if GU.match(n)]
    if not gu_here: continue
    out={}
    with safe_open(f"{DST}/{shard}", framework="pt", device="cpu") as f:
        for n in names: out[n]=f.get_tensor(n)
    # re-quantize each gate/up from BF16 source * s
    for n in gu_here:
        L=int(GU.match(n).group(1))
        sh=src_idx[n]
        with safe_open(f"{SRC}/{sh}", framework="pt", device="cpu") as fs:
            wbf=fs.get_tensor(n).float()
        wsm=wbf * s[L].unsqueeze(0)  # scale input dim
        q,sc=qc(wsm)
        out[n]=q; out[n.replace(".weight",".weight_scale")]=sc
    save_file(out, f"{DST}/{shard}", metadata={"format":"pt"})
    print(f"[sq] requantized {len(gu_here)} gate/up in {shard}", flush=True)

# store smooth_scale buffers in a small extra shard
extra={}
for L in sorted(s): extra[f"model.layers.{L}.mlp.smooth_scale"]=s[L].to(torch.float32)
save_file(extra, f"{DST}/model-smoothscale.safetensors", metadata={"format":"pt"})
for k in extra: wm[k]="model-smoothscale.safetensors"
idx["weight_map"]=wm
json.dump(idx, open(f"{DST}/model.safetensors.index.json","w"), indent=2)
print(f"[sq] DONE: smoothed {len(s)} layers, stored {len(extra)} smooth_scale buffers", flush=True)
