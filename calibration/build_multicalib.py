#!/usr/bin/env python3
"""
Build LongCat-Next-NVFP4-multicalib-bf16mla by MERGE-applying recalibrated
input_scales onto the existing -bf16mla checkpoint.

MERGE rule (per input_scale tensor, matched by json_key + ".input_scale"):
  - json value MEASURED (!= DEFAULT) -> overwrite existing with new value
  - json value == DEFAULT (cold expert) OR no json key -> KEEP existing value

Preserves -bf16mla MLA BF16 (self_attn) work. Rewrites only shards whose
input_scale tensors actually change; symlinks the rest; rebuilds index/aux files.
"""
import json, os, glob, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC   = "/models/output/LongCat-Next-NVFP4-bf16mla"          # build FROM (preserves MLA bf16)
SCALES= os.environ.get("SCALES", "/models/output/LongCat-Next-NVFP4-multicalib/input_scales.json")
OUT   = os.environ.get("OUT", "/models/output/LongCat-Next-NVFP4-multicalib-bf16mla")
DEFAULT = 0.0003720238095238095
def is_default(v): return abs(float(v) - DEFAULT) < 1e-9

new = json.load(open(SCALES))
print(f"new scales json: {len(new)} keys")

# resolve real path of a (possibly symlinked) shard file in SRC
def real(fn): return os.path.realpath(os.path.join(SRC, fn))

idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
isc_tensors = [k for k in wm if k.endswith(".input_scale")]
print(f"input_scale tensors: {len(isc_tensors)}")

# decide per-tensor action, group by shard
from collections import defaultdict, Counter
shard_updates = defaultdict(dict)   # shard_fn -> {tensor_name: new_value}
updated = kept_cold = kept_nokey = 0
for t in isc_tensors:
    base = t[:-len(".input_scale")]
    shard = wm[t]
    if base in new and not is_default(new[base]):
        shard_updates[shard][t] = float(new[base]); updated += 1
    elif base in new:
        kept_cold += 1
    else:
        kept_nokey += 1
print(f"UPDATED(measured)={updated}  KEPT_cold(default)={kept_cold}  KEPT_nokey={kept_nokey}")
print("shards with changes:")
for s in sorted(shard_updates): print(f"  {s}: {len(shard_updates[s])}")

os.makedirs(OUT, exist_ok=True)

# all shard filenames
all_shards = sorted({fn for fn in wm.values()})
changed = set(shard_updates)
unchanged = [s for s in all_shards if s not in changed]

# 1) symlink every unchanged shard to its SRC real path
for fn in unchanged:
    dst = os.path.join(OUT, fn)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(real(fn), dst)
print(f"symlinked {len(unchanged)} unchanged shards")

# 2) rewrite changed shards: read from SRC real path, patch the input_scale tensors
for fn in sorted(changed):
    srcpath = real(fn)
    upd = shard_updates[fn]
    tensors = {}
    patched = 0
    with safe_open(srcpath, framework="pt") as sf:
        meta = sf.metadata()
        for k in sf.keys():
            tt = sf.get_tensor(k)
            if k in upd:
                tt = torch.tensor(upd[k], dtype=tt.dtype).reshape(tt.shape)
                patched += 1
            tensors[k] = tt
    dst = os.path.join(OUT, fn)
    if os.path.lexists(dst): os.remove(dst)
    save_file(tensors, dst, metadata=meta if meta else {"format": "pt"})
    print(f"  wrote {fn}: patched {patched}/{len(upd)} expected")

# 3) copy/symlink all non-shard files (config, tokenizer, index, json, decoder dirs, etc.)
for fn in os.listdir(SRC):
    if fn in all_shards: continue
    src = os.path.join(SRC, fn)
    dst = os.path.join(OUT, fn)
    if os.path.lexists(dst): continue
    if os.path.islink(src):
        os.symlink(os.path.realpath(src), dst)
    elif os.path.isdir(src):
        os.symlink(os.path.realpath(src), dst)
    else:
        shutil.copy2(src, dst)
# overwrite the stale input_scales.json copy with the new one (cosmetic provenance)
shutil.copy2(SCALES, os.path.join(OUT, "input_scales.json"))
print("copied aux files + index")
print("DONE build")
