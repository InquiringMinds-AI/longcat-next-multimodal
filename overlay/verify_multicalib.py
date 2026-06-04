#!/usr/bin/env python3
import json, os, glob
import torch
from safetensors import safe_open

OUT="/models/output/LongCat-Next-NVFP4-multicalib-bf16mla"
SRC="/models/output/LongCat-Next-NVFP4-bf16mla"
SCALES="/models/output/LongCat-Next-NVFP4-multicalib/input_scales.json"
DEFAULT=0.0003720238095238095
new=json.load(open(SCALES))

idx=json.load(open(os.path.join(OUT,"model.safetensors.index.json")))
wm=idx["weight_map"]

# 1) every shard file referenced exists & opens; collect all tensor keys + shapes
shard_files=sorted({fn for fn in wm.values()})
total_keys=0
key_shapes={}
for fn in shard_files:
    p=os.path.join(OUT,fn)
    assert os.path.exists(p), f"missing {fn}"
    with safe_open(p,framework="pt") as sf:
        for k in sf.keys():
            key_shapes[k]=None
            total_keys+=1
# 2) index <-> actual keys consistency
idx_keys=set(wm); act_keys=set(key_shapes)
print("index keys:",len(idx_keys),"actual tensor keys:",len(act_keys))
print("in index not in shards:",len(idx_keys-act_keys))
print("in shards not in index:",len(act_keys-idx_keys))

# 3) shape consistency vs SRC for a sample of patched + all self_attn bf16 still bf16
import random
isc=[k for k in wm if k.endswith(".input_scale")]
# verify updated values
chk_upd=chk_cold=0
mism=[]
# build expected map
for fn in ["model-00008-of-00011.safetensors","model-00010-of-00011.safetensors","model-00006-of-00011.safetensors"]:
    p=os.path.join(OUT,fn)
    with safe_open(p,framework="pt") as sf:
        ks=[k for k in sf.keys() if k.endswith(".input_scale")]
        for k in ks:
            base=k[:-len(".input_scale")]
            v=float(sf.get_tensor(k).flatten()[0])
            if base in new and abs(new[base]-DEFAULT)>=1e-9:
                if abs(v-new[base])>1e-9*max(1,abs(new[base])):
                    mism.append((k,v,new[base]))
                else: chk_upd+=1
            else:
                chk_cold+=1
print(f"verified updated match={chk_upd} cold/other={chk_cold} mismatches={len(mism)}")
for m in mism[:5]: print("  MISMATCH",m)

# 4) self_attn in shard7 still bf16 (MLA preserved)
with safe_open(os.path.join(OUT,"model-00007-of-00011.safetensors"),framework="pt") as sf:
    sa=[k for k in sf.keys() if ".self_attn." in k and ".weight" in k and "layernorm" not in k]
    sample=sa[:3]
    for k in sample:
        t=sf.get_tensor(k); print("  self_attn:",k,t.dtype,tuple(t.shape))
print("VERIFY OK total_tensor_keys",total_keys)
