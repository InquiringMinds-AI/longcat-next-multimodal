#!/usr/bin/env python3
"""Encode specific PNGs -> raw [324,8] codebook ids, save dict to /tmp/real_codes.pt.
Reuses the visual tokenizer setup. Runs in the vision container."""
import os, json, glob, inspect, torch
from safetensors import safe_open
from PIL import Image
from transformers import Qwen2VLImageProcessor

ENC_P = "/models/output/LongCat-Next-NVFP4-bf16mla"
device, dtype = "cuda", torch.bfloat16
TOK_H = TOK_W = 18
SIDE = TOK_H * 2 * 14

class Cfg:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, str): setattr(self, k, Cfg(v) if isinstance(v, dict) else v)
    def get(self, k, default=None): return getattr(self, k, default)

enc_cfg = json.load(open(ENC_P + "/config.json"))
full = Cfg(enc_cfg); vc = enc_cfg.get("visual_config")
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
sig = inspect.signature(Qwen2_5_VLVisionConfig.__init__)
full.visual_config._hf_vision_config = Qwen2_5_VLVisionConfig(**{k: v for k, v in vc.items() if isinstance(k, str) and k in sig.parameters})
from sglang.srt.models.longcat_next_visual import LongcatNextVisualTokenizer
tok = LongcatNextVisualTokenizer(full).eval()
idx = json.load(open(glob.glob(ENC_P + "/*index*.json")[0]))["weight_map"]
sd = tok.state_dict(); want = {k for k in idx if k.startswith("model.visual_tokenizer.")}
byshard = {}
for k in want: byshard.setdefault(idx[k], []).append(k)
new = {}
for sf, keys in byshard.items():
    with safe_open(os.path.realpath(os.path.join(ENC_P, sf)), framework="pt") as f:
        for k in keys:
            local = k[len("model.visual_tokenizer."):]
            if local in sd: new[local] = f.get_tensor(k).to(sd[local].dtype)
tok.load_state_dict(new, strict=False); tok = tok.to(device).to(dtype)
ip = Qwen2VLImageProcessor.from_pretrained(ENC_P)

# images to encode: red circle (exact prompt match) + 2 food photos (natural)
targets = [("red_circle", "/tmp/red_rect.png")]
for i, p in enumerate(sorted(glob.glob("/tmp/calib_build/media_color/*.png"))[:2]):
    targets.append((f"food{i}", p))

out = {}
for name, path in targets:
    if not os.path.exists(path):
        print(f"  SKIP {name}: {path} missing"); continue
    img = Image.open(path).convert("RGB").resize((SIDE, SIDE), Image.BICUBIC)
    proc = ip(images=[img], return_tensors="pt", min_pixels=SIDE*SIDE, max_pixels=SIDE*SIDE)
    pv = proc["pixel_values"].to(device).to(dtype); thw = proc["image_grid_thw"]
    with torch.no_grad():
        vids = tok.encode(pv, thw)
    vids = (vids if isinstance(vids, torch.Tensor) else vids[0]).long().cpu()
    assert vids.shape == (324, 8), f"{name}: {tuple(vids.shape)}"
    out[name] = vids
    l0 = torch.unique(vids[:, 0]).numel()
    print(f"  {name}: [324,8] level0_unique={l0} from {os.path.basename(path)}")

torch.save(out, "/tmp/real_codes.pt")
print(f"saved {len(out)} real-code sets -> /tmp/real_codes.pt")
