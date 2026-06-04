#!/usr/bin/env python3
"""Build FORMAT-MATCHED image-gen calibration sequences (raw-completion, no chat template,
canonical space before <img_token_size>) -> ~/calib_assets/image_gen_calib_color.pt.
Encodes /tmp/calib_build/media_color/*.png via the visual tokenizer. Runs in the vision container."""
import os, json, glob, inspect, torch
from safetensors import safe_open
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLImageProcessor

ENC_P = "/models/output/LongCat-Next-NVFP4-bf16mla"
device, dtype = "cuda", torch.bfloat16
IMG_START, IMG_END, IMG_NEWLINE = 131106, 131107, 131109
CB, VOFF = 16384, 150581
VOFF_VALS = [VOFF + i*CB for i in range(8)]
TOK_H = TOK_W = 18

class Cfg:
    def __init__(self, d):
        for k, v in d.items():
            if not isinstance(k, str): continue
            setattr(self, k, Cfg(v) if isinstance(v, dict) else v)
    def get(self, k, default=None): return getattr(self, k, default)

print("[1] building encoder tokenizer ...", flush=True)
enc_cfg = json.load(open(ENC_P + "/config.json"))
full = Cfg(enc_cfg); vc = enc_cfg.get("visual_config")
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
sig = inspect.signature(Qwen2_5_VLVisionConfig.__init__)
full.visual_config._hf_vision_config = Qwen2_5_VLVisionConfig(**{k:v for k,v in vc.items() if isinstance(k,str) and k in sig.parameters})
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
miss = tok.load_state_dict(new, strict=False)
print(f"    loaded {len(new)} encoder tensors; missing={len(miss.missing_keys)}", flush=True)
tok = tok.to(device).to(dtype)

txt = AutoTokenizer.from_pretrained(ENC_P, trust_remote_code=True)
ip = Qwen2VLImageProcessor.from_pretrained(ENC_P)
SIDE = TOK_H * 2 * 14  # 504

PROMPTS = ["A vivid color photograph of a colorful dish of food.","A bright saturated close-up color photo.","A richly colored photograph of a meal.","A vibrant color photo with strong reds and greens.","A glossy colorful food photograph.","A high-color natural photograph.","A saturated, colorful close-up image.","A vivid photograph full of color."]

pngs = sorted(glob.glob("/tmp/calib_build/media_color/*.png"))
print(f"[2] encoding {len(pngs)} images -> raw-format sequences ...", flush=True)
seqs = []
for i, p in enumerate(pngs):
    img = Image.open(p).convert("RGB").resize((SIDE, SIDE), Image.BICUBIC)
    proc = ip(images=[img], return_tensors="pt", min_pixels=SIDE*SIDE, max_pixels=SIDE*SIDE)
    pv = proc["pixel_values"].to(device).to(dtype); thw = proc["image_grid_thw"]
    with torch.no_grad():
        vids = tok.encode(pv, thw)
    vids = vids if isinstance(vids, torch.Tensor) else vids[0]
    assert vids.shape == (TOK_H*TOK_W, 8), f"{p}: {tuple(vids.shape)}"
    raw = vids.long().cpu()
    prompt = PROMPTS[i % len(PROMPTS)]
    head_text = prompt.rstrip() + " <longcat_img_token_size>18 18</longcat_img_token_size>"  # RAW, canonical space
    head = txt.encode(head_text)
    seq = list(head) + [IMG_START]
    off = (raw + torch.tensor(VOFF_VALS)).tolist()  # [324][8]
    k = 0
    for r in range(TOK_H):
        for c in range(TOK_W):
            seq.extend(off[k]); k += 1
        seq.append(IMG_NEWLINE)
    seq.append(IMG_END)
    seqs.append(torch.tensor([seq], dtype=torch.long))
    if i < 2 or i % 10 == 0:
        print(f"    [{i}] {os.path.basename(p)} head={len(head)}tok seqlen={len(seq)} headtext={head_text!r}", flush=True)

out = os.path.expanduser("~/calib_assets/image_gen_calib_color.pt")
torch.save(seqs, out)
print(f"[3] saved {len(seqs)} sequences -> {out}", flush=True)
print(f"    seqlen range [{min(s.shape[1] for s in seqs)}, {max(s.shape[1] for s in seqs)}]", flush=True)
