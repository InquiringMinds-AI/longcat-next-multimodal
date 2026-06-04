#!/usr/bin/env python3
"""
LongCat-Next IMAGE DECODE round-trip test.

real image -> visual tokenizer encode -> [N,8] codebook ids
           -> decode_image(...) -> reconstructed PNG

Runs inside the lc-vision container.
- Encoder: LongcatNextVisualTokenizer (sglang overlay), weights from NVFP4 model
  at /models/output/LongCat-Next-NVFP4-bf16mla. Provides .visual_bridge_model
  with the VQ codebooks that decode_image indexes.
- Decoder + refiner: inference-repo omni_gen2_new modules, copied to /tmp/lcdec.
  Weights from /models/LongCat-Next/image_decoder/image_decoder.safetensors.
- visual_decoder_config: from /models/LongCat-Next/nmm_infer/config.json.
"""
import json, inspect, glob, os, sys, traceback
import torch
from safetensors import safe_open
from PIL import Image

# ---- make the inference-repo decoder modules importable as a package ----
# The modules use absolute imports `from processor.decoder.omni_gen2_new.X import ...`.
# Build a fake package tree pointing at /tmp/lcdec so those imports resolve.
import types, importlib
sys.path.insert(0, "/tmp")
for name in ["processor", "processor.decoder", "processor.decoder.omni_gen2_new"]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
pkg = sys.modules["processor.decoder.omni_gen2_new"]
pkg.__path__ = ["/tmp/lcdec"]  # make it a package whose submodules load from lcdec
for sub in ["refiner_modules", "image_refiner", "modular_longcat_next_visual"]:
    m = importlib.import_module("lcdec." + sub)
    sys.modules["processor.decoder.omni_gen2_new." + sub] = m
    setattr(pkg, sub, m)

from lcdec.modular_longcat_next_visual import VisionTransformerDecoder, decode_image
from lcdec.refiner_modules import FlowMatchEulerDiscreteScheduler
from lcdec.image_refiner import ImageRefinerContainer, RefinerPipeline

# ---- flash_attn shim: the container's flash_attn is a hollow stub, so the
# refiner's varlen attention symbols are None. Provide pure-PyTorch drop-ins. ----
import lcdec.refiner_modules as _rm
import torch.nn.functional as _F


def _index_first_axis(t, indices):
    # t: [N, ...]; indices: [M] -> [M, ...]
    return t[indices]


def _unpad_input(hidden_states, attention_mask):
    # hidden_states: [B, S, ...]; attention_mask: [B, S] (1=keep)
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item())
    cu = _F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    bsz, slen = hidden_states.shape[0], hidden_states.shape[1]
    flat = hidden_states.reshape(bsz * slen, *hidden_states.shape[2:])
    return flat[indices], indices, cu, max_seqlen


def _pad_input(hidden_states, indices, batch, seqlen):
    # hidden_states: [total, ...] -> [batch*seqlen, ...] scatter -> [batch, seqlen, ...]
    out = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.reshape(batch, seqlen, *hidden_states.shape[1:])


def _flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                            max_seqlen_q, max_seqlen_k,
                            dropout_p=0.0, causal=False, softmax_scale=None, **kw):
    # q:[Tq,H,D] k,v:[Tk,H,D] packed varlen. Segment by cu_seqlens, run SDPA per seq.
    outs = []
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qs = q[cq[i]:cq[i + 1]]          # [sq,H,D]
        ks = k[ck[i]:ck[i + 1]]          # [sk,H,D]
        vs = v[ck[i]:ck[i + 1]]
        qh = qs.permute(1, 0, 2).unsqueeze(0)  # [1,H,sq,D]
        kh = ks.permute(1, 0, 2).unsqueeze(0)
        vh = vs.permute(1, 0, 2).unsqueeze(0)
        o = _F.scaled_dot_product_attention(qh, kh, vh, dropout_p=dropout_p,
                                            is_causal=causal, scale=softmax_scale)
        outs.append(o.squeeze(0).permute(1, 0, 2))  # [sq,H,D]
    return torch.cat(outs, dim=0)


_rm.index_first_axis = _index_first_axis
_rm.unpad_input = _unpad_input
_rm.pad_input = _pad_input
_rm.flash_attn_varlen_func = _flash_attn_varlen_func
print("[shim] installed pure-PyTorch flash_attn replacements in refiner_modules", flush=True)

ENC_P = "/models/output/LongCat-Next-NVFP4-bf16mla"          # encoder weights + image processor
DEC_WEIGHTS = "/models/LongCat-Next/image_decoder/image_decoder.safetensors"
DEC_CFG_DIR = "/models/LongCat-Next/nmm_infer"               # has visual_decoder_config

device = "cuda"
dtype = torch.bfloat16


# ---------- attribute-accessible config wrapper ----------
class Cfg:
    def __init__(self, d):
        for k, v in d.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, dict) and all(isinstance(kk, str) for kk in v):
                v = Cfg(v)
            setattr(self, k, v)
    def get(self, k, default=None):
        return getattr(self, k, default)


# ===================== 1. BUILD ENCODER TOKENIZER =====================
print("[1] building encoder tokenizer ...", flush=True)
enc_cfg_dict = json.load(open(ENC_P + "/config.json"))
full = Cfg(enc_cfg_dict)
vc = enc_cfg_dict.get("visual_config")
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
sig = inspect.signature(Qwen2_5_VLVisionConfig.__init__)
full.visual_config._hf_vision_config = Qwen2_5_VLVisionConfig(
    **{k: v for k, v in vc.items() if isinstance(k, str) and k in sig.parameters}
)
from sglang.srt.models.longcat_next_visual import LongcatNextVisualTokenizer
tok = LongcatNextVisualTokenizer(full).eval()

idx = json.load(open(glob.glob(ENC_P + "/*index*.json")[0]))["weight_map"]
sd = tok.state_dict()
want = {k for k in idx if k.startswith("model.visual_tokenizer.")}
byshard = {}
for k in want:
    byshard.setdefault(idx[k], []).append(k)
new = {}
for sf, keys in byshard.items():
    with safe_open(os.path.realpath(os.path.join(ENC_P, sf)), framework="pt") as f:
        for k in keys:
            local = k[len("model.visual_tokenizer."):]
            if local in sd:
                new[local] = f.get_tensor(k).to(sd[local].dtype)
miss = tok.load_state_dict(new, strict=False)
print(f"    loaded {len(new)} encoder tensors; missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}", flush=True)
tok = tok.to(device).to(dtype)


# ===================== 2. BUILD DECODER + REFINER =====================
print("[2] building decoder + refiner from decoder safetensors ...", flush=True)
vdc_dict = json.load(open(os.path.join(DEC_CFG_DIR, "config.json")))["visual_decoder_config"]
vd_config = Cfg(vdc_dict)

image_decoder = VisionTransformerDecoder.from_pretrained(
    vd_config.image_decoder_config, DEC_WEIGHTS
).to(device=device, dtype=dtype)

image_refiner = ImageRefinerContainer.from_pretrained(
    vd_config, DEC_WEIGHTS
).to(device=device, dtype=dtype)

sc = vd_config.scheduler_config
scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=sc.num_train_timesteps,
    dynamic_time_shift=sc.dynamic_time_shift,
)
refiner_pipeline = RefinerPipeline(
    vae=image_refiner.vae,
    transformer=image_refiner.base_transformer,
    scheduler=scheduler,
    cond_proj=image_refiner.cond_proj,
)
refiner_pipeline.set_progress_bar_config(disable=False)
print("    decoder spatial_merge_size =", image_decoder.spatial_merge_size, flush=True)


# ===================== 3. ENCODE + DECODE PER IMAGE =====================
from transformers import Qwen2VLImageProcessor
ip = Qwen2VLImageProcessor.from_pretrained(ENC_P)

# 18 merged tokens * spatial_merge_size(2) * patch(14) = 504 px -> grid_thw [1,36,36] -> [324,8] ids
TOKENS_H = TOKENS_W = 18
SIDE = TOKENS_H * image_decoder.spatial_merge_size * image_decoder.patch_size  # 504


def run_one(in_path, out_path):
    print(f"\n=== {in_path} -> {out_path} ===", flush=True)
    img = Image.open(in_path).convert("RGB").resize((SIDE, SIDE), Image.BICUBIC)
    proc = ip(images=[img], return_tensors="pt", min_pixels=SIDE * SIDE, max_pixels=SIDE * SIDE)
    pv = proc["pixel_values"].to(device).to(dtype)
    thw = proc["image_grid_thw"]
    print("    pixel_values:", tuple(pv.shape), "grid_thw:", thw.tolist(), flush=True)
    with torch.no_grad():
        vids = tok.encode(pv, thw)
    vids = vids if isinstance(vids, torch.Tensor) else vids[0]
    print("    ids shape:", tuple(vids.shape), "dtype:", vids.dtype,
          "min/max:", int(vids.min()), int(vids.max()), flush=True)
    assert vids.shape == (TOKENS_H * TOKENS_W, 8), f"expected [{TOKENS_H*TOKENS_W},8], got {tuple(vids.shape)}"

    with torch.no_grad():
        refined = decode_image(vids, tok, image_decoder, refiner_pipeline, TOKENS_H, TOKENS_W)
    refined[0].save(out_path)
    print("    SAVED", out_path, "size:", refined[0].size, "mode:", refined[0].mode, flush=True)
    return refined[0].size


results = {}
for inp, outp in [("/tmp/red_rect.png", "/tmp/recon_red_rect.png"),
                  ("/tmp/lc_test.png", "/tmp/recon_ironing.png")]:
    try:
        results[outp] = run_one(inp, outp)
    except Exception:
        print("    ERROR:")
        traceback.print_exc()
        results[outp] = None

print("\n===== SUMMARY =====")
for k, v in results.items():
    print(f"  {k}: {'OK '+str(v) if v else 'FAILED'}")
