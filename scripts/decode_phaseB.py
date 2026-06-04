#!/usr/bin/env python3
"""
PHASE B — decode /tmp/gen_ids.pt ([324,8] raw codebook ids) -> /tmp/gen_image.png.
Reuses decode_roundtrip.py's PROVEN decode half verbatim:
  encoder tokenizer (for its VQ codebooks) + VisionTransformerDecoder + RefinerPipeline,
  then decode_image(ids, tok, image_decoder, refiner_pipeline, 18, 18).
Runs INSIDE the lc-vision container. Loads NOTHING of the backbone (memory-safe).
"""
import json, inspect, glob, os, sys, types, importlib, traceback
import torch
from safetensors import safe_open
from PIL import Image

# ---- make the inference-repo decoder modules importable (same as decode_roundtrip) ----
sys.path.insert(0, "/tmp")
for name in ["processor", "processor.decoder", "processor.decoder.omni_gen2_new"]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
pkg = sys.modules["processor.decoder.omni_gen2_new"]
pkg.__path__ = ["/tmp/lcdec"]
for sub in ["refiner_modules", "image_refiner", "modular_longcat_next_visual"]:
    m = importlib.import_module("lcdec." + sub)
    sys.modules["processor.decoder.omni_gen2_new." + sub] = m
    setattr(pkg, sub, m)

from lcdec.modular_longcat_next_visual import VisionTransformerDecoder, decode_image
from lcdec.refiner_modules import FlowMatchEulerDiscreteScheduler
from lcdec.image_refiner import ImageRefinerContainer, RefinerPipeline

# ---- pure-PyTorch flash_attn shims for the refiner (container flash_attn is a stub) ----
import lcdec.refiner_modules as _rm
import torch.nn.functional as _F


def _index_first_axis(t, indices):
    return t[indices]


def _unpad_input(hidden_states, attention_mask):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item())
    cu = _F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    bsz, slen = hidden_states.shape[0], hidden_states.shape[1]
    flat = hidden_states.reshape(bsz * slen, *hidden_states.shape[2:])
    return flat[indices], indices, cu, max_seqlen


def _pad_input(hidden_states, indices, batch, seqlen):
    out = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.reshape(batch, seqlen, *hidden_states.shape[1:])


def _flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                            dropout_p=0.0, causal=False, softmax_scale=None, **kw):
    outs = []
    cq = cu_seqlens_q.tolist(); ck = cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qs = q[cq[i]:cq[i + 1]]; ks = k[ck[i]:ck[i + 1]]; vs = v[ck[i]:ck[i + 1]]
        qh = qs.permute(1, 0, 2).unsqueeze(0); kh = ks.permute(1, 0, 2).unsqueeze(0)
        vh = vs.permute(1, 0, 2).unsqueeze(0)
        o = _F.scaled_dot_product_attention(qh, kh, vh, dropout_p=dropout_p,
                                            is_causal=causal, scale=softmax_scale)
        outs.append(o.squeeze(0).permute(1, 0, 2))
    return torch.cat(outs, dim=0)


_rm.index_first_axis = _index_first_axis
_rm.unpad_input = _unpad_input
_rm.pad_input = _pad_input
_rm.flash_attn_varlen_func = _flash_attn_varlen_func
print("[shim] installed pure-PyTorch flash_attn replacements in refiner_modules", flush=True)

ENC_P = os.environ.get("MODEL_PATH", "/models/output/LongCat-Next-NVFP4-bf16mla")
DEC_WEIGHTS = "/models/LongCat-Next/image_decoder/image_decoder.safetensors"
DEC_CFG_DIR = "/models/LongCat-Next/nmm_infer"
device = "cuda"; dtype = torch.bfloat16


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


def main():
    print("[B.1] building encoder tokenizer (for VQ codebooks decode_image indexes) ...", flush=True)
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

    def _real(rel):
        c = os.path.join(ENC_P, rel)
        if os.path.exists(c):
            return os.path.realpath(c)
        return os.path.join("/models/output/LongCat-Next-NVFP4", os.path.basename(rel))

    new = {}
    for sf, keys in byshard.items():
        with safe_open(_real(sf), framework="pt") as f:
            for k in keys:
                local = k[len("model.visual_tokenizer."):]
                if local in sd:
                    new[local] = f.get_tensor(k).to(sd[local].dtype)
    miss = tok.load_state_dict(new, strict=False)
    print(f"    loaded {len(new)} encoder tensors; missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}", flush=True)
    tok = tok.to(device).to(dtype)

    print("[B.2] building decoder + refiner ...", flush=True)
    vdc_dict = json.load(open(os.path.join(DEC_CFG_DIR, "config.json")))["visual_decoder_config"]
    vd_config = Cfg(vdc_dict)
    image_decoder = VisionTransformerDecoder.from_pretrained(
        vd_config.image_decoder_config, DEC_WEIGHTS).to(device=device, dtype=dtype)
    image_refiner = ImageRefinerContainer.from_pretrained(
        vd_config, DEC_WEIGHTS).to(device=device, dtype=dtype)
    sc = vd_config.scheduler_config
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=sc.num_train_timesteps, dynamic_time_shift=sc.dynamic_time_shift)
    refiner_pipeline = RefinerPipeline(
        vae=image_refiner.vae, transformer=image_refiner.base_transformer,
        scheduler=scheduler, cond_proj=image_refiner.cond_proj)
    refiner_pipeline.set_progress_bar_config(disable=False)
    print("    decoder spatial_merge_size =", image_decoder.spatial_merge_size, flush=True)

    _tag = os.environ.get("GEN_TAG", "")
    _in = f"/tmp/gen_ids_{_tag}.pt" if _tag else "/tmp/gen_ids.pt"
    print(f"[B.3] loading {_in} ...", flush=True)
    gen_ids = torch.load(_in).to(device)
    print(f"    gen_ids shape={tuple(gen_ids.shape)} dtype={gen_ids.dtype} "
          f"min={int(gen_ids.min())} max={int(gen_ids.max())}", flush=True)
    assert gen_ids.shape == (324, 8), f"expected [324,8], got {tuple(gen_ids.shape)}"

    print("[B.4] decode_image ...", flush=True)
    with torch.no_grad():
        refined = decode_image(gen_ids, tok, image_decoder, refiner_pipeline, 18, 18)
    _outpng = f"/tmp/gen_image_{_tag}.png" if _tag else "/tmp/gen_image.png"
    refined[0].save(_outpng)
    # objective color stats so the headless caller can judge chroma
    import numpy as _np
    _arr = _np.asarray(refined[0].convert("RGB")).astype(_np.float32)
    _mean = _arr.reshape(-1, 3).mean(axis=0)
    _spread = float(_mean.max() - _mean.min())
    print(f"=== SAVED {_outpng} size={refined[0].size} mode={refined[0].mode} ===", flush=True)
    print(f"=== MEANRGB R={_mean[0]:.1f} G={_mean[1]:.1f} B={_mean[2]:.1f} "
          f"spread={_spread:.1f} chromatic={_spread>25} ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
