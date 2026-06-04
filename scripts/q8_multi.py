#!/usr/bin/env python3
"""Generate ONE image with the REAL BF16 LongCat-Next weights (disk-offloaded) via the
model's native generate(). Canonical visual settings (cfg 3.0), token_h/w=18 to reuse our
decoder. Saves raw visual_ids -> /tmp/gen_ids_bf16toad.pt for decode_phaseB."""
import os, time, torch
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q
if not hasattr(_q, "Qwen2RMSNorm") and hasattr(_q, "Qwen2_5_VLRMSNorm"):
    _q.Qwen2RMSNorm = _q.Qwen2_5_VLRMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

MP = "/home/magi/models/LongCat-Next"
OFF = "/home/magi/lc_offload"; os.makedirs(OFF, exist_ok=True)
PROMPT = os.environ.get("PROMPT", "A photograph of a toad sitting on grass.")
TAG = os.environ.get("TAG", "bf16toad")

print(f"[load] transformers {__import__('transformers').__version__}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["router","classifier","lm_head","visual_head","audio_head","visual_tokenizer","audio_tokenizer"])
model = AutoModelForCausalLM.from_pretrained(
    MP, trust_remote_code=True, torch_dtype=torch.bfloat16,
    quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa",
)
model.eval()
model.text_tokenizer = tok  # prepare_inputs uses self.text_tokenizer
print(f"[load] DONE {time.time()-t0:.0f}s", flush=True)

vcfg = model.config.visual_config
img_start = vcfg.image_start_token_id
print(f"[cfg] image_start_token_id={img_start}", flush=True)

gc = GenerationConfig.from_pretrained(MP)
# 18x18 to reuse our proven decoder; keep canonical sampling (cfg 3.0, temp 0.5, top_p .75, top_k 1024)
gc.visual_generation_config["custom_params"]["token_h"] = 18
gc.visual_generation_config["custom_params"]["token_w"] = 18
gc.max_new_tokens = 420
print(f"[cfg] visual_generation_config={gc.visual_generation_config}", flush=True)


# --- install SDPA varlen fallback for the depth-transformer (container has no flash_attn) ---
import sys as _sys, torch.nn.functional as _F
def _sdpa_varlen(q, k, v, cu_q, cu_k, max_q, max_k=None, causal=False, *a, **kw):
    outs = []; cq = cu_q.tolist(); ck = cu_k.tolist()
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i+1]].unsqueeze(0).transpose(1, 2)
        ki = k[ck[i]:ck[i+1]].unsqueeze(0).transpose(1, 2)
        vi = v[ck[i]:ck[i+1]].unsqueeze(0).transpose(1, 2)
        oi = _F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
        outs.append(oi.transpose(1, 2).squeeze(0))
    return torch.cat(outs, dim=0)
_npatch = 0
for _name, _m in list(_sys.modules.items()):
    if _name.endswith("modular_longcat_next") and hasattr(_m, "flash_attn_varlen_func"):
        _m.flash_attn_varlen_func = _sdpa_varlen; _npatch += 1
        print(f"[shim] patched flash_attn_varlen_func in {_name}", flush=True)
print(f"[shim] patched {_npatch} module(s)", flush=True)

PROMPTS = [
    ("q8_cat",  "A photograph of an orange tabby cat sitting on a windowsill."),
    ("q8_land", "A photograph of a mountain landscape under a blue sky."),
    ("q8_apple","A photograph of a red apple on a wooden table."),
    ("q8_circle","A single large red circle on a white background."),
]
import torch as _t
voff = model.model.visual_offset_vals.detach().cpu()
for tag, prompt in PROMPTS:
    ids = tok(prompt, return_tensors="pt").input_ids
    ids = _t.cat([ids, _t.tensor([[img_start]], dtype=ids.dtype)], dim=1).to("cuda")
    print(f"[gen] {tag} prompt={prompt!r} input {tuple(ids.shape)}", flush=True)
    import time as _tm; t1=_tm.time()
    with _t.no_grad():
        out = model.generate(input_ids=ids, generation_config=gc, return_dict_in_generate=False)
    vis = (out[1] if isinstance(out,(tuple,list)) else out.visual_ids).detach().cpu()
    if int(vis.max()) >= int(voff[0]):
        vis = vis - voff.view(1,-1)
    vis = vis.long().clamp(min=0)
    n = vis.shape[0]
    print(f"[gen] {tag} DONE {(_tm.time()-t1)/60:.1f}min  pos={n} l0u={_t.unique(vis[:,0]).numel()} min/max={int(vis.min())}/{int(vis.max())}", flush=True)
    if n >= 324:
        _t.save(vis[:324].contiguous(), f"/tmp/gen_ids_{tag}.pt")
        print(f"[gen] saved /tmp/gen_ids_{tag}.pt", flush=True)
    else:
        _t.save(vis.contiguous(), f"/tmp/gen_ids_{tag}_partial.pt"); print(f"[gen] partial {n}", flush=True)
print("Q8 MULTI DONE", flush=True)
