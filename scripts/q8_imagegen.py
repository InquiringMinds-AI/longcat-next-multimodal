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

ids = tok(PROMPT, return_tensors="pt").input_ids
ids = torch.cat([ids, torch.tensor([[img_start]], dtype=ids.dtype)], dim=1).to("cuda")
print(f"[gen] prompt={PROMPT!r} input {tuple(ids.shape)} tail={ids[0,-4:].tolist()}", flush=True)

t1 = time.time()
with torch.no_grad():
    out = model.generate(input_ids=ids, generation_config=gc, return_dict_in_generate=False)
dt = time.time() - t1
# out = (input_ids, visual_ids, audio_ids, audio_text_ids)
visual_ids = out[1] if isinstance(out, (tuple, list)) else out.visual_ids
visual_ids = visual_ids.detach().cpu()
print(f"[gen] DONE {dt/60:.1f}min  visual_ids {tuple(visual_ids.shape)} dtype={visual_ids.dtype} min={int(visual_ids.min())} max={int(visual_ids.max())}", flush=True)

# raw-ify: if offset (>=150581) subtract per-level visual_offset_vals
voff = model.model.visual_offset_vals.detach().cpu()
if int(visual_ids.max()) >= int(voff[0]):
    print(f"[gen] subtracting visual_offset_vals {voff.tolist()}", flush=True)
    visual_ids = visual_ids - voff.view(1, -1)
visual_ids = visual_ids.long().clamp(min=0)
n = visual_ids.shape[0]
print(f"[gen] {n} positions; level0_unique={torch.unique(visual_ids[:,0]).numel()} raw min/max={int(visual_ids.min())}/{int(visual_ids.max())}", flush=True)
if n >= 324:
    torch.save(visual_ids[:324].contiguous(), f"/tmp/gen_ids_{TAG}.pt")
    print(f"[gen] saved [324,8] -> /tmp/gen_ids_{TAG}.pt", flush=True)
else:
    torch.save(visual_ids.contiguous(), f"/tmp/gen_ids_{TAG}_partial.pt")
    print(f"[gen] only {n}<324 positions; saved partial", flush=True)
print("BF16 GEN DONE", flush=True)
