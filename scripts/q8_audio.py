#!/usr/bin/env python3
"""8-bit (int8) canonical Speech-Synthesis (voice clone) with LongCat-Next, per the model card.
Tests whether 8-bit precision lifts the audio drone (same RVQ level-0 collapse as image tiling).
Loads once, synthesizes several texts cloning a reference voice. Saves wavs to /tmp/."""
import os, time, torch
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q
if not hasattr(_q, "Qwen2RMSNorm") and hasattr(_q, "Qwen2_5_VLRMSNorm"):
    _q.Qwen2RMSNorm = _q.Qwen2_5_VLRMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig, BitsAndBytesConfig

MP = "/home/magi/models/LongCat-Next"
REF = os.environ.get("REF", "/home/magi/Projects/LongCat-Next-inference/example/spk_syn.wav")

print(f"[load] transformers {__import__('transformers').__version__}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, fix_mistral_regex=True)
bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["router","classifier","lm_head","visual_head","audio_head","visual_tokenizer","audio_tokenizer"])
model = AutoModelForCausalLM.from_pretrained(
    MP, trust_remote_code=True, torch_dtype=torch.bfloat16,
    quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa",
)
model.eval()
model.text_tokenizer = tok
processor = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
print(f"[load] DONE {time.time()-t0:.0f}s", flush=True)

# --- SDPA varlen fallback into BOTH modular_longcat_next and ..._audio (container flash_attn=None) ---
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
_np = 0
for _name, _m in list(_sys.modules.items()):
    if "modular_longcat" in _name and hasattr(_m, "flash_attn_varlen_func") and getattr(_m, "flash_attn_varlen_func") is None:
        _m.flash_attn_varlen_func = _sdpa_varlen; _np += 1
        print(f"[shim] patched flash_attn_varlen_func in {_name}", flush=True)
print(f"[shim] patched {_np} module(s)", flush=True)

gc = GenerationConfig.from_pretrained(MP)
print(f"[cfg] audio_generation_config={gc.audio_generation_config}", flush=True)

SYNTH = [
    ("zh", "明天的meeting在三楼的Conference Room举行。"),
    ("en", "The quick brown fox jumps over the lazy dog."),
    ("en2", "Hello, this is a test of the voice cloning system."),
]

for tag, text in SYNTH:
    messages = [
        {"role": "system", "content": f"Replicate the voice in the audio clip to formulate an answer:<longcat_audio_start>{REF}<longcat_audio_end>"},
        {"role": "user", "content": f"用这个声音合成以下内容：{text}<longcat_audiogen_start>"},
    ]
    text_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_inputs, visual_inputs, audio_inputs = processor(text=text_input, return_tensors="pt")
    text_inputs = text_inputs.to(model.device)
    if audio_inputs is not None:
        audio_inputs = audio_inputs.to(model.device)
    print(f"[gen] {tag} text={text!r} input_ids {tuple(text_inputs['input_ids'].shape)} audio_inputs={'yes' if audio_inputs is not None else 'NONE'}", flush=True)
    t1 = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids=text_inputs["input_ids"], audio_inputs=audio_inputs,
                                 generation_config=gc, return_dict_in_generate=True)
    dt = time.time() - t1
    at_ids = outputs.audio_text_ids; a_ids = outputs.audio_ids
    audio_text = tok.decode(at_ids[0], skip_special_tokens=True) if at_ids.size(-1) > 0 else "(none)"
    print(f"[gen] {tag} DONE {dt/60:.1f}min  audio_ids {tuple(a_ids.shape)}  audio_text={audio_text!r}", flush=True)
    if a_ids.size(0) > 0:
        torch.save(a_ids.detach().cpu(), f"/tmp/q8_audio_{tag}_ids.pt")  # backup before decode
        try:
            paths = model.model.decode_audio_ids_and_save(a_ids, save_prefix=f"/tmp/q8_audio_{tag}",
                        **gc.audio_generation_config["custom_params"])
            print(f"[gen] {tag} saved {paths}", flush=True)
        except Exception as _e:
            import traceback; print(f"[gen] {tag} DECODE FAILED: {_e}", flush=True); traceback.print_exc()
    else:
        print(f"[gen] {tag} NO audio_ids produced", flush=True)
print("Q8 AUDIO DONE", flush=True)
