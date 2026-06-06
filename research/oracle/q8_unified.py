#!/usr/bin/env python3
"""Unified 8-bit LongCat-Next server: load ONCE, route any modality by request type.
Exercises all 5 task types (text, image-gen, image-und, audio-gen, audio-und) on one load,
using the model's native generate() (routes by trigger token) + built-in decoders.
Also prints GPU footprint after load and mid-generation (the headroom measurement)."""
import os, time, json, torch
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q
if not hasattr(_q, "Qwen2RMSNorm") and hasattr(_q, "Qwen2_5_VLRMSNorm"):
    _q.Qwen2RMSNorm = _q.Qwen2_5_VLRMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig, BitsAndBytesConfig

MP = os.environ.get("LONGCAT_MODEL", os.path.expanduser("~/models/LongCat-Next"))
def gb(): return torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9

print(f"[load] transformers {__import__('transformers').__version__}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, fix_mistral_regex=True)
bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["router","classifier","lm_head","visual_head","audio_head","visual_tokenizer","audio_tokenizer"])
model = AutoModelForCausalLM.from_pretrained(MP, trust_remote_code=True, torch_dtype=torch.bfloat16,
    quantization_config=bnb, device_map={"": 0}, attn_implementation="sdpa")
model.eval(); model.text_tokenizer = tok
processor = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
# SDPA varlen shim — patch ANY module whose flash_attn_varlen_func is None (incl. lazily-loaded refiner)
import sys as _sys, torch.nn.functional as _F
def _sdpa(q,k,v,cu_q,cu_k,mq,mk=None,causal=False,*a,**kw):
    o=[];cq=cu_q.tolist();ck=cu_k.tolist()
    for i in range(len(cq)-1):
        qi=q[cq[i]:cq[i+1]].unsqueeze(0).transpose(1,2);ki=k[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2);vi=v[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2)
        o.append(_F.scaled_dot_product_attention(qi,ki,vi,is_causal=causal).transpose(1,2).squeeze(0))
    return torch.cat(o,dim=0)
def patch_flash():
    n=0
    for nm,m in list(_sys.modules.items()):
        if getattr(m,"flash_attn_varlen_func",1) is None:
            m.flash_attn_varlen_func=_sdpa; n+=1
    return n
print(f"[shim] patched {patch_flash()} modules at load", flush=True)
a,r = gb(); print(f"[load] DONE {time.time()-t0:.0f}s  GPU weights: alloc={a:.1f}G reserved={r:.1f}G", flush=True)

gc = GenerationConfig.from_pretrained(MP)
gc.visual_generation_config["custom_params"]["token_h"] = 18
gc.visual_generation_config["custom_params"]["token_w"] = 18
IMG_START = model.config.visual_config.image_start_token_id
REF = os.environ.get("LONGCAT_REF_WAV", os.path.expanduser("~/Projects/LongCat-Next-inference/example/spk_syn.wav"))
_first = [True]

@torch.no_grad()
def serve(req):
    typ = req["type"]; tag = req["tag"]
    if typ == "image_gen":  # raw text + image_start token (no chat template)
        ids = tok(req["prompt"], return_tensors="pt").input_ids
        ids = torch.cat([ids, torch.tensor([[IMG_START]],dtype=ids.dtype)],dim=1).to(model.device)
        ti = {"input_ids": ids}; vis_in = aud_in = None
    else:
        if typ == "text":
            messages=[{"role":"user","content":req["prompt"]}]
        elif typ == "image_und":
            messages=[{"role":"user","content":f"{req['prompt']}<longcat_img_start>{req['image']}<longcat_img_end>"}]
        elif typ == "audio_und":
            messages=[{"role":"user","content":f"<longcat_audio_start>{req['audio']}<longcat_audio_end>"}]
        elif typ == "audio_gen":
            messages=[{"role":"system","content":f"Replicate the voice in the audio clip to formulate an answer:<longcat_audio_start>{req.get('ref',REF)}<longcat_audio_end>"},
                      {"role":"user","content":f"用这个声音合成以下内容：{req['text']}<longcat_audiogen_start>"}]
        text_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ti, vis_in, aud_in = processor(text=text_input, return_tensors="pt")
        ti = ti.to(model.device)
        if vis_in is not None: vis_in = vis_in.to(model.device)
        if aud_in is not None: aud_in = aud_in.to(model.device)
    t1=time.time()
    out = model.generate(input_ids=ti["input_ids"], visual_inputs=vis_in, audio_inputs=aud_in,
                         generation_config=gc, return_dict_in_generate=True)
    dt=time.time()-t1
    if _first[0]:
        a,r=gb(); print(f"[mem] mid-gen GPU: alloc={a:.1f}G reserved={r:.1f}G (weights+act+decoders)", flush=True); _first[0]=False
    res={"tag":tag,"type":typ,"sec":round(dt,1)}
    txt = tok.decode(out.sequences[0][ti["input_ids"].shape[1]:], skip_special_tokens=True)
    res["text"]=txt
    if out.visual_ids.size(0) > 0:
        try:
            p = model.model.decode_visual_ids_and_save(out.visual_ids, save_prefix=f"/tmp/uni_{tag}", **gc.visual_generation_config["custom_params"])
        except TypeError:
            print(f"[shim] retry-patched {patch_flash()} modules (lazy refiner)", flush=True)
            p = model.model.decode_visual_ids_and_save(out.visual_ids, save_prefix=f"/tmp/uni_{tag}", **gc.visual_generation_config["custom_params"])
        res["image"]=p
    if out.audio_ids.size(0) > 0:
        atxt = tok.decode(out.audio_text_ids[0], skip_special_tokens=True) if out.audio_text_ids.size(-1)>0 else ""
        p = model.model.decode_audio_ids_and_save(out.audio_ids, save_prefix=f"/tmp/uni_{tag}", **gc.audio_generation_config["custom_params"])
        res["audio"]=p; res["audio_text"]=atxt
    print(f"[serve] {json.dumps(res, ensure_ascii=False)}", flush=True)
    return res

REQUESTS = [
    {"type":"text","tag":"txt","prompt":"In one sentence, what is a transformer in machine learning?"},
    {"type":"image_und","tag":"iund","prompt":"What animal is in this image?","image":"/tmp/calib_build/media_color/img_001.png"},
    {"type":"audio_und","tag":"aund","audio":REF},
    {"type":"image_gen","tag":"igen","prompt":"A photograph of a golden retriever sitting in a park."},
    {"type":"audio_gen","tag":"agen","text":"Unified serving is working."},
]
print("=== UNIFIED SERVE: 5 task types on one load ===", flush=True)
for req in REQUESTS:
    try: serve(req)
    except Exception as e:
        import traceback; print(f"[ERR {req['tag']}] {e}", flush=True); traceback.print_exc()
print("UNIFIED DONE", flush=True)
