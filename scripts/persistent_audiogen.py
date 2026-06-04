#!/usr/bin/env python3
"""
MILESTONE 2 — LongCat-Next standalone AUDIO-GENERATION backbone loop (spk_syn voice-clone TTS).

Mirrors gen_image_standalone.py's ModelRunner driver, adapted for the dual-stream audio
generation path (canonical recipe reverse-engineered from the inference repo's
modules/output_processor.py + modules/input_processor.py + modules/context.py):

  prefill spk_syn prompt (ref voice encoded as UNDERSTANDING audio input, spliced at
  audio_pad positions) -> capture LAST hidden -> per-frame:
     (a) audio_head 8-codebook depth loop on the hidden  (canonical audio sampling)
     (b) main-stream text logits -> sample audiotext token (the spoken words)
     (c) STOP when level-0 == AUDIO_END_FLAG (8192) or audiogen_end, or max frames
     (d) A1 feedback (input_processor.get_audio_embeddings):
            next_embed = ext_ids_emb + text_tok_embed*mask + sum_i audio_embed_layers[i](prev_ids[i])
         with audio_embs masked when level-0 in {0, 8192}.
  Accumulate [T,8] audio codebook ids -> /tmp/gen_audio_ids.pt.

Runs INSIDE lc-vision container image with ~/lc_overlay/* mounts (same as image gen).
"""
import os, sys, json, glob, gc, traceback, types
import numpy as np
import torch

NVFP4 = os.environ.get("MODEL_PATH", "/models/output/LongCat-Next-NVFP4-bf16mla")
ORIG  = "/models/LongCat-Next"
CFG_DIR = os.path.join(NVFP4, "nmm_infer") if os.path.exists(os.path.join(NVFP4, "nmm_infer")) \
          else os.path.join(ORIG, "nmm_infer")
REPO = "/home/magi/Projects/LongCat-Next-inference"
sys.path.insert(0, REPO)

HIDDEN = 3072
RMS_EPS = 1e-5
# audio token ids (from config / special_token.py)
AUDIO_OFFSET = 131125
AUDIO_PAD = 131105
AUDIO_START = 131103
AUDIO_END = 131104
AUDIOGEN_START = 131123
AUDIOGEN_END = 131124
AUDIOTEXT_START = 131120
AUDIOTEXT_PAD = 131122
AUDIOTEXT_END = 131121
EOS_ID = None  # filled from config

# canonical audio gen sampling (differs from text/image; rep 1.1 NOT 1.3)
A_TEMP = float(os.environ.get("A_TEMP", "0.2"))
A_TOP_K = int(os.environ.get("A_TOP_K", "20"))
A_TOP_P = float(os.environ.get("A_TOP_P", "0.85"))
A_REP = float(os.environ.get("A_REP", "1.1"))
# main-stream (audiotext) sampling — canonical demo defaults (run_test sampling_params)
T_TEMP = float(os.environ.get("T_TEMP", "0.5"))
T_TOP_K = int(os.environ.get("T_TOP_K", "5"))
T_TOP_P = float(os.environ.get("T_TOP_P", "0.85"))
T_REP = float(os.environ.get("T_REP", "1.3"))
# diagnostic: if set, do NOT break on level-0==8192; keep generating to MAX_FRAMES and log
DIAG_NO_STOP = os.environ.get("DIAG_NO_STOP", "0") == "1"
# layer-3 fix: require this many CONSECUTIVE post-text_end level-0==8192 frames to confirm
# the genuine end-cluster (rejects isolated stray end-flags). Set 1 for pure canonical "first 8192".
END_CONFIRM = int(os.environ.get("END_CONFIRM", "2"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "1000"))
REF_WAV = os.environ.get("REF_WAV", "/tmp/spk_syn.wav")
SYN_TEXT = os.environ.get("SYN_TEXT", "明天的meeting在三楼的Conference Room举行。")
OUT_IDS = os.environ.get("OUT_IDS", "/tmp/gen_audio_ids.pt")

PROMPT = ("<longcat_system>Replicate the voice in the audio clip to formulate an answer."
          "<longcat_audio_start>{\"path\": \"" + REF_WAV + "\"}<longcat_audio_end>"
          "<longcat_user>用这个声音合成以下内容：" + SYN_TEXT +
          "<longcat_assistant><longcat_audiogen_start>")


def log(*a):
    print(*a, flush=True)


# ---------------- import-time shims (same as M1) ----------------
def _install_shims():
    if "deepspeed" not in sys.modules:
        _ds = types.ModuleType("deepspeed"); _ds.__path__ = []
        _zero = types.ModuleType("deepspeed.zero")
        _zero.register_external_parameter = lambda *a, **k: None
        _comm = types.ModuleType("deepspeed.comm")
        for n in ("is_initialized","get_rank","get_world_size","barrier","all_reduce","all_gather","broadcast"):
            setattr(_comm, n, (lambda *a, **k: (0 if n in ("get_rank",) else (1 if n=="get_world_size" else None))))
        _ds.zero = _zero; _ds.comm = _comm
        sys.modules["deepspeed"] = _ds; sys.modules["deepspeed.zero"] = _zero; sys.modules["deepspeed.comm"] = _comm
    try:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
        if not hasattr(_q25, "Qwen2RMSNorm") and hasattr(_q25, "Qwen2_5_VLRMSNorm"):
            _q25.Qwen2RMSNorm = _q25.Qwen2_5_VLRMSNorm
    except Exception:
        pass
    import importlib.util
    for name, attrs in [("decord",["VideoReader","cpu"]),("cv2",None),("av",None),("imagesize",None),("cairosvg",["svg2png"])]:
        if importlib.util.find_spec(name) is None:
            m = types.ModuleType(name)
            for a in (attrs or []):
                setattr(m, a, (lambda *args, **kw: None))
            sys.modules[name] = m
    # sglang ships a namespace 'utils' package that shadows the inference repo's utils/.
    # Force-load the repo's utils.model_utils from file and register it so the module-level
    # `from utils.model_utils import ...` inside modeling_longcat_oe resolves to the repo copy.
    import importlib.util as _ilu
    _mu_path = os.path.join(REPO, "utils", "model_utils.py")
    if os.path.exists(_mu_path):
        _pkg = types.ModuleType("utils"); _pkg.__path__ = [os.path.join(REPO, "utils")]
        sys.modules["utils"] = _pkg
        _spec = _ilu.spec_from_file_location("utils.model_utils", _mu_path)
        _mu = _ilu.module_from_spec(_spec)
        sys.modules["utils.model_utils"] = _mu
        _spec.loader.exec_module(_mu)
        _pkg.model_utils = _mu


def memfree():
    fa, ta = torch.cuda.mem_get_info()
    return fa / 1e9, ta / 1e9


# ============================================================================
# Build the gen-side audio modules: audio_head + audio_embed_layers + (ref-voice) audio_tokenizer
# ============================================================================
def _index(path):
    idxf = glob.glob(os.path.join(path, "*index*.json"))[0]
    return json.load(open(idxf))["weight_map"], path


def _resolve(path, rel):
    cand = os.path.join(path, rel)
    if os.path.exists(cand):
        return os.path.realpath(cand)
    return os.path.join("/models/output/LongCat-Next-NVFP4", os.path.basename(rel))


def load_tensor(idx, path, key):
    from safetensors import safe_open
    sf = _resolve(path, idx[key])
    with safe_open(sf, framework="pt") as f:
        return f.get_tensor(key)


def build_audio_gen_modules(audio_codebook_sizes, audio_cfg):
    """OmniAudioHead (from audio_head.* with flash rename) + audio_embed_layers sliced from
    ORIGINAL embed_tokens at AUDIO_OFFSET (per-level cumulative; the 'use_oe'/flash path)."""
    from modules.image_head import OmniAudioHead
    import modules.image_head as _ih
    # install SDPA varlen fallback for the depth-head attention (container flash_attn is hollow)
    if getattr(_ih, "flash_attn_varlen_func", None) is None:
        import torch.nn.functional as _F
        def _sdpa_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, **kw):
            outs = []; cq = cu_q.tolist(); ck = cu_k.tolist()
            for i in range(len(cq)-1):
                qi = q[cq[i]:cq[i+1]].unsqueeze(0).transpose(1,2)
                ki = k[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2)
                vi = v[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2)
                oi = _F.scaled_dot_product_attention(qi, ki, vi, is_causal=bool(causal))
                outs.append(oi.transpose(1,2).squeeze(0))
            return torch.cat(outs, dim=0)
        _ih.flash_attn_varlen_func = _sdpa_varlen
        log("[A.modules] installed SDPA varlen fallback in image_head (for audio depth head)")

    head = OmniAudioHead(
        hidden_size=HIDDEN,
        codebook_sizes=list(audio_codebook_sizes),
        audio_head_transformer_ffn_scale=audio_cfg.audio_head_transformer_ffn_scale,
        audio_head_transformer_dims=audio_cfg.audio_head_transformer_dims,
        audio_head_transformer_layers=audio_cfg.audio_head_transformer_layers,
        audio_head_enable=False,
    )
    nidx, npath = _index(NVFP4)
    head_sd = {}
    for k in nidx:
        if k.startswith("audio_head."):
            local = k[len("audio_head."):]
            head_sd[local] = load_tensor(nidx, npath, k)
    # flash model names hidden_in_proj; OmniAudioHead expects hidden_proj
    head_sd = {kk.replace("hidden_in_proj", "hidden_proj"): vv for kk, vv in head_sd.items()}
    miss = head.load_state_dict(head_sd, strict=False)
    log(f"[A.modules] audio_head loaded; n={len(head_sd)} missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
    if miss.missing_keys:
        log("   head missing:", miss.missing_keys[:10])
    if miss.unexpected_keys:
        log("   head unexpected:", miss.unexpected_keys[:10])
    head = head.to("cuda").to(torch.bfloat16).eval()

    # audio_embed_layers: Embedding(codedim+1, hidden) per level, sliced from ORIGINAL embed_tokens
    # at AUDIO_OFFSET, cumulative (8192,4096,2048,1024x5)
    oidx, opath = _index(ORIG)
    et = load_tensor(oidx, opath, "model.embed_tokens.weight")  # [282624,3072]
    log(f"[A.modules] orig embed_tokens {tuple(et.shape)} for audio_embed_layers slice")
    emb_layers = torch.nn.ModuleList([
        torch.nn.Embedding(cd + 1, HIDDEN) for cd in audio_codebook_sizes
    ])
    sd = {}
    offset = AUDIO_OFFSET
    for i, cd in enumerate(audio_codebook_sizes):
        sd[f"{i}.weight"] = et[offset:offset + cd + 1, :].clone()
        offset += cd
    emb_layers.load_state_dict(sd, strict=True)
    del et; gc.collect()
    emb_layers = emb_layers.to("cuda").to(torch.bfloat16).eval()
    log(f"[A.modules] audio_embed_layers ready ({len(emb_layers)} levels)")
    return head, emb_layers


# ============================================================================
# sampling
# ============================================================================
def sample_logits(logits, temp, top_p, top_k, rep_penalty=1.0, past_ids=None):
    logits = logits.float()
    if rep_penalty and rep_penalty != 1.0 and past_ids is not None and past_ids.numel() > 0:
        for b in range(logits.shape[0]):
            uids = torch.unique(past_ids[b][past_ids[b] >= 0])
            if uids.numel() > 0:
                sc = logits[b, uids]
                logits[b, uids] = torch.where(sc > 0, sc / rep_penalty, sc * rep_penalty)
    logits = logits / max(temp, 1e-6)
    if top_k and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    csum = torch.cumsum(sp, dim=-1)
    mask = csum - sp > top_p
    sp = sp.masked_fill(mask, 0.0)
    sp = sp / sp.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    idx = torch.multinomial(sp, 1)
    return si.gather(-1, idx).squeeze(-1)


@torch.no_grad()
def audio_depth_step(hidden, head, emb_layers, codebook_sizes, past_multi_ids):
    """Run the audio depth head over 8 codebooks. hidden [1,HIDDEN].
    OmniAudioHead returns a LIST of per-codebook logits; we re-run per level feeding
    progressively-filled next_ids (matches depth_transformer_forward_new AUDIO branch).
    Level-0 EOS slot (8192) is NOT masked (it is the natural stop). Levels>0 mask the +1 EOS slot."""
    bs = hidden.shape[0]
    NUM = len(codebook_sizes)
    next_ids = torch.zeros(bs, NUM, dtype=torch.long, device="cuda")
    for i in range(NUM):
        logits = head(hidden, next_ids, emb_layers, bs)[i]   # [bs, cb_i+1]
        NEG = torch.finfo(logits.dtype).min
        # BUG-1 FIX: per-level range masking. Audio RVQ codebooks are NOT uniform.
        # codebook_sizes = [8192,4096,2048,1024,1024,1024,1024,1024]; each head emits cb_i+1 logits.
        # Level 0: keep index 8192 (== AUDIO_END_FLAG, the natural stop) sampleable; mask anything above.
        # Levels 1-7: valid ids are [0, cb_i); the +1 slot (index cb_i) is an EOS slot that must NOT be
        # sampled mid-utterance, so mask from cb_i onward.
        if i == 0:
            if logits.shape[-1] > codebook_sizes[0] + 1:
                logits[:, codebook_sizes[0] + 1:] = NEG
        else:
            logits[:, codebook_sizes[i]:] = NEG
        past_i = past_multi_ids[:, :, i] if past_multi_ids is not None and past_multi_ids.numel() else None
        next_ids[:, i] = sample_logits(logits, A_TEMP, A_TOP_P, A_TOP_K, A_REP, past_i)
    return next_ids  # [1,8]


@torch.no_grad()
def audio_relevel0_no_end(hidden, head, emb_layers, codebook_sizes, past_multi_ids):
    """Re-sample ONLY level-0 with the AUDIO_END_FLAG (8192) slot masked out.
    Used for layer-3 fix: a stray 8192 sampled at level-0 BEFORE the transcript is complete
    is not a real end -> force the model to pick a real acoustic code so the frame keeps
    carrying speech content. Returns a scalar level-0 id."""
    bs = hidden.shape[0]
    next_ids = torch.zeros(bs, len(codebook_sizes), dtype=torch.long, device="cuda")
    logits = head(hidden, next_ids, emb_layers, bs)[0]   # [bs, cb0+1]
    NEG = torch.finfo(logits.dtype).min
    logits[:, codebook_sizes[0]:] = NEG  # mask 8192 (the end flag) AND anything above
    past_i = past_multi_ids[:, :, 0] if past_multi_ids is not None and past_multi_ids.numel() else None
    return int(sample_logits(logits, A_TEMP, A_TOP_P, A_TOP_K, A_REP, past_i)[0])


def main():
    global SYN_TEXT, OUT_IDS
    _install_shims()
    global EOS_ID
    log(f"[cfg] MODEL_PATH={NVFP4} CFG_DIR={CFG_DIR}")
    log(f"[cfg] REF_WAV={REF_WAV} SYN_TEXT={SYN_TEXT!r}")
    log(f"[cfg] sampling temp={A_TEMP} top_k={A_TOP_K} top_p={A_TOP_P} rep={A_REP} max_frames={MAX_FRAMES}")
    fa, ta = memfree(); log(f"[mem] cuda free {fa:.1f}/{ta:.1f} GB at start")

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs, PortArgs
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
    from sglang.srt.managers.schedule_batch import ScheduleBatch, Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer
    from types import SimpleNamespace

    override = {
        "architectures": ["LongcatFlashForCausalLM"],
        "use_ngram_embedding": True,
        "ngram_embedding_m": 10223616, "ngram_embedding_n": 5, "ngram_embedding_k": 3,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "disable_quant_module": ["self_attn"],
    }
    server_args = ServerArgs(
        model_path=NVFP4, tokenizer_path=NVFP4, quantization="modelopt_fp4",
        json_model_override_args=json.dumps(override),
        attention_backend="flashinfer", mem_fraction_static=0.5,
        max_total_tokens=8192, dtype="bfloat16", disable_cuda_graph=True,
        disable_radix_cache=True, skip_server_warmup=True, trust_remote_code=True,
        context_length=8192, chunked_prefill_size=8192, device="cuda",
    )
    log("[A.runner] building ModelConfig + ModelRunner (mem_fraction_static=0.5) ...")
    model_config = ModelConfig.from_server_args(server_args)
    port_args = PortArgs.init_new(server_args)
    mr = ModelRunner(
        model_config=model_config, mem_fraction_static=0.5, gpu_id=0, tp_rank=0, tp_size=1,
        moe_ep_rank=0, moe_ep_size=1, pp_rank=0, pp_size=1,
        nccl_port=port_args.nccl_port, server_args=server_args,
    )
    log(f"[A.runner] ModelRunner up. max_total_num_tokens={mr.max_total_num_tokens}")
    tok = get_tokenizer(NVFP4, trust_remote_code=True)
    model = mr.model
    fa, _ = memfree(); log(f"[mem] cuda free {fa:.1f} GB after ModelRunner")
    # mem_fraction_static=0.5 reserves ~60GB for the KV pool, so ~18GB unallocated is expected
    # and safe (same config as the proven image gen). Only abort if essentially nothing is left.
    if fa < 3:
        log("[ABORT] cuda free < 3GB after ModelRunner"); raise RuntimeError("mem abort")

    # audio codebook sizes + audio config from the backbone's own config
    hf = mr.model_config.hf_config
    ac = getattr(hf, "audio_config", None)
    if ac is None or not hasattr(ac, "vq_config"):
        # build from json (ngram override stripped audio_config from hf_config sometimes)
        from processor.flash_omni.configuration_omni import OmniConfig
        try:
            from transformers import CLIPVisionConfig as _CVC
            if hasattr(_CVC, "__validators__"): _CVC.__validators__ = {}
            if hasattr(_CVC, "validate"): _CVC.validate = staticmethod(lambda self: None)
        except Exception:
            pass
        oc = OmniConfig(**json.load(open(os.path.join(CFG_DIR, "config.json"))))
        ac = oc.audio_config
    codebook_sizes = list(ac.vq_config.codebook_sizes)
    EOS_ID = getattr(hf, "eos_token_id", None)
    log(f"[cfg] audio codebook_sizes={codebook_sizes} eos_id={EOS_ID}")

    head, emb_layers = build_audio_gen_modules(codebook_sizes, ac)
    log("=== MILESTONE 2a: ModelRunner + audio gen modules loaded ===")
    fa, _ = memfree(); log(f"[mem] cuda free {fa:.1f} GB after audio modules")

    # ---- reference voice -> understanding audio embeds (reuse backbone get_audio_feature) ----
    # Build an mm audio item the way get_audio_feature expects (feature=mel, model_specific_data
    # carries encoder_length/bridge_length). We compute the mel via OmniAudioProcessor.
    from processor.flash_omni.processor_omni import OmniAudioProcessor
    # patch SDPA fallback into audio modeling modules used by the backbone audio_tokenizer encode
    import torch.nn.functional as _F
    import processor.flash_omni.audio_modeling_omni as _am
    import processor.flash_omni.matcha_transformer as _mt
    def _sdpa_varlen2(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, **kw):
        outs = []; cq = cu_q.tolist(); ck = cu_k.tolist()
        for i in range(len(cq)-1):
            qi = q[cq[i]:cq[i+1]].unsqueeze(0).transpose(1,2)
            ki = k[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2)
            vi = v[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2)
            oi = _F.scaled_dot_product_attention(qi, ki, vi, is_causal=bool(causal))
            outs.append(oi.transpose(1,2).squeeze(0))
        return torch.cat(outs, dim=0)
    for _m in (_am, _mt):
        if getattr(_m, "flash_attn_varlen_func", None) is None:
            _m.flash_attn_varlen_func = _sdpa_varlen2

    apr = OmniAudioProcessor(ac)
    wav = apr.load_audio_waveform(REF_WAV, return_tensors=True)
    fbank, valid = apr.extract_fbank_features(wav)
    enc_len, br_len = apr.inference_output_length(ac, valid)
    log(f"[A.ref] ref wav {wav.shape[1]/ac.sampling_rate:.2f}s fbank={fbank.shape} "
        f"encoder_length={enc_len} bridge_length={br_len}")

    # ref VQ ids via a standalone LongcatAudioTokenizer (the proven M1 encoder path), since the
    # backbone's grafted audio_tokenizer may be None when the ngram override strips audio_config.
    atok = getattr(model, "audio_tokenizer", None)
    if atok is None and hasattr(model, "model"):
        atok = getattr(model.model, "audio_tokenizer", None)
    if atok is None:
        log("[A.ref] backbone audio_tokenizer absent; building standalone LongcatAudioTokenizer (M1 path)")
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        sys.path.insert(0, REPO)  # ensure utils.model_utils resolves to REPO/utils
        from processor.flash_omni.modeling_longcat_oe import LongcatAudioTokenizer
        from processor.flash_omni.configuration_omni import OmniConfig
        from utils.model_utils import load_weights_from_safetensors_helper
        try:
            from transformers import CLIPVisionConfig as _CVC
            if hasattr(_CVC, "__validators__"): _CVC.__validators__ = {}
            if hasattr(_CVC, "validate"): _CVC.validate = staticmethod(lambda self: None)
        except Exception:
            pass
        oc = OmniConfig(**json.load(open(os.path.join(CFG_DIR, "config.json"))))
        atok = LongcatAudioTokenizer(oc)
        sds = load_weights_from_safetensors_helper(NVFP4, ["model.audio_tokenizer."], "cpu")
        atok.load_state_dict(sds[0], strict=False)
        atok = atok.to("cuda").to(torch.bfloat16).eval()
        log("[A.ref] standalone audio_tokenizer ready")
    dev = next(atok.parameters()).device
    mel = torch.as_tensor(fbank, dtype=torch.float32, device=dev).unsqueeze(0)
    ref_ids = atok.encode(mel, torch.tensor([enc_len], device=dev), torch.tensor([br_len], device=dev))
    log(f"[A.ref] ref VQ ids {tuple(ref_ids.shape)} uniq={ref_ids.unique().numel()}")
    # cumulative audio offsets: cumsum([AUDIO_OFFSET] + codebook_sizes[:-1])
    offs = [AUDIO_OFFSET]
    for cs in codebook_sizes[:-1]:
        offs.append(offs[-1] + cs)
    offs_t = torch.tensor(offs, device=ref_ids.device, dtype=ref_ids.dtype)
    ref_ids_off = ref_ids + offs_t  # [actual, 8] offset ids
    # understanding embed = sum over levels of mm_embed_rows[offset_id - 131125]
    # (mirrors longcat_flash._embed_visual_ids / get_audio_feature; NO post-projection for audio).
    from safetensors import safe_open as _so
    mm_path = None
    for c in ["/models/lc_mm_embed_rows.safetensors",
              os.path.join(NVFP4, "mm_embed_rows.safetensors"),
              "/models/output/LongCat-Next-NVFP4-bf16mla/mm_embed_rows.safetensors"]:
        if os.path.exists(c):
            mm_path = c; break
    if mm_path is None:
        raise RuntimeError("mm_embed_rows.safetensors not found")
    with _so(mm_path, framework="pt") as f:
        mm_embed_rows = f.get_tensor("mm_embed_rows").to("cuda").to(torch.bfloat16)
    log(f"[A.ref] loaded mm_embed_rows {tuple(mm_embed_rows.shape)} from {mm_path}")
    CODEBOOK_BASE = 131125
    def _embed_audio_understanding(ids_off):
        summed = None
        for lev in range(ids_off.shape[1]):
            idx = (ids_off[:, lev].long() - CODEBOOK_BASE).clamp(0, mm_embed_rows.shape[0]-1)
            e = mm_embed_rows[idx]
            summed = e if summed is None else summed + e
        return summed  # [N, HIDDEN]
    ref_embeds = _embed_audio_understanding(ref_ids_off)  # [actual, HIDDEN]
    actual = ref_embeds.shape[0]
    if actual < br_len:
        ref_embeds = torch.cat([ref_embeds, torch.zeros(br_len-actual, HIDDEN, dtype=ref_embeds.dtype, device=ref_embeds.device)])
    elif actual > br_len:
        ref_embeds = ref_embeds[:br_len]
    log(f"[A.ref] ref_embeds {tuple(ref_embeds.shape)} mean|x|={ref_embeds.abs().mean().item():.4f}")

    import json as _json, traceback as _tb
    _REQUESTS = _json.load(open(os.environ['PROMPTS_JSON'])) if os.environ.get('PROMPTS_JSON') else [{'text': SYN_TEXT, 'instr': os.environ.get('INSTR','用这个声音合成以下内容：'), 'out': os.environ.get('OUT_IDS','/tmp/gen_audgen_ids.pt')}]
    log(f'[persistent] {len(_REQUESTS)} request(s); model+modules loaded ONCE')
    for _ri, _req in enumerate(_REQUESTS):
        SYN_TEXT = _req['text']; _INSTR = _req.get('instr','用这个声音合成以下内容：'); OUT_IDS = _req.get('out', f'/tmp/batch/ids_{_ri}.pt')
        log(f'\n===== REQUEST {_ri}/{len(_REQUESTS)} instr={_INSTR[:16]!r} text={SYN_TEXT[:60]!r} -> {OUT_IDS} =====')
        for _alloc in (mr.token_to_kv_pool_allocator, mr.req_to_token_pool):
            for _m in ('clear','reset','free_all'):
                if hasattr(_alloc,_m):
                    try: getattr(_alloc,_m)(); break
                    except Exception as _e: log(f'[reset] {_alloc.__class__.__name__}.{_m} -> {_e}')
            else:
                if _ri==0: log(f'[reset] NO clear/reset on {_alloc.__class__.__name__}; methods={[x for x in dir(_alloc) if not x.startswith(chr(95))][:25]}')
        try:
            # ---- build the prompt input_ids: system + ref-audio(pad x br_len) + user + assistant + audiogen_start
            # Replicate processor: audio_start + audio_pad*br_len + audio_end inside the system audio span.
            # Tokenize with the placeholders expanded to br_len audio_pad tokens.
            sys_pre = "<longcat_system>Replicate the voice in the audio clip to formulate an answer. <longcat_audio_start>"
            sys_post = "<longcat_audio_end> <longcat_user>" + _INSTR + SYN_TEXT + " <longcat_assistant><longcat_audiogen_start>\n"
            pre_ids = tok.encode(sys_pre, add_special_tokens=False)
            pad_ids = tok.encode("<longcat_audio_pad>", add_special_tokens=False) * br_len
            post_ids = tok.encode(sys_post, add_special_tokens=False)
            input_ids = pre_ids + pad_ids + post_ids
            log(f"[A.prompt] {len(input_ids)} tokens (pre={len(pre_ids)} pad={len(pad_ids)} post={len(post_ids)}); tail={input_ids[-6:]}")
            audio_pad_positions = [i for i, t in enumerate(input_ids) if t == AUDIO_PAD]
            log(f"[A.prompt] audio_pad positions in prompt: {len(audio_pad_positions)} (expect {br_len})")

            # === ScheduleBatch driver (bs=1) ===
            class _TC(SimpleNamespace):
                def supports_swa(self): return False
                def supports_mamba(self): return False
                def is_chunk_cache(self): return False
                def is_tree_cache(self): return True
                def evict(self, *a, **k): pass
            dummy_tc = _TC(page_size=server_args.page_size, device=mr.device,
                           token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator)

            spp = SamplingParams(temperature=0, max_new_tokens=MAX_FRAMES + 8)
            spp.normalize(tok)
            rq = Req(rid=0, origin_input_text=PROMPT, origin_input_ids=list(input_ids), sampling_params=spp)
            rq.fill_ids = list(input_ids)
            rq.logprob_start_len = -1
            rq.set_extend_input_len(len(rq.fill_ids) - len(rq.prefix_indices))
            BS = 1
            batch = ScheduleBatch.init_new(
                reqs=[rq], req_to_token_pool=mr.req_to_token_pool,
                token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
                tree_cache=dummy_tc, model_config=mr.model_config,
                enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
            )

            # ngram token-table plumbing (copied from gen_image_standalone)
            from sglang.jit_kernel.ngram_embedding import update_token_table
            NE_N = mr.model_config.hf_config.ngram_embedding_n
            token_table = mr.token_table
            tv = getattr(mr.model_config.hf_config, "text_vocab_size", 131072)
            tvp = getattr(mr.model_config.hf_config, "text_vocab_plus_multimodal_special_token_size", 131125) or 131125

            def _zero_mm(t):
                if tvp > tv:
                    m = (t >= tv) & (t < tvp); t = t.clone(); t[m] = 0
                return t

            def prefill_token_table():
                batch.ne_token_table = token_table
                all_tokens, column_starts, req_lens = [], [], []
                for rqi in batch.reqs:
                    start = len(rqi.prefix_indices); end = start + rqi.extend_input_len
                    fill = rqi.origin_input_ids + rqi.output_ids
                    if start == 0: toks = fill[start:end]; cs = 0
                    elif start < NE_N: toks = fill[0:end]; cs = 0
                    else: toks = fill[start-NE_N+1:end]; cs = start-NE_N+1
                    all_tokens.extend(toks); column_starts.append(cs); req_lens.append(len(toks))
                dev2 = token_table.device
                tt = _zero_mm(torch.tensor(all_tokens, dtype=token_table.dtype, device=dev2))
                update_token_table(ne_token_table=token_table, tokens=tt, row_indices=batch.req_pool_indices,
                                    column_starts=torch.tensor(column_starts, dtype=torch.int32, device=dev2),
                                    req_lens=torch.tensor(req_lens, dtype=torch.int32, device=dev2), ignore_tokens=None)

            def append_token_table(fb, next_tok_id):
                dev2 = token_table.device
                nrows = fb.req_pool_indices.shape[0]
                tt = _zero_mm(torch.full((nrows,), next_tok_id, dtype=torch.int32, device=dev2))
                update_token_table(ne_token_table=token_table, tokens=tt, row_indices=fb.req_pool_indices,
                                   column_starts=fb.seq_lens.to(torch.int32),
                                   req_lens=torch.ones_like(fb.seq_lens, dtype=torch.int32), ignore_tokens=None)

            @torch.no_grad()
            def run_forward(input_embeds_override=None, splice_audio=False):
                mwb = batch.get_model_worker_batch(); mwb.ne_token_table = token_table
                fb = ForwardBatch.init_new(mwb, mr)
                fb.capture_hidden_mode = CaptureHiddenMode.LAST
                mr.attn_backend.init_forward_metadata(fb)
                if input_embeds_override is not None:
                    lo = model.forward(None, fb.positions, fb, input_embeds=input_embeds_override)
                else:
                    lo = model.forward(fb.input_ids, fb.positions, fb, input_embeds=None)
                return lo, fb

            # ---- PREFILL: build input_embeds with ref audio spliced at audio_pad, then forward ----
            batch.prepare_for_extend()
            prefill_token_table()
            # build prefill embeds: word/ngram embed then overwrite audio_pad positions with ref_embeds
            iid = torch.tensor(input_ids, dtype=torch.long, device=mr.device)
            # use the backbone's ngram-aware embedder
            try:
                mwb0 = batch.get_model_worker_batch(); mwb0.ne_token_table = token_table
                fb0 = ForwardBatch.init_new(mwb0, mr); mr.attn_backend.init_forward_metadata(fb0)
                prefill_embeds = model.model.embed_tokens(fb0.input_ids, fb0)
            except Exception as e:
                log(f"[A.prefill] ngram embed_tokens(fb) failed ({e}); falling back to plain embed")
                prefill_embeds = model.model.embed_tokens(iid)
            # zero + splice audio
            apm = (iid == AUDIO_PAD)
            if apm.any():
                prefill_embeds[apm] = 0
                n = int(apm.sum())
                prefill_embeds[apm] = ref_embeds[:n].to(prefill_embeds.dtype)
                log(f"[A.prefill] spliced {n} ref-audio embeds at audio_pad positions")
            lo, fb = run_forward(input_embeds_override=prefill_embeds)
            log(f"[A.prefill] hidden {tuple(lo.hidden_states.shape)} dtype {lo.hidden_states.dtype}")
            log("=== MILESTONE 2b: prefill returned a hidden ===")

            def last_hidden(lo):
                return lo.hidden_states.reshape(-1, HIDDEN)[-1:].to(torch.bfloat16)

            word_embed = model.model.embed_tokens.word_embeder if hasattr(model.model.embed_tokens, "word_embeder") else model.model.embed_tokens

            @torch.no_grad()
            def audio_feedback_embed(prev_ids, main_text_id, ext_id):
                """Canonical per-step audio decode embedding, matching the inference repo's
                input_processor.forward_decode + get_audio_embeddings + decode_oe_with_sp_new:

                    base + ext_ids_emb + audio_embs
                  base        = word_embed(AUDIO_PAD)  (decode token is AUDIO_PAD, a SPECIAL token, so
                                decode_oe_with_sp_new uses the plain word_embeder with NO ngram OE; the
                                base term is KEPT because AUDIO_PAD != AUDIOTEXT_PAD -> input_ids_mask=1)
                  ext_ids_emb = word_embed(ext_id)     (0 when ext_id == AUDIOTEXT_PAD)
                  audio_embs  = sum_i emb_layers[i](prev_ids[i])  (0 when level-0 in {0, AUDIO_END_FLAG})

                BUGFIX (2026-06-01, teacher-forcing bisection): the prior version added
                word_embed(SAMPLED main_text_id) as the base term and OMITTED word_embed(AUDIO_PAD).
                The spoken audiotext is an OUTPUT (read from the LM logits), NOT fed back as the decode
                embedding. After text_end the sampled-text term went to 0, leaving the feedback as
                audio_embs ONLY, so the backbone lost its constant per-frame structural anchor and the
                decode hiddens drifted. A teacher-forcing test (real hiddens + real prev codes) proved
                the audio_head itself is HEALTHY (level-0 top-1 53%, p_real 0.40, entropy 1.65 nats vs
                9.0 max), so the fault was here in the feedback content, not the head."""
                # base term: word-embed of the constant decode token (AUDIO_PAD)
                base = word_embed(torch.tensor([AUDIO_PAD], device="cuda"))
                # ext id embed (audiotext start/pad marker)
                ei = torch.tensor([ext_id], device="cuda")
                ext = word_embed(torch.clamp(ei, min=0))
                if ext_id == AUDIOTEXT_PAD:
                    ext = ext * 0
                # audio embeds (A1 feedback from the previous frame's codes)
                valid = torch.clamp(prev_ids, min=0)  # [1,8]
                aud = None
                for i, layer in enumerate(emb_layers):
                    e = layer(valid[:, i])
                    aud = e if aud is None else aud + e
                lvl0 = int(prev_ids[0, 0])
                if lvl0 == 0 or lvl0 == codebook_sizes[0]:
                    aud = aud * 0
                return (base + ext + aud).to(torch.bfloat16)  # [1,HIDDEN]

            # ---- DUAL-STREAM DECODE LOOP ----
            all_audio = []           # list of [8] tensors
            all_text = []            # main-stream sampled token ids (the audiotext)
            past_multi = torch.full((BS, MAX_FRAMES, len(codebook_sizes)), -1, dtype=torch.long, device="cuda")
            past_text = torch.full((BS, MAX_FRAMES), -1, dtype=torch.long, device="cuda")
            h = last_hidden(lo)
            text_end = False
            delay = 0
            audio_start = False
            end_run = 0  # consecutive post-text_end level-0==8192 frames (layer-3 end-cluster confirm)
            stop_reason = "max_frames"
            fa, _ = memfree(); log(f"[A.mem] before loop cuda free {fa:.1f} GB")

            for step in range(MAX_FRAMES):
                # (a) audio depth head -> 8 ids
                cur = audio_depth_step(h, head, emb_layers, codebook_sizes,
                                       past_multi[:, :step, :] if step > 0 else None)  # [1,8]
                lvl0 = int(cur[0, 0])
                # (b) main-stream text token from the backbone's LM logits
                try:
                    text_logits = lo.next_token_logits.float()
                    past_t = past_text[:, :step] if step > 0 else None
                    main_tok = int(sample_logits(text_logits, T_TEMP, T_TOP_P, T_TOP_K, T_REP, past_t)[0])
                except Exception as e:
                    main_tok = AUDIOTEXT_PAD
                # text_end detection: first AUDIOTEXT_PAD or EOS on the main stream marks the spoken
                # text complete. After that the main stream is FORCED to AUDIOTEXT_PAD (so its embed is
                # masked to 0) and only audio frames keep advancing for the rest of the utterance.
                if not text_end and (main_tok == AUDIOTEXT_PAD or main_tok == EOS_ID):
                    text_end = True
                if text_end:
                    main_tok = AUDIOTEXT_PAD
                # ext id schedule (delay=0): step0 -> AUDIOTEXT_START, else PAD
                ext_id = AUDIOTEXT_START if step == delay else AUDIOTEXT_PAD

                # ============================================================================
                # LAYER-3 FIX — premature AUDIO_END_FLAG (8192) at level-0.
                #
                # Canonical state_machine.GenAudioStageStage ends the audio stage on the FIRST
                # frame where audio_start AND multi_ids[0]==AUDIO_END_FLAG_ID(8192). With the
                # canonical (ngram-aware, full-context) conditioning, P(8192) is ~0 until the
                # utterance is genuinely finished, so "first 8192" == real end.
                #
                # Under THIS standalone's slightly-thinner conditioning (plain word-embed feedback,
                # no decode-side ngram OE), 8192 is sampleable as a low-probability stray at
                # level-0 (it is a valid sample slot at top_k=20). A single stray 8192 MID-UTTERANCE
                # was killing generation after ~5-10 frames. Empirically proven via DIAG_NO_STOP:
                # the first 8192 fires at ~frame 5-10 (while text_end is still False, i.e. the model
                # has NOT finished emitting the transcript), the model RECOVERS and keeps producing
                # varied, speech-like codes, and the GENUINE end is a *cluster* of 8192 at ~frame 55-65
                # (≈4.5-5.2s @12.5fps for the 9-word sentence — the expected duration).
                #
                # Faithful guard (matches the canonical interleave invariant: the acoustic utterance
                # cannot terminate before the spoken transcript is fully emitted):
                #   (1) An 8192 BEFORE text_end is NEVER an end -> it is a stray sample. Re-sample
                #       level-0 with the 8192 slot masked out so the frame still carries valid audio
                #       content (the model wants to keep speaking; honor that).
                #   (2) An 8192 AFTER text_end is a candidate end. Require END_CONFIRM (default 2)
                #       CONSECUTIVE raw-8192 draws to confirm the genuine end-cluster, rejecting an
                #       isolated post-text stray. On confirmation, stop (BEFORE storing the flag).
                #       A tentative (unconfirmed) 8192 is ALSO re-sampled to a real acoustic code so
                #       NO 8192 row ever enters the stored body (the decoder truncates at the first
                #       level-0==8192 row, so an interior end-flag would silently chop the utterance).
                # ============================================================================
                if audio_start and lvl0 == codebook_sizes[0]:
                    end_run += 1  # consecutive RAW-8192 draws (pre- or post-text_end)
                    confirmed_end = text_end and end_run >= END_CONFIRM
                    if DIAG_NO_STOP:
                        log(f"[A.DIAG] level-0==8192 (text_end={text_end}) run={end_run} at step {step}; continuing (DIAG_NO_STOP)")
                    elif confirmed_end:
                        stop_reason = f"audio_end_flag@{step}(run={end_run})"; break
                    # not a confirmed end (stray pre-text_end, OR unconfirmed isolated post-text):
                    # re-sample level-0 without the 8192 slot so the frame carries real speech content.
                    relog = audio_relevel0_no_end(h, head, emb_layers, codebook_sizes,
                                                  past_multi[:, :step, :] if step > 0 else None)
                    cur[0, 0] = relog
                    lvl0 = int(cur[0, 0])
                    if step < 60:
                        log(f"[A.frame {step}] STRAY-8192 (text_end={text_end} run={end_run}) -> re-sampled lvl0={lvl0}")
                else:
                    end_run = 0  # any non-8192 frame breaks the run
                if step >= delay:
                    audio_start = True

                if step < 60 or step % 25 == 0:
                    log(f"[A.frame {step}] lvl0={lvl0} main_tok={main_tok} text_end={text_end} ext={ext_id} end_run={end_run}")

                all_audio.append(cur.reshape(-1).clone())
                all_text.append(main_tok)
                past_multi[:, step, :] = cur
                past_text[:, step] = main_tok

                # (d) CANONICAL feedback (get_audio_embeddings): the decode token IS the sampled
                # audiotext token (next_token_ids = text_ids), embedded via the model's NGRAM-OE path,
                # masked at AUDIOTEXT_PAD; plus ext_ids_emb + Sum_i audio_emb_layer(prev codes).
                dec_tok = main_tok if main_tok >= 0 else AUDIOTEXT_PAD
                batch.output_ids = torch.full((BS,), dec_tok, dtype=torch.long, device=mr.device)
                batch.prepare_for_decode()
                mwb = batch.get_model_worker_batch(); mwb.ne_token_table = token_table
                fbi = ForwardBatch.init_new(mwb, mr); fbi.capture_hidden_mode = CaptureHiddenMode.LAST
                mr.attn_backend.init_forward_metadata(fbi)
                # base = ngram-OE embed of the decode (audiotext) token; canonical input_ids_mask
                base = model.model.embed_tokens(fbi.input_ids, fbi).reshape(-1, HIDDEN)[-1:]
                if main_tok == AUDIOTEXT_PAD:
                    base = base * 0
                # ext_ids_emb (audiotext start/pad marker), masked at PAD
                ei = torch.tensor([ext_id], device="cuda"); ext = word_embed(torch.clamp(ei, min=0))
                if ext_id == AUDIOTEXT_PAD:
                    ext = ext * 0
                # audio_embs: Sum_i audio_emb_layer(prev frame codes); row-mask when lvl0 in {0, 8192}
                valid = torch.clamp(cur, min=0); aud = None
                for i, layer in enumerate(emb_layers):
                    e = layer(valid[:, i]); aud = e if aud is None else aud + e
                if lvl0 == 0 or lvl0 == codebook_sizes[0]:
                    aud = aud * 0
                feed = (ext + base + aud).to(torch.bfloat16)
                lo = model.forward(None, fbi.positions, fbi, input_embeds=feed)
                append_token_table(fbi, dec_tok)   # ngram context advances with the audiotext token
                h = last_hidden(lo)
                if (step + 1) % 50 == 0:
                    fa, _ = memfree()
                    log(f"[A.loop] step {step+1}/{MAX_FRAMES} lvl0={lvl0} main_tok={main_tok} text_end={text_end} cuda free {fa:.1f} GB")
                    if fa < 1.0:
                        log("[A.ABORT] cuda free < 1.0GB"); stop_reason="mem_abort"; break

            # trim any trailing level-0==8192 END-MARKER frames that were appended during the
            # tentative end-run before confirmation (these are end flags, not acoustic content;
            # the decoder appends its own single 8192 terminator).
            while all_audio and int(all_audio[-1][0]) == codebook_sizes[0]:
                all_audio.pop(); all_text.pop()
            n = len(all_audio)
            log(f"[A.done] generated {n} audio frames (after end-flag trim); stop_reason={stop_reason}")
            if n == 0:
                log("[A.WARN] zero frames generated");
            gen_ids = torch.stack(all_audio, dim=0).cpu() if n else torch.zeros(0, len(codebook_sizes), dtype=torch.long)
            log("=== MILESTONE 2c: audio loop done ===")
            log(f"[A.stats] frames={n} shape={tuple(gen_ids.shape)}")
            if n:
                log(f"[A.stats] overall min={int(gen_ids.min())} max={int(gen_ids.max())}")
                for lvl in range(len(codebook_sizes)):
                    u = torch.unique(gen_ids[:, lvl])
                    log(f"  level {lvl}: unique={u.numel()} range=[{int(u.min())},{int(u.max())}] cb_size={codebook_sizes[lvl]}")
                # level-0 diversity is the squeal tell
                u0 = torch.unique(gen_ids[:, 0]).numel()
                log(f"[A.SQUEAL-TELL] level-0 unique={u0} over {n} frames "
                    f"-> {'COLLAPSED (squeal risk)' if u0 < max(3, n//20) else 'VARIED (speech-like)'}")
            # main-stream text decode for sanity
            try:
                txt_ids = [t for t in all_text if t not in (AUDIOTEXT_PAD, AUDIOTEXT_START, AUDIOGEN_END, -1) and t < tv]
                decoded = tok.decode(txt_ids, skip_special_tokens=True) if txt_ids else ""
                log(f"[A.text] main-stream decoded ({len(txt_ids)} toks): {decoded!r}")
            except Exception as e:
                log(f"[A.text] decode failed: {e}")
            torch.save(gen_ids, OUT_IDS)
            log(f"=== MILESTONE 2d: saved {OUT_IDS} ({tuple(gen_ids.shape)}) ===")
            fa, ta = memfree(); log(f"[A.mem] end cuda free {fa:.1f}/{ta:.1f} GB")
        except Exception:
            _tb.print_exc(); log(f'[REQUEST {_ri}] FAILED — continuing')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
