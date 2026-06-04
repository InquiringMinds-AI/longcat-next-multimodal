#!/usr/bin/env python3
"""
MILESTONE 3 — decode generated audio codebook ids -> wav.

Loads /tmp/gen_audio_ids.pt ([T,8] raw audio codebook ids from M2), splits into segments
at level-0 == AUDIO_END_FLAG (8192) the way demo.py does, ensures each segment ends with an
8192 EOS row, and runs the proven M1 decode path (flow-matching decoder + Cosy24k vocoder).

Memory-safe: builds ONLY LongcatAudioTokenizer + Cosy24kVocoder (no backbone).
"""
import os, sys, json, traceback, types
import numpy as np
import torch

REPO = "/home/magi/Projects/LongCat-Next-inference"
sys.path.insert(0, REPO)
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/output/LongCat-Next-NVFP4-bf16mla")
CFG_DIR = os.path.join(MODEL_PATH, "nmm_infer") if os.path.exists(os.path.join(MODEL_PATH, "nmm_infer")) \
          else "/models/LongCat-Next/nmm_infer"
VOCODER_PATH = os.environ.get("VOCODER_PATH", "/models/LongCat-Next/cosy24k_vocoder/hift.pt")
IN_IDS = os.environ.get("IN_IDS", "/tmp/gen_audio_ids.pt")
OUT_WAV = os.environ.get("OUT_WAV", "/tmp/gen_audio.wav")


def log(*a):
    print(*a, flush=True)


# import-time shims (same as M1)
if "deepspeed" not in sys.modules:
    _ds = types.ModuleType("deepspeed"); _ds.__path__ = []
    _zero = types.ModuleType("deepspeed.zero"); _zero.register_external_parameter = lambda *a, **k: None
    _comm = types.ModuleType("deepspeed.comm")
    for n in ("is_initialized","get_rank","get_world_size","barrier","all_reduce","all_gather","broadcast"):
        setattr(_comm, n, (lambda *a, **k: None))
    _ds.zero = _zero; _ds.comm = _comm
    sys.modules["deepspeed"] = _ds; sys.modules["deepspeed.zero"] = _zero; sys.modules["deepspeed.comm"] = _comm
try:
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
    if not hasattr(_q25, "Qwen2RMSNorm") and hasattr(_q25, "Qwen2_5_VLRMSNorm"):
        _q25.Qwen2RMSNorm = _q25.Qwen2_5_VLRMSNorm
except Exception:
    pass
import importlib.util as _ilu
for name, attrs in [("decord",["VideoReader","cpu"]),("cv2",None),("av",None),("imagesize",None),("cairosvg",["svg2png"])]:
    if _ilu.find_spec(name) is None:
        m = types.ModuleType(name)
        for a in (attrs or []):
            setattr(m, a, (lambda *args, **kw: None))
        sys.modules[name] = m


def main():
    log(f"[cfg] IN_IDS={IN_IDS} -> OUT_WAV={OUT_WAV}")
    try:
        from transformers import CLIPVisionConfig as _CVC
        if hasattr(_CVC, "__validators__"): _CVC.__validators__ = {}
        if hasattr(_CVC, "validate"): _CVC.validate = staticmethod(lambda self: None)
    except Exception:
        pass
    from processor.flash_omni.configuration_omni import OmniConfig
    from processor.flash_omni.modeling_longcat_oe import LongcatAudioTokenizer
    from processor.decoder.cosy24k_vocoder.cosy24k_vocoder import Cosy24kVocoder
    from processor.decoder.audio_decode import decode_save_concat
    from utils.model_utils import load_weights_from_safetensors_helper

    # SDPA varlen fallback for the flow-matching decoder / matcha transformer
    import torch.nn.functional as _F
    import processor.flash_omni.audio_modeling_omni as _am
    import processor.flash_omni.matcha_transformer as _mt
    def _sdpa(q, k, v, cu_q, cu_k, mq, mk, causal=False, **kw):
        outs = []; cq = cu_q.tolist(); ck = cu_k.tolist()
        for i in range(len(cq)-1):
            qi = q[cq[i]:cq[i+1]].unsqueeze(0).transpose(1,2)
            ki = k[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2)
            vi = v[ck[i]:ck[i+1]].unsqueeze(0).transpose(1,2)
            outs.append(_F.scaled_dot_product_attention(qi, ki, vi, is_causal=bool(causal)).transpose(1,2).squeeze(0))
        return torch.cat(outs, dim=0)
    for _m in (_am, _mt):
        if getattr(_m, "flash_attn_varlen_func", None) is None:
            _m.flash_attn_varlen_func = _sdpa

    config = OmniConfig(**json.load(open(os.path.join(CFG_DIR, "config.json"))))
    codebook_sizes = list(config.audio_config.vq_config.codebook_sizes)
    log(f"[1] codebook_sizes={codebook_sizes}")

    atok = LongcatAudioTokenizer(config)
    sds = load_weights_from_safetensors_helper(MODEL_PATH, ["model.audio_tokenizer."], "cpu")
    atok.load_state_dict(sds[0], strict=False)
    atok = atok.to("cuda").to(torch.bfloat16).eval()
    vocoder = Cosy24kVocoder.from_pretrained(VOCODER_PATH).cuda()
    log("[2] audio_tokenizer + vocoder loaded")

    # patch torchaudio.save -> soundfile
    import torchaudio as _ta, soundfile as _sf
    def _save(path, tensor, sr, **kw):
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        if arr.ndim == 2: arr = arr.T
        _sf.write(path, arr, int(sr))
    _ta.save = _save

    gen = torch.load(IN_IDS).to("cuda").long()  # [T,8]
    log(f"[3] loaded gen ids {tuple(gen.shape)} min={int(gen.min())} max={int(gen.max())}")
    EOS = codebook_sizes[0]  # 8192 == AUDIO_END_FLAG_ID

    # BUG-2 FIX (decode side): the generation loop now stops on the FIRST level-0==8192 (the END
    # marker) and does NOT store that frame, so gen is ONE contiguous code sequence with no 8192
    # rows. We do NOT split into multiple segments. decode_wave_vocoder() finds the audio length
    # via the FIRST row whose level-0 == codebook_sizes[0]; so we append exactly ONE [8192]*8 row
    # as the terminator and decode the single sequence.
    # Defensive: if any stray 8192 rows leaked into the body (older runs), truncate at the first.
    # strip LEADING end-flag rows: canonical delay-step emits no audio -> shows as a leading 8192
    _start = 0
    while _start < gen.shape[0] and int(gen[_start, 0]) == EOS:
        _start += 1
    if _start > 0:
        log(f'[4] skipped {_start} leading end-flag row(s) (delay-step no-audio)')
        gen = gen[_start:]
    body_end = gen.shape[0]
    eos_rows = (gen[:, 0] == EOS).nonzero(as_tuple=True)[0]
    if eos_rows.numel() > 0:
        body_end = int(eos_rows[0].item())
        log(f"[4] stray level-0==8192 at row {body_end}; truncating body there")
    body = gen[:body_end]
    if body.shape[0] == 0:
        log("[ERR] empty audio body (first frame was the end flag)"); raise RuntimeError("no frames")
    seq = torch.cat([body, torch.full((1, len(codebook_sizes)), EOS, dtype=torch.long, device=gen.device)], dim=0)
    log(f"[4] single contiguous sequence: body={body.shape[0]} frames + 1 EOS row -> {seq.shape[0]}")
    segments = [seq]

    processed = [s.unsqueeze(0).to("cuda") for s in segments]
    with torch.no_grad():
        decode_save_concat(
            response_list=processed, vocoder=vocoder, audio_tokenizer=atok,
            codebook_sizes=codebook_sizes, path=OUT_WAV, sampling_rate=24000, wave_concat_overlap=1200,
        )

    if os.path.exists(OUT_WAV):
        arr, sr = _sf.read(OUT_WAV)
        wf = torch.from_numpy(np.asarray(arr)).float()
        samples = wf.shape[0]; ch = 1 if wf.dim() == 1 else wf.shape[1]
        rms = float(wf.pow(2).mean().sqrt()); peak = float(wf.abs().max())
        log(f"=== M3 SAVED {OUT_WAV} ===")
        log(f"=== WAVSTATS sr={sr} samples={samples} dur={samples/sr:.2f}s channels={ch} "
            f"rms={rms:.5f} peak={peak:.5f} nonsilent={rms>1e-4} ===")
    else:
        raise RuntimeError("no output wav")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
