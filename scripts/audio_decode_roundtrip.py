#!/usr/bin/env python3
"""
MILESTONE 1 — LongCat-Next AUDIO decode round-trip (de-risk vocoder on new stack).

Mirrors image's decode_phaseB philosophy: build ONLY the audio tokenizer
(audio_model encoder + audio_bridge_model VQ + audio_decoder + flow-matching decoder)
plus the Cosy24k HiFi-GAN vocoder. Loads NONE of the LongCat LLM backbone -> memory-safe.

Chain:
  wav -> OmniAudioProcessor.extract_fbank_features -> [128, frames] fbank
      -> audio_tokenizer.encode(fbank, encoder_length, bridge_length) -> RVQ codes [T, 8]
      -> append [8192]*8 EOS row (decode_wave_vocoder slices at level0==codebook_sizes[0])
      -> decode_save_concat(..., Cosy24kVocoder, sampling_rate=24000, overlap=1200)
      -> /tmp/audio_recon.wav

Runs INSIDE lc-vision container image with the inference repo on path.
"""
import os, sys, json, glob, traceback, types
import numpy as np
import torch

REPO = "/home/magi/Projects/LongCat-Next-inference"
sys.path.insert(0, REPO)

# --- minimal deepspeed stub (only deepspeed.zero.register_external_parameter is touched,
#     a no-op outside ZeRO-3 training). Avoids a heavy deepspeed install in the container. ---
if "deepspeed" not in sys.modules:
    _ds = types.ModuleType("deepspeed")
    _ds.__path__ = []  # mark as package so submodule imports resolve
    _zero = types.ModuleType("deepspeed.zero")
    def _register_external_parameter(module, parameter):  # no-op for inference
        return None
    _zero.register_external_parameter = _register_external_parameter
    _comm = types.ModuleType("deepspeed.comm")
    # distributed no-ops / single-process defaults (only used on training paths)
    _comm.is_initialized = lambda *a, **k: False
    _comm.get_rank = lambda *a, **k: 0
    _comm.get_world_size = lambda *a, **k: 1
    _comm.barrier = lambda *a, **k: None
    _comm.all_reduce = lambda *a, **k: None
    _comm.all_gather = lambda *a, **k: None
    _comm.broadcast = lambda *a, **k: None
    _ds.zero = _zero
    _ds.comm = _comm
    sys.modules["deepspeed"] = _ds
    sys.modules["deepspeed.zero"] = _zero
    sys.modules["deepspeed.comm"] = _comm

# --- transformers 5.6 renamed Qwen2RMSNorm -> Qwen2_5_VLRMSNorm in the qwen2_5_vl module.
#     navit_vq_model.py imports the old name; alias it back so the import chain resolves. ---
try:
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
    if not hasattr(_q25, "Qwen2RMSNorm") and hasattr(_q25, "Qwen2_5_VLRMSNorm"):
        _q25.Qwen2RMSNorm = _q25.Qwen2_5_VLRMSNorm
except Exception as _e:
    print(f"[shim] qwen2_5_vl alias skipped: {_e}", flush=True)

# --- processor_omni imports video/SVG deps at module top-level (decord, cv2, av, imagesize,
#     cairosvg) that are only used on image/video paths. Stub them so we can import the
#     OmniAudioProcessor (audio path) without installing those heavy/niche packages. ---
def _ensure_stub(name, attrs=None):
    import importlib.util
    if importlib.util.find_spec(name) is not None:
        return  # really installed; use the real one
    m = types.ModuleType(name)
    for a in (attrs or []):
        setattr(m, a, (lambda *args, **kw: None))
    if "." in name:
        m.__path__ = []
    sys.modules[name] = m

_ensure_stub("decord", ["VideoReader", "cpu"])
_ensure_stub("cv2")
_ensure_stub("av")
_ensure_stub("imagesize")
_ensure_stub("cairosvg", ["svg2png"])

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/output/LongCat-Next-NVFP4-bf16mla")
# audio_tokenizer weights live in BOTH checkpoints (bf16); use bf16mla (the served gen model)
AUDIO_WEIGHTS = MODEL_PATH
CFG_DIR = os.path.join(MODEL_PATH, "nmm_infer") if os.path.exists(os.path.join(MODEL_PATH, "nmm_infer")) \
          else "/models/LongCat-Next/nmm_infer"
VOCODER_PATH = os.environ.get("VOCODER_PATH", "/models/LongCat-Next/cosy24k_vocoder/hift.pt")
IN_WAV = os.environ.get("IN_WAV", "/tmp/aud_qa.wav")
OUT_WAV = os.environ.get("OUT_WAV", "/tmp/audio_recon.wav")


def log(*a):
    print(*a, flush=True)


def memfree():
    fa, ta = torch.cuda.mem_get_info()
    return fa / 1e9, ta / 1e9


def main():
    log(f"[cfg] MODEL_PATH={MODEL_PATH}")
    log(f"[cfg] CFG_DIR={CFG_DIR}")
    log(f"[cfg] VOCODER_PATH={VOCODER_PATH}")
    log(f"[cfg] IN_WAV={IN_WAV} -> OUT_WAV={OUT_WAV}")
    fa, ta = memfree(); log(f"[mem] cuda free {fa:.1f}/{ta:.1f} GB at start")

    # transformers 5.6 made CLIPVisionConfig a strict huggingface_hub dataclass that rejects
    # this checkpoint's visual_config (hidden 1280 / 12 heads). We only need the AUDIO config,
    # so neutralize the strict validators on CLIPVisionConfig before building OmniConfig.
    try:
        from transformers import CLIPVisionConfig as _CVC
        # huggingface_hub strict dataclass stores validators in a per-field dict
        # (__validators__) plus class-level validators (__class_validators__). Empty both.
        if hasattr(_CVC, "__validators__"):
            try:
                _CVC.__validators__ = {}
            except Exception:
                pass
        if hasattr(_CVC, "__class_validators__"):
            try:
                _CVC.__class_validators__ = []
            except Exception:
                pass
        if hasattr(_CVC, "validate"):
            _CVC.validate = staticmethod(lambda self: None)
        log("[shim] neutralized CLIPVisionConfig strict validation")
    except Exception as _e:
        log(f"[shim] CLIPVisionConfig validator patch skipped: {_e}")

    from processor.flash_omni.modeling_longcat_oe import LongcatAudioTokenizer
    from processor.flash_omni.processor_omni import OmniAudioProcessor
    from processor.decoder.cosy24k_vocoder.cosy24k_vocoder import Cosy24kVocoder
    from processor.decoder.audio_decode import decode_save_concat
    from utils.model_utils import load_weights_from_safetensors_helper

    # The container's flash_attn is a hollow stub -> flash_attn_varlen_func is None in the
    # audio encoder/decoder and the matcha (flow-matching) transformer. Install a pure-PyTorch
    # SDPA varlen fallback into both modules (mirrors the proven image_head fallback).
    import torch.nn.functional as _F
    import processor.flash_omni.audio_modeling_omni as _am
    import processor.flash_omni.matcha_transformer as _mt

    def _sdpa_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, **kw):
        # q,k,v: [total_tokens, num_heads, head_dim]; cu_*: [bs+1] cumulative lengths
        outs = []
        cq = cu_q.tolist(); ck = cu_k.tolist()
        for i in range(len(cq) - 1):
            qi = q[cq[i]:cq[i + 1]].unsqueeze(0).transpose(1, 2)  # [1,H,sq,D]
            ki = k[ck[i]:ck[i + 1]].unsqueeze(0).transpose(1, 2)
            vi = v[ck[i]:ck[i + 1]].unsqueeze(0).transpose(1, 2)
            oi = _F.scaled_dot_product_attention(qi, ki, vi, is_causal=bool(causal))
            outs.append(oi.transpose(1, 2).squeeze(0))           # [sq,H,D]
        return torch.cat(outs, dim=0)

    n_patched = 0
    for _mod in (_am, _mt):
        if getattr(_mod, "flash_attn_varlen_func", "missing") is None or \
           getattr(_mod, "flash_attn_varlen_func", None) is None:
            _mod.flash_attn_varlen_func = _sdpa_varlen
            n_patched += 1
    log(f"[shim] installed SDPA varlen fallback into {n_patched} audio module(s)")

    log("[1] building OmniConfig from nmm_infer/config.json ...")
    from processor.flash_omni.configuration_omni import OmniConfig
    cfg_dict = json.load(open(os.path.join(CFG_DIR, "config.json")))
    config = OmniConfig(**cfg_dict)
    ac = config.audio_config
    codebook_sizes = list(ac.vq_config.codebook_sizes)
    log(f"    codebook_sizes={codebook_sizes}")
    log(f"    audio sr={ac.sampling_rate} n_fft={ac.n_fft} hop={ac.hop_length} "
        f"mel={ac.num_mel_bins} max_s={ac.max_audio_seconds} pooler={ac.avg_pooler} "
        f"kernel={ac.kernel_size} stride={ac.stride_size}")

    log("[2] building LongcatAudioTokenizer + loading model.audio_tokenizer.* weights ...")
    atok = LongcatAudioTokenizer(config)
    sds = load_weights_from_safetensors_helper(AUDIO_WEIGHTS, ["model.audio_tokenizer."], "cpu")
    miss = atok.load_state_dict(sds[0], strict=False)
    log(f"    loaded {len(sds[0])} audio_tokenizer tensors; "
        f"missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
    if miss.missing_keys:
        log("    first missing:", miss.missing_keys[:8])
    if miss.unexpected_keys:
        log("    first unexpected:", miss.unexpected_keys[:8])
    atok = atok.to("cuda").to(torch.bfloat16).eval()
    fa, _ = memfree(); log(f"[mem] cuda free {fa:.1f} GB after audio_tokenizer load")

    log("[3] building Cosy24kVocoder from hift.pt ...")
    vocoder = Cosy24kVocoder.from_pretrained(VOCODER_PATH).cuda()
    fa, _ = memfree(); log(f"[mem] cuda free {fa:.1f} GB after vocoder load")
    if fa < 20:
        log("[ABORT] cuda free < 20GB after loads"); raise RuntimeError("mem abort")

    log("[4] OmniAudioProcessor: load + fbank extract ...")
    ap = OmniAudioProcessor(ac)
    wav = ap.load_audio_waveform(IN_WAV, return_tensors=True)  # mono, resampled to ac.sampling_rate
    log(f"    waveform shape={tuple(wav.shape)} (channels, samples) @ {ac.sampling_rate}Hz "
        f"-> {wav.shape[1]/ac.sampling_rate:.2f}s")
    fbank, valid_frames = ap.extract_fbank_features(wav)  # (mel, frames) numpy, padded to max_audio_seconds
    enc_len, bridge_len = ap.inference_output_length(ac, valid_frames)
    log(f"    fbank shape={fbank.shape} valid_frames={valid_frames} "
        f"encoder_length={enc_len} bridge_length={bridge_len}")

    # shape audios like the processor's collate: [B, mel, frames]
    audios = torch.from_numpy(fbank).unsqueeze(0).to("cuda").to(torch.bfloat16)  # [1,128,F]
    encoder_length = torch.tensor([enc_len], device="cuda")
    bridge_length = torch.tensor([bridge_len], device="cuda")

    log("[5] encode -> RVQ codes ...")
    with torch.no_grad():
        codes = atok.forward(audios, encoder_length=encoder_length, bridge_length=bridge_length)
    log(f"    raw codes shape={tuple(codes.shape)} dtype={codes.dtype}")
    codes = codes.long().squeeze(0) if codes.dim() == 3 else codes.long()
    # ensure [T, 8]
    if codes.dim() != 2 or codes.shape[-1] != len(codebook_sizes):
        # some encoders return [B, T, n_cb]; flatten batch
        codes = codes.reshape(-1, len(codebook_sizes))
    log(f"    codes -> [T,8] = {tuple(codes.shape)}; "
        f"min={int(codes.min())} max={int(codes.max())}")
    for lvl in range(len(codebook_sizes)):
        u = torch.unique(codes[:, lvl])
        log(f"    level {lvl}: unique={u.numel()} range=[{int(u.min())},{int(u.max())}] cb_size={codebook_sizes[lvl]}")

    # append EOS row [8192]*8 (decode_wave_vocoder finds end via level0==codebook_sizes[0])
    eos = torch.full((1, len(codebook_sizes)), codebook_sizes[0], dtype=torch.long, device=codes.device)
    seg = torch.cat([codes, eos], dim=0)  # [T+1, 8]
    log(f"    segment with EOS = {tuple(seg.shape)} (last row level0={int(seg[-1,0])} == {codebook_sizes[0]})")

    # This torchaudio build routes save() through torchcodec (absent). Patch torchaudio.save
    # to use soundfile, which is installed. full_wave is [channels, samples] -> sf wants [samples, channels].
    import torchaudio as _ta
    import soundfile as _sf
    def _sf_save(path, tensor, sr, **kw):
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        if arr.ndim == 2:
            arr = arr.T  # [samples, channels]
        _sf.write(path, arr, int(sr))
    _ta.save = _sf_save
    log("[shim] patched torchaudio.save -> soundfile.write")

    log("[6] decode_save_concat (flow-matching decoder + Cosy24k vocoder) ...")
    processed_sequences = [seg.unsqueeze(0).to(atok.device if hasattr(atok, 'device') else 'cuda')]
    fa, _ = memfree(); log(f"[mem] cuda free {fa:.1f} GB before decode")
    with torch.no_grad():
        decode_save_concat(
            response_list=processed_sequences,
            vocoder=vocoder,
            audio_tokenizer=atok,
            codebook_sizes=codebook_sizes,
            path=OUT_WAV,
            sampling_rate=24000,
            wave_concat_overlap=1200,
        )
    fa, _ = memfree(); log(f"[mem] cuda free {fa:.1f} GB after decode")

    # report objective wav stats
    if os.path.exists(OUT_WAV):
        arr, sr = _sf.read(OUT_WAV)  # [samples] or [samples, channels]
        wf = torch.from_numpy(np.asarray(arr)).float()
        samples = wf.shape[0]
        channels = 1 if wf.dim() == 1 else wf.shape[1]
        rms = float(wf.pow(2).mean().sqrt())
        peak = float(wf.abs().max())
        dur = samples / sr
        log(f"=== M1 SAVED {OUT_WAV} ===")
        log(f"=== WAVSTATS sr={sr} samples={samples} dur={dur:.2f}s "
            f"channels={channels} rms={rms:.5f} peak={peak:.5f} "
            f"nonsilent={rms>1e-4} ===")
    else:
        log(f"[ERR] {OUT_WAV} was not written")
        raise RuntimeError("no output wav")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
