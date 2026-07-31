#!/usr/bin/env python3
"""Extract the codec frame ids the audio tokenizer assigns to SILENCE.

Run INSIDE the container: python3 /workspace/outputs/extract_silence_frames.py
Replicates the processor's mel pipeline on 2s of digital silence, loads the
audio tokenizer (encoder+bridge) on CPU from the checkpoint shards, encodes,
and prints per-frame codebook ids. The modal frame = the canonical
"silence frame" for first-frame conditioning injection.
"""
import json
import torch

MODEL = "/workspace/model"

cfg = json.load(open(f"{MODEL}/config.json"))
acfg = cfg["audio_config"]
sr = acfg.get("sampling_rate", 16000)
n_fft = acfg.get("n_fft", 400)
hop = acfg.get("hop_length", 160)
n_mels = acfg.get("num_mel_bins", 128)
kernel = acfg.get("kernel_size", 3)
stride = acfg.get("stride_size", 2)
pool = acfg.get("avg_pooler", 4)
max_s = acfg.get("max_audio_seconds", 30)

# --- mel of 2s silence, exactly like the processor ---
wave = torch.zeros(1, 2 * sr)
seg_len = max_s * sr
valid_frames = min(seg_len // hop, wave.shape[1] // hop + 1)
wave = torch.nn.functional.pad(wave, (0, seg_len - wave.shape[1]))
from transformers.audio_utils import mel_filter_bank
mf = mel_filter_bank(num_frequency_bins=1 + n_fft // 2, num_mel_filters=n_mels,
                     min_frequency=0.0, max_frequency=sr / 2.0, sampling_rate=sr,
                     norm="slaney", mel_scale="slaney")
stft = torch.stft(wave, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True)
mag = stft[..., :-1].abs() ** 2
mel = torch.from_numpy(mf).float().T @ mag
log_spec = torch.clamp(mel, min=1e-10).log10()
mx = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
log_spec = torch.maximum(log_spec, mx - 8.0)
log_spec = (log_spec + 4.0) / 4.0
feats = log_spec[0]
feats[:, int(valid_frames):] = 0.0
enc_len = (int(valid_frames) + 2 * (kernel // 2) - kernel) // 1 + 1
enc_len = (enc_len + 2 * (kernel // 2) - kernel) // stride + 1
br_len = enc_len // pool if pool > 1 else enc_len
print(f"mel {tuple(feats.shape)}  encoder_length={enc_len}  bridge_length={br_len}")

# --- audio tokenizer on CPU from shards ---
import sys, shutil as _sh, os as _os
_pkg = "/tmp/lcfg_pkg/lcfg"
_os.makedirs(_pkg, exist_ok=True)
for f in ("configuration_longcat_next.py", "configuration_longcat_ngram.py"):
    _sh.copy(f"{MODEL}/{f}", f"{_pkg}/{f}")
open(f"{_pkg}/__init__.py", "w").close()
sys.path.insert(0, "/tmp/lcfg_pkg")
from lcfg.configuration_longcat_next import LongcatNextConfig
full_cfg = LongcatNextConfig.from_pretrained(MODEL)
from sglang.srt.models.longcat_next_audio import LongcatNextAudioTokenizer
tok = LongcatNextAudioTokenizer(full_cfg)
tok.eval()

from safetensors import safe_open
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
prefix = "model.audio_tokenizer."
by_shard = {}
for k, shard in idx["weight_map"].items():
    if k.startswith(prefix):
        by_shard.setdefault(shard, []).append(k)
sd = {}
for shard, keys in by_shard.items():
    with safe_open(f"{MODEL}/{shard}", framework="pt") as f:
        for k in keys:
            sd[k[len(prefix):]] = f.get_tensor(k)
missing, unexpected = tok.load_state_dict(sd, strict=False)
print(f"loaded {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")

with torch.no_grad():
    ids = tok.encode(feats.unsqueeze(0).float(),
                     torch.tensor([enc_len]), torch.tensor([br_len]))
print(f"encoded silence -> {tuple(ids.shape)}")
frames = ids.tolist()
from collections import Counter
counted = Counter(tuple(f) for f in frames)
print("first 10 frames:")
for f in frames[:10]:
    print("  ", f)
modal, count = counted.most_common(1)[0]
print(f"modal frame ({count}/{len(frames)}): {list(modal)}")
print("LCN_TTS_SILENCE_FRAME=" + ",".join(str(x) for x in modal))
