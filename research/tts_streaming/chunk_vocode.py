#!/usr/bin/env python3
"""Chunked-vocode experiment: the streaming-TTS quality gate, offline and PAIRED.

Loads dumped codebook ids (LCN_TTS_DUMP_IDS=1 artifact) and vocodes the SAME ids
three ways, so the vocoder windowing is the only variable:

  full     — the shipping path: one decode over the whole utterance (reference)
  prefix   — growing-prefix streaming: window [0 : emit_end + lookahead] each round,
             emit only the new region. Every emitted sample has FULL left context;
             quality ceiling for streaming, cost quadratic in utterance length.
  window   — sliding-window streaming: [emit_start - left_ctx : emit_end + lookahead].
             Production shape: bounded cost per chunk, bounded left context.

Boundaries between emitted regions are joined with the same linear cross-fade the
shipping path uses between utterance segments (decode_save_concat2's overlap logic).

Run INSIDE the serving image with the model dir mounted and the server DOWN
(loads its own copy of the audio tower):

  docker run --rm --gpus all -v $WEIGHTS:/workspace/model:ro \
      -v ~/longcat-outputs:/workspace/outputs \
      --entrypoint python3 longcat-next-gb10:v0516-ttsdump \
      /workspace/scripts/research_chunk_vocode.py /workspace/outputs/<tag>.ids.pt

Emits <tag>.full.wav / <tag>.prefix.wav / <tag>.window.wav beside the input, plus
per-chunk decode timings (the number that decides whether the serving loop can
afford to vocode inline).
"""
import json
import sys
import time

import torch

MODEL_DIR = "/workspace/model"
CHUNK_FRAMES = int(__import__("os").environ.get("CHUNK_FRAMES", "50"))
LOOKAHEAD = int(__import__("os").environ.get("LOOKAHEAD", "12"))
LEFT_CTX = int(__import__("os").environ.get("LEFT_CTX", "100"))
FADE = 800  # samples, matches AUDIO_GEN_WAVE_OVERLAP's shipping default
SR = 24000


def build_tokenizer():
    from sglang.srt.models.longcat_next_audio import LongcatNextAudioTokenizer
    from sglang.srt.models.longcat_next_mm import LongcatNextForCausalLM

    cfg_dict = json.load(open(f"{MODEL_DIR}/config.json"))

    class _Shim:
        def to_dict(self):
            return cfg_dict

    full_cfg = LongcatNextForCausalLM._make_full_config(object(), _Shim())
    tok = LongcatNextAudioTokenizer(full_cfg)

    # Weights live in the main checkpoint under model.audio_tokenizer.*
    from safetensors import safe_open
    index = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))
    prefix = "model.audio_tokenizer."
    by_shard = {}
    for name, shard in index["weight_map"].items():
        if name.startswith(prefix):
            by_shard.setdefault(shard, []).append(name)
    sd = {}
    for shard, names in by_shard.items():
        with safe_open(f"{MODEL_DIR}/{shard}", framework="pt") as f:
            for n in names:
                sd[n[len(prefix):]] = f.get_tensor(n)
    missing, unexpected = tok.load_state_dict(sd, strict=False, assign=True)
    print(f"tokenizer weights: {len(sd)} loaded, {len(missing)} missing, "
          f"{len(unexpected)} unexpected", flush=True)
    if missing:
        print("  missing (first 8):", missing[:8], flush=True)

    # Resolve the vocoder weight path the way the server does
    # EXACTLY the server's candidate list (_ensure_vocoder_path) — an earlier version
    # invented its own fallback glob, which matched cosy24k_vocoder.py and loaded a
    # Python source file as weights. Mirror the shipping resolution, never improvise it.
    voc_cfg = tok.config.audio_config.cosy24kvocoder_config
    import os
    if not os.path.exists(getattr(voc_cfg, "weight_path", "") or ""):
        for cand in (f"{MODEL_DIR}/cosy24k_vocoder/hift.pt", f"{MODEL_DIR}/hift.pt"):
            if os.path.exists(cand):
                voc_cfg.weight_path = cand
                break
    print("vocoder weight_path:", voc_cfg.weight_path, flush=True)

    tok = tok.to("cuda").eval()
    return tok


@torch.no_grad()
def vocode(tok, ids):
    """ids [n_frames, codebooks] -> wave tensor [1, samples] (the full shipping math)."""
    if tok.cosy24kvocoder is None:
        from sglang.srt.models.cosy24k_vocoder import Cosy24kVocoder
        # hift.pt is a legacy pickle from the checkpoint the server already runs;
        # torch>=2.6 defaults weights_only=True and refuses it here (the serving
        # process loads it through its own path). Same trust boundary either way.
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        try:
            tok.cosy24kvocoder = Cosy24kVocoder.from_pretrained(
                tok.config.audio_config.cosy24kvocoder_config.weight_path).to("cuda")
        finally:
            torch.load = _orig_load
    ids = ids.to("cuda")
    ret = tok.decode(ids, bridge_length=torch.tensor([ids.shape[0]], device="cuda"))
    mel = ret.flow_matching_mel[0][: ret.flow_matching_mel_lengths[0], :]
    wave = tok.cosy24kvocoder.decode(mel.transpose(0, 1).to(torch.float32).unsqueeze(0))
    return wave.cpu()


def crossfade_join(pieces):
    """Join wave pieces with the shipping linear cross-fade."""
    out = [pieces[0]]
    for w in pieces[1:]:
        if out[-1].shape[1] > FADE and w.shape[1] > FADE:
            ramp_d = torch.linspace(1.0, 0.0, FADE)[None, :]
            ramp_u = torch.linspace(0.0, 1.0, FADE)[None, :]
            out[-1] = out[-1].clone()
            faded = out[-1][:, -FADE:] * ramp_d + w[:, :FADE] * ramp_u
            out[-1] = out[-1][:, :-FADE]
            out.append(faded)
            out.append(w[:, FADE:])
        else:
            out.append(w)
    return torch.cat(out, dim=1)


def save(path, wave):
    import numpy as np
    import scipy.io.wavfile as wavfile
    x = wave.squeeze(0).float().clamp(-1, 1).numpy()
    wavfile.write(path, SR, (x * 32767).astype(np.int16))
    print(f"saved {path} ({x.shape[0]/SR:.2f}s)", flush=True)


def main():
    ids_path = sys.argv[1]
    ids = torch.load(ids_path)  # [n_frames, codebooks], raw (offset-free), clamped
    n = ids.shape[0]
    print(f"{ids_path}: {n} frames, {ids.shape[1]} codebooks", flush=True)
    tok = build_tokenizer()
    base = ids_path[: -len(".ids.pt")]

    # --- full (reference: the shipping math, minus segment splitting) ---
    t0 = time.time()
    wave_full = vocode(tok, ids)
    print(f"full: one decode of {n} frames in {time.time()-t0:.2f}s", flush=True)
    save(base + ".full.wav", wave_full)
    spf = wave_full.shape[1] / n  # samples per frame, measured not assumed
    print(f"samples/frame: {spf:.1f} ({SR/spf:.1f} frames/s)", flush=True)

    if __import__("os").environ.get("FULL_ONLY", "0") == "1":
        return  # pair-member production for prosody-vs-mechanism listening; the
                # prefix/window variants were already owner-cleared 2026-08-11

    # --- growing prefix ---
    pieces, emitted, times = [], 0, []
    while emitted < n:
        emit_end = min(emitted + CHUNK_FRAMES, n)
        win_end = min(emit_end + LOOKAHEAD, n)
        t0 = time.time()
        w = vocode(tok, ids[:win_end])
        times.append(time.time() - t0)
        s0, s1 = int(emitted * spf), int(emit_end * spf)
        pieces.append(w[:, s0:min(s1, w.shape[1])])
        emitted = emit_end
    save(base + ".prefix.wav", crossfade_join(pieces))
    print(f"prefix: {len(times)} chunks, per-chunk s: "
          + " ".join(f"{t:.2f}" for t in times), flush=True)

    # --- sliding window ---
    pieces, emitted, times = [], 0, []
    while emitted < n:
        emit_end = min(emitted + CHUNK_FRAMES, n)
        win_start = max(0, emitted - LEFT_CTX)
        win_end = min(emit_end + LOOKAHEAD, n)
        t0 = time.time()
        w = vocode(tok, ids[win_start:win_end])
        times.append(time.time() - t0)
        s0 = int((emitted - win_start) * spf)
        s1 = int((emit_end - win_start) * spf)
        pieces.append(w[:, s0:min(s1, w.shape[1])])
        emitted = emit_end
    save(base + ".window.wav", crossfade_join(pieces))
    print(f"window: {len(times)} chunks (left_ctx={LEFT_CTX}, lookahead={LOOKAHEAD}), "
          "per-chunk s: " + " ".join(f"{t:.2f}" for t in times), flush=True)


if __name__ == "__main__":
    main()
