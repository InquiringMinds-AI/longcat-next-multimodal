#!/usr/bin/env python3
"""Segment-join A/B for multi-round TTS: same ids, two join treatments.

The 2026-08-14 owner listen found the joins in a 4-sentence recitation
"really awkward". Measurement showed WHY they could be: the inter-sentence
pause is whatever trailing/leading silence the model generated around each
end flag, minus the 50 ms cross-fade — measured anywhere from ~0 ms
(sentences slammed together) to ~510 ms across the delivered clips. This
harness discriminates pause-inconsistency from the other candidate
(per-round intonation reset, which no join treatment can fix):

  ship — the shipping join: linear cross-fade, AUDIO_GEN_WAVE_OVERLAP samples
         (mirrors the server's _stream_emit drain + reference decode_save_concat2)
  gap  — silence-normalized join: trim each segment's trailing AND leading
         silence to zero (10 ms windows, 2%-of-peak RMS, the server's trim
         thresholds), then insert a fixed GAP_MS pause of true silence.

If "gap" fixes the awkwardness, the serving fix is pause normalization at
segment boundaries; if both variants sound the same, the awkwardness is the
model's per-round delivery and joins are innocent.

Run inside the serving image, server DOWN (loads its own audio tower):
  docker run --rm --gpus all -v $WEIGHTS:/workspace/model:ro \
      -v ~/longcat-outputs:/workspace/outputs \
      --entrypoint python3 longcat-next-gb10:v0516-multitts4 \
      /workspace/scripts/research_join_variants.py /workspace/outputs/<tag>.ids.pt

Emits <tag>.join-ship.wav and <tag>.join-gap.wav beside the input.
"""
import os
import sys

import torch

# Reuse the offline tower loader + vocode math from the chunk_vocode harness —
# same directory inside the image (scripts/), same weights resolution.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research_chunk_vocode import build_tokenizer, vocode, save, SR

FADE = int(os.environ.get("FADE", "1200"))     # AUDIO_GEN_WAVE_OVERLAP shipping default
GAP_MS = int(os.environ.get("GAP_MS", "300"))  # normalized inter-sentence pause


def split_segments(ids, marker):
    """ids [n, cb] with marker rows (level-0 == codebook_sizes[0]) BETWEEN segments."""
    ends = (ids[:, 0] == marker).nonzero().view(-1).tolist()
    segs, start = [], 0
    for e in ends:
        if e > start:
            segs.append(ids[start:e])
        start = e + 1
    if start < ids.shape[0]:
        segs.append(ids[start:])
    return segs


def crossfade(a, b, fade):
    if a.shape[1] > fade and b.shape[1] > fade:
        ramp_d = torch.linspace(1.0, 0.0, fade)[None, :]
        ramp_u = torch.linspace(0.0, 1.0, fade)[None, :]
        mid = a[:, -fade:] * ramp_d + b[:, :fade] * ramp_u
        return torch.cat([a[:, :-fade], mid, b[:, fade:]], dim=1)
    return torch.cat([a, b], dim=1)


def trim_edges(w):
    """Cut leading and trailing silence to zero (server trim thresholds)."""
    x = w.squeeze(0)
    win = SR // 100
    n_win = x.shape[0] // win
    if n_win < 3:
        return w
    rms = x[: n_win * win].view(n_win, win).pow(2).mean(dim=1).sqrt()
    peak = rms.max()
    if peak <= 0:
        return w
    active = (rms > 0.02 * peak).nonzero().view(-1)
    if active.numel() == 0:
        return w
    lo = int(active[0]) * win
    hi = min(x.shape[0], (int(active[-1]) + 1) * win)
    return w[:, lo:hi]


def main():
    tok = build_tokenizer()
    marker = tok.config.audio_config.vq_config.codebook_sizes[0]
    gap = torch.zeros(1, SR * GAP_MS // 1000)
    for ids_path in sys.argv[1:]:
        ids = torch.load(ids_path)
        segs = split_segments(ids, marker)
        print(f"{ids_path}: {len(segs)} segment(s), frames "
              f"{[int(s.shape[0]) for s in segs]}", flush=True)
        waves = [vocode(tok, s) for s in segs]
        base = ids_path[: -len(".ids.pt")]

        ship = waves[0]
        for w in waves[1:]:
            ship = crossfade(ship, w, FADE)
        save(base + ".join-ship.wav", ship)

        trimmed = [trim_edges(w) for w in waves]
        pieces = [trimmed[0]]
        for w in trimmed[1:]:
            pieces += [gap, w]
        save(base + ".join-gap.wav", torch.cat(pieces, dim=1))


if __name__ == "__main__":
    main()
