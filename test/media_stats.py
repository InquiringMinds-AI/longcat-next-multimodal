#!/usr/bin/env python3
"""Objective descriptors for generated artifacts, shared by the test battery.

Extracted from selftest.py so the SAME measurements can be re-run offline against
saved artifacts (which is how the tail-vs-cadence question was settled) instead of
only ever at generation time. No verdicts live here — only measurements.
"""
import wave


def image_stats(raw):
    """Objective descriptors of a PNG. `pixel_std` and `distinct_colors` are the
    ones that separate a real photograph from a featureless smudge — reported as
    numbers, with no verdict attached."""
    st = {"bytes": len(raw), "is_png": raw[:4].hex() == "89504e47"}
    try:
        import cv2, numpy as np
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            st["decode"] = "FAILED"
            return st
        st["dims"] = f"{arr.shape[1]}x{arr.shape[0]}"
        st["pixel_std"] = round(float(arr.std()), 2)
        st["mean_rgb"] = [int(v) for v in arr.reshape(-1, 3).mean(axis=0)[::-1]]
        # distinct colours at 5-bit depth: a smudge collapses to very few
        q = (arr >> 3).astype("uint32")
        st["distinct_colors_5bit"] = int(len(np.unique(q[:, :, 0] * 1024 + q[:, :, 1] * 32 + q[:, :, 2])))
    except Exception as e:
        st["stats_error"] = str(e)[:80]
    return st


def audio_stats(raw, text=None):
    """Objective descriptors of a WAV.

    Duration alone does NOT diagnose a trailing artifact: the same sentence read at a
    slower cadence is legitimately longer, and total rms/peak move with cadence too
    (inter-word pauses dilute rms without any tail existing). So this separates the
    two explanations instead of conflating them, via a 20ms energy envelope:

      lead_ms / trail_ms  — silence BEFORE first and AFTER last speech frame. A tail
                            is trail_ms, and nothing else is.
      speech_sec          — span from first to last speech frame; the part cadence owns.
      cps                 — characters of input text per speech_sec = a cadence proxy.
                            Stable cps + growing trail_ms => tail. Falling cps with
                            flat trail_ms => the model simply read it slower.
      max_gap_ms          — widest INTERIOR silence. Normal speech has some (this
                            sentence has a comma), so read it against its own baseline.
      after_gap_ms        — voiced audio following that widest gap. This pair is what
                            can see "silence ... um?", which trail_ms CANNOT: the
                            filler is voiced, so it ends the trail and reclassifies the
                            silence as interior.

    MEASURED BASELINE (2026-08-10, 10 renders of "Self test, all systems nominal.",
    all owner-unheard, so this is the shape of ORDINARY output, not of the defect):

      cps 6.1-14.5   speech_sec 2.14-5.12   trail_ms 0-620   max_gap_ms 80-1300

    Two findings that shape how these are read:
      * trail_ms is NEGATIVELY correlated with duration — the longest renders had the
        smallest trails. Long renders are not padded ones. Duration was tried as a
        tail proxy and RETIRED; it pointed the wrong way.
      * max_gap_ms is strongly anti-correlated with cps (1300ms gap at cps 6.1, 80ms
        at cps 14.5). The cadence swing is INTERIOR PAUSE LENGTH, not word rate.

    THE DEFECT METRIC IS `end_jump`, and it is the only one here validated against
    owner-adjudicated ground truth (8 renders, 3 defective, 2026-08-10):

        BAD   0.130  0.131  0.138      (click / lost syllable / cut mid-consonant)
        GOOD  0.001  0.002  0.005  0.011  0.012

    A 10x gap. It works because it measures the defect DIRECTLY: the owner hears a click,
    a click IS a waveform discontinuity, so measure the discontinuity. `end_level` and a
    tail-decay ratio also split cleanly but with narrower margins.

    `trail_ms` DOES NOT WORK for this and a prediction based on it FAILED: a defective clip
    and a clean one both sat at 200ms, and a clean one sat at 100ms below two defective
    ones. The reason is instructive — a clip can click and THEN pad out to normal trailing
    silence, so the stop is loud while the ending is quiet. Keep trail_ms for what it does
    measure (padding), not as a defect signal.

    Caveat kept attached: n=8 with 3 positives, so a clean split could be luck. What earns
    end_jump more trust than the metrics that failed today is that its mechanism is
    physical rather than statistical.

    Deliberately NOT provided: a flag for "this render has the defect". A first attempt
    (>=400ms gap with <=600ms of audio after) fired on 2 of these 10 ordinary renders.
    With zero owner-adjudicated defective samples to calibrate against, any threshold
    here is a guess wearing a measurement's clothes. Report the numbers; let a human
    adjudicate. (Contrast image_stats' palette advisory, which HAS an adjudicated
    bad sample on one side and three good ones on the other.)
    """
    st = {"bytes": len(raw)}
    try:
        import io
        with wave.open(io.BytesIO(raw)) as w:
            frames, rate = w.getnframes(), w.getframerate()
            st["seconds"] = round(frames / rate, 2) if rate else -1
            st["sample_rate"] = rate
            st["channels"] = w.getnchannels()
            pcm = w.readframes(frames)
        import numpy as np
        a = np.frombuffer(pcm, dtype=np.int16).astype("float32")
        if a.size:
            st["peak"] = round(float(np.abs(a).max()) / 32768, 3)
            st["rms"] = round(float(np.sqrt((a ** 2).mean())) / 32768, 4)
            ch = max(1, st.get("channels", 1))
            mono = a.reshape(-1, ch).mean(axis=1) if ch > 1 else a
            win = max(1, int(rate * 0.02))
            nw = mono.size // win
            if nw:
                env = np.sqrt((mono[:nw * win].reshape(nw, win) ** 2).mean(axis=1))
                # Relative floor: 2% of the loudest frame, with an absolute guard so a
                # silent file doesn't threshold itself into "all speech".
                thr = max(env.max() * 0.02, 20.0)
                voiced = np.flatnonzero(env > thr)
                if voiced.size:
                    st["lead_ms"] = int(voiced[0] * 20)
                    st["trail_ms"] = int((nw - 1 - voiced[-1]) * 20)
                    sp = (voiced[-1] - voiced[0] + 1) * 0.02
                    st["speech_sec"] = round(sp, 2)
                    if text and sp > 0:
                        st["cps"] = round(len(text) / sp, 1)
                    # The owner-heard artifact was silence FOLLOWED BY "um?" — and that
                    # filler is voiced, so it terminates the trail and hides the silence
                    # INSIDE the speech span. trail_ms alone cannot see it, hence these.
                    if voiced.size > 1:
                        gaps = np.diff(voiced) - 1
                        gi = int(np.argmax(gaps))
                        st["max_gap_ms"] = int(gaps[gi] * 20)
                        # Voiced audio (not wall-clock span) following that widest gap.
                        st["after_gap_ms"] = int(voiced.size - gi - 1) * 20
                    # end_jump: the largest sample-to-sample discontinuity in the last 60ms
                    # of voiced audio. THE validated defect metric — see the block comment.
                    _e = (voiced[-1] + 1) * win
                    _seg = mono[max(0, _e - int(rate * 0.06)):_e]
                    if _seg.size > 1:
                        st["end_jump"] = round(float(np.abs(np.diff(_seg)).max()) / 32768, 3)
                    # How loud the signal still is where it ENDS, relative to the clip peak.
                    st["end_level"] = round(float(env[voiced[-1]] / max(env.max(), 1e-6)), 3)
                else:
                    st["trail_ms"] = -1  # no frame above threshold: silent render
    except Exception as e:
        st["stats_error"] = str(e)[:80]
    return st
