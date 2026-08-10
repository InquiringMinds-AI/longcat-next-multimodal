#!/usr/bin/env python3
"""Does the TTS render every word of its input? (round-trip: generate -> transcribe)

WHY THIS EXISTS. The TTS defect on this port was chased for a long time as a TRAILING
artifact, measured by duration. Both were wrong. The owner adjudicated two renders of
the same sentence on 2026-08-10:

    slow render  -> "very slow cadence, and ends with 'all systems' skipping nominal"
    fast render  -> "no missing words or extra tail"

So the failure is CONTENT LOSS (and, in an earlier owner-heard sample, content GAIN:
silence plus a stray "um?"). Those are the same mechanism with opposite signs — the
acoustic phase ending too early or too late — and NO silence-geometry measurement can
see the first one at all. A truncated render looks perfectly healthy to trail_ms.

WHAT IT MEASURES. Each render is transcribed by this same server's audio-understanding
path and compared word-by-word against the input text. `missing` / `extra` are the
signal; the silence stats come along so truncation can be correlated against them.

⚠ TWO HONEST LIMITS, do not launder them away:
  * The transcriber is a model. A missing word in the transcript is EVIDENCE of a
    missing word in the audio, not proof — ASR drops words too. Treat a hit as "listen
    to this one", which is why every render is saved.
  * `cps` is computed against the INPUT text, so it is only a cadence measure when the
    render is complete. On a truncated render the numerator is too large and cps
    OVERSTATES the rate. That confound is exactly why low cps was misread as a
    stylistic slow read rather than as a truncation signature.

    docker exec longcat-next python3 /workspace/scripts/tts_roundtrip.py
Env: RT_RUNS (default 8), RT_TEXT, RT_GLOB (analyze existing WAVs instead of generating).
"""
import glob as globmod
import base64, json, os, re, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from media_stats import audio_stats

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
KEY = os.environ.get("LCN_API_KEY", "")
RUNS = int(os.environ.get("RT_RUNS", "8"))
TEXT = os.environ.get("RT_TEXT", "Self test, all systems nominal.")
GLOB = os.environ.get("RT_GLOB", "")
OUT = os.environ.get("LCN_OUTPUT_DIR", "/workspace/outputs")


def post(path, payload, timeout=900, binary=False):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw if binary else json.loads(raw)


def words(s):
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w]


def transcribe(raw):
    b = base64.b64encode(raw).decode()
    r = post("/v1/chat/completions", {
        "model": "longcat-next", "max_tokens": 80, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe this audio exactly. Output only the words spoken."},
            {"type": "input_audio", "input_audio": {"data": b, "format": "wav"}}]}]})
    return r["choices"][0]["message"]["content"] or ""


def report(tag, raw, path=""):
    st = audio_stats(raw, TEXT)
    try:
        tr = transcribe(raw)
    except Exception as e:
        tr = f"<transcribe failed: {type(e).__name__}: {str(e)[:60]}>"
    want, got = words(TEXT), words(tr)
    missing = [w for w in want if w not in got]
    extra = [w for w in got if w not in want]
    keys = ("seconds", "speech_sec", "cps", "lead_ms", "trail_ms", "max_gap_ms")
    stats = " ".join(f"{k}={st[k]}" for k in keys if k in st)
    print(f"{tag}: {stats}", flush=True)
    print(f"    transcript: {tr.strip()[:120]!r}", flush=True)
    print(f"    missing={missing} extra={extra}"
          + (f" saved={path}" if path else ""), flush=True)
    return bool(missing), st.get("cps"), missing


def main():
    print(f"tts_roundtrip: text={TEXT!r}", flush=True)
    rows = []
    if GLOB:
        files = sorted(globmod.glob(GLOB))
        print(f"analyzing {len(files)} existing renders\n", flush=True)
        for f in files:
            rows.append(report(os.path.basename(f), open(f, "rb").read()))
    else:
        for i in range(1, RUNS + 1):
            raw = post("/v1/audio/speech",
                       {"model": "longcat-next", "voice": "en", "input": TEXT}, binary=True)
            p = os.path.join(OUT, f"roundtrip_{i:02d}.wav")
            try:
                open(p, "wb").write(raw)
            except Exception:
                p = ""
            rows.append(report(f"run {i:02d}", raw, p))

    bad = [r for r in rows if r[0]]
    print(f"\nRESULT: {len(bad)}/{len(rows)} renders missing at least one word", flush=True)
    if rows and all(r[1] is not None for r in rows):
        lo = [r[1] for r in rows if r[0]]
        hi = [r[1] for r in rows if not r[0]]
        if lo and hi:
            # Reported as ranges, not a threshold: this is the correlation the
            # truncation hypothesis predicts, and it needs its own evidence.
            print(f"cps of truncated renders: {sorted(lo)}", flush=True)
            print(f"cps of complete  renders: {sorted(hi)}", flush=True)
    print("LISTEN before trusting any of this — the transcriber is a model too.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
