#!/usr/bin/env python3
"""Does the prefix cache serve one request's AUDIO to another request?

HYPOTHESIS. Multimodal content does not live in the token IDs — the processor writes
placeholder pads and the real content arrives as embeddings. The radix prefix cache keys
on token IDs. sglang guards against the resulting collision by deriving each item's
`pad_value` from a hash of its content (`MultimodalDataItem.set_pad_value`, whose
docstring says "Each item has its own hash and pad_value, enabling per-image
RadixAttention caching"). Our processor assigns a CONSTANT `pad_value` first, and
`set_pad_value()` early-returns when `pad_value is not None` — so `hash` is never
computed and every audio item shares one pad value.

If that matters, two requests with the same prompt text and the same audio token count
produce identical input_ids, and the second can be served the first's cached KV — i.e.
it transcribes the PREVIOUS clip.

DESIGN. Two clips of EXACTLY equal sample length (so the bridge-token count matches) cut
from different parts of a reference recording, so their content differs. Then:

  phase 1  A, B, B, A with an identical prompt   -> collision possible
  phase 2  same clips, each with a UNIQUE prompt prefix -> shared prefix broken

A content bleed in phase 1 that disappears in phase 2 is the signature. If both phases
agree, the cache is not the culprit and the transcription errors are something else.

Interpretation is by CONTENT DIFFERENCE, not by correctness — we do not need to know what
the clips say, only whether the answer tracks the clip that was actually sent. That
sidesteps trusting the transcriber's accuracy, which is separately in question.

    docker exec longcat-next python3 /workspace/scripts/probe_mm_cache.py
"""
import base64, io, json, os, sys, urllib.request, wave

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
KEY = os.environ.get("LCN_API_KEY", "")
VOICE = os.environ.get("PROBE_VOICE", "/workspace/scripts/voices/en_reference.wav")


def post(payload, timeout=600):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def segment(path, start_s, dur_s):
    """Byte-exact equal-length cut, so both clips yield the same audio token count."""
    with wave.open(path) as w:
        rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(int(start_s * rate))
        frames = w.readframes(int(dur_s * rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(ch); o.setsampwidth(sw); o.setframerate(rate)
        o.writeframes(frames)
    return buf.getvalue()


def ask(raw, prompt):
    b = base64.b64encode(raw).decode()
    r = post({"model": "longcat-next", "max_tokens": 60, "temperature": 0,
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "input_audio", "input_audio": {"data": b, "format": "wav"}}]}]})
    return (r["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")


def main():
    with wave.open(VOICE) as w:
        total = w.getnframes() / w.getframerate()
    print(f"reference: {VOICE} ({total:.2f}s)", flush=True)
    if total < 7:
        print("reference too short for two distinct 3s segments", flush=True)
        return 1

    dur = 3.0
    A = segment(VOICE, 0.0, dur)
    B = segment(VOICE, total - dur - 0.1, dur)
    assert len(A) == len(B), (len(A), len(B))
    print(f"clip A = first {dur}s, clip B = last {dur}s, both {len(A)} bytes\n", flush=True)

    P = "Transcribe this audio exactly. Output only the words spoken."
    print("--- phase 1: IDENTICAL prompt (prefix shared, collision possible) ---", flush=True)
    p1 = []
    for tag, clip in (("A", A), ("B", B), ("B", B), ("A", A)):
        t = ask(clip, P)
        p1.append((tag, t))
        print(f"  {tag}: {t[:110]!r}", flush=True)

    print("\n--- phase 2: UNIQUE prompt prefix per request (prefix broken) ---", flush=True)
    p2 = []
    for i, (tag, clip) in enumerate((("A", A), ("B", B), ("B", B), ("A", A)), 1):
        t = ask(clip, f"Request {i}. " + P)
        p2.append((tag, t))
        print(f"  {tag}: {t[:110]!r}", flush=True)

    def consistent(rows):
        """Do both presentations of the same clip agree, and do the two clips differ?"""
        a = [t for g, t in rows if g == "A"]
        b = [t for g, t in rows if g == "B"]
        return (a[0] == a[1], b[0] == b[1], a[0] != b[0])

    for name, rows in (("phase 1", p1), ("phase 2", p2)):
        sa, sb, diff = consistent(rows)
        print(f"\n{name}: A self-consistent={sa}  B self-consistent={sb}  A differs from B={diff}",
              flush=True)

    print("\nREAD: a bleed in phase 1 that disappears in phase 2 implicates the prefix "
          "cache. Both phases alike => cache is NOT the cause; look elsewhere.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
