#!/usr/bin/env python3
"""End-to-end self-test for the LongCat-Next server — exercises ALL modalities via the
OpenAI-compatible endpoints. Run INSIDE the container:

    docker exec longcat-next python3 /workspace/scripts/selftest.py

Verifies: text, image generation, image understanding, audio generation, audio
understanding, video understanding, tool calling. Exit code 0 iff every modality passes.

WHAT A PASS DOES AND DOES NOT MEAN
----------------------------------
These checks verify that each modality RESPONDS and returns well-formed data of the
right type. For the two GENERATION paths they cannot verify that the output is
CORRECT, and the difference has bitten this project repeatedly:

  * a 1040x1040 PNG containing a featureless smudge scored PASS on "is it a PNG";
  * a TTS clip with several seconds of trailing silence and a stray "um?" scored
    PASS on "is it a WAV over 1KB".

So this script now does three things beyond PASS/FAIL:

  1. prints OBJECTIVE STATISTICS for every artifact (image: size, pixel spread,
     distinct colours; audio: duration, sample rate, peak/RMS level) — these are
     measurements, NOT quality judgments, and they are chosen because they move
     when the known silent-degradation modes occur (a smudge has near-zero pixel
     spread; a trailing artifact shows up as duration);
  2. SAVES every generated artifact to LCN_OUTPUT_DIR so a human can actually look
     and listen — generation changes are only ever closed by human eyes/ears;
  3. on failure, DUMPS THE RAW RESPONSE (content, finish_reason, status) instead of
     a one-line verdict, because the two failure modes behind "no tool_calls" — an
     unhandled tool-call dialect versus the model not attempting a call at all —
     look identical in a pass/fail line and have opposite fixes.

Env: PORT, LCN_API_KEY, LCN_OUTPUT_DIR, SELFTEST_KEEP (default 1; 0 deletes artifacts).
"""
import base64, json, os, sys, time, wave
import requests

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
# authenticated deployments (LCN_API_KEY set) require the bearer on every request
_KEY = os.environ.get("LCN_API_KEY", "").strip()
_session = requests.Session()
if _KEY:
    _session.headers["Authorization"] = "Bearer " + _KEY
requests = _session  # route module-level requests.post/get through the session
OUT = os.environ.get("LCN_OUTPUT_DIR", "/tmp")
KEEP = os.environ.get("SELFTEST_KEEP", "1") != "0"
VOICE = "/workspace/scripts/voices/en_reference.wav"
RUN = time.strftime("%Y%m%d-%H%M%S")
results = []


def rec(name, ok, detail="", evidence=None):
    """Record a check. `evidence` is printed whenever the check FAILS — it is the
    material needed to diagnose, and discarding it is how a defect gets mislabeled
    'flaky' and closed."""
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok and evidence:
        for k, v in evidence.items():
            print(f"         {k}: {v}", flush=True)


def save(data, suffix):
    """Persist an artifact for human inspection; returns the path (or '' if off)."""
    if not KEEP:
        return ""
    path = os.path.join(OUT, f"selftest_{RUN}_{suffix}")
    try:
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        return f"(save failed: {e})"


def resp_evidence(r):
    """Everything worth knowing about a response that did not do what we expected."""
    ev = {"http_status": r.status_code}
    try:
        j = r.json()
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        ev["finish_reason"] = repr(ch.get("finish_reason"))
        ev["content"] = repr(msg.get("content"))
        if msg.get("tool_calls"):
            ev["tool_calls"] = json.dumps(msg["tool_calls"])[:400]
        if j.get("usage"):
            ev["usage"] = json.dumps(j["usage"])
        if j.get("error"):
            ev["error"] = json.dumps(j["error"])[:400]
    except Exception:
        ev["body"] = repr(r.text[:400])
    return ev


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


def audio_stats(raw):
    """Objective descriptors of a WAV. `seconds` is the one that surfaces trailing
    artifacts — the same sentence rendering much longer than usual is measurable
    even though 'does it sound right' is not."""
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
    except Exception as e:
        st["stats_error"] = str(e)[:80]
    return st


print(f"selftest run {RUN}  ->  artifacts in {OUT}" if KEEP else f"selftest run {RUN} (artifacts not kept)",
      flush=True)

# 1. text
try:
    r = requests.post(BASE + "/v1/chat/completions", json={"model": "longcat-next",
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 8, "temperature": 0}, timeout=120)
    t = r.json()["choices"][0]["message"]["content"]
    rec("text", r.status_code == 200 and len(t) > 0, repr(t[:80]), resp_evidence(r))
except Exception as e:
    rec("text", False, str(e))

# 2. image generation
img_b64 = None
try:
    r = requests.post(BASE + "/v1/images/generations",
        json={"prompt": "A photograph of a red apple on a wooden table.", "response_format": "b64_json"}, timeout=900)
    img_b64 = r.json()["data"][0]["b64_json"]
    raw = base64.b64decode(img_b64)
    st = image_stats(raw)
    path = save(raw, "image.png")
    # PASS = well-formed PNG. It is NOT a statement about what the picture shows;
    # read pixel_std / distinct_colors and then LOOK at the saved file.
    rec("image_generation", r.status_code == 200 and st.get("is_png"),
        " ".join(f"{k}={v}" for k, v in st.items()) + (f" saved={path}" if path else ""),
        resp_evidence(r))
    # ADVISORY, never a verdict and never an exit-code gate: the one silent failure
    # this project has actually produced (a featureless smudge that scored PASS)
    # collapses the palette. Measured on owner-adjudicated samples —
    #   smudge:            distinct_colors_5bit = 185,  pixel_std = 29.8
    #   three good images: distinct_colors_5bit = 1855-2743, pixel_std = 45.8-55.6
    # so the threshold sits with wide margin on both sides. n is small (1 bad, 3
    # good): treat a trip as "go look", not as "it is broken", and treat silence as
    # no evidence either way.
    _dc = st.get("distinct_colors_5bit")
    if isinstance(_dc, int) and _dc < 800:
        print(f"         NOTE: distinct_colors_5bit={_dc} is far below the range seen in "
              f"known-good output (1855-2743); a collapsed palette is what the "
              f"'white smudge' failure looked like. Inspect {path or 'the image'}.",
              flush=True)
except Exception as e:
    rec("image_generation", False, str(e))

# 3. image understanding (feed the generated image back)
try:
    assert img_b64
    r = requests.post(BASE + "/v1/chat/completions", json={"model": "longcat-next", "max_tokens": 40,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What object is shown?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}]}]}, timeout=300)
    t = r.json()["choices"][0]["message"]["content"]
    rec("image_understanding", r.status_code == 200 and len(t) > 0, repr(t[:160]), resp_evidence(r))
except Exception as e:
    rec("image_understanding", False, str(e))

# 4. audio generation
try:
    r = requests.post(BASE + "/v1/audio/speech",
        json={"input": "Self test, all systems nominal.", "voice": "en"}, timeout=900)
    ok = r.status_code == 200 and r.headers.get("content-type") == "audio/wav" and len(r.content) > 1000
    st = audio_stats(r.content) if ok else {}
    path = save(r.content, "audio.wav") if ok else ""
    # PASS = well-formed WAV. Onset garble and trailing artifacts both pass this;
    # check `seconds` against the usual range for the sentence, then LISTEN.
    rec("audio_generation", ok,
        " ".join(f"{k}={v}" for k, v in st.items()) + (f" saved={path}" if path else ""),
        resp_evidence(r))
except Exception as e:
    rec("audio_generation", False, str(e))

# 5. audio understanding (bundled reference clip)
try:
    ab = base64.b64encode(open(VOICE, "rb").read()).decode()
    r = requests.post(BASE + "/v1/chat/completions", json={"model": "longcat-next", "max_tokens": 60,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe this audio."},
            {"type": "input_audio", "input_audio": {"data": ab, "format": "wav"}}]}]}, timeout=300)
    t = r.json()["choices"][0]["message"]["content"]
    rec("audio_understanding", r.status_code == 200 and len(t) > 0, repr(t[:160]), resp_evidence(r))
except Exception as e:
    rec("audio_understanding", False, str(e))

# 6. video understanding (build a tiny clip from the generated image)
try:
    import cv2, numpy as np
    assert img_b64
    arr = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
    arr = cv2.resize(arr, (512, 512))
    vp = OUT + "/_selftest_video.mp4"
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 5, (512, 512))
    for _ in range(10): vw.write(arr)
    vw.release()
    vb = base64.b64encode(open(vp, "rb").read()).decode()
    r = requests.post(BASE + "/v1/chat/completions", json={"model": "longcat-next", "max_tokens": 40,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this video?"},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + vb}}]}]}, timeout=300)
    t = r.json()["choices"][0]["message"]["content"]
    rec("video_understanding", r.status_code == 200 and len(t) > 0, repr(t[:160]), resp_evidence(r))
    os.remove(vp)
except Exception as e:
    rec("video_understanding", False, str(e))

# 7. tool calling
try:
    tools = [{"type": "function", "function": {"name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r = requests.post(BASE + "/v1/chat/completions", json={"model": "longcat-next",
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "tools": tools, "tool_choice": "auto", "max_tokens": 150, "temperature": 0}, timeout=300)
    tc = r.json()["choices"][0]["message"].get("tool_calls")
    ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "Tokyo" in tc[0]["function"]["arguments"]
    # On failure the gateway leaves the model's RAW emission in `content` (it only
    # rewrites content when parsing succeeded), so resp_evidence recovers exactly
    # what the model said — which decides whether this is an unhandled dialect (fix
    # parse_tool_calls) or no call attempted at all (a parser shim cannot help).
    rec("tool_calling", ok, json.dumps(tc) if tc else "no tool_calls parsed", resp_evidence(r))
except Exception as e:
    rec("tool_calling", False, str(e))

n_pass = sum(1 for _, ok, _ in results if ok)
print(f"\n=== {n_pass}/{len(results)} modalities passed ===")
if KEEP:
    print(f"Generated artifacts saved under {OUT} (prefix selftest_{RUN}_).")
    print("PASS on the two generation checks means well-formed output, NOT correct "
          "output — look at the image and listen to the audio before calling a "
          "generation-path change good.")
sys.exit(0 if n_pass == len(results) else 1)
