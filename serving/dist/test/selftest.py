#!/usr/bin/env python3
"""End-to-end self-test for the LongCat-Next server — exercises ALL modalities via the
OpenAI-compatible endpoints and prints PASS/FAIL. Run INSIDE the container:

    docker exec longcat-next python3 /workspace/scripts/selftest.py

Verifies: text, image generation, image understanding, audio generation, audio understanding,
video understanding. Exit code 0 iff every modality passes."""
import base64, json, os, sys, time
import requests

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
OUT = os.environ.get("LCN_OUTPUT_DIR", "/tmp")
VOICE = "/workspace/scripts/voices/en_reference.wav"
results = []
def rec(name, ok, detail=""):
    results.append((name, ok, detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

# 1. text
try:
    r = requests.post(BASE + "/v1/chat/completions", json={"model": "longcat-next",
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 8, "temperature": 0}, timeout=120)
    t = r.json()["choices"][0]["message"]["content"]
    rec("text", r.status_code == 200 and len(t) > 0, repr(t[:40]))
except Exception as e:
    rec("text", False, str(e))

# 2. image generation
img_b64 = None
try:
    r = requests.post(BASE + "/v1/images/generations",
        json={"prompt": "A photograph of a red apple on a wooden table.", "response_format": "b64_json"}, timeout=900)
    img_b64 = r.json()["data"][0]["b64_json"]
    raw = base64.b64decode(img_b64)
    rec("image_generation", r.status_code == 200 and raw[:4].hex() == "89504e47", f"{len(raw)} bytes PNG")
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
    rec("image_understanding", r.status_code == 200 and len(t) > 0, repr(t[:60]))
except Exception as e:
    rec("image_understanding", False, str(e))

# 4. audio generation
try:
    r = requests.post(BASE + "/v1/audio/speech",
        json={"input": "Self test, all systems nominal.", "voice": "en"}, timeout=900)
    ok = r.status_code == 200 and r.headers.get("content-type") == "audio/wav" and len(r.content) > 1000
    rec("audio_generation", ok, f"{len(r.content)} bytes wav")
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
    rec("audio_understanding", r.status_code == 200 and len(t) > 0, repr(t[:60]))
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
    rec("video_understanding", r.status_code == 200 and len(t) > 0, repr(t[:60]))
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
    rec("tool_calling", ok, json.dumps(tc) if tc else "no tool_calls")
except Exception as e:
    rec("tool_calling", False, str(e))

n_pass = sum(1 for _, ok, _ in results if ok)
print(f"\n=== {n_pass}/{len(results)} modalities passed ===")
sys.exit(0 if n_pass == len(results) else 1)
