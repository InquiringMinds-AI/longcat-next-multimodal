#!/usr/bin/env python3
"""Reproduce selftest's intermittent tool_calling failure AND capture the raw output.

selftest reports only "no tool_calls" and discards the model's text, which is the
one thing needed to decide whether this is shimmable. The gateway leaves unparsed
output in `content` (parse_tool_calls returns (text, []) and the content field is
untouched), so the raw emission IS recoverable -- this replays selftest's exact
modality sequence, then issues selftest's exact tool-call request and dumps
everything on failure.

Two outcomes, very different consequences:
  * model emitted a tool call in an unhandled dialect  -> fix in parse_tool_calls
  * model emitted ordinary prose / no call attempt     -> a parser shim cannot help

    docker exec longcat-next python3 /workspace/scripts/probe_toolcall_sequence.py
Env: SEQ_ROUNDS (default 3).
"""
import base64, json, os, sys, urllib.request

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
KEY = os.environ.get("LCN_API_KEY", "")
ROUNDS = int(os.environ.get("SEQ_ROUNDS", "3"))
VOICE = "/workspace/scripts/voices/en_reference.wav"

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def post(path, payload, timeout=1200, binary=False):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return {"bytes": len(raw)} if binary else json.loads(raw)


def chat(messages, **kw):
    body = {"model": "longcat-next", "max_tokens": 40, "messages": messages}
    body.update(kw)
    return post("/v1/chat/completions", body)


def run_sequence(i):
    """selftest's FULL order, up to but not including tool_calling.

    Fidelity to the sequence is the whole point — the trigger is cumulative state,
    so any omitted step turns a clean run into a non-result.
    """
    chat([{"role": "user", "content": "Say ready."}])
    img = post("/v1/images/generations", {
        "model": "longcat-next", "response_format": "b64_json",
        "prompt": f"A photograph of a red apple on a wooden table. Take {i}."})
    b64 = img["data"][0]["b64_json"]
    chat([{"role": "user", "content": [
        {"type": "text", "text": "What object is shown?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}])
    post("/v1/audio/speech", {"model": "longcat-next", "voice": "en",
                              "input": "Self test, all systems nominal."}, binary=True)
    ab = base64.b64encode(open(VOICE, "rb").read()).decode()
    chat([{"role": "user", "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "input_audio", "input_audio": {"data": ab, "format": "wav"}}]}], max_tokens=60)
    # Video understanding — selftest step 6, the step IMMEDIATELY BEFORE the failing
    # tool call and the only predecessor never tested in isolation. Omitting it left
    # this probe unable to eliminate the most likely trigger, so a clean run here
    # previously proved nothing.
    try:
        import cv2, numpy as np
        arr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        arr = cv2.resize(arr, (512, 512))
        vp = "/tmp/_probe_seq_video.mp4"
        vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 5, (512, 512))
        for _ in range(10):
            vw.write(arr)
        vw.release()
        vb = base64.b64encode(open(vp, "rb").read()).decode()
        chat([{"role": "user", "content": [
            {"type": "text", "text": "What is in this video?"},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + vb}}]}])
        os.remove(vp)
    except Exception as e:
        # Loud, not silent: a skipped video step makes a clean result meaningless.
        print(f"  WARNING: video step FAILED to run ({type(e).__name__}: {str(e)[:90]}) "
              f"— a PASS this round does not exonerate video understanding", flush=True)
    return b64


def main():
    fails = 0
    for i in range(1, ROUNDS + 1):
        print(f"=== round {i}: replaying selftest sequence ===", flush=True)
        try:
            run_sequence(i)
        except Exception as e:
            print(f"  sequence error: {e}", flush=True)
        r = post("/v1/chat/completions", {
            "model": "longcat-next", "max_tokens": 150, "temperature": 0,
            "tools": TOOLS, "tool_choice": "auto",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
        msg = r["choices"][0]["message"]
        tc = msg.get("tool_calls")
        if tc and tc[0]["function"]["name"] == "get_weather":
            print("  tool_calling: PASS", flush=True)
            continue
        fails += 1
        print("  tool_calling: FAIL — RAW MODEL OUTPUT FOLLOWS", flush=True)
        print("  finish_reason: %r" % r["choices"][0].get("finish_reason"), flush=True)
        print("  content: %r" % (msg.get("content"),), flush=True)
        print("  full message: %s" % json.dumps(msg)[:1200], flush=True)
    print(f"\nRESULT: {fails}/{ROUNDS} rounds failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
