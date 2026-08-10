#!/usr/bin/env python3
"""Does a preceding generation break the next tool call?

selftest scores tool_calling FAIL intermittently (2 of 3 runs under NGRAM), and in
selftest that check runs immediately after image gen -> audio gen -> video
understanding. Standalone tool calls pass 3/3. So the suspect is state left behind
by a generation, not tool calling itself.

Alternates: cold tool call, then generation, then tool call again -- N times, and
reports the failure rate in each position. Any asymmetry localizes the trigger.

    docker exec longcat-next python3 /workspace/scripts/probe_toolcall.py
Env: PROBE_ROUNDS (default 5), PROBE_GEN (audio|image, default audio).
"""
import json, os, sys, urllib.request

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
KEY = os.environ.get("LCN_API_KEY", "")
ROUNDS = int(os.environ.get("PROBE_ROUNDS", "5"))
GEN = os.environ.get("PROBE_GEN", "audio")

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


def tool_call(tag):
    """Exactly selftest's check: temperature 0, same tools, same prompt."""
    try:
        r = post("/v1/chat/completions", {
            "model": "longcat-next", "max_tokens": 150, "temperature": 0,
            "tools": TOOLS, "tool_choice": "auto",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
        m = r["choices"][0]["message"]
        tc = m.get("tool_calls")
        ok = bool(tc) and tc[0]["function"]["name"] == "get_weather"
        detail = "" if ok else f" content={(m.get('content') or '')[:70]!r}"
        print(f"  {tag}: {'PASS' if ok else 'FAIL'}{detail}", flush=True)
        return ok
    except Exception as e:
        print(f"  {tag}: ERROR {e}", flush=True)
        return False


def generate(i):
    if GEN == "understand":
        # The ACTUAL predecessor of tool_calling in selftest is a multimodal
        # UNDERSTANDING request (video), not a generation. Cheap to test.
        import base64
        png = base64.b64encode(open("/workspace/scripts/voices/probe.png", "rb").read()).decode() \
            if os.path.exists("/workspace/scripts/voices/probe.png") else None
        if png is None:
            import glob
            cands = sorted(glob.glob("/workspace/outputs/longcat_img_*_refined.png"))
            png = base64.b64encode(open(cands[-1], "rb").read()).decode()
        post("/v1/chat/completions", {"model": "longcat-next", "max_tokens": 40,
             "messages": [{"role": "user", "content": [
                 {"type": "text", "text": "What object is shown?"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}]}]})
    elif GEN == "image":
        post("/v1/images/generations",
             {"model": "longcat-next", "n": 1, "prompt": f"A red apple, take {i}."})
    else:
        post("/v1/audio/speech", {"model": "longcat-next", "voice": "en_reference",
                                  "input": f"Probe round {i}."}, binary=True)


def main():
    print(f"probe_toolcall: {ROUNDS} rounds, generation={GEN}", flush=True)
    cold, after = 0, 0
    for i in range(1, ROUNDS + 1):
        print(f"round {i}", flush=True)
        cold += tool_call("cold (no preceding generation)")
        generate(i)
        after += tool_call("after generation          ")
    print(f"\nRESULT: cold {cold}/{ROUNDS} passed, after-generation {after}/{ROUNDS} passed",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
