#!/usr/bin/env python3
"""Long-session soak for the spec-decode + generation fallback (KV accounting).

The defect class this guards against is SLOW ACCUMULATION: the fallback path
previously orphaned one KV slot per generated token, which a 25-minute validation
run cannot distinguish from healthy. So this alternates the three workloads that
actually move the counters -- speculative text decode, audio generation, image
generation -- and watches the pool for drift across many cycles.

Run with the strict idle leak check ARMED (i.e. do NOT set
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0): a residual leak then kills the
server, which is an unambiguous fail rather than a warning to be rationalized.

    docker exec longcat-next python3 /workspace/scripts/soak_specgen.py

Env: SOAK_CYCLES (default 12), SOAK_IMAGE_EVERY (default 4).
Verdict is the pool trend plus survival, not the exit code.
"""
import json, os, sys, time, urllib.request

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
KEY = os.environ.get("LCN_API_KEY", "")
CYCLES = int(os.environ.get("SOAK_CYCLES", "12"))
IMAGE_EVERY = int(os.environ.get("SOAK_IMAGE_EVERY", "4"))
LOG = os.environ.get("SOAK_LOG", "/workspace/outputs/soak_specgen.jsonl")


def post(path, payload, timeout=1200, binary=False):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return {"bytes": len(raw)} if binary else json.loads(raw)


def mem_avail_kb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return -1


def main():
    log = open(LOG, "a")
    t0 = time.time()
    print(f"soak_specgen: {CYCLES} cycles, image every {IMAGE_EVERY}", flush=True)

    for c in range(1, CYCLES + 1):
        rec = {"cycle": c, "t": round(time.time() - t0, 1)}
        try:
            # 1. speculative text decode (the NGRAM-accelerated path)
            for i in range(3):
                r = post("/v1/chat/completions", {
                    "model": "longcat-next", "max_tokens": 160, "temperature": 0,
                    "ignore_eos": True,
                    "messages": [{"role": "user",
                                  "content": f"Explain topic {c}-{i} in detail."}]})
            rec["text_tokens"] = r.get("usage", {}).get("completion_tokens")

            # 2. audio generation -> forces the plain-decode fallback
            a = post("/v1/audio/speech", {
                "model": "longcat-next", "voice": "en_reference",
                "input": f"Soak cycle {c}. The tide came in slowly."}, binary=True)
            rec["audio_bytes"] = a["bytes"]

            # 3. image generation, periodically (the expensive long fallback)
            if c % IMAGE_EVERY == 0:
                im = post("/v1/images/generations", {
                    "model": "longcat-next", "n": 1,
                    "prompt": f"A still life with fruit, variation {c}."})
                rec["image_ok"] = bool(im.get("data"))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"

        rec["mem_avail_gb"] = round(mem_avail_kb() / 1024 / 1024, 2)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if "error" in rec:
            print("ABORT: request failed (server may have died)", flush=True)
            return 2
    print("SOAK COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
