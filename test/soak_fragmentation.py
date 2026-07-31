#!/usr/bin/env python3
"""Fragmentation soak test for LongCat-Next on unified-memory hosts (DGX Spark GB10).

Reproduces the reported failure: long multi-turn conversations with input images
leak host memory via PyTorch CUDA allocator fragmentation in the vision encoder
(fragmented segments are never returned to the system; on unified memory this
consumes system RAM until the node freezes / earlyoom kills the server).

Method: many chat turns, each carrying a fresh noise image at a CYCLING
resolution/aspect ratio (varied anyres tiling -> varied allocation sizes ->
maximum fragmentation pressure), with a rolling conversation history. Host
MemAvailable is sampled every turn and logged as JSONL. A safety floor aborts
the run long before the host is endangered.

Run INSIDE the container (deps: requests, cv2, numpy — same as selftest.py):

    docker exec longcat-next python3 /workspace/outputs/soak_fragmentation.py

Env knobs: SOAK_TURNS (default 120), SOAK_MIN_AVAIL_GB (default 6),
SOAK_LOG (default /workspace/outputs/soak_log.jsonl), SOAK_HISTORY (default 6).
The verdict is in the log trend, not the exit code: flat MemAvailable = healthy,
monotonic decline = fragmentation confirmed. Exit 2 = safety abort (leak severe).
"""
import base64, json, os, sys, time
import cv2, numpy as np
import requests

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
HEADERS = ({"Authorization": "Bearer " + os.environ["LCN_API_KEY"]}
           if os.environ.get("LCN_API_KEY") else {})
TURNS = int(os.environ.get("SOAK_TURNS", "120"))
MIN_AVAIL_KB = float(os.environ.get("SOAK_MIN_AVAIL_GB", "6")) * 1024 * 1024
LOG = os.environ.get("SOAK_LOG", "/workspace/outputs/soak_log.jsonl")
HISTORY = int(os.environ.get("SOAK_HISTORY", "6"))  # user/assistant turn PAIRS kept
# 1 = history retains the IMAGES too (each turn re-encodes the whole rolling window in one
# request — the reported real-world failure shape). 0 = text-only history: exactly one
# encoder invocation per turn, the clean per-turn leak signal.
KEEP_IMAGES = os.environ.get("SOAK_KEEP_IMAGES", "0") == "1"

# Varied geometries: different tile counts + aspect ratios each turn.
RESOLUTIONS = [(512, 512), (1024, 768), (768, 1344), (1536, 640), (896, 896),
               (640, 480), (1280, 1280), (448, 1120)]

def mem_available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1])
    return -1

def noise_image_b64(w, h, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()

def main():
    log = open(LOG, "a")
    start_avail = mem_available_kb()
    print(f"soak start: {TURNS} turns, MemAvailable={start_avail/1048576:.1f} GiB, "
          f"floor={MIN_AVAIL_KB/1048576:.1f} GiB", flush=True)
    history = []
    for turn in range(1, TURNS + 1):
        w, h = RESOLUTIONS[turn % len(RESOLUTIONS)]
        img = noise_image_b64(w, h, seed=turn)
        user_msg = {"role": "user", "content": [
            {"type": "text", "text": f"Turn {turn}: briefly describe the texture of this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}
        messages = history + [user_msg]
        t0 = time.time()
        try:
            r = requests.post(BASE + "/v1/chat/completions", headers=HEADERS, json={
                "model": "longcat-next", "messages": messages,
                "max_tokens": 48, "temperature": 0.7}, timeout=600)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            err = None
        except Exception as e:
            reply, err = None, str(e)
        dt = time.time() - t0
        avail = mem_available_kb()
        rec = {"turn": turn, "ts": time.time(), "res": f"{w}x{h}",
               "mem_available_kb": avail, "latency_s": round(dt, 2),
               "delta_from_start_mb": round((avail - start_avail) / 1024), "error": err}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"turn {turn:3d} {w}x{h:>5} avail={avail/1048576:6.1f} GiB "
              f"drift={rec['delta_from_start_mb']:+6d} MB lat={dt:5.1f}s"
              + (f" ERR={err}" if err else ""), flush=True)
        if err is None:
            history.append(user_msg if KEEP_IMAGES else
                           {"role": "user", "content": [user_msg["content"][0]]})
            history.append({"role": "assistant", "content": reply})
            history = history[-2 * HISTORY:]
        if avail < MIN_AVAIL_KB:
            print(f"SAFETY ABORT: MemAvailable {avail/1048576:.1f} GiB below floor — "
                  f"leak reproduced hard, stopping to protect the host", flush=True)
            log.close(); sys.exit(2)
    log.close()
    end_avail = mem_available_kb()
    print(f"soak done: drift {(end_avail - start_avail)/1024:+.0f} MB over {TURNS} turns", flush=True)

if __name__ == "__main__":
    main()
