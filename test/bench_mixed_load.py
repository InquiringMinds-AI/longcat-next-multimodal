#!/usr/bin/env python3
"""Mixed-modality concurrency: what versatility actually costs.

The point of holding this model resident is that ONE process serves text,
vision, audio understanding, image generation and voice cloning. Single-modality
throughput says nothing about whether those coexist — and image generation
lazily allocates ~25GB outside --mem-fraction-static and runs for minutes, so
the interesting question is what happens to everything else while it does.

Measures text decode throughput in three phases: alone, during a concurrent
image generation, and during concurrent audio generation + understanding. A
graceful system shows text degrading proportionally and recovering; a fragile
one shows collapse, timeouts, or a dead scheduler.

  docker exec <container> python3 /workspace/scripts/bench_mixed_load.py
"""
import argparse, json, os, statistics, threading, time, urllib.request

def post(url, path, key, payload, timeout, binary=False):
    """POST and decode. /v1/audio/speech returns a raw WAV, not JSON — parsing
    it as JSON raises a UTF-8 decode error that looks like a server failure."""
    req = urllib.request.Request(
        f"{url}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if binary:
        return {"bytes": len(raw)}
    return json.loads(raw)


def text_stream(url, key, nonce, idx, max_tokens, timeout, out, lock, stop):
    """Repeated short text decodes; records tok/s per completion until stop."""
    i = 0
    while not stop.is_set():
        i += 1
        try:
            t0 = time.perf_counter()
            p = post(url, "/v1/chat/completions", key, {
                "model": "longcat-next",
                "messages": [{"role": "user",
                              "content": f"Explain idea {nonce}-{idx}-{i} in detail."}],
                "max_tokens": max_tokens, "temperature": 0,
                "ignore_eos": True, "stream": False}, timeout)
            dt = time.perf_counter() - t0
            n = p.get("usage", {}).get("completion_tokens") or 0
            with lock:
                out.append(n / dt if dt else 0)
        except Exception as e:
            with lock:
                out.append(-1.0)
            return


def measure(url, key, seconds, conc, max_tokens, timeout, label, background=None):
    out, lock, stop = [], threading.Lock(), threading.Event()
    threads = [threading.Thread(target=text_stream,
                                args=(url, key, f"{label}{time.time_ns()}", i,
                                      max_tokens, timeout, out, lock, stop))
               for i in range(conc)]
    bg = None
    if background:
        bg = threading.Thread(target=background)
        bg.start()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=timeout)
    if bg:
        bg.join(timeout=timeout)
    good = [x for x in out if x > 0]
    errs = sum(1 for x in out if x < 0)
    med = statistics.median(good) if good else 0.0
    print(f"  {label:<34} text {med:6.2f} tok/s per stream  "
          f"({len(good)} completions, {errs} errors)")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("LCN_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--key", default=os.environ.get("LCN_API_KEY", ""))
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    print(f"bench_mixed_load: {args.conc} text streams, {args.seconds}s phases")
    state = {}

    def gen_image():
        t0 = time.perf_counter()
        try:
            post(args.url, "/v1/images/generations", args.key, {
                "model": "longcat-next",
                "prompt": "a lighthouse on a rocky shore at dusk", "n": 1},
                args.timeout)
            state["image"] = f"ok in {time.perf_counter()-t0:.0f}s"
        except Exception as e:
            state["image"] = f"FAILED after {time.perf_counter()-t0:.0f}s: {str(e)[:60]}"

    def gen_audio():
        t0 = time.perf_counter()
        try:
            r = post(args.url, "/v1/audio/speech", args.key, {
                "model": "longcat-next",
                "input": "The tide came in slowly, and the boats began to lift.",
                "voice": "en_reference"}, args.timeout, binary=True)
            state["audio"] = f"ok in {time.perf_counter()-t0:.0f}s ({r['bytes']} bytes wav)"
        except Exception as e:
            state["audio"] = f"FAILED after {time.perf_counter()-t0:.0f}s: {str(e)[:60]}"

    base = measure(args.url, args.key, args.seconds, args.conc,
                   args.max_tokens, args.timeout, "text alone (baseline)")
    with_img = measure(args.url, args.key, args.seconds, args.conc,
                       args.max_tokens, args.timeout, "text + image generation",
                       background=gen_image)
    with_aud = measure(args.url, args.key, args.seconds, args.conc,
                       args.max_tokens, args.timeout, "text + audio generation",
                       background=gen_audio)
    after = measure(args.url, args.key, args.seconds, args.conc,
                    args.max_tokens, args.timeout, "text alone (recovery)")

    print()
    print(f"  image gen: {state.get('image', 'not run')}")
    print(f"  audio gen: {state.get('audio', 'not run')}")
    if base:
        print(f"  text retained during image gen: {with_img/base*100:.0f}% of baseline")
        print(f"  text retained during audio gen: {with_aud/base*100:.0f}% of baseline")
        print(f"  recovery after both:            {after/base*100:.0f}% of baseline")


if __name__ == "__main__":
    main()
