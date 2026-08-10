#!/usr/bin/env python3
"""Aggregate-throughput bench across concurrency levels.

Batch-size-1 decode is the worst case for this box: each token re-reads the
active expert weights from unified memory at ~270 GB/s and the GPU is mostly
idle waiting. Batching amortises that read across concurrent requests, so
aggregate throughput should climb well past the single-stream number — this
measures how far, and where it stops.

Every request carries a unique nonce: identical prompts would share radix-cache
prefixes and inflate the result.

Note the CUDA-graph interaction — graphs are captured for a fixed set of batch
sizes (LCN_CUDAGRAPH_BS, default 8). Concurrency above that falls back to eager
decode, which may show as a knee in the curve.

  docker exec <container> python3 /workspace/scripts/bench_concurrency.py
"""
import argparse, json, os, statistics, threading, time, urllib.request

PROMPT = ("Write a thorough explanation of how %s works, with concrete examples "
          "and a discussion of trade-offs. Session %s.")
TOPICS = ["a write-ahead log", "consistent hashing", "a bloom filter",
          "copy-on-write snapshots", "vector clocks", "rate limiting",
          "a skip list", "leader election"]


def one_request(url, key, idx, nonce, max_tokens, timeout, results, lock):
    body = json.dumps({
        "model": "longcat-next",
        "messages": [{"role": "user",
                      "content": PROMPT % (TOPICS[idx % len(TOPICS)], f"{nonce}-{idx}")}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        elapsed = time.perf_counter() - t0
        n = payload.get("usage", {}).get("completion_tokens") or 0
        with lock:
            results.append((n, elapsed))
    except Exception as e:
        with lock:
            results.append((0, time.perf_counter() - t0, str(e)[:60]))


def run_level(url, key, n_conc, max_tokens, timeout, nonce):
    results, lock, threads = [], threading.Lock(), []
    t0 = time.perf_counter()
    for i in range(n_conc):
        t = threading.Thread(target=one_request,
                             args=(url, key, i, nonce, max_tokens, timeout, results, lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if len(r) == 2 and r[0] > 0]
    failed = len(results) - len(ok)
    total_tokens = sum(r[0] for r in ok)
    per_req = [r[0] / r[1] for r in ok] if ok else [0]
    return {
        "conc": n_conc,
        "wall": wall,
        "aggregate": total_tokens / wall if wall else 0,
        "per_req_median": statistics.median(per_req),
        "ok": len(ok),
        "failed": failed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("LCN_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--key", default=os.environ.get("LCN_API_KEY", ""))
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    tag = f"  [{args.label}]" if args.label else ""
    print(f"bench_concurrency: levels {levels}, {args.max_tokens} tok/request, "
          f"temp 0, ignore_eos, unique prompts{tag}")

    run_level(args.url, args.key, 1, 32, args.timeout, "warm")
    print("warmup done")
    print(f"{'conc':>5} {'aggregate tok/s':>16} {'per-req tok/s':>14} "
          f"{'wall s':>8} {'ok':>4} {'fail':>5}")
    base = None
    for n in levels:
        r = run_level(args.url, args.key, n, args.max_tokens, args.timeout, f"n{n}-{time.time_ns()}")
        if base is None:
            base = r["aggregate"]
        scale = r["aggregate"] / base if base else 0
        print(f"{r['conc']:>5} {r['aggregate']:>16.1f} {r['per_req_median']:>14.2f} "
              f"{r['wall']:>8.1f} {r['ok']:>4} {r['failed']:>5}   {scale:.2f}x vs conc=1")


if __name__ == "__main__":
    main()
