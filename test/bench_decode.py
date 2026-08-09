#!/usr/bin/env python3
"""Decode-throughput bench for LongCat-Next.

Isolates decode: short prompts, long fixed-length outputs, ignore_eos so every
run emits exactly MAX_TOKENS and the token counts are comparable across builds.
temp=0 so output (and therefore work per token) does not vary between runs.

Reports per-prompt medians and an overall median of medians. Compare two builds
by running this against each with identical arguments — the absolute numbers are
only meaningful relative to a run of THIS script on the same machine.

  docker exec longcat-next python3 /workspace/scripts/bench_decode.py [--runs 3]
"""
import argparse, json, os, statistics, time, urllib.request

PROMPTS = [
    ("essay", "Write a detailed essay about the history of movable type printing."),
    ("technical", "Explain how a modern branch predictor works, in depth."),
    ("narrative", "Tell a long story about a lighthouse keeper who finds a map."),
]


def one_run(url, key, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": "longcat-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    elapsed = time.perf_counter() - t0
    usage = payload.get("usage", {})
    out_tokens = usage.get("completion_tokens")
    if not out_tokens:
        raise RuntimeError(f"no completion_tokens in usage: {usage}")
    return out_tokens, elapsed, out_tokens / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("LCN_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--key", default=os.environ.get("LCN_API_KEY", ""))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    print(f"bench_decode: {args.runs} runs x {len(PROMPTS)} prompts, "
          f"{args.max_tokens} tokens each, temp 0, ignore_eos"
          + (f"  [{args.label}]" if args.label else ""))

    # warmup: first request pays lazy-init costs that belong to neither build
    one_run(args.url, args.key, "Say hello.", 32, args.timeout)
    print("warmup done")

    medians = []
    for name, prompt in PROMPTS:
        rates, counts = [], set()
        for i in range(args.runs):
            n, elapsed, rate = one_run(args.url, args.key, prompt,
                                       args.max_tokens, args.timeout)
            rates.append(rate)
            counts.add(n)
            print(f"  {name} run {i+1}: {n} tok in {elapsed:.2f}s = {rate:.2f} tok/s")
        med = statistics.median(rates)
        medians.append(med)
        spread = (max(rates) - min(rates)) / med * 100
        print(f"  {name}: median {med:.2f} tok/s  (spread {spread:.1f}%, "
              f"token counts {sorted(counts)})")

    overall = statistics.median(medians)
    print(f"OVERALL median-of-medians: {overall:.2f} tok/s"
          + (f"  [{args.label}]" if args.label else ""))


if __name__ == "__main__":
    main()
