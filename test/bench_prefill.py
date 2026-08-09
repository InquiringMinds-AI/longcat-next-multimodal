#!/usr/bin/env python3
"""Prefill-throughput bench for LongCat-Next.

Decode at batch size 1 only exercises the M=1 MoE config. Prefill runs the
large-M shapes (up to chunked_prefill_size), which is where most of the tuning
ladder's cost went — so this is the measurement that says what the big configs
bought.

Each request carries a unique nonce prefix: identical prompts would hit the
radix cache and skip prefill entirely, silently measuring nothing.

  docker exec <container> python3 /workspace/scripts/bench_prefill.py
"""
import argparse, json, os, statistics, time, urllib.request

FILLER = (
    "The manufacture of paper from wood pulp transformed publishing in ways "
    "that contemporaries did not anticipate, altering the economics of print, "
    "the physical durability of books, and the archival practices of libraries. "
)


def make_prompt(target_words, nonce):
    reps = max(1, target_words // len(FILLER.split()))
    return f"[session {nonce}] " + (FILLER * reps) + "\nSummarize the passage in one word."


def one_run(url, key, prompt, timeout):
    body = json.dumps({
        "model": "longcat-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
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
    ptok = usage.get("prompt_tokens")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if not ptok:
        raise RuntimeError(f"no prompt_tokens in usage: {usage}")
    return ptok, cached, elapsed, ptok / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("LCN_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--key", default=os.environ.get("LCN_API_KEY", ""))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--words", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    tag = f"  [{args.label}]" if args.label else ""
    print(f"bench_prefill: {args.runs} runs, ~{args.words} words/prompt, "
          f"max_tokens=1, unique nonce per run{tag}")

    one_run(args.url, args.key, make_prompt(200, "warmup"), args.timeout)
    print("warmup done")

    rates = []
    for i in range(args.runs):
        ptok, cached, elapsed, rate = one_run(
            args.url, args.key, make_prompt(args.words, f"n{i}-{time.time_ns()}"),
            args.timeout)
        rates.append(rate)
        flag = "  <-- CACHE HIT, result invalid" if cached else ""
        print(f"  run {i+1}: {ptok} prompt tok ({cached} cached) in {elapsed:.2f}s "
              f"= {rate:.0f} tok/s{flag}")

    med = statistics.median(rates)
    spread = (max(rates) - min(rates)) / med * 100
    print(f"PREFILL median: {med:.0f} tok/s  (spread {spread:.1f}%){tag}")


if __name__ == "__main__":
    main()
