#!/usr/bin/env python3
"""Agent-workload decode bench: verbatim reproduction.

NGRAM speculative decode wins on text the model has already seen in its context
— which is the dominant shape of agent work: re-emitting a file with one edit,
quoting a diff back, restating a tool result. The essay probe in bench_decode.py
is the opposite shape (novel prose) and shows the ~7-9% overhead instead.

Run it against a server with LCN_NGRAM=1 and one without, same script, to get
the real agent-mode number.

  docker exec <container> python3 /workspace/scripts/bench_agent.py
"""
import argparse, json, os, statistics, time, urllib.request

# A chunk of plausible source, given in the prompt and then asked for verbatim.
SOURCE = '''def resolve_config(name, overrides=None):
    """Resolve a config by name, applying overrides in order."""
    base = dict(DEFAULTS.get(name, {}))
    if overrides:
        for key, value in overrides.items():
            if value is None:
                base.pop(key, None)
            else:
                base[key] = value
    if "timeout" in base and base["timeout"] < 0:
        raise ValueError(f"negative timeout for {name}: {base['timeout']}")
    return base


def merge_configs(primary, secondary):
    """Merge two resolved configs; primary wins on conflict."""
    merged = dict(secondary)
    merged.update(primary)
    return merged
'''


def one_run(url, key, max_tokens, timeout):
    prompt = (
        "Here is a Python module:\n\n```python\n" + SOURCE + "```\n\n"
        "Reproduce the module above exactly, character for character, inside a "
        "```python code fence. Do not add commentary."
    )
    body = json.dumps({
        "model": "longcat-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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
    out = usage.get("completion_tokens") or 0
    text = payload["choices"][0]["message"].get("content") or ""
    # how much of the source actually came back — a speedup on garbage is not a win
    fidelity = sum(1 for line in SOURCE.splitlines() if line and line in text)
    total = sum(1 for line in SOURCE.splitlines() if line)
    return out, elapsed, out / elapsed, fidelity / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("LCN_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--key", default=os.environ.get("LCN_API_KEY", ""))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    tag = f"  [{args.label}]" if args.label else ""
    print(f"bench_agent: verbatim reproduction, {args.runs} runs, temp 0{tag}")
    one_run(args.url, args.key, 32, args.timeout)
    print("warmup done")

    rates, fids = [], []
    for i in range(args.runs):
        n, elapsed, rate, fid = one_run(args.url, args.key, args.max_tokens, args.timeout)
        rates.append(rate)
        fids.append(fid)
        print(f"  run {i+1}: {n} tok in {elapsed:.2f}s = {rate:.2f} tok/s "
              f"(source lines reproduced: {fid*100:.0f}%)")

    med = statistics.median(rates)
    print(f"AGENT median: {med:.2f} tok/s  (fidelity {statistics.median(fids)*100:.0f}%){tag}")


if __name__ == "__main__":
    main()
