#!/usr/bin/env python3
"""Kernel-timeline gap analysis for the generation-loop GPU-idle budget.

Aggregate views failed six times on this question (op totals overlap GPU work;
removing 22ms of host ops bought 1.4ms). This reads the TIMELINE instead:

  1. Merge all GPU-track events (kernels, memcpy, memset) into busy intervals.
  2. Inside each `step[DECODE...]` annotation span, list idle gaps >= MIN_GAP.
  3. For each gap: the flanking kernels and the host-side events (cpu_op +
     cuda_runtime) that overlap the gap — what the CPU was doing while the GPU
     waited, at that exact moment.
  4. Cluster gaps by (prev-kernel, next-kernel) signature; report count, mean,
     and total ms/step so the structural gap (same place every step) separates
     from one-off stalls.

Usage: analyze_gap_trace.py <trace.json[.gz]> [min_gap_ms]
"""
import gzip
import json
import sys
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    path = sys.argv[1]
    min_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5  # ms
    data = load(path)
    events = data["traceEvents"] if isinstance(data, dict) else data

    gpu_busy, steps, host = [], [], []
    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        ts, dur = e.get("ts", 0), e.get("dur", 0)
        if cat in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"):
            if cat != "gpu_user_annotation":
                gpu_busy.append((ts, ts + dur))
                kernels.append((ts, ts + dur, e.get("name", "?")))
        elif cat == "user_annotation" and e.get("name", "").startswith("step"):
            steps.append((ts, ts + dur, e.get("name")))
        elif cat in ("cpu_op", "cuda_runtime", "user_annotation"):
            host.append((ts, ts + dur, e.get("name", "?"), cat))

    gpu_busy.sort()
    merged = []
    for s, e in gpu_busy:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    kernels.sort()
    host.sort()
    steps.sort()
    print(f"{len(steps)} step spans, {len(kernels)} gpu events, "
          f"{len(host)} host events; min_gap {min_gap}ms")

    import bisect
    kstarts = [k[0] for k in kernels]
    sig = defaultdict(lambda: [0, 0.0, defaultdict(float)])  # count, total_ms, host_ms
    step_tot = []
    for (ss, se, _name) in steps:
        idx = bisect.bisect_left(merged, [ss, ss]) - 1
        idx = max(idx, 0)
        cur = ss
        tot = 0.0
        while idx < len(merged) and merged[idx][0] < se:
            bs, be = merged[idx]
            if bs > cur:
                gs, ge = max(cur, ss), min(bs, se)
                gap_ms = (ge - gs) / 1000.0
                if gap_ms >= min_gap:
                    tot += gap_ms
                    ki = bisect.bisect_left(kstarts, gs) - 1
                    prev = kernels[ki][2][:60] if 0 <= ki < len(kernels) else "<start>"
                    kj = bisect.bisect_left(kstarts, ge)
                    nxt = kernels[kj][2][:60] if kj < len(kernels) else "<end>"
                    entry = sig[(prev, nxt)]
                    entry[0] += 1
                    entry[1] += gap_ms
                    hi = bisect.bisect_left(host, (gs - 200000,)) if host else 0
                    for (hs, he, hname, hcat) in host[hi:]:
                        if hs >= ge:
                            break
                        ov = (min(he, ge) - max(hs, gs)) / 1000.0
                        if ov > 0:
                            entry[2][f"{hcat}:{hname[:70]}"] += ov
            cur = max(cur, be)
            idx += 1
        step_tot.append(tot)
    if step_tot:
        st = sorted(step_tot)
        print(f"\nidle(>={min_gap}ms gaps) per step: mean "
              f"{sum(step_tot)/len(step_tot):.2f}ms  median {st[len(st)//2]:.2f}ms  "
              f"max {st[-1]:.2f}ms over {len(step_tot)} steps")

    print("\n=== gap signatures (top by total ms) ===")
    nsteps = max(len(steps), 1)
    for (prev, nxt), (cnt, tot, hops) in sorted(
            sig.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"\n[{tot/nsteps:6.2f} ms/step  n={cnt}  mean {tot/cnt:5.2f}ms]")
        print(f"  after : {prev}")
        print(f"  before: {nxt}")
        for hname, ms in sorted(hops.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    host {ms/nsteps:6.2f} ms/step  {hname}")


if __name__ == "__main__":
    main()
