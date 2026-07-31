#!/usr/bin/env python3
"""Degenerate-behavior probe: looping, inability to complete, runaway generation.

Checks BEHAVIOR, not just output presence:
  - short factual answers must stop EARLY (finish_reason=stop, well under the cap)
  - long generations must not tail-loop (repeated-suffix n-gram detection)
  - multi-turn conversation must stay coherent and terminate per turn (the n-gram
    embedding's eos-boundary handling is exercised exactly here)
Run INSIDE the container:  docker exec longcat-next python3 /workspace/scripts/degeneracy_probe.py
"""
import os, sys, requests

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
H = {"Authorization": "Bearer " + os.environ.get("LCN_API_KEY", "")}
results = []
def rec(name, ok, detail=""):
    results.append((name, ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

def chat(messages, max_tokens, temperature=0.7):
    r = requests.post(BASE + "/v1/chat/completions", headers=H, json={
        "model": "longcat-next", "temperature": temperature,
        "messages": messages, "max_tokens": max_tokens}, timeout=600)
    j = r.json()
    ch = j["choices"][0]
    return (ch["message"].get("content") or ""), ch.get("finish_reason"), j.get("usage", {}).get("completion_tokens", 0)

def tail_loop_score(text, tail_words=120, n=8):
    """Fraction of duplicated n-grams in the tail — near 1.0 = the model is looping."""
    w = text.split()[-tail_words:]
    if len(w) < n * 2:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)

# 1. short factual answer must terminate early on its own
txt, fr, ct = chat([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], 200, 0)
rec("early_stop", fr == "stop" and ct < 60, f"finish={fr} tokens={ct} {txt[:60]!r}")

# 2/3. long generations: must finish or cap cleanly, and must not tail-loop
for name, prompt in [("essay", "Write a 400-word essay on the history of navigation."),
                     ("code", "Write a python CSV parsing module with three functions and docstrings.")]:
    txt, fr, ct = chat([{"role": "user", "content": prompt}], 700)
    loop = tail_loop_score(txt)
    rec(f"long_{name}", fr in ("stop", "length") and loop < 0.30 and ct > 100,
        f"finish={fr} tokens={ct} tail_loop={loop:.2f}")

# 4. multi-turn: eos boundaries in history (the n-gram eos change's exact surface)
hist = []
ok_turns, details = True, []
for q in ["Name three planets.", "Which of those is largest?", "How many moons does it have? One sentence."]:
    hist.append({"role": "user", "content": q})
    txt, fr, ct = chat(hist, 150)
    hist.append({"role": "assistant", "content": txt})
    loop = tail_loop_score(txt, tail_words=60, n=5)
    if fr != "stop" or loop >= 0.30 or not txt.strip():
        ok_turns = False
    details.append(f"{fr}/{ct}t/loop{loop:.2f}")
rec("multiturn", ok_turns, " | ".join(details))
ans = hist[-1]["content"].lower()
rec("multiturn_coherent", "jupiter" in (hist[3]["content"].lower() + ans) or "moon" in ans,
    hist[-1]["content"][:80])

# 5. repetition-bait: a prompt that invites looping
txt, fr, ct = chat([{"role": "user", "content": "Repeat the word 'data' exactly 5 times, then stop."}], 300, 0)
rec("repetition_bait", fr == "stop" and ct < 120, f"finish={fr} tokens={ct} {txt[:60]!r}")

n_pass = sum(1 for _, ok in results if ok)
print(f"\n=== {n_pass}/{len(results)} degeneracy checks passed ===")
sys.exit(0 if n_pass == len(results) else 1)
