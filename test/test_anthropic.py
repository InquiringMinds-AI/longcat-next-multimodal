#!/usr/bin/env python3
"""Self-test for the Anthropic Messages route (/v1/messages) — text, streaming, and the
full tool round-trip (call -> tool_result -> final answer). Run INSIDE the container:

    docker exec longcat-next python3 /workspace/scripts/test_anthropic.py

Exit 0 iff every check passes."""
import json, os, sys
import requests

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
HEADERS = {"x-api-key": os.environ.get("LCN_API_KEY", ""), "anthropic-version": "2023-06-01"}
TOOLS = [{"name": "get_weather", "description": "Get the current weather in a city",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}]
results = []
def rec(name, ok, detail=""):
    results.append((name, ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

# 1. plain text
try:
    r = requests.post(BASE + "/v1/messages", headers=HEADERS, json={
        "model": "longcat-next", "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}, timeout=300)
    j = r.json()
    txt = "".join(b.get("text", "") for b in j.get("content", []))
    rec("text", r.status_code == 200 and j.get("type") == "message" and len(txt) > 0,
        f"stop={j.get('stop_reason')} {txt[:40]!r}")
except Exception as e:
    rec("text", False, str(e))

# 2. tool call (turn 1 of the round-trip)
tool_use = None
try:
    r = requests.post(BASE + "/v1/messages", headers=HEADERS, json={
        "model": "longcat-next", "max_tokens": 200, "temperature": 0,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}, timeout=300)
    j = r.json()
    tool_use = next((b for b in j.get("content", []) if b.get("type") == "tool_use"), None)
    ok = (r.status_code == 200 and j.get("stop_reason") == "tool_use" and tool_use
          and tool_use["name"] == "get_weather" and tool_use["input"].get("city"))
    rec("tool_call", bool(ok), json.dumps(tool_use) if tool_use else json.dumps(j)[:200])
except Exception as e:
    rec("tool_call", False, str(e))

# 3. tool result -> final answer (turn 2)
try:
    assert tool_use
    r = requests.post(BASE + "/v1/messages", headers=HEADERS, json={
        "model": "longcat-next", "max_tokens": 100, "temperature": 0,
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {"role": "assistant", "content": [tool_use]},
            {"role": "user", "content": [{"type": "tool_result",
                "tool_use_id": tool_use["id"], "content": "22C and sunny"}]}]}, timeout=300)
    j = r.json()
    txt = "".join(b.get("text", "") for b in j.get("content", []))
    ok = r.status_code == 200 and ("22" in txt or "sunny" in txt.lower())
    rec("tool_roundtrip", ok, txt[:80])
except Exception as e:
    rec("tool_roundtrip", False, str(e))

# 4. streaming SSE shape — INCREMENTAL: a multi-sentence answer must arrive as
# MULTIPLE text deltas (buffered-then-synthesized emitted exactly one)
try:
    r = requests.post(BASE + "/v1/messages", headers=HEADERS, json={
        "model": "longcat-next", "max_tokens": 200, "stream": True,
        "messages": [{"role": "user", "content": "In two or three sentences, what is a lighthouse for?"}]},
        timeout=300, stream=True)
    events = [ln.split(": ", 1)[1] for ln in r.iter_lines(decode_unicode=True)
              if ln and ln.startswith("event: ")]
    need = ["message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop"]
    n_deltas = events.count("content_block_delta")
    ok = all(e in events for e in need) and n_deltas >= 3
    rec("streaming", ok, "%d text deltas; %s" % (n_deltas, "->".join(dict.fromkeys(events))))
except Exception as e:
    rec("streaming", False, str(e))

# 5. count_tokens
try:
    r = requests.post(BASE + "/v1/messages/count_tokens", headers=HEADERS, json={
        "model": "longcat-next",
        "messages": [{"role": "user", "content": "hello there"}]}, timeout=60)
    n = r.json().get("input_tokens", 0)
    rec("count_tokens", r.status_code == 200 and n > 0, f"{n} tokens")
except Exception as e:
    rec("count_tokens", False, str(e))

n_pass = sum(1 for _, ok in results if ok)
print(f"\n=== {n_pass}/{len(results)} anthropic-route checks passed ===")
sys.exit(0 if n_pass == len(results) else 1)
