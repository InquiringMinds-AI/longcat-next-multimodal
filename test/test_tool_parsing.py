#!/usr/bin/env python3
"""Offline regression tests for longcat_tools.parse_tool_calls — no server, no GPU.

Every case here is a REAL emission observed from the model, not an invented one. The
model has four known ways of expressing a call, and a dialect the parser does not
recognize is dropped silently: the request looks like an ordinary text answer, which is
how syntax 4 spent a long time being misfiled as an intermittent "tool_calling is flaky"
failure rather than a parser gap.

    python3 test/test_tool_parsing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from longcat_tools import parse_tool_calls  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]

CASES = [
    # (label, raw model output, expected name, expected args)
    ("syntax 1: XML arg pairs",
     '<longcat_tool_call>functions.get_weather\n'
     '<longcat_arg_key>city</longcat_arg_key>\n'
     '<longcat_arg_value>Tokyo</longcat_arg_value>\n</longcat_tool_call>',
     "get_weather", {"city": "Tokyo"}),

    ("syntax 2: TS-style call",
     '<longcat_tool_call>functions.get_weather({city: "Tokyo"})',
     "get_weather", {"city": "Tokyo"}),

    ("syntax 3: Claude-imitation JSON array",
     '<function_calls>[{"name": "get_weather", "parameters": {"city": "Tokyo"}}]</function_calls>',
     "get_weather", {"city": "Tokyo"}),

    # Captured 2026-08-10 by selftest's raw-emission dump, finish_reason='stop'.
    # This is the case that used to parse to zero calls.
    ("syntax 4: args as one JSON object after <longcat_arg_key>",
     '<longcat_tool_call>functions.get_weather\n<longcat_arg_key>{"city": "Tokyo"}',
     "get_weather", {"city": "Tokyo"}),

    ("syntax 4 with trailing junk (no closing tag)",
     '<longcat_tool_call>functions.get_weather\n<longcat_arg_key>{"city": "Tokyo"} </longcat',
     "get_weather", {"city": "Tokyo"}),

    ("syntax 4 with unquoted keys",
     '<longcat_tool_call>functions.get_weather\n<longcat_arg_key>{city: "Tokyo"}',
     "get_weather", {"city": "Tokyo"}),
]

NON_CALLS = [
    ("plain prose stays prose", "The weather in Tokyo is currently sunny and mild."),
    # A call was never attempted; a parser must NOT invent one.
    ("prose mentioning the tool name", "I could use get_weather(city) but I will not."),
]


def main():
    import json
    failed = 0
    for label, raw, want_name, want_args in CASES:
        text, calls = parse_tool_calls(raw, TOOLS)
        ok = (len(calls) == 1
              and calls[0]["function"]["name"] == want_name
              and json.loads(calls[0]["function"]["arguments"]) == want_args)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failed += 1
            print(f"         got calls={calls!r} text={text!r}")

    for label, raw in NON_CALLS:
        text, calls = parse_tool_calls(raw, TOOLS)
        ok = not calls
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failed += 1
            print(f"         got calls={calls!r}")

    total = len(CASES) + len(NON_CALLS)
    print(f"\n=== {total - failed}/{total} passed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
