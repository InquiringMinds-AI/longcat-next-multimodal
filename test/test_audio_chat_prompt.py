#!/usr/bin/env python3
"""Offline checks for multi-turn audio-chat prompt construction (no server needed).

The bug these lock down: every message's text was concatenated with no role markers and
only the LAST clip was kept, then the whole conversation was rebuilt as one
<longcat_user> turn -- so on turn 2 the model saw its own prior reply as user speech.

Each case prints the CONSTRUCTED PROMPT, not just a verdict. A conversation that
collapses turns and one that preserves them both produce "1 clip, no error"; only the
prompt text tells them apart.
"""
import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("LCN_OUTPUT_DIR", "/tmp")
from audio_chat import extract_audio_chat as _extract_audio_chat  # noqa: E402

A1 = base64.b64encode(b"CLIP-ONE").decode()
A2 = base64.b64encode(b"CLIP-TWO").decode()
PAIR = "<longcat_audio_start><longcat_audio_end>"


def audio_msg(role, text, data):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if data:
        content.append({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})
    return {"role": role, "content": content}


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("single turn keeps the proven text-then-clip layout")
def _():
    p, b = _extract_audio_chat([audio_msg("user", "What is said here?", A1)])
    assert b == [b"CLIP-ONE"], b
    assert p == "<longcat_user>What is said here?" + PAIR + "<longcat_assistant>", p
    return p


@case("multi-turn preserves role boundaries and BOTH clips")
def _():
    p, b = _extract_audio_chat([
        audio_msg("user", "Who is speaking?", A1),
        {"role": "assistant", "content": "A man with a low voice."},
        audio_msg("user", "And in this one?", A2),
    ])
    # Both clips survive, in prompt order -- the old code returned only the last.
    assert b == [b"CLIP-ONE", b"CLIP-TWO"], b
    # The assistant's reply is marked as the ASSISTANT's, not folded into user text.
    assert p.count("<longcat_user>") == 2, p
    assert "<longcat_assistant>A man with a low voice." in p, p
    # One placeholder pair per clip, so the processor can fill them positionally.
    assert p.count(PAIR) == 2, p
    return p


@case("system turn is marked as system")
def _():
    p, b = _extract_audio_chat([
        {"role": "system", "content": "You are terse."},
        audio_msg("user", "", A1),
    ])
    assert p.startswith("<longcat_system>You are terse.<longcat_user>"), p
    return p


@case("audio with no text anywhere gets the transcribe instruction")
def _():
    p, b = _extract_audio_chat([audio_msg("user", "", A1)])
    assert "Transcribe this audio." in p, p
    return p


@case("string-content turns are marked, not concatenated")
def _():
    p, b = _extract_audio_chat([
        {"role": "user", "content": "Context sentence."},
        audio_msg("user", "Now this clip.", A1),
    ])
    assert p.count("<longcat_user>") == 2, p
    return p


@case("no decodable audio returns (None, []) rather than crashing the caller")
def _():
    p, b = _extract_audio_chat([{"role": "user", "content": "just text"}])
    assert p is None and b == [], (p, b)
    return "(None, [])"


@case("an undecodable clip is skipped without sinking the conversation")
def _():
    msgs = [audio_msg("user", "first", A1),
            {"role": "user", "content": [{"type": "input_audio",
                                          "input_audio": {"data": "!!!not base64!!!"}}]}]
    p, b = _extract_audio_chat(msgs)
    assert b == [b"CLIP-ONE"], b
    # Placeholder count must match the clips that actually survived, or the processor
    # would leave an empty pair for a clip that was never sent.
    assert p.count(PAIR) == 1, p
    return p


@case("trailing assistant turn is not given a second marker")
def _():
    p, b = _extract_audio_chat([audio_msg("user", "q", A1),
                                {"role": "assistant", "content": "partial"}])
    assert not p.endswith("<longcat_assistant>"), p
    return p


if __name__ == "__main__":
    fails = 0
    for name, fn in CASES:
        try:
            shown = fn()
            print("PASS  %s\n      %s" % (name, shown))
        except AssertionError as e:
            fails += 1
            print("FAIL  %s\n      %s" % (name, str(e)[:400]))
    print("\n%d/%d passed" % (len(CASES) - fails, len(CASES)))
    sys.exit(1 if fails else 0)
