"""Audio-chat prompt construction for LongCat-Next.

Kept out of gateway.py deliberately: this is a pure function over the request body with
no framework dependencies, so it can be tested offline (test/test_audio_chat_prompt.py)
without fastapi/httpx/transformers or a running model -- the same split that makes
longcat_tools and stream_tools checkable in isolation.
"""
import base64

ROLE_MARK = {"system": "<longcat_system>", "user": "<longcat_user>",
             "assistant": "<longcat_assistant>"}
AUDIO_PAIR = "<longcat_audio_start><longcat_audio_end>"


def extract_audio_chat(messages):
    """Render an audio chat into LongCat's native turn markers, keeping EVERY clip.

    Returns (prompt, [clip_bytes, ...]) with the clips in prompt order, or (None, [])
    when the request carries no decodable audio.

    What this replaces: the previous version concatenated every message's text into one
    string with no role markers and overwrote the clip on each pass, then the caller
    rebuilt the whole conversation as a single <longcat_user> turn. Two consequences on
    any multi-turn audio conversation -- the model saw its OWN prior replies as part of
    the user's utterance, and every clip but the last was silently discarded.

    One empty AUDIO_PAIR is emitted per clip. The processor fills them positionally: it
    scans for the first pair that is still EMPTY, so once clip N's pads are inserted that
    pair no longer matches and clip N+1 lands in the next one (see the `starts` loop in
    new_files/processors/longcat_next.py).

    Intra-turn order is text-then-clip, matching the single-turn form this replaces --
    that is the shape known to work against this checkpoint, so the multi-turn case
    extends it rather than inventing a new layout.
    """
    turns, blobs = [], []
    for m in messages:
        mark = ROLE_MARK.get(m.get("role", "user"), "<longcat_user>")
        c = m.get("content")
        text, n_audio = "", 0
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text += p.get("text", "")
                elif p.get("type") == "input_audio":
                    try:
                        blobs.append(base64.b64decode(p["input_audio"]["data"], validate=True))
                        n_audio += 1
                    except Exception:
                        pass  # one unreadable clip must not sink the whole conversation
        text = text.strip()
        if text or n_audio:
            turns.append((mark, text, n_audio))
    if not blobs:
        return None, []
    if not any(t for _, t, _ in turns):
        # Audio with no text anywhere: supply the instruction the single-turn path used,
        # attached to the last turn that actually carries a clip.
        for i in range(len(turns) - 1, -1, -1):
            if turns[i][2]:
                turns[i] = (turns[i][0], "Transcribe this audio.", turns[i][2])
                break
    prompt = "".join(mk + tx + AUDIO_PAIR * na for mk, tx, na in turns)
    if turns[-1][0] != "<longcat_assistant>":
        prompt += "<longcat_assistant>"
    return prompt, blobs
