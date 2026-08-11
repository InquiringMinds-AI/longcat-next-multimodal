"""Anthropic Messages API for LongCat-Next — POST /v1/messages (+ count_tokens).

Lets Anthropic-native clients — Claude Code via ANTHROPIC_BASE_URL — drive the model,
including tool calling. Translation strategy:

  Anthropic request  ->  internal OpenAI-ish chat body  ->  SGLang /v1/chat/completions
                                                         -> parse <longcat_tool_call> XML
                     <-  Anthropic response / SSE stream <-

Tool handling REUSES the proven longcat_tools path (TS-namespace system block + XML
parse). Assistant tool_use HISTORY is pre-rendered into message CONTENT as the canonical
XML: the model's chat template calls `arguments.items()` (expects a dict) so passing
OpenAI-style tool_calls (arguments = JSON string) through SGLang renders garbage —
content-side rendering sidesteps that entirely. Tool RESULTS go through as role:"tool"
messages, which the template renders correctly (<longcat_tool_response>).

Streaming is INCREMENTAL (real deltas): text passes through live; a tool marker
silences pass-through mid-stream and the buffered tail parses into tool_use blocks at
stream end (stream_tools.ToolStreamFilter, shared with the OpenAI route).
"""
import json, os, time, uuid
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from longcat_tools import build_tools_system_block, parse_tool_calls, render_tool_call_xml
from stream_tools import ToolStreamFilter, MARKERS as _MARKERS
from stream_util import open_upstream_stream

SGLANG = "http://localhost:%s" % os.environ.get("SGLANG_INTERNAL_PORT", "30000")
router = APIRouter()
_client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0))


# ---------- request translation ----------

def _flatten_text(content):
    """Anthropic content (str | [blocks]) -> plain text (text blocks only)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""


def _tool_result_text(block):
    txt = _flatten_text(block.get("content", ""))
    return ("ERROR: " + txt) if block.get("is_error") else txt


def _tools_to_openai(tools):
    return [{"type": "function", "function": {"name": t["name"],
             "description": t.get("description", ""), "parameters": t.get("input_schema", {})}}
            for t in tools or [] if t.get("name")]


def _to_openai_messages(system, messages):
    out = []
    for m in messages or []:
        role, c = m.get("role"), m.get("content")
        if isinstance(c, str):
            out.append({"role": role, "content": c})
            continue
        if role == "user":
            parts, tool_msgs = [], []
            for b in c or []:
                t = b.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif t == "image":
                    src = b.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({"type": "image_url", "image_url": {
                            "url": "data:%s;base64,%s" % (src.get("media_type", "image/png"), src.get("data", ""))}})
                    elif src.get("type") == "url":
                        parts.append({"type": "image_url", "image_url": {"url": src.get("url", "")}})
                elif t == "tool_result":
                    tool_msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                                      "content": _tool_result_text(b)})
            # tool results answer the PREVIOUS assistant turn -> they precede this turn's text
            out.extend(tool_msgs)
            if parts:
                out.append({"role": "user", "content": parts})
        elif role == "assistant":
            txt = ""
            for b in c or []:
                if b.get("type") == "text":
                    txt += b.get("text", "")
                elif b.get("type") == "tool_use":
                    txt += render_tool_call_xml(b.get("name", ""), b.get("input") or {})
            out.append({"role": "assistant", "content": txt})
    sys_txt = _flatten_text(system or "")
    if sys_txt:
        out.insert(0, {"role": "system", "content": sys_txt})
    return out


# ---------- response translation ----------

def _anthropic_content(normal, calls):
    content = []
    if normal:
        content.append({"type": "text", "text": normal})
    for c in calls:
        try:
            args = json.loads(c["function"]["arguments"])
        except Exception:
            args = {}
        content.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:24],
                        "name": c["function"]["name"], "input": args})
    return content


_STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
# Finish reasons meaning the generation did NOT complete. Anthropic's stop_reason vocabulary
# has no value for this, so _STOP_MAP's default silently rendered them as "end_turn" -- a
# truncated reply arriving as a normal, complete one. Measured: aborting a stream mid-flight
# produced 181 tokens with stop_reason "end_turn" and no error event, which a client has no
# way to distinguish from a finished answer. Reported as an explicit error instead.
_ABNORMAL_FINISH = {"abort", "abort_request", "error", "cancelled"}
_ABNORMAL_MSG = "generation did not complete: the backend aborted it (finish_reason=%s)"


def _sse(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


async def _stream_live(body, upstream, oai_tools):
    """REAL incremental SSE (ROADMAP #2): text deltas pass through as they arrive;
    a tool marker silences pass-through and the buffered tail parses into tool_use
    blocks at the end. The old buffered-then-synthesized path remains for
    stream=false only."""
    filt = ToolStreamFilter()
    mid = "msg_" + uuid.uuid4().hex[:24]
    model = body.get("model", "longcat-next")
    head = {"id": mid, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}
    yield _sse("message_start", {"type": "message_start", "message": head})
    finish_reason, usage = "stop", {}
    text_open = False
    idx = 0

    def text_delta(t):
        return _sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                    "delta": {"type": "text_delta", "text": t}})

    try:
        try:
            async for line in upstream.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                ch = (j.get("choices") or [{}])[0] if j.get("choices") else {}
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                out = filt.feed((ch.get("delta") or {}).get("content") or "")
                if out:
                    if not text_open:
                        yield _sse("content_block_start", {"type": "content_block_start",
                                   "index": idx, "content_block": {"type": "text", "text": ""}})
                        text_open = True
                    yield text_delta(out)
        finally:
            await upstream.aclose()
    except Exception as e:
        # message_start is already on the wire, so the status cannot change -- but the
        # stream must still terminate in a shape the client understands rather than dying
        # mid-message. Anthropic's SSE schema has a dedicated `error` event for this.
        yield _sse("error", {"type": "error", "error": {"type": "api_error",
                   "message": "stream failed: " + str(e)[:200]}})
        if text_open:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        yield _sse("message_delta", {"type": "message_delta",
                   "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                   "usage": {"output_tokens": 0}})
        yield _sse("message_stop", {"type": "message_stop"})
        return
    leftover, raw = filt.finish()
    calls = []
    tail_text = leftover
    if filt.saw_marker:
        if oai_tools:
            _normal, calls = parse_tool_calls(raw, oai_tools)
        if not calls:
            # Marker without parseable calls (or no tools declared): release the text
            i = min((raw.find(m) for m in _MARKERS if m in raw), default=-1)
            if i != -1:
                tail_text = raw[i:]
    if tail_text:
        if not text_open:
            yield _sse("content_block_start", {"type": "content_block_start", "index": idx,
                       "content_block": {"type": "text", "text": ""}})
            text_open = True
        yield text_delta(tail_text)
    if text_open:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        idx += 1
    for c in calls:
        try:
            args = json.loads(c["function"]["arguments"])
        except Exception:
            args = {}
        yield _sse("content_block_start", {"type": "content_block_start", "index": idx,
                   "content_block": {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:24],
                                     "name": c["function"]["name"], "input": {}}})
        yield _sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                   "delta": {"type": "input_json_delta",
                             "partial_json": json.dumps(args, ensure_ascii=False)}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        idx += 1
    abnormal = finish_reason in _ABNORMAL_FINISH
    if abnormal:
        # The 200 is long since committed, so this cannot become an error STATUS -- but the
        # stream must still say the reply is incomplete rather than terminate as if it
        # finished. Emitted before the terminators so the stream still ends well-formed.
        yield _sse("error", {"type": "error", "error": {
            "type": "api_error", "message": _ABNORMAL_MSG % finish_reason}})
    # stop_reason stays NULL on an abnormal finish. Emitting the error event but still
    # claiming "end_turn" left the stream asserting both that it failed and that it ended
    # normally, and a client reading stop_reason -- the field that exists to answer exactly
    # this -- would still have been told the reply was complete. Null is the honest value:
    # the turn has no normal stop reason because it did not stop normally.
    stop = None if abnormal else ("tool_use" if calls else _STOP_MAP.get(finish_reason, "end_turn"))
    yield _sse("message_delta", {"type": "message_delta",
               "delta": {"stop_reason": stop, "stop_sequence": None},
               "usage": {"input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                         "output_tokens": int(usage.get("completion_tokens", 0) or 0)}})
    yield _sse("message_stop", {"type": "message_stop"})


# ---------- routes ----------

@router.post("/v1/messages")
async def messages(req: Request):
    body = await req.json()
    tools = body.get("tools") or []
    # tool_choice was previously read on the OpenAI route but not here, so an Anthropic
    # client asking for {"type": "none"} still had tools offered and could still receive a
    # tool call it had explicitly forbidden.
    tc = body.get("tool_choice")
    tc_type = (tc.get("type") if isinstance(tc, dict) else tc) or "auto"
    if tc_type == "none":
        tools = []
    oai_tools = _tools_to_openai(tools)
    msgs = _to_openai_messages(body.get("system"), body.get("messages"))
    if oai_tools:
        block = build_tools_system_block(oai_tools)
        # "any"/"tool" are honoured by INSTRUCTION, not by constrained decoding: this model
        # emits tool calls as free-form XML that is parsed afterwards, so there is no grammar
        # to force. Stated plainly because a caller that needs a hard guarantee does not have
        # one here -- silently accepting the field would imply otherwise.
        if tc_type == "any":
            block += "\n\nYou MUST call one of the available tools in your reply."
        elif tc_type == "tool" and isinstance(tc, dict) and tc.get("name"):
            block += ("\n\nYou MUST call the tool `%s` in your reply." % tc["name"])
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = block + "\n\n" + msgs[0]["content"]
        else:
            msgs.insert(0, {"role": "system", "content": block})
    sg_body = {"model": body.get("model", "longcat-next"), "messages": msgs,
               "max_tokens": body.get("max_tokens", 4096), "stream": False}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"),
                     ("top_k", "top_k"), ("stop_sequences", "stop")):
        if body.get(src) is not None:
            sg_body[dst] = body[src]
    if body.get("stream"):
        sg2 = dict(sg_body)
        sg2["stream"] = True
        sg2["stream_options"] = {"include_usage": True}
        # Open and status-check BEFORE returning a StreamingResponse: once that response
        # starts, its 200 is committed and a backend failure can only be reported inside
        # the stream. A non-200 body would otherwise yield no "data:" lines and reach the
        # client as an empty but successful message.
        upstream, err = await open_upstream_stream(_client, SGLANG + "/v1/chat/completions", sg2)
        if err:
            status = 529 if err[0] == 503 else err[0]
            return JSONResponse({"type": "error", "error": {
                "type": "overloaded_error" if status == 529 else "api_error",
                "message": err[1]}}, status_code=status)
        return StreamingResponse(_stream_live(body, upstream, oai_tools),
                                 media_type="text/event-stream")
    try:
        r = await _client.post(SGLANG + "/v1/chat/completions", json=sg_body)
    except httpx.ConnectError:
        return JSONResponse({"type": "error", "error": {"type": "overloaded_error",
                            "message": "backend unavailable (model may still be loading)"}}, status_code=529)
    try:
        j = r.json()
        choice = j["choices"][0]
    except Exception:
        return JSONResponse({"type": "error", "error": {"type": "api_error",
                            "message": "backend error: " + r.text[:300]}}, status_code=502)
    raw = choice["message"].get("content") or ""
    normal, calls = parse_tool_calls(raw, oai_tools) if oai_tools else (raw, [])
    fr = "tool_calls" if calls else (choice.get("finish_reason") or "stop")
    if fr in _ABNORMAL_FINISH:
        # Non-streaming has the luxury of a real status code, so use it rather than
        # returning a partial message that claims to have ended normally.
        return JSONResponse({"type": "error", "error": {
            "type": "api_error", "message": _ABNORMAL_MSG % fr}}, status_code=500)
    usage = j.get("usage", {}) or {}
    msg = {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
           "model": body.get("model", "longcat-next"),
           "content": _anthropic_content(normal, calls),
           "stop_reason": _STOP_MAP.get(fr, "end_turn"), "stop_sequence": None,
           "usage": {"input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                     "output_tokens": int(usage.get("completion_tokens", 0) or 0)}}
    return JSONResponse(msg)


@router.post("/v1/messages/count_tokens")
async def count_tokens(req: Request):
    body = await req.json()
    text = _flatten_text(body.get("system") or "")
    for m in body.get("messages") or []:
        text += "\n" + _flatten_text(m.get("content"))
    for t in body.get("tools") or []:
        text += "\n" + t.get("name", "") + " " + t.get("description", "") + json.dumps(t.get("input_schema", {}))
    from gateway import tok  # deferred: gateway imports this module at startup
    return JSONResponse({"input_tokens": len(tok(text, add_special_tokens=False).input_ids)})
