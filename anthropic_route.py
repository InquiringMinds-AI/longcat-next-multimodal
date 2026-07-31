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

Streaming is buffered-then-emitted: the tool path needs the full completion to parse
XML, so we generate non-streaming and then synthesize the Anthropic SSE event sequence.
Anthropic clients (Claude Code included) tolerate chunky streams.
"""
import json, os, time, uuid
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from longcat_tools import build_tools_system_block, parse_tool_calls

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


def _render_tool_call_xml(name, args):
    """Canonical <longcat_tool_call> XML — mirrors the chat template's own rendering,
    with the TS namespace prefix the model uses when calling."""
    s = "<longcat_tool_call>functions." + name + "\n"
    for k, v in (args or {}).items():
        s += ("<longcat_arg_key>%s</longcat_arg_key>\n<longcat_arg_value>%s</longcat_arg_value>\n"
              % (k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)))
    return s + "</longcat_tool_call>\n"


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
                    txt += _render_tool_call_xml(b.get("name", ""), b.get("input") or {})
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


def _sse(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


async def _stream_events(msg):
    """Synthesize the Anthropic SSE sequence from a complete message dict."""
    head = {k: msg[k] for k in ("id", "type", "role", "model")}
    head.update({"content": [], "stop_reason": None, "stop_sequence": None,
                 "usage": {"input_tokens": msg["usage"]["input_tokens"], "output_tokens": 0}})
    yield _sse("message_start", {"type": "message_start", "message": head})
    for i, block in enumerate(msg["content"]):
        if block["type"] == "text":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": {"type": "text_delta", "text": block["text"]}})
        else:  # tool_use
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                       "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                       "delta": {"type": "input_json_delta",
                                 "partial_json": json.dumps(block["input"], ensure_ascii=False)}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    yield _sse("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                                 "usage": {"output_tokens": msg["usage"]["output_tokens"]}})
    yield _sse("message_stop", {"type": "message_stop"})


# ---------- routes ----------

@router.post("/v1/messages")
async def messages(req: Request):
    body = await req.json()
    tools = body.get("tools") or []
    oai_tools = _tools_to_openai(tools)
    msgs = _to_openai_messages(body.get("system"), body.get("messages"))
    if oai_tools:
        block = build_tools_system_block(oai_tools)
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
    usage = j.get("usage", {}) or {}
    msg = {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
           "model": body.get("model", "longcat-next"),
           "content": _anthropic_content(normal, calls),
           "stop_reason": _STOP_MAP.get(fr, "end_turn"), "stop_sequence": None,
           "usage": {"input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                     "output_tokens": int(usage.get("completion_tokens", 0) or 0)}}
    if body.get("stream"):
        return StreamingResponse(_stream_events(msg), media_type="text/event-stream")
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
