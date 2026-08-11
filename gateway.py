#!/usr/bin/env python3
"""OpenAI-compatible gateway for LongCat-Next — ALL modalities, concurrency-safe.

SGLang's own OpenAI surface handles text, image-understanding (image_url) and video-understanding
(video_url) in /v1/chat/completions. This gateway adds the gaps so the WHOLE feature set is reachable
via standard OpenAI endpoints, and proxies everything else to SGLang unchanged:
  - POST /v1/chat/completions  : proxied (incl. SSE when stream=true); input_audio parts (audio
                                 understanding, which SGLang's schema rejects) -> native /generate.
  - POST /v1/images/generations: text-to-image (supports n); returns b64_json or url.
  - POST /v1/audio/speech      : voice-clone TTS (voice = en | zh | a container path); returns wav.
  - GET  /health               : 200 when the SGLang backend is ready, 503 while still loading.

CONCURRENCY: fully async (httpx); a long image gen never blocks the event loop. Generated artifacts
are retrieved by the per-request SGLang id (meta_info.id -> the model names the file via
ForwardBatch.rids), i.e. exact-name lookup — no globbing, no lock — correct under concurrent load.

NOTES / current limits (documented honestly):
  - /v1/images/generations ignores `size`/`quality`/`style` (model emits a fixed 37x37 token grid).
  - /v1/audio/speech returns WAV (the image has no mp3 encoder); `response_format` other than wav/pcm
    is best-effort returned as wav.
  - Per-request generation sampling (CFG/temp/top_k) is server-configured via env (IMAGE_GEN_* /
    AUDIO_GEN_*), not per-call (the generation heads read module/env config, not request params).
"""
import os, time, base64, asyncio, uuid, json, hmac, logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoTokenizer
from longcat_tools import build_tools_system_block, parse_tool_calls
from stream_tools import ToolStreamFilter, MARKERS as STREAM_MARKERS
from audio_chat import extract_audio_chat
from stream_util import open_upstream_stream
from anthropic_route import router as anthropic_router

SGLANG = "http://localhost:%s" % os.environ.get("SGLANG_INTERNAL_PORT", "30000")
MODEL = os.environ.get("MODEL_PATH", "/workspace/model")
OUT = os.environ.get("LCN_OUTPUT_DIR", "/tmp")
VOICES = {
    "en": "/workspace/scripts/voices/en_reference.wav",
    "english": "/workspace/scripts/voices/en_reference.wav",
    "zh": "/workspace/scripts/voices/zh_reference.wav",
    "chinese": "/workspace/scripts/voices/zh_reference.wav",
}
DEFAULT_VOICE = VOICES["en"]
IMG_START = 131106
ANYRES = "<longcat_img_token_size>37 37</longcat_img_token_size>"
AUDIO_INSTR = "用这个声音合成以下内容："

# Optional bearer-token auth. Unset (default) => no auth, which is safe ONLY because run.sh/compose
# publish to 127.0.0.1 by default. If you expose the port on a network, set LCN_API_KEY.
API_KEY = os.environ.get("LCN_API_KEY", "").strip()
# LCN_AGENT=1 = agentic/understanding profile: image+audio GENERATION endpoints are
# refused so their heads never lazily allocate (~25GB on a 128GB GB10) — that budget is
# instead spent on the full-context KV pool (see entrypoint.sh). Understanding of all
# modalities (image/audio/video INPUT) still works.
AGENT_MODE = os.environ.get("LCN_AGENT", "0").strip() == "1"
# Catch-all proxy is DEFAULT-DENY: only inference/read-only SGLang endpoints pass through. The
# mutating admin surface (/flush_cache, /update_weights*, /release_memory_occupation, /*_profile,
# session + expert-distribution control, …) is NOT exposed — it could DoS or hijack the server.
PROXY_ALLOW = {"generate", "get_model_info", "get_server_info", "health", "health_generate",
               "v1/models", "v1/completions", "v1/embeddings", "encode", "classify"}
if AGENT_MODE:
    # raw /generate could carry generation-head prompts (<longcat_img_start> etc.) and
    # lazily allocate the ~25GB heads agent mode exists to avoid — close it too.
    PROXY_ALLOW = PROXY_ALLOW - {"generate"}
# Custom TTS reference clips must resolve UNDER one of these dirs (the bundled voices, or the
# user-mounted output dir) — a raw `voice` path would otherwise read any file in the container.
VOICE_DIRS = tuple(os.path.realpath(d) for d in
                   (os.path.dirname(VOICES["en"]), OUT, os.environ.get("LCN_VOICE_DIR", "")) if d)

logger = logging.getLogger("lcn.gateway")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
app = FastAPI(title="LongCat-Next OpenAI gateway")
app.include_router(anthropic_router)  # Anthropic Messages API (/v1/messages) — Claude Code etc.
_client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0))

# Global admission control for GENERATION requests (image + TTS).
# The per-request cap on `n` bounds ONE request to 4 concurrent generations and nothing
# bounded the server: two clients at n=4, or five clients at n=1, submitted 8+ simultaneous
# generations. On a box that hard-powers-off past ~110-115GB claimed this is a safety
# property, not just a throughput one. The cap preserves the previously ADVERTISED limit
# (4) as a real server-wide limit; requests queue rather than being rejected, because a
# client that waits is better than a node that powers off.
MAX_CONCURRENT_GEN = max(1, int(os.environ.get("LCN_MAX_CONCURRENT_GEN", "4")))
_gen_slots = asyncio.Semaphore(MAX_CONCURRENT_GEN)


def _discard_artifact(path):
    """Delete a generated artifact after it has been read into the response.

    Generated PNGs/WAVs were never unlinked, so every successful generation leaked a file
    and output storage grew without bound until generation started failing for lack of
    space. Set LCN_KEEP_ARTIFACTS=1 to retain them (the test battery does this to let a
    human inspect what was produced).
    """
    if os.environ.get("LCN_KEEP_ARTIFACTS", "0").strip() == "1":
        return
    try:
        os.unlink(path)
    except OSError:
        pass


@app.middleware("http")
async def _auth(request: Request, call_next):
    # /health is always open (orchestrator liveness probes shouldn't need the key).
    if API_KEY and request.url.path != "/health":
        hdr = request.headers.get("authorization", "")
        token = hdr[7:].strip() if hdr[:7].lower() == "bearer " else ""
        # Anthropic-native clients (Claude Code with ANTHROPIC_API_KEY) send x-api-key instead.
        token = token or request.headers.get("x-api-key", "").strip()
        if not hmac.compare_digest(token, API_KEY):
            return JSONResponse({"error": {"message": "invalid or missing API key"}}, status_code=401)
    return await call_next(request)


def _san(s):
    return "".join(c for c in str(s) if c.isalnum() or c in "-_")[:64]


def _json_or_text(r):
    """SGLang normally returns JSON, but a crash/error can yield a non-JSON body. Returns
    (parsed, None) on JSON, or (None, raw_text) so callers proxy the backend body instead of 500ing."""
    try:
        return r.json(), None
    except Exception:
        return None, r.text


def _resolve_voice(voice):
    """Named voice (en/zh/…) always; a custom path only if it stays under an allowed dir."""
    key = voice.lower()
    if key in VOICES:
        return VOICES[key]
    rp = os.path.realpath(voice)
    if any(rp == d or rp.startswith(d + os.sep) for d in VOICE_DIRS) and os.path.isfile(rp):
        return rp
    return DEFAULT_VOICE


async def _read_when_ready(path, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        await asyncio.sleep(0.2)
    return None


async def _backend_up():
    # readiness = model loaded & serving. /get_model_info returns 200 only once the model
    # is up (fast, no generation). /health_generate runs a real gen and can exceed a short
    # timeout right after load -> false "loading".
    try:
        r = await _client.get(SGLANG + "/get_model_info", timeout=10.0)
        return r.status_code == 200
    except Exception:
        return False


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"}) if await _backend_up() \
        else JSONResponse({"status": "loading"}, status_code=503)


# --- prewarm ------------------------------------------------------------------------
# The image/audio generation heads allocate ~25GB lazily on FIRST USE, and the visual
# decoder + refiner lazy-load on the first decode. Measured cost of that surprise:
# the first image after a load takes ~338s against ~196s warm. Whoever sends the first
# request pays it, with no signal that anything unusual is happening.
#
# LCN_PREWARM=1 issues one real image generation and one short TTS at startup so the
# cost lands during load, where an operator already expects to wait. Opt-in, because it
# adds minutes to startup and an agent-mode deployment never wants it at all.
PREWARM = os.environ.get("LCN_PREWARM", "0").strip() == "1"
_prewarm_state = {"enabled": PREWARM, "status": "disabled" if not PREWARM else "pending",
                  "image_s": None, "audio_s": None, "error": None}


async def _prewarm_task():
    """Warm the generation paths once the backend is up. Never fatal."""
    import time as _t
    if AGENT_MODE:
        # Agent mode 403s these endpoints precisely so the heads never allocate;
        # prewarming would spend the memory the profile exists to protect.
        _prewarm_state.update(status="skipped (agent mode)")
        logger.info("[prewarm] skipped: LCN_AGENT=1 deliberately avoids the generation heads")
        return
    _prewarm_state.update(status="waiting for backend")
    for _ in range(240):                      # up to ~20 min of model load
        if await _backend_up():
            break
        await asyncio.sleep(5)
    else:
        _prewarm_state.update(status="failed", error="backend never came up")
        return

    _prewarm_state.update(status="running")
    try:
        t0 = _t.monotonic()
        data, err = await _gen_one_image("A photograph of a wooden chair.",
                                         {"max_new_tokens": 2048, "temperature": 0.5})
        _prewarm_state["image_s"] = round(_t.monotonic() - t0, 1)
        if err:
            raise RuntimeError(err)
        logger.info("[prewarm] image path warm in %.1fs", _prewarm_state["image_s"])
    except Exception as e:                     # noqa: BLE001 - prewarm must never take the server down
        _prewarm_state.update(status="failed", error=f"image: {e}")
        logger.warning("[prewarm] image warmup failed (serving continues): %s", e)
        return

    # Audio too: the TTS head and the cosy24k vocoder have their own lazy allocation, and
    # a deployment that prewarms images while leaving the first TTS caller to pay is only
    # half a fix. Kept SHORT -- a few words is enough to fault in the weights, and warmup
    # should not cost more startup than it saves.
    try:
        t0 = _t.monotonic()
        async with _gen_slots:
            r = await _client.post(SGLANG + "/generate", json={
                "text": _tts_prompt("Ready."), "audio_data": [DEFAULT_VOICE],
                "sampling_params": {"max_new_tokens": 1200, "temperature": 0.5,
                                    "top_k": 5, "top_p": 0.85}})
        if r.status_code != 200:
            raise RuntimeError("backend error: " + r.text[:200])
        rj, raw = _json_or_text(r)
        if rj is None:
            raise RuntimeError("backend error: " + raw[:200])
        path = "%s/longcat_tts_%s.wav" % (OUT, _san(rj.get("meta_info", {}).get("id", "")))
        if await _read_when_ready(path) is not None:
            _discard_artifact(path)
        _prewarm_state["audio_s"] = round(_t.monotonic() - t0, 1)
        logger.info("[prewarm] audio path warm in %.1fs", _prewarm_state["audio_s"])
    except Exception as e:                     # noqa: BLE001
        # Image is already warm, which is the expensive half — report the miss, keep going.
        _prewarm_state.update(error=f"audio: {e}")
        logger.warning("[prewarm] audio warmup failed (serving continues): %s", e)

    _prewarm_state.update(status="ready")
    logger.info("[prewarm] complete: image=%ss audio=%ss",
                _prewarm_state["image_s"], _prewarm_state["audio_s"])


@app.on_event("startup")
async def _on_startup():
    if PREWARM:
        asyncio.create_task(_prewarm_task())


@app.get("/status")
async def status():
    """Build id + EFFECTIVE config + readiness, in one place.

    Exists because 'which build is actually running, with which flags?' was repeatedly
    answered by reading launcher scripts and guessing. The values below are read from the
    live process, not from what a script intended to set.
    """
    return JSONResponse({
        "build": os.environ.get("LCN_BUILD", "unknown"),
        "backend_up": await _backend_up(),
        "prewarm": _prewarm_state,
        "config": {
            "agent_mode": AGENT_MODE,
            "auth_required": bool(API_KEY),
            "max_concurrent_gen": MAX_CONCURRENT_GEN,
            "output_dir": OUT,
            "keep_artifacts": os.environ.get("LCN_KEEP_ARTIFACTS", "0").strip() == "1",
            "radix": os.environ.get("LCN_RADIX", "1"),
            "cudagraph": os.environ.get("LCN_CUDAGRAPH", "0"),
            "ngram": os.environ.get("LCN_NGRAM", "0"),
            "yarn": os.environ.get("LCN_YARN", "0"),
            "kv_dtype": os.environ.get("LCN_KV_DTYPE", "") or "bf16 (default)",
            "head_batch": os.environ.get("LCN_HEAD_BATCH", "1") != "0",
            "refiner_fast": os.environ.get("LCN_REFINER_FAST", "1") != "0",
            "refiner_steps": os.environ.get("REFINER_STEPS", "10"),
        },
    })


async def _gen_one_image(prompt, sampling):
    ids = (tok(prompt, add_special_tokens=False).input_ids
           + tok(ANYRES, add_special_tokens=False).input_ids + [IMG_START])
    try:
        async with _gen_slots:
            r = await _client.post(SGLANG + "/generate", json={"input_ids": ids, "sampling_params": sampling})
    except httpx.ConnectError:
        return None, "backend unavailable (model may still be loading)"
    if r.status_code != 200:
        return None, "backend error: " + r.text[:200]
    rj, raw = _json_or_text(r)
    if rj is None:
        return None, "backend error: " + raw[:200]
    rid = _san(rj.get("meta_info", {}).get("id", ""))
    path = "%s/longcat_img_%s_refined.png" % (OUT, rid)
    data = await _read_when_ready(path)
    if data is None:
        return None, "image generation produced no output"
    _discard_artifact(path)
    return data, None


@app.post("/v1/images/generations")
async def images_generations(req: Request):
    if AGENT_MODE:
        return JSONResponse({"error": {"message": "image generation is disabled in agent mode "
                            "(LCN_AGENT=1); its memory budget funds the full-context KV pool"}}, status_code=403)
    body = await req.json()
    if body.get("response_format") == "url":
        # Reject BEFORE generating: we have no public file server, so "url" could only return an
        # unfetchable file:// path that leaks a container path and leaves the PNG uncleaned.
        return JSONResponse({"error": {"message": "response_format 'url' is not supported; use "
                            "'b64_json' (the default)"}}, status_code=400)
    prompt = body.get("prompt", "")
    n = max(1, min(int(body.get("n", 1)), 4))
    sampling = {"max_new_tokens": 1500, "temperature": 0.5, "top_k": 1024, "top_p": 0.75}
    results = await asyncio.gather(*[_gen_one_image(prompt, sampling) for _ in range(n)])
    data = []
    for img, err in results:
        if err:
            return JSONResponse({"error": {"message": err}}, status_code=503 if "loading" in err else 500)
        data.append({"b64_json": base64.b64encode(img).decode()})
    return {"created": int(time.time()), "data": data}


def _tts_prompt(text: str) -> str:
    """The voice-clone TTS prompt. Shared by the endpoint and prewarm.

    Extracted rather than duplicated: prewarm must exercise the SAME path a real request
    takes, and a copy would silently stop matching the moment either side is edited —
    warming a path nobody uses is worse than not warming at all, because it looks warm.
    """
    return ("<longcat_system>Replicate the voice in the audio clip to formulate an answer:"
            "<longcat_audio_start><longcat_audio_end>"
            "<longcat_user>" + AUDIO_INSTR + text +
            "<longcat_assistant><longcat_audiogen_start>")


@app.post("/v1/audio/speech")
async def audio_speech(req: Request):
    if AGENT_MODE:
        return JSONResponse({"error": {"message": "audio generation is disabled in agent mode "
                            "(LCN_AGENT=1); its memory budget funds the full-context KV pool"}}, status_code=403)
    body = await req.json()
    text = body.get("input", "")
    voice = str(body.get("voice", "en"))
    ref = _resolve_voice(voice)
    prompt = _tts_prompt(text)
    try:
        async with _gen_slots:
            r = await _client.post(SGLANG + "/generate", json={"text": prompt, "audio_data": [ref],
                "sampling_params": {"max_new_tokens": 1200, "temperature": 0.5, "top_k": 5, "top_p": 0.85}})
    except httpx.ConnectError:
        return JSONResponse({"error": {"message": "backend unavailable (model may still be loading)"}}, status_code=503)
    rj, raw = _json_or_text(r)
    if rj is None:
        return JSONResponse({"error": {"message": "backend error: " + raw[:200]}}, status_code=502)
    rid = _san(rj.get("meta_info", {}).get("id", ""))
    path = "%s/longcat_tts_%s.wav" % (OUT, rid)
    data = await _read_when_ready(path)
    if data is None:
        return JSONResponse({"error": {"message": "audio generation produced no output"}}, status_code=500)
    _discard_artifact(path)
    return Response(content=data, media_type="audio/wav")


async def _stream_chat(r):
    """Pass an already-opened, already-status-checked upstream stream through verbatim."""
    try:
        async for chunk in r.aiter_bytes():
            yield chunk
    finally:
        await r.aclose()


async def _sse_deltas(r):
    """Yield (content_piece, finish_reason, usage) triples from an open OpenAI SSE stream.

    Takes an OPEN response rather than opening one: the status must already have been
    settled by stream_util.open_upstream_stream, because a non-200 body yields no
    "data:" lines here and would otherwise read as a clean, empty, successful stream.
    """
    try:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                j = json.loads(data)
            except Exception:
                continue
            ch = (j.get("choices") or [{}])[0] if j.get("choices") else {}
            piece = (ch.get("delta") or {}).get("content") or ""
            yield piece, ch.get("finish_reason"), j.get("usage")
    finally:
        await r.aclose()


def _earliest_marker(text):
    idxs = [text.find(m) for m in STREAM_MARKERS]
    idxs = [i for i in idxs if i != -1]
    return min(idxs) if idxs else -1


async def _stream_chat_with_tools(upstream, tools, model):
    """Live token streaming on the tools path: pass text through, go silent at the
    first tool marker, emit parsed tool_calls at the end (ROADMAP #2)."""
    filt = ToolStreamFilter()
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def chunk(delta, finish=None):
        return "data: " + json.dumps(
            {"id": cid, "object": "chat.completion.chunk", "created": created,
             "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
            ensure_ascii=False) + "\n\n"

    yield chunk({"role": "assistant"})
    finish_reason = "stop"
    try:
        async for piece, fr, _usage in _sse_deltas(upstream):
            if fr:
                finish_reason = fr
            out = filt.feed(piece)
            if out:
                yield chunk({"content": out})
    except Exception as e:
        # The 200 and its headers are already on the wire, so this cannot become an error
        # STATUS -- but it must not become a silent truncation either. Emit an SSE error
        # object (the shape OpenAI clients surface) and terminate the stream properly,
        # rather than letting the exception kill the connection with no [DONE].
        yield "data: " + json.dumps({"error": {"message": "stream failed: " + str(e)[:200],
                                     "type": "upstream_error"}}) + "\n\n"
        yield chunk({}, "error")
        yield "data: [DONE]\n\n"
        return
    leftover, raw = filt.finish()
    if leftover:
        yield chunk({"content": leftover})
    calls = []
    if filt.saw_marker:
        _normal, calls = parse_tool_calls(raw, tools)
        if not calls:
            # Marker never parsed into calls — release the swallowed text
            i = _earliest_marker(raw)
            if i != -1:
                yield chunk({"content": raw[i:]})
    if calls:
        for i, c in enumerate(calls):
            yield chunk({"tool_calls": [{"index": i, "id": c["id"], "type": "function",
                        "function": {"name": c["function"]["name"],
                                     "arguments": c["function"]["arguments"]}}]})
        yield chunk({}, "tool_calls")
    else:
        yield chunk({}, finish_reason)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    has_audio = any(isinstance(m.get("content"), list)
                    and any(p.get("type") == "input_audio" for p in m["content"])
                    for m in msgs)
    tools = body.get("tools")
    tool_choice = body.get("tool_choice", "auto")
    if not has_audio and tools and tool_choice != "none":
        # Tool calling: inject the CANONICAL TS-namespace tools block into the system prompt
        # (the format the model was trained on — SGLang's jinja tool rendering produces garbage),
        # don't pass `tools` to SGLang, then parse the <longcat_tool_call> XML output -> tool_calls.
        block = build_tools_system_block(tools)
        msgs2 = [dict(m) for m in msgs]
        if msgs2 and msgs2[0].get("role") == "system" and isinstance(msgs2[0].get("content"), str):
            msgs2[0]["content"] = block + "\n\n" + msgs2[0]["content"]
        else:
            msgs2 = [{"role": "system", "content": block}] + msgs2
        b2 = dict(body); b2["messages"] = msgs2; b2.pop("tools", None); b2.pop("tool_choice", None)
        if b2.pop("stream", False):
            b3 = dict(b2); b3["stream"] = True
            up, err = await open_upstream_stream(_client, SGLANG + "/v1/chat/completions", b3)
            if err:
                return JSONResponse({"error": {"message": err[1]}}, status_code=err[0])
            return StreamingResponse(
                _stream_chat_with_tools(up, tools, body.get("model", MODEL)),
                media_type="text/event-stream")
        try:
            r = await _client.post(SGLANG + "/v1/chat/completions", json=b2)
        except httpx.ConnectError:
            return JSONResponse({"error": {"message": "backend unavailable (model may still be loading)"}}, status_code=503)
        j, raw = _json_or_text(r)
        if j is None:
            return Response(content=raw, status_code=r.status_code,
                            media_type=r.headers.get("content-type", "text/plain"))
        try:
            msg = j["choices"][0]["message"]
            normal, calls = parse_tool_calls(msg.get("content") or "", tools)
            if calls:
                msg["content"] = normal or None
                msg["tool_calls"] = calls
                j["choices"][0]["finish_reason"] = "tool_calls"
        except Exception:
            pass
        return JSONResponse(j, status_code=r.status_code)
    if not has_audio:
        if tools:  # tool_choice == "none" reaches here — strip tools so SGLang doesn't apply its broken jinja tool rendering
            body = {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}
        if body.get("stream"):
            up, err = await open_upstream_stream(_client, SGLANG + "/v1/chat/completions", body)
            if err:
                return JSONResponse({"error": {"message": err[1]}}, status_code=err[0])
            return StreamingResponse(_stream_chat(up), media_type="text/event-stream")
        try:
            r = await _client.post(SGLANG + "/v1/chat/completions", json=body)
        except httpx.ConnectError:
            return JSONResponse({"error": {"message": "backend unavailable (model may still be loading)"}}, status_code=503)
        j, raw = _json_or_text(r)
        if j is None:
            return Response(content=raw, status_code=r.status_code,
                            media_type=r.headers.get("content-type", "text/plain"))
        return JSONResponse(j, status_code=r.status_code)
    # audio understanding: SGLang's chat schema rejects input_audio -> native /generate
    prompt, blobs = extract_audio_chat(msgs)
    if not blobs:
        return JSONResponse({"error": {"message": "no decodable audio in request; "
                            "input_audio.data must be base64"}}, status_code=400)
    paths = []
    for blob in blobs:
        p = "%s/_in_%s.wav" % (OUT, uuid.uuid4().hex)
        with open(p, "wb") as f:
            f.write(blob)
        paths.append(p)
    sp = {"max_new_tokens": int(body.get("max_tokens", 256)), "temperature": body.get("temperature", 0.2)}
    for k in ("top_p", "top_k", "frequency_penalty", "presence_penalty", "repetition_penalty", "stop"):
        if body.get(k) is not None:
            sp[k] = body[k]
    try:
        r = await _client.post(SGLANG + "/generate", json={"text": prompt, "audio_data": paths, "sampling_params": sp})
        rj, raw = _json_or_text(r)
    finally:
        for p in paths:
            try: os.remove(p)
            except OSError: pass
    if rj is None:
        return JSONResponse({"error": {"message": "backend error: " + raw[:200]}}, status_code=502)
    txt = rj.get("text", "")
    meta = rj.get("meta_info", {}) or {}
    fr = meta.get("finish_reason")
    fr = fr.get("type") if isinstance(fr, dict) else (fr or "stop")
    pt, ct = int(meta.get("prompt_tokens", 0) or 0), int(meta.get("completion_tokens", 0) or 0)
    return {"object": "chat.completion", "created": int(time.time()), "model": body.get("model", MODEL),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": fr}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, req: Request):
    if path not in PROXY_ALLOW:
        return JSONResponse({"error": {"message": "endpoint '%s' is not exposed by the "
                            "LongCat-Next gateway" % path}}, status_code=404)
    url = SGLANG + "/" + path
    try:
        if req.method == "GET":
            r = await _client.get(url, params=dict(req.query_params))
        else:
            r = await _client.post(url, content=await req.body(),
                                   headers={"content-type": req.headers.get("content-type", "application/json")})
    except httpx.ConnectError:
        return JSONResponse({"error": {"message": "backend unavailable (model may still be loading)"}}, status_code=503)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))
