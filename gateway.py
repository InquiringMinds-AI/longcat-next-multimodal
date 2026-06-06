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
import os, time, base64, asyncio, uuid, json, hmac
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoTokenizer
from longcat_tools import build_tools_system_block, parse_tool_calls

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
# Catch-all proxy is DEFAULT-DENY: only inference/read-only SGLang endpoints pass through. The
# mutating admin surface (/flush_cache, /update_weights*, /release_memory_occupation, /*_profile,
# session + expert-distribution control, …) is NOT exposed — it could DoS or hijack the server.
PROXY_ALLOW = {"generate", "get_model_info", "get_server_info", "health", "health_generate",
               "v1/models", "v1/completions", "v1/embeddings", "encode", "classify"}
# Custom TTS reference clips must resolve UNDER one of these dirs (the bundled voices, or the
# user-mounted output dir) — a raw `voice` path would otherwise read any file in the container.
VOICE_DIRS = tuple(os.path.realpath(d) for d in
                   (os.path.dirname(VOICES["en"]), OUT, os.environ.get("LCN_VOICE_DIR", "")) if d)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
app = FastAPI(title="LongCat-Next OpenAI gateway")
_client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0))


@app.middleware("http")
async def _auth(request: Request, call_next):
    # /health is always open (orchestrator liveness probes shouldn't need the key).
    if API_KEY and request.url.path != "/health":
        hdr = request.headers.get("authorization", "")
        token = hdr[7:].strip() if hdr[:7].lower() == "bearer " else ""
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


async def _gen_one_image(prompt, sampling):
    ids = (tok(prompt, add_special_tokens=False).input_ids
           + tok(ANYRES, add_special_tokens=False).input_ids + [IMG_START])
    try:
        r = await _client.post(SGLANG + "/generate", json={"input_ids": ids, "sampling_params": sampling})
    except httpx.ConnectError:
        return None, "backend unavailable (model may still be loading)"
    if r.status_code != 200:
        return None, "backend error: " + r.text[:200]
    rj, raw = _json_or_text(r)
    if rj is None:
        return None, "backend error: " + raw[:200]
    rid = _san(rj.get("meta_info", {}).get("id", ""))
    data = await _read_when_ready("%s/longcat_img_%s_refined.png" % (OUT, rid))
    if data is None:
        return None, "image generation produced no output"
    return data, None


@app.post("/v1/images/generations")
async def images_generations(req: Request):
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


@app.post("/v1/audio/speech")
async def audio_speech(req: Request):
    body = await req.json()
    text = body.get("input", "")
    voice = str(body.get("voice", "en"))
    ref = _resolve_voice(voice)
    prompt = ("<longcat_system>Replicate the voice in the audio clip to formulate an answer:"
              "<longcat_audio_start><longcat_audio_end>"
              "<longcat_user>" + AUDIO_INSTR + text +
              "<longcat_assistant><longcat_audiogen_start>")
    try:
        r = await _client.post(SGLANG + "/generate", json={"text": prompt, "audio_data": [ref],
            "sampling_params": {"max_new_tokens": 1200, "temperature": 0.5, "top_k": 5, "top_p": 0.85}})
    except httpx.ConnectError:
        return JSONResponse({"error": {"message": "backend unavailable (model may still be loading)"}}, status_code=503)
    rj, raw = _json_or_text(r)
    if rj is None:
        return JSONResponse({"error": {"message": "backend error: " + raw[:200]}}, status_code=502)
    rid = _san(rj.get("meta_info", {}).get("id", ""))
    data = await _read_when_ready("%s/longcat_tts_%s.wav" % (OUT, rid))
    if data is None:
        return JSONResponse({"error": {"message": "audio generation produced no output"}}, status_code=500)
    return Response(content=data, media_type="audio/wav")


def _extract_audio_chat(messages):
    q, audio_b = "", None
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            q += c
        elif isinstance(c, list):
            for p in c:
                if p.get("type") == "text":
                    q += p.get("text", "")
                elif p.get("type") == "input_audio":
                    audio_b = base64.b64decode(p["input_audio"]["data"])
    return (q.strip() or "Transcribe this audio."), audio_b


async def _stream_chat(body):
    async with _client.stream("POST", SGLANG + "/v1/chat/completions", json=body) as r:
        async for chunk in r.aiter_bytes():
            yield chunk


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
        b2 = dict(body); b2["messages"] = msgs2; b2.pop("tools", None); b2.pop("tool_choice", None); b2["stream"] = False
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
            return StreamingResponse(_stream_chat(body), media_type="text/event-stream")
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
    q, audio_b = _extract_audio_chat(msgs)
    path = "%s/_in_%s.wav" % (OUT, uuid.uuid4().hex)
    with open(path, "wb") as f:
        f.write(audio_b)
    prompt = "<longcat_user>" + q + "<longcat_audio_start><longcat_audio_end><longcat_assistant>"
    sp = {"max_new_tokens": int(body.get("max_tokens", 256)), "temperature": body.get("temperature", 0.2)}
    for k in ("top_p", "top_k", "frequency_penalty", "presence_penalty", "repetition_penalty", "stop"):
        if body.get(k) is not None:
            sp[k] = body[k]
    try:
        r = await _client.post(SGLANG + "/generate", json={"text": prompt, "audio_data": [path], "sampling_params": sp})
        rj, raw = _json_or_text(r)
    finally:
        try: os.remove(path)
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
