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
                "sampling_params": {"max_new_tokens": 2048, "temperature": 0.5,
                                    "top_k": 5, "top_p": 0.85}})
        if r.status_code != 200:
            raise RuntimeError("backend error: " + r.text[:200])
        rj, raw = _json_or_text(r)
        if rj is None:
            raise RuntimeError("backend error: " + raw[:200])
        path = "%s/longcat_tts_%s.wav" % (OUT, _san(rj.get("meta_info", {}).get("id", "")))
        # A missing artifact is the FAILURE the endpoint 500s on — prewarm must not
        # swallow it and report "ready" over a broken TTS path.
        if await _read_when_ready(path) is None:
            raise RuntimeError("audio generation produced no output")
        _discard_artifact(path)
        _discard_artifact(path[:-len(".wav")] + ".pcm.part")  # streamed-PCM sibling
        _prewarm_state["audio_s"] = round(_t.monotonic() - t0, 1)
        logger.info("[prewarm] audio path warm in %.1fs", _prewarm_state["audio_s"])
    except Exception as e:                     # noqa: BLE001
        # Image is already warm, which is the expensive half — report the miss, keep going.
        _prewarm_state.update(error=f"audio: {e}")
        logger.warning("[prewarm] audio warmup failed (serving continues): %s", e)

    # "ready" must mean BOTH paths verified. A recorded audio error downgrades to
    # "degraded" rather than being readable only by someone who checks the error
    # field — a status that says ready while a path is broken is the same shape as
    # a warmup that looks warm without being exercised.
    _prewarm_state.update(status="degraded" if _prewarm_state["error"] else "ready")
    logger.info("[prewarm] complete (%s): image=%ss audio=%ss",
                _prewarm_state["status"], _prewarm_state["image_s"], _prewarm_state["audio_s"])


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
            # Default mirrors the entrypoint's (which also exports the effective value,
            # so this fallback only fires when the gateway runs outside the entrypoint).
            "cudagraph": os.environ.get("LCN_CUDAGRAPH", "1"),
            "ngram": os.environ.get("LCN_NGRAM", "0"),
            "yarn": os.environ.get("LCN_YARN", "0"),
            "kv_dtype": os.environ.get("LCN_KV_DTYPE", "") or "bf16 (default)",
            "head_batch": os.environ.get("LCN_HEAD_BATCH", "1") != "0",
            "refiner_fast": os.environ.get("LCN_REFINER_FAST", "1") != "0",
            "refiner_steps": os.environ.get("REFINER_STEPS", "10"),
            "refiner_cfg_range": os.environ.get("LCN_REFINER_CFG_RANGE", "1.0,0.0"),
            "tts_stream": os.environ.get("LCN_TTS_STREAM", "1") != "0",
            "tts_multi": os.environ.get("LCN_TTS_MULTI", "1") != "0",
            "tts_chunk_frames": os.environ.get("LCN_TTS_CHUNK_FRAMES", "25"),
            "int8_heads": os.environ.get("LCN_INT8_HEADS", "audio"),
            "head_graph": os.environ.get("LCN_HEAD_GRAPH", "1") == "1",
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
    # The lcntts rid prefix marks a RECITATION request to the model's content
    # stops. The model also detects the TTS instruction in the prompt, but that
    # check reads only the final chunked-prefill extend — a >8192-token input
    # would escape it; the rid survives chunking.
    sg_body = {"text": prompt, "audio_data": [ref],
               "rid": "lcntts" + uuid.uuid4().hex,
               "sampling_params": {"max_new_tokens": 2048, "temperature": 0.5,
                                   "top_k": 5, "top_p": 0.85}}
    if body.get("stream"):
        return await _audio_speech_stream(sg_body, body)
    _reqtext = _write_reqtext_sidecar(sg_body["rid"], text)
    try:
        async with _gen_slots:
            r = await _client.post(SGLANG + "/generate", json=sg_body)
    except httpx.ConnectError:
        _discard_artifact(_reqtext)
        return JSONResponse({"error": {"message": "backend unavailable (model may still be loading)"}}, status_code=503)
    _discard_artifact(_reqtext)  # the model consumed it at prefill; this covers paths that never did
    rj, raw = _json_or_text(r)
    if rj is None:
        return JSONResponse({"error": {"message": "backend error: " + raw[:200]}}, status_code=502)
    rid = _san(rj.get("meta_info", {}).get("id", ""))
    path = "%s/longcat_tts_%s.wav" % (OUT, rid)
    data = await _read_when_ready(path)
    if data is None:
        return JSONResponse({"error": {"message": "audio generation produced no output"}}, status_code=500)
    _discard_artifact(path)
    # The streaming vocoder (LCN_TTS_STREAM=1, default) also writes <rid>.pcm.part
    # and the model leaves it for the gateway to clean up — without this, every
    # non-streaming TTS request leaked one PCM file into the output dir.
    _discard_artifact(path[:-len(".wav")] + ".pcm.part")
    return Response(content=data, media_type="audio/wav")


def _write_reqtext_sidecar(rid, text):
    """Authoritative recitation text for the model's transcript coverage stop.

    The model's in-prompt capture reads only the prefill EXTEND region, and the radix
    cache can shrink that to a single token on a repeated prompt — the coverage stop
    then mis-reads honest recitation as invention and closes the request with no audio.
    The sidecar survives any cache split; the model reads and deletes it when it opens
    the audio-generation state. Returns the path (for belt-and-braces cleanup)."""
    path = "%s/longcat_tts_%s.reqtext" % (OUT, rid)
    try:
        with open(path, "w") as f:
            f.write(text)
    except OSError:
        pass  # coverage stop falls back to the extend-derived text
    return path


def _wav_stream_header(sample_rate=24000, bits=16, channels=1):
    """A WAV header with unknown length (0xFFFFFFFF chunk sizes) for chunked streaming.
    Players and decoders accept it; the true length is whatever arrives before EOF."""
    import struct
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                    byte_rate, block_align, bits)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


async def _audio_speech_stream(sg_body, body):
    """Stream TTS as it is generated (requires LCN_TTS_STREAM=1 on the model side).

    The model appends PCM to <rid>.pcm.part DURING generation; the gateway supplies the
    rid (SGLang's /generate accepts a client rid), fires the backend call as a task, and
    tails the file concurrently — first audio reaches the client ~2s in, instead of after
    the full generation (~2.6s of compute per second of audio).

    The finished .wav (assembled from the SAME streamed chunks) is the completion signal;
    the model leaves the .part in place for the gateway to finish draining, and the
    gateway deletes both. NOTE: generation runs ~2.2x slower than realtime on this box,
    so a client that plays immediately will drain its buffer mid-clip — buffer ~half the
    expected clip, or accept the pause. Time-to-first-audio is what this buys.
    """
    rid = "lcntts" + uuid.uuid4().hex  # lcntts prefix = recitation marker (see audio_speech)
    sg_body = dict(sg_body)
    sg_body["rid"] = rid
    part = "%s/longcat_tts_%s.pcm.part" % (OUT, rid)
    wav = "%s/longcat_tts_%s.wav" % (OUT, rid)
    reqtext = _write_reqtext_sidecar(rid, str(body.get("input", "")))
    fmt = str(body.get("response_format", "wav")).lower()

    # Fire the backend BEFORE committing the response: a dead backend errors within
    # milliseconds, and catching that here turns "HTTP 200 with an empty WAV" into an
    # honest 502. The slot is acquired manually so it survives into the generator.
    await _gen_slots.acquire()
    task = asyncio.create_task(_client.post(SGLANG + "/generate", json=sg_body))
    done, _pending = await asyncio.wait({task}, timeout=0.4)
    if done:
        err = None
        try:
            r = task.result()
            if r.status_code != 200:
                err = "backend %d: %s" % (r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            err = str(e)[:200]
        if err:
            _gen_slots.release()
            _discard_artifact(part)
            _discard_artifact(reqtext)
            return JSONResponse({"error": {"message": err}}, status_code=502)

    # Slot/artifact ownership: normally gen()'s finally releases. But a client
    # that disconnects before the response body is ever consumed leaves gen()
    # UNSTARTED — its finally never runs. The done-callback covers that orphan
    # case at generation end; the once-guard prevents a double release.
    _state = {"owned": False, "released": False}

    def _release_once():
        if not _state["released"]:
            _state["released"] = True
            _gen_slots.release()
            _discard_artifact(part)
            _discard_artifact(wav)
            _discard_artifact(reqtext)

    task.add_done_callback(lambda _t: None if _state["owned"] else _release_once())

    async def gen():
        _state["owned"] = True
        try:
            if fmt != "pcm":
                yield _wav_stream_header()
            pos = 0
            while True:
                try:
                    size = os.path.getsize(part)
                except OSError:
                    size = 0
                if size > pos:
                    with open(part, "rb") as f:
                        f.seek(pos)
                        chunk = f.read(size - pos)
                    pos += len(chunk)
                    yield chunk
                    continue
                if task.done():
                    # Generation over: drain whatever landed between the last read
                    # and finalize, then stop. The .wav's existence marks finalize.
                    if os.path.exists(wav):
                        try:
                            final_size = os.path.getsize(part)
                        except OSError:
                            final_size = pos
                        if final_size > pos:
                            continue
                        # If the finalized wav carries MORE PCM than was streamed —
                        # LCN_TTS_STREAM=0 server-side (no .part is ever written) or
                        # the stream path failed and the model fell back to a full
                        # decode — serve the remainder from the wav payload instead
                        # of ending a header-only/truncated stream and deleting the
                        # only good audio. (The wav's TAIL may legitimately be
                        # shorter than the part: the tail trim; that is not a
                        # shortfall.) 44 = the standard PCM WAV header the model's
                        # finalize writes.
                        try:
                            with open(wav, "rb") as f:
                                payload = f.read()[44:]
                            if len(payload) > pos:
                                logger.info("[tts-stream] rid=%s serving %d bytes from "
                                            "the finalized wav (streamed %d)",
                                            rid, len(payload) - pos, pos)
                                yield payload[pos:]
                                pos = len(payload)
                        except OSError:
                            pass
                        break
                    # task done, no wav: with the wav fallback above this is a
                    # GENUINE backend failure. The 200 is committed, so abort the
                    # connection (client sees a reset/truncation) rather than
                    # faking a clean end-of-stream.
                    err = None
                    try:
                        r = task.result()
                        if r.status_code != 200:
                            err = f"backend {r.status_code}"
                    except Exception as e:  # noqa: BLE001
                        err = str(e)[:120]
                    logger.warning("[tts-stream] rid=%s FAILED without a finalized wav "
                                   "(%s) — aborting stream after %d PCM bytes",
                                   rid, err or "no error", pos)
                    raise RuntimeError("tts stream failed: " + (err or "no wav"))
                await asyncio.sleep(0.1)
        finally:
            if not task.done():
                task.cancel()
            _release_once()

    media = "audio/wav" if fmt != "pcm" else "application/octet-stream"
    return StreamingResponse(gen(), media_type=media)


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
    prompt, blobs, dropped = extract_audio_chat(msgs)
    if not blobs:
        return JSONResponse({"error": {"message": "no decodable audio in request; "
                            "input_audio.data must be base64"}}, status_code=400)
    if dropped and os.environ.get("LCN_LENIENT_MEDIA", "0").strip() != "1":
        # Same fail-loud policy as the processor's audio/video decode: a request that
        # attached media the server cannot read gets an ERROR, not a fluent answer
        # quietly generated from the readable subset. Previously only the all-clips-bad
        # case 400'd, so one good clip + one corrupt clip sailed through on the good
        # clip alone with nothing telling the caller. LCN_LENIENT_MEDIA=1 restores the
        # drop-and-continue behaviour, gateway and processor together.
        return JSONResponse({"error": {"message":
                            "%d audio clip(s) could not be decoded (input_audio.data must "
                            "be base64); refusing to answer from partial media. Set "
                            "LCN_LENIENT_MEDIA=1 server-side to allow." % dropped}},
                            status_code=400)
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
