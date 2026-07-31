<!-- LANG -->
**English** | [中文](README.zh-CN.md)

# LongCat-Next on DGX Spark (GB10) — all-modality serving

[Meituan **LongCat-Next**](https://huggingface.co/meituan-longcat) — a 75B-total / ~A3B-active
any-to-any multimodal MoE (LongCat-Flash backbone + MLA attention + N-gram over-embedding, native
RVQ tokenizers for vision and audio) — running **every modality through a single SGLang process on
a single NVIDIA GB10 system (`sm_121`)**, behind an **OpenAI-compatible API**:

| capability | OpenAI endpoint | status |
|---|---|:--:|
| text generation (+ `stream`) | `POST /v1/chat/completions` | ✅ |
| image / audio / video **understanding** | `POST /v1/chat/completions` | ✅ |
| **image generation** (text → image) | `POST /v1/images/generations` | ✅ |
| **voice-clone audio generation** | `POST /v1/audio/speech` | ✅ |
| tool / function calling | `POST /v1/chat/completions` (`tools`) | ✅ |

<sub>(LongCat-Next has no video *generation* — video is understanding-only.)</sub>

Quantized to **`w8a8_int8`** (8-bit weights + per-token int8 activations) — switching to 8-bit is what
made image and audio generation coherent (4-bit collapsed both). We didn't re-test 4-bit after later
fixes, so it's the *validated* setting, not a proven minimum. One self-contained ~90 GB model that runs
comfortably on a GB10, validated end-to-end by a [7/7 self-test](#self-test). See `examples/` for a
sample generated image and voice clip before you download anything.

> **How this was built** — getting all of this working took two separate fixes that looked like one
> bug (incoherent generation): switching to 8-bit, then a **structural** fix in the serving path (a
> dropped spatial anchor) — plus an adversarial multi-agent review that caught a silent MoE scaling
> bug. The full arc is in **[research/FINDINGS.md](research/FINDINGS.md)**.

> Built for the GB10 superchip (`sm_121`) — validated on a DGX Spark, expected to run on any
> GB10-based system (the dependency is the chip, not the product). The cu130 SGLang base is the one
> whose Triton compiles for `sm_121`; not expected to run on other GPUs.

## Prerequisites
- NVIDIA **GB10 system** (e.g. DGX Spark), driver + **NVIDIA Container Toolkit** (`--gpus all` works), **Docker**
- **~100 GB free disk** for the weights
- Run **headless** (screen off, remote/SSH) for maximum memory headroom

## 1. Download the weights (Hugging Face)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download InquiringMinds-AI/LongCat-Next-w8a8-int8-GB10 --local-dir ./longcat-next-gb10-weights
```
The weights directory is **self-contained** (~90 GB): quantized backbone + tokenizers + image
decoder + audio vocoder. Nothing else to fetch.

Then extract the codebook-embedding sidecar the generation heads need (one-time, ~931 MB,
sliced from the unquantized `embed_tokens` already in the shards):
```bash
python3 quantize/extract_codebook_embeddings.py ./longcat-next-gb10-weights
```
Without `codebook_embeddings.safetensors` the server still runs, but prior-level codebook
conditioning in the audio/image generation heads silently degrades to zero vectors
(watch for `codebook_embeddings.safetensors not found` in the logs).

## 2. Build the image
```bash
docker build -t longcat-next-gb10 .
```
Layers the LongCat-Next overlay + GB10 fixes onto `lmsysorg/sglang:v0.5.12.post1-cu130` (the base
pull is the only large download here).

## 3. Run the server
```bash
./run.sh ./longcat-next-gb10-weights
```
First start loads ~90 GB (a few minutes). When you see `The server is fired up and ready to roll!`,
the API is live on `http://localhost:8090` and is **OpenAI-compatible across every modality**
(works with the `openai` SDK / LangChain):

| modality | OpenAI endpoint |
|---|---|
| text | `POST /v1/chat/completions` |
| image / video / audio **understanding** | `POST /v1/chat/completions` (`image_url` / `video_url` / `input_audio` content parts) |
| **image generation** | `POST /v1/images/generations` |
| **voice-clone TTS** | `POST /v1/audio/speech` (`voice`: `en`, `zh`, or a container path to a reference clip) |

The native SGLang `/generate` is also exposed (passthrough); the bundled `gen_*`/`understand_*`
scripts use it. Generated files also land in `./outputs/`.

## Security

This server has **no built-in authentication**, so the defaults keep it off the network:

- **Loopback by default.** `run.sh` and `docker-compose.yml` publish the port on `127.0.0.1:8090`
  only — reachable from the host, not the LAN.
- **To expose it on a network**, set both an interface and a key:
  ```bash
  LCN_BIND=0.0.0.0 LCN_API_KEY=$(openssl rand -hex 24) ./run.sh ./longcat-next-gb10-weights
  ```
  With `LCN_API_KEY` set, every endpoint except `GET /health` requires `Authorization: Bearer <key>`.
  (`run.sh` warns if you bind off-loopback without a key.)
- **The native SGLang admin surface is not exposed.** The passthrough proxy is default-deny: only
  inference/read endpoints (`/generate`, `/get_model_info`, `/v1/models`, …) pass through; mutating
  control endpoints (`/flush_cache`, `/update_weights*`, profiling, etc.) return `404`.
- **TTS reference clips are path-contained.** A custom `voice` path must resolve under the bundled
  voices dir or the mounted output dir (or `LCN_VOICE_DIR`); arbitrary container paths are rejected.

## 4. Test each modality (OpenAI endpoints)

**Text**
```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"longcat-next","messages":[{"role":"user","content":"Name two oceans."}],"max_tokens":24}'
```

**Image generation** (returns base64 PNG, OpenAI images schema)
```bash
curl -s localhost:8090/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"A photograph of a red apple on a wooden table.","response_format":"b64_json"}'
```

**Voice-clone TTS** (returns audio/wav; `voice`=`en`|`zh`|a container path)
```bash
curl -s localhost:8090/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"The quick brown fox jumps over the lazy dog.","voice":"en"}' -o speech.wav
```

**Image / video / audio understanding** — `/v1/chat/completions` with an `image_url`,
`video_url`, or `input_audio` content part (standard OpenAI multimodal messages), e.g.:
```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"longcat-next","max_tokens":80,
  "messages":[{"role":"user","content":[
    {"type":"text","text":"Describe this image."},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}]}]}'
```

> Also available: the bundled scripts `gen_image.py`, `gen_audio.py`, `understand_video.py`
> (under `/workspace/scripts/`) and the native SGLang `/generate` endpoint.
> Text chat supports **`stream: true`** (SSE), like the OpenAI API.

## Self-test

Verify every modality works end-to-end on your machine:
```bash
docker exec longcat-next python3 /workspace/scripts/selftest.py
```
Prints PASS/FAIL for text, image gen, image understanding, audio gen, audio understanding, and video
understanding; exits non-zero if any fail.

## Context length

Defaults to the model's **native 128k** (`max_total_tokens` 131072). MLA makes the KV cache cheap
(~16 KB/token), so long context is nearly free — the limiter is just `--mem-fraction-static`, which is
set to fit the full 128k pool.

Set **`LCN_YARN=1`** to extend to **256k via YaRN** (RoPE factor 2). It's opt-in because YaRN can
slightly affect short-context / generation quality; the default keeps the unscaled 128k path. Both
modes pass the self-test and stay well under the GB10's memory headroom (128k peaks ~95 GB, 256k
~101 GB during a full generation pass). Verified: a 28k-token prompt recalls a fact planted at its start.

## Memory stability on unified memory (important)

Two fixes ship in the container/launchers — keep both if you write your own launcher:

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (entrypoint default). Without it, the
  vision encoder fragments the CUDA caching allocator across long image-bearing conversations and
  the fragmented segments are never returned — on unified-memory hosts (GB10) that eats *system*
  RAM until the node freezes or an OOM killer fires. Measured on an 80-turn multi-image soak:
  −7.6 GB and still declining without the flag, −2.9 GB converging to flat with it.
- **`--shm-size=32g`** (run.sh / compose / your `docker run`). SGLang moves multimodal pixel
  tensors between processes via `/dev/shm`; Docker's 64 MB default SIGBUS-crashes the server on the
  first request that carries several images. tmpfs allocates lazily — the ceiling is free.

Also measured: the image/audio **generation** heads lazily allocate **~25 GB on first use**,
outside `--mem-fraction-static`. The all-modality default budgets for that (see Agent mode for the
alternative).

## Agent mode & Claude Code

`LCN_AGENT=1` trades the generation heads for the **full native context**: image/audio *generation*
endpoints return 403 (so their ~25 GB is never allocated) and `MEM_FRACTION` defaults to 0.75,
which funds the entire 131072-token KV pool (~3.9 GB; the all-modality default 0.72 profiles to
~55k tokens). Understanding of image/audio/video **input still works**. With the radix cache this
is the profile for agentic clients that resend a large system prompt every turn — measured: a
15.6k-token prefix costs ~5.9 s cold and **~0.36 s warm** (16×).

The gateway also speaks the **Anthropic Messages API** (`POST /v1/messages`, streaming and tool
calling included), so Anthropic-native clients work directly. Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://<host>:8090
export ANTHROPIC_AUTH_TOKEN=<your LCN_API_KEY>
export ANTHROPIC_MODEL=longcat-next
export ANTHROPIC_SMALL_FAST_MODEL=longcat-next
claude
```

Honest calibration: the model handles real agentic tool loops (write/read/run, error recovery,
multi-turn) but is a 75B-A3B — expect occasional path/argument slips on long random strings, and
review its work. `test/test_anthropic.py` self-tests the route end-to-end.

## Tuning (env vars)

Set at `docker run -e …` (or in `docker-compose.yml`):
`MEM_FRACTION` (0.72; 0.74 under `LCN_YARN`; +0.03 with `LCN_AGENT=1`),
`MAX_TOTAL_TOKENS` (131072; 262144 under `LCN_YARN`),
`LCN_YARN` (0), `LCN_AGENT` (0) — full-context agent profile, disables generation endpoints,
`LCN_RADIX` (1) — radix/prefix cache; set 0 to restore the old disabled behavior,
`LCN_KV_DTYPE` (unset) — e.g. `fp8_e4m3` halves KV bytes → ~2× token capacity at the same
mem-fraction; validate output quality on your workload first,
`LCN_CUDAGRAPH` (0) — **leave off**: graph capture fails on this port ("operation failed during
capture"); the gate exists for retesting after overlay changes,
`LCN_NGRAM` (0) — **leave off**: NGRAM speculative decoding CUDA-faults in the model's n-gram
input embedding (draft positions violate its history-hash indexing); needs a draft-aware layer,
`IMAGE_GEN_CFG_SCALE` (3.0),
`IMAGE_GEN_TEMPERATURE`/`IMAGE_GEN_TOP_K`/`IMAGE_GEN_TOP_P`, `AUDIO_GEN_TEMPERATURE`/`AUDIO_GEN_TOP_K`,
`REFINER_STEPS` (10; raise toward 28 for max image fidelity at ~1.5× latency),
and `LCN_VERBOSE=1` for per-step debug logging.

## Example outputs

See `examples/` for a sample generated image and voice-clone clip, so you know the expected quality
before downloading the weights.

## Troubleshooting

- **Cold start ~5–8 min** (loads ~90 GB). `GET /health` returns `503 {"status":"loading"}` until
  ready, then `200 {"status":"ok"}`. A `503 "backend unavailable"` from any endpoint means it's still loading.
- **Box powers off mid-run** → you're using too much GPU memory; serve headless, don't run other
  heavy GPU work alongside, and don't raise `MEM_FRACTION`.
- **First image is slow (~4–5 min)** — 1369 visual tokens + diffusion refine; audio is near-real-time.

## Notes
- **Optimized for headless GB10 operation** at time of publishing — serve with the screen off /
  remote-only for maximum memory headroom.
- **Audio length is model-decided** — output runs as long as the text requires, with no task-length
  floor; a ~40s (1000-frame) safety backstop only guards against runaway generation.
- **First image** ~4–5 min (1369 visual tokens + diffusion refine); audio is near-real-time.

## Repository layout

```
.                       the runnable package (this README, Dockerfile, run.sh, …)
├── gateway.py          OpenAI-compatible gateway fronting SGLang (all modalities + tools)
├── anthropic_route.py  Anthropic Messages API (/v1/messages) — Claude Code & friends
├── longcat_tools.py    tool-calling: TS-namespace prompt build + <longcat_tool_call> parse (both syntaxes)
├── entrypoint.sh       SGLang + gateway process supervision
├── new_files/          the LongCat-Next SGLang overlay (models / layers / processors)
├── patches/            container build patches
├── quantize/           the w8a8_int8 export tooling (how the weights were made)
├── test/               selftest.py + per-modality example clients
├── voices/             TTS reference clips (en: public-domain LibriVox, zh: Meituan MIT)
├── examples/           a sample generated image + voice clip
└── research/           HOW THIS WAS BUILT — the engineering narrative + proof tooling
    ├── FINDINGS.md       the debugging arc (start here)
    └── oracle/           the bnb-int8 capability proof + soundness probes
```

## Credits & license
Model: **Meituan LongCat-Next** (MIT). Serving stack: **SGLang**. English demo voice: public-domain
**LibriVox** narration. Chinese demo voice: Meituan's LongCat example clip (MIT). See [LICENSE](LICENSE).
