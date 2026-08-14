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
| **voice-clone TTS** (multi-sentence, + `stream`) | `POST /v1/audio/speech` | ✅ |
| tool / function calling | `POST /v1/chat/completions` (`tools`) | ✅ |
| Anthropic Messages API (Claude Code) | `POST /v1/messages` | ✅ |

<sub>(LongCat-Next has no video *generation* — video is understanding-only.)</sub>

Quantized to **`w8a8_int8`** (8-bit weights + per-token int8 activations) — switching to 8-bit is what
made image and audio generation coherent (4-bit collapsed both). One self-contained ~90 GB model that
runs comfortably on a GB10, validated end-to-end by a [7/7 self-test](#self-test). See `examples/`
for a sample generated image and voice clip before you download anything.

> **How this was built** — the debugging and optimization arc (two fixes that looked like one bug,
> an adversarial multi-agent review, a launch-latency hunt that overturned two of its own recorded
> verdicts, and every performance number below with the measurement that produced it) lives in
> **[research/FINDINGS.md](research/FINDINGS.md)**. Claims in this README that carry a number are
> measured there, not estimated.

> Built for the GB10 superchip (`sm_121`) — validated on a DGX Spark, expected to run on any
> GB10-based system (the dependency is the chip, not the product). The cu130 SGLang base
> (`v0.5.16-cu130`) is the one whose Triton compiles for `sm_121`; not expected to run on other GPUs.

## Performance (measured on GB10, single request unless noted)

| path | at first publish | now (defaults) | how |
|---|---|---|---|
| single image, warm (1040×1040 + refine) | ~4–5 min | **~2.4 min** | refiner guidance-off default, per-level head CUDA graphs, dense-SDPA head attention |
| TTS generation rate | ~2.2–2.4× slower than realtime | **~1.4×** | int8 audio-head FFN (−34–36 %/frame) |
| TTS first audio (streaming) | ~32 s (whole clip) | **~6.5 s** | sliding-window chunk vocoding, streamed PCM |
| text decode | — | +6 % (single), **+13.6 % aggregate @16 concurrent** | CUDA graphs default-on, max-bs 32 |
| warm agent prefix (15.6k tokens) | ~5.9 s | **~0.36 s** | radix cache default-on |

What ships **on by default** because it measured faster with no quality cost: CUDA graphs
(`LCN_CUDAGRAPH=1`), per-level generation-head graph replay (`LCN_HEAD_GRAPH=1`, math-identical by
capture-time proof), dense-SDPA head attention (bit-identical), radix cache, refiner guidance-off
(`LCN_REFINER_CFG_RANGE=1.0,0.0`, −17 % per image, output human-adjudicated "fine").

What ships on by default with a **known, human-adjudicated trade**: `LCN_INT8_HEADS=audio` — the
audio generation head runs int8 (the TTS speedup above; no audible change found). The **visual**
head deliberately stays bf16: int8 there measurably increased spatial-geometry failures in a
5-vs-5 same-prompt comparison. `both` re-enables full int8 (~8 % faster images) if you accept that.
An int4 audio variant exists in the code and was **rejected on listening** — recorded as a dead end.

Known ceiling: TTS cannot reach faster-than-realtime on this box at acceptable quality — int4 was
the lever that could have closed the gap and it failed the ear test. ~1.4× with streaming is the
honest steady state.

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
Layers the LongCat-Next overlay + GB10 fixes onto `lmsysorg/sglang:v0.5.16-cu130` (the base
pull is the only large download here).

## 3. Run the server
```bash
./run.sh ./longcat-next-gb10-weights
```
First start loads ~90 GB (a few minutes). When you see `The server is fired up and ready to roll!`,
the API is live on `http://localhost:8090` and is **OpenAI-compatible across every modality**
(works with the `openai` SDK / LangChain). `run.sh` forwards the whole `LCN_*`/tuning environment,
so `LCN_YARN=1 ./run.sh …` works as expected.

The native SGLang `/generate` is also exposed (passthrough); the bundled `gen_*`/`understand_*`
scripts use it. Generated files also land in `./outputs/` **only if `LCN_KEEP_ARTIFACTS=1`** —
by default artifacts are served to the client and deleted.

## Choosing a serving profile

| you are serving… | set | what you get |
|---|---|---|
| everything (default) | *(nothing)* | all modalities; full native 128k context; generation heads lazily allocate ~22 GB on first use (or at startup with `LCN_PREWARM=1`) |
| an agentic / text client (Claude Code, tool loops) | `LCN_AGENT=1` | generation endpoints 403 so their ~22 GB is never allocated; understanding still works; radix cache makes resent system prompts ~free |
| many long sessions at once | `LCN_AGENT=1 MAX_TOTAL_TOKENS=917504 MEM_FRACTION=0.88` | **~6 full 128k contexts** of KV pool (measured: 800 557 tokens, 7.5 GB headroom steady, 36k-token prompt prefilled at ~1.8k tok/s with exact recall) |
| 256k context | `LCN_YARN=1` | RoPE-YaRN ×2; opt-in because YaRN can slightly affect short-context / generation quality |

KV is MLA-compressed at **~31.5 KB/token** — one full 128k context is 3.94 GB, which is why the
multi-context recipe is possible at all. The default `MEM_FRACTION=0.72` already fits the full
131 072-token pool; the binding constraint on this box is *physical* headroom for the generation
heads, not the fraction. **Do not raise `MEM_FRACTION` casually** — the GB10 hard-powers-off
(needs a physical power button) when total memory runs out; there is no graceful OOM.

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
Add **`"stream": true`** to receive audio as it is generated (WAV with a streaming header, or raw
PCM s16le/24 kHz with `"response_format":"pcm"`) — first audio in ~6.5 s instead of after the whole
clip. **Multi-sentence input is spoken in full**: the model recites sentence-by-sentence in rounds
and chooses its own stop when the text is exhausted (the serving loop guards against re-reads,
invented continuations, and runaway rounds — recitation semantics apply only to TTS-shaped
requests, so open-ended voice generation is not constrained).

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
Prints PASS/FAIL for text, image gen, image understanding, audio gen, audio understanding, video
understanding, and tool calling; exits non-zero if any fail. PASS on the generation checks means
*well-formed* output — look at the image and listen to the audio yourself before trusting a change.

## Agent mode & Claude Code

`LCN_AGENT=1` trades the generation heads for guaranteed headroom: image/audio *generation*
endpoints return 403 (so their ~22 GB is never allocated). Understanding of image/audio/video
**input still works**. With the radix cache this is the profile for agentic clients that resend a
large system prompt every turn — measured: a 15.6k-token prefix costs ~5.9 s cold and
**~0.36 s warm** (16×). For many concurrent sessions, add the multi-context recipe from
[Choosing a serving profile](#choosing-a-serving-profile).

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

## Operating the server (humans and agents)

**`GET /health`** — `503 {"status":"loading"}` until the backend is up, then `200 {"status":"ok"}`.
Never requires auth.

**`GET /status`** — the machine-readable source of truth for the *effective* configuration: every
tuning flag as the running process sees it (not as your launcher intended), plus the prewarm state.
If a deployment mystery involves "which config is this box actually running", read `/status`, not
the launch script.

**`LCN_PREWARM=1`** runs one real image generation and one short TTS at startup, so the ~22 GB
lazy head allocation and first-call compile costs are paid before the first user request.
`/status → prewarm.status` goes `pending → running → ready`; **`degraded` means a generation path
failed its warmup** — the server still serves, but treat it as a broken path, not a warning.

**Failure signatures → causes:**

| symptom | cause / action |
|---|---|
| `503 backend unavailable` on any endpoint | still loading (~5–8 min cold); poll `/health` |
| `403` from images/audio-speech | `LCN_AGENT=1` — generation is disabled by profile, not broken |
| first image takes ~3.5 min, later ones ~2.4 min | first call pays head allocation + graph capture; use `LCN_PREWARM=1` to front-load it |
| whole box powers OFF (needs physical power button) | total memory exhausted — lower ambitions, never raise `MEM_FRACTION` to "fix" capacity; serve headless; don't co-run GPU work |
| generated file missing from `./outputs/` | intended — artifacts are deleted after serving unless `LCN_KEEP_ARTIFACTS=1` |

## Tuning (env vars)

All forwarded by `run.sh` and readable back from `GET /status`. Defaults in parentheses.

**Capacity / profile**
- `MEM_FRACTION` (0.72; 0.74 under YaRN; agent profiles +0.03) — SGLang static fraction. See the
  power-off warning above before raising.
- `MAX_TOTAL_TOKENS` (131072; 262144 under YaRN) — KV pool size; `917504` with `MEM_FRACTION=0.88`
  and `LCN_AGENT=1` is the measured multi-context recipe.
- `LCN_AGENT` (0) — agent profile: generation endpoints 403, head memory never allocated.
- `LCN_YARN` (0) — 256k context via YaRN ×2.
- `LCN_RADIX` (1) — radix/prefix cache. Needs the `expandable_segments` allocator (entrypoint
  default) to be leak-safe on unified memory.
- `LCN_KV_DTYPE` (unset) — e.g. `fp8_e4m3` halves KV bytes → ~2× token capacity; validate quality
  on your workload first.

**Speed (all measured; see FINDINGS for the runs)**
- `LCN_CUDAGRAPH` (1) + `LCN_CUDAGRAPH_BS` (32) — decode CUDA graphs. +6 % single-stream, +13.6 %
  aggregate at 16 concurrent. `0` restores eager.
- `LCN_HEAD_GRAPH` (1) — per-level CUDA-graph replay of the generation heads. Math-identical
  (capture-time replay-vs-eager equality check; falls back to eager on any capture failure).
- `LCN_INT8_HEADS` (`audio`) — per-head int8 FFN: `audio` | `visual` | `both` | `0`. `audio` is the
  adjudicated default (TTS −34–36 %/frame, no audible change); `both` adds ~8 % image speed at a
  measured spatial-quality cost; `audio4` (int4) exists and was rejected on listening.
- `LCN_NGRAM` (0) — N-gram speculative decoding (chain drafts, `LCN_NGRAM_DRAFT`=4). Opt-in;
  coexists with generation (speculation pauses during a generation batch, resumes after).

**Image generation**
- `LCN_REFINER_CFG_RANGE` (`1.0,0.0` = guidance off, −17 %/image, output adjudicated equal;
  `0.0,1.0` restores the original guided refiner) · `REFINER_STEPS` (10; toward 28 for maximum
  fidelity at ~1.5× latency) · `IMAGE_GEN_CFG_SCALE` (3.0) ·
  `IMAGE_GEN_TEMPERATURE` / `IMAGE_GEN_TOP_K` / `IMAGE_GEN_TOP_P`.

**TTS**
- `LCN_TTS_STREAM` (1) — sliding-window chunk vocoding + streamed PCM; the final `.wav` is
  assembled from exactly the streamed bytes.
- `LCN_TTS_MULTI` (1) — multi-sentence rounds (see the TTS section). `0` restores one-round-only.
- `LCN_TTS_SILENCE_FRAMES` (1) and `LCN_TTS_TRIM_LEAD_MS` (150) — onset conditioning: absorb the
  model's first-frame garble, then trim the rendered lead-in.
- `LCN_TTS_TRIM_TAIL_MS` (250) — the model generates trailing silence *as content*; the assembled
  wav is cut back to this much tail. `0` disables.
- `AUDIO_GEN_TEMPERATURE` / `AUDIO_GEN_TOP_K`.

**Operations**
- `LCN_PREWARM` (0) — warm both generation paths at startup; see Operating the server.
- `LCN_KEEP_ARTIFACTS` (0) — keep generated files in the output dir instead of deleting after serving.
- `LCN_VERBOSE` (0) — per-step generation debug logging.
- `LCN_MODEL_NAME` (`longcat-next`), `LCN_VOICE_DIR`, `LCN_NGRAM_EOS` (−1, legacy hashing —
  the checkpoint was trained under it).

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

## Example outputs

See `examples/` for a sample generated image and voice-clone clip, so you know the expected quality
before downloading the weights.

## Notes
- **Optimized for headless GB10 operation** — serve with the screen off / remote-only for maximum
  memory headroom.
- **Audio length is model-decided** — output runs as long as the text requires; a ~40 s
  (1000-frame) safety backstop only guards against runaway generation.
- **Image quality calibration**: simple compositions are reliable; dense scenes (a full café,
  crowds) show the model's compositional limits at this quant. That is the checkpoint, not the
  serving stack — the serving defaults here were each gated on not making it worse.

## Repository layout

```
.                       the runnable package (this README, Dockerfile, run.sh, …)
├── gateway.py          OpenAI-compatible gateway fronting SGLang (all modalities + tools + streaming TTS)
├── anthropic_route.py  Anthropic Messages API (/v1/messages) — Claude Code & friends
├── longcat_tools.py    tool-calling: TS-namespace prompt build + <longcat_tool_call> parse (both syntaxes)
├── entrypoint.sh       SGLang + gateway process supervision + the tuning-default surface
├── new_files/          the LongCat-Next SGLang overlay (models / layers / processors,
│                       incl. int8_head_ffn.py and lcn_head_graph.py — the measured speed levers)
├── patches/            container build patches
├── quantize/           the w8a8_int8 export tooling (how the weights were made)
├── test/               selftest.py + per-modality example clients + unit gates
├── voices/             TTS reference clips (en: public-domain LibriVox, zh: Meituan MIT)
├── examples/           a sample generated image + voice clip
└── research/           HOW THIS WAS BUILT — the engineering narrative + proof tooling
    ├── FINDINGS.md       the full arc: bugs, fixes, benchmarks, overturned verdicts (start here)
    ├── int8_heads/       head quantization + launch-latency benches and trace analyzers
    ├── tts_streaming/    the chunk-vocoding quality-gate harness
    └── oracle/           the bnb-int8 capability proof + soundness probes
```

## Credits & license
Model: **Meituan LongCat-Next** (MIT). Serving stack: **SGLang**. English demo voice: public-domain
**LibriVox** narration. Chinese demo voice: Meituan's LongCat example clip (MIT). See [LICENSE](LICENSE).
