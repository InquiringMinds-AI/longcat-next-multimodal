#!/bin/bash
# LongCat-Next all-modality server (GB10 / w8a8_int8).
# SGLang runs on an internal port; an OpenAI-compatible gateway (all modalities) serves PORT.
# If EITHER process exits, the container is torn down (don't serve a dead backend) so an
# orchestrator/restart policy can recover it.
# Env: MODEL_PATH, PORT, MEM_FRACTION, MAX_TOTAL_TOKENS, LCN_YARN, LCN_RADIX.
# Allocator: expandable segments, so freed vision-encoder allocations return to the
# system instead of accumulating as fragmented CUDA segments. On unified-memory hosts
# (GB10/DGX Spark) that fragmentation consumes SYSTEM RAM across long image-bearing
# conversations until the node freezes — reported in the field, root-caused to the
# PyTorch CUDA caching allocator. Overridable by setting the var before launch.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INTERNAL="${SGLANG_INTERNAL_PORT:-30000}"
export SGLANG_INTERNAL_PORT="$INTERNAL"
export MODEL_PATH="${MODEL_PATH:-/workspace/model}"

# NGRAM spec decode and generation COEXIST as of 2026-08-10 — the old
# "LCN_NGRAM=1 implies LCN_AGENT=1" coupling is gone. A batch with an active
# image/audio generation falls back to plain decode (one token per step, the
# model's Python state machines run), and speculation resumes when it finishes;
# see patches/spec_gen_fallback.patch and research/FINDINGS.md for the three
# relayed-state bugs that had to be fixed to get there (input_ids, seq_lens,
# kv_committed_len).
# Validated with the strict idle KV-leak check ARMED: selftest 7/7, anthropic
# 5/5, degeneracy 6/6, and a 12-cycle generation soak (4000+ fallback steps, 3
# full image generations) with FLAT memory and zero leak reports.
# LCN_NGRAM_AGENT_COUPLE=1 restores the old always-paired behavior.
if [ "${LCN_NGRAM:-0}" = "1" ] && [ "${LCN_NGRAM_AGENT_COUPLE:-0}" = "1" ] \
   && [ "${LCN_AGENT:-0}" != "1" ]; then
  echo "[entrypoint] LCN_NGRAM_AGENT_COUPLE=1 — enabling LCN_AGENT=1 (legacy pairing)"
  export LCN_AGENT=1
fi

# Context length. Default is the model's NATIVE 128k (max_position_embeddings=131072). Set
# LCN_YARN=1 to extend to 256k via YaRN (RoPE factor 2) — opt-in because YaRN can slightly
# affect short-context / generation quality. KV is cheap here (MLA, ~16 KB/token), so the
# limiter is --mem-fraction-static (weights ~88 GB + KV pool); we raise it just enough.
OVERRIDE='{"architectures":["LongcatNextForCausalLM"]}'
DEFAULT_TOKENS=131072
# MEASURED BUDGET (128GB GB10, re-measured 2026-08-11 from live logs): weights land at
# 76.26GB (not the ~88GB an earlier version of this comment claimed), so at 0.72 the KV
# pool hits its --max-total-tokens CAP, not the fraction: the FULL 131072-token pool
# (3.94GB, exactly one native context at ~31.5KB/token MLA-compressed KV) allocates in
# every profile. The binding constraint is physical headroom, not the fraction: the
# image/audio GENERATION heads lazily allocate ~25GB on first use OUTSIDE
# --mem-fraction-static, and with both heads warm the box settles at ~3-4GB MemAvailable
# — one context is all that fits, by ~0.6GB. LCN_AGENT=1 disables generation at the
# gateway so the heads never allocate, and that ~25GB funds ~5 MORE full contexts.
# MEASURED RECIPE (2026-08-11): MAX_TOTAL_TOKENS=917504 MEM_FRACTION=0.88 ->
# pool 800557 tokens (24.05GB, 6.1 contexts), 7.5GB MemAvailable steady, and a 36k-token
# needle prompt prefilled at ~1.8k tok/s with exact retrieval. BOTH knobs are needed:
# the fraction applies to SGLang's DETECTED budget (~114GB at load start), not the
# physical 128GB, so 0.82 only bought 4.2 contexts. 0.91 would reach the full 7 but
# lands ~4GB free — the thin regime — for no qualitative gain. Not the default;
# deliberately opt-in for multi-session/radix-depth workloads.
DEFAULT_MEMFRAC=0.72
[ "${LCN_AGENT:-0}" = "1" ] && DEFAULT_MEMFRAC=0.75
if [ "${LCN_YARN:-0}" = "1" ]; then
  # Override rope_parameters (NOT rope_scaling): transformers 4.57 rebuilds rope_parameters from a
  # rope_scaling override and drops rope_theta, which the model reads -> KeyError. Setting
  # rope_parameters directly keeps rope_theta AND carries the yarn fields (also aliased to rope_scaling).
  OVERRIDE='{"architectures":["LongcatNextForCausalLM"],"max_position_embeddings":262144,"rope_parameters":{"rope_type":"yarn","rope_theta":10000000,"factor":2.0,"original_max_position_embeddings":131072}}'
  DEFAULT_TOKENS=262144
  DEFAULT_MEMFRAC=0.74
  [ "${LCN_AGENT:-0}" = "1" ] && DEFAULT_MEMFRAC=0.78
fi

# CUDA graphs (decode): the overlay's decode path was made capture-safe in the
# 2026-07 ngram rework — the layer's ignored-mask branch is now branch-free and
# the multimodal state machines are gated behind the gen-trigger latch (zero
# host syncs on steady-state text decode; generation batches veto graph replay
# via patches/decode_graph_gen_veto.patch and run eager). Prefill graphs stay
# OFF — the multimodal prefill forward (encoders, gen-entry detection) is
# host-driven and has not been made capture-safe.
# ON by default since 2026-08-10, on measurement: +3.8% decode untuned / +6.0%
# with the tuned MoE configs, and +13.6% AGGREGATE at concurrency 16 (166.6 ->
# 189.2 tok/s) for 0.17GB of extra capture memory. Capture costs ~30s of startup
# and ~1.7GB total; measured MemAvailable after generation warmup with graphs on
# was 9.95GB, so the all-modality budget absorbs it. Max-bs 32 (not 8) is what
# buys the concurrency win — the curve keeps climbing to 64 concurrent requests.
# LCN_CUDAGRAPH=0 restores the old always-eager behavior.
# (--cuda-graph-max-bs-decode, not --cuda-graph-max-bs: the latter is deprecated
# upstream and warns on every start.)
# Exported so GET /status reports the EFFECTIVE state. Before this export, a
# deployment that left the var unset ran with graphs ON (this default) while the
# gateway — reading the raw env — reported cudagraph "0": exactly the
# script-vs-live-process mismatch /status exists to prevent.
export LCN_CUDAGRAPH="${LCN_CUDAGRAPH:-1}"
GRAPH_FLAG="--cuda-graph-max-bs-decode ${LCN_CUDAGRAPH_BS:-32} --disable-prefill-cuda-graph"
[ "$LCN_CUDAGRAPH" = "0" ] && GRAPH_FLAG="--disable-cuda-graph"

# NGRAM lookup speculative decoding: reworked 2026-07 — verify batches now get
# correct hash-table geometry (patches/ngram_spec_verify.patch: TARGET_VERIFY
# column_starts + draft/accept table writes). Implies LCN_AGENT=1 (see above).
# CHAIN drafts only: the hash kernel walks history linearly, so bfs breadth is
# pinned to 1 (a branching tree would corrupt the hash context).
NGRAM_FLAGS=""
[ "${LCN_NGRAM:-0}" = "1" ] && NGRAM_FLAGS="--speculative-algorithm NGRAM --speculative-num-draft-tokens ${LCN_NGRAM_DRAFT:-4} --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1"

# KV cache dtype (opt-in): LCN_KV_DTYPE=fp8_e4m3 halves KV bytes -> ~2x token capacity
# at the same mem-fraction. Validate quality on your workload before trusting it.
KV_FLAGS=""
[ -n "${LCN_KV_DTYPE:-}" ] && KV_FLAGS="--kv-cache-dtype ${LCN_KV_DTYPE}"

# n-gram embedding EOS semantics: -1 (legacy cross-boundary hashing) by default — the
# checkpoint was trained/validated under it; sglang >=0.5.16's eos-exclusion measurably
# degrades image generation. Set LCN_NGRAM_EOS=<token_id> to opt into upstream behavior.
export LCN_NGRAM_EOS="${LCN_NGRAM_EOS:--1}"

# TTS onset conditioning (owner-validated defaults, 2026-07-31): inject 1 encoded-
# silence frame before the acoustic head's first free sample (absorbs the first-
# frame onset garble: "these all work"), and trim the rendered leading silence
# back to ~150ms (the model extends injected silence by momentum). 0/0 disables.
export LCN_TTS_SILENCE_FRAMES="${LCN_TTS_SILENCE_FRAMES:-1}"
export LCN_TTS_TRIM_LEAD_MS="${LCN_TTS_TRIM_LEAD_MS:-150}"
# Trailing twin of the lead trim. The model generates trailing silence AS CONTENT
# before (and while dithering around) its end flag — measured up to ~2.9s on short
# clips, telemetry 2026-08-14 — so the rendered tail is cut back to 250ms after the
# last active audio. Applies to the assembled .wav; a live stream has already sent
# its bytes. 0 disables.
export LCN_TTS_TRIM_TAIL_MS="${LCN_TTS_TRIM_TAIL_MS:-250}"

# int8 depth-head FFN: ON by default since 2026-08-14, on measurement + owner
# A/B ("all in all, the examples are all up to standard"): TTS -34-36%/frame
# (~2.2x slower-than-realtime -> ~1.4x), single image -7.9%, +3.3GB memory
# (bf16 FFN weights freed after per-slot int8 quantization). Attention and the
# per-level output heads stay bf16 — see int8_head_ffn.py for the measured
# scope. LCN_INT8_HEADS=0 restores the reference einsum path, byte-identical
# to pre-feature behavior. Exported so GET /status reports the EFFECTIVE state.
export LCN_INT8_HEADS="${LCN_INT8_HEADS:-1}"

# Per-level CUDA-graph replay of the depth-head forward: ON by default since
# 2026-08-14 — the generation loop was launch-latency-bound; graphs are
# math-identical by construction (capture-time replay-vs-eager torch.equal
# proof in lcn_head_graph.py) and measured −5.6% single image on top of the
# int8/SDPA wins. Any capture failure falls back to eager permanently for
# that head. LCN_HEAD_GRAPH=0 disables.
export LCN_HEAD_GRAPH="${LCN_HEAD_GRAPH:-1}"

# Radix (prefix) cache: ON by default — warm-prefix reuse is what makes agentic clients
# (Claude Code re-sends a ~15k-token system prompt every turn) responsive. Viable only
# WITH the expandable_segments allocator above (radix keeps KV resident, which amplified
# the fragmentation leak). LCN_RADIX=0 restores the old disabled behavior.
RADIX_FLAG="--disable-radix-cache"
[ "${LCN_RADIX:-1}" = "1" ] && RADIX_FLAG=""

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "${LCN_MODEL_NAME:-longcat-next}" \
  --port "$INTERNAL" --host 127.0.0.1 \
  --trust-remote-code \
  --json-model-override-args "$OVERRIDE" \
  --mem-fraction-static "${MEM_FRACTION:-$DEFAULT_MEMFRAC}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-$DEFAULT_TOKENS}" \
  --attention-backend flashinfer \
  $GRAPH_FLAG $RADIX_FLAG $NGRAM_FLAGS $KV_FLAGS --skip-server-warmup \
  --watchdog-timeout 600 &
SGLANG_PID=$!

# OpenAI-compatible gateway (text + image/audio/video understanding + image/audio gen + tools)
uvicorn gateway:app --host 0.0.0.0 --port "${PORT:-8090}" --app-dir /workspace/scripts &
GATEWAY_PID=$!

_term() { kill -TERM "$SGLANG_PID" "$GATEWAY_PID" 2>/dev/null; }
trap _term TERM INT

# Wait for whichever exits first; then tear down both (graceful, then forced) and exit non-zero.
wait -n "$SGLANG_PID" "$GATEWAY_PID"
echo "[entrypoint] a managed process exited — shutting down container"
kill -TERM "$SGLANG_PID" "$GATEWAY_PID" 2>/dev/null
sleep 8
kill -KILL "$SGLANG_PID" "$GATEWAY_PID" 2>/dev/null
exit 1
