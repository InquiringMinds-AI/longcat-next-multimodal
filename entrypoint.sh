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

# Context length. Default is the model's NATIVE 128k (max_position_embeddings=131072). Set
# LCN_YARN=1 to extend to 256k via YaRN (RoPE factor 2) — opt-in because YaRN can slightly
# affect short-context / generation quality. KV is cheap here (MLA, ~16 KB/token), so the
# limiter is --mem-fraction-static (weights ~88 GB + KV pool); we raise it just enough.
OVERRIDE='{"architectures":["LongcatNextForCausalLM"]}'
DEFAULT_TOKENS=131072
# MEASURED BUDGET (128GB GB10): image/audio GENERATION heads lazily allocate ~25GB on
# first use, OUTSIDE --mem-fraction-static. All-modality serving therefore keeps the
# static fraction at 0.72 (~55k-token KV pool, ~5GB headroom after generation warmup).
# LCN_AGENT=1 (agentic/understanding profile): generation endpoints are disabled at the
# gateway, freeing that ~25GB to fund the FULL native context — 0.75 = 131072 tokens
# (~3.9GB KV). Raising the fraction in an all-modality deployment ends near 1GB free
# after the first image generation — global-OOM territory on unified memory.
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

# CUDA graphs: LEAVE DISABLED — capture fails on this model port ("operation failed
# during capture" even at bs 8; the overlay's custom decode path is not capture-safe).
# The env gate remains for retesting after engine/overlay changes.
GRAPH_FLAG="--disable-cuda-graph"
[ "${LCN_CUDAGRAPH:-0}" = "1" ] && GRAPH_FLAG="--cuda-graph-max-bs ${LCN_CUDAGRAPH_BS:-8}"

# NGRAM lookup speculative decoding: BROKEN on this port — the model's n-gram input
# embedding (n_gram_embedding.py) hits a CUDA illegal memory access when the NGRAM
# worker's verify batches flow through it (draft positions violate its history-hash
# indexing). Gate kept for retesting if the embedding layer is made draft-aware.
NGRAM_FLAGS=""
[ "${LCN_NGRAM:-0}" = "1" ] && NGRAM_FLAGS="--speculative-algorithm NGRAM --speculative-num-draft-tokens ${LCN_NGRAM_DRAFT:-4}"

# KV cache dtype (opt-in): LCN_KV_DTYPE=fp8_e4m3 halves KV bytes -> ~2x token capacity
# at the same mem-fraction. Validate quality on your workload before trusting it.
KV_FLAGS=""
[ -n "${LCN_KV_DTYPE:-}" ] && KV_FLAGS="--kv-cache-dtype ${LCN_KV_DTYPE}"

# n-gram embedding EOS semantics: -1 (legacy cross-boundary hashing) by default — the
# checkpoint was trained/validated under it; sglang >=0.5.16's eos-exclusion measurably
# degrades image generation. Set LCN_NGRAM_EOS=<token_id> to opt into upstream behavior.
export LCN_NGRAM_EOS="${LCN_NGRAM_EOS:--1}"

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
