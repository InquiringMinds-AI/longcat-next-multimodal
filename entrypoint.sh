#!/bin/bash
# LongCat-Next all-modality server (GB10 / w8a8_int8).
# SGLang runs on an internal port; an OpenAI-compatible gateway (all modalities) serves PORT.
# If EITHER process exits, the container is torn down (don't serve a dead backend) so an
# orchestrator/restart policy can recover it.
# Env: MODEL_PATH, PORT, MEM_FRACTION, MAX_TOTAL_TOKENS, LCN_YARN.
INTERNAL="${SGLANG_INTERNAL_PORT:-30000}"
export SGLANG_INTERNAL_PORT="$INTERNAL"
export MODEL_PATH="${MODEL_PATH:-/workspace/model}"

# Context length. Default is the model's NATIVE 128k (max_position_embeddings=131072). Set
# LCN_YARN=1 to extend to 256k via YaRN (RoPE factor 2) — opt-in because YaRN can slightly
# affect short-context / generation quality. KV is cheap here (MLA, ~16 KB/token), so the
# limiter is --mem-fraction-static (weights ~88 GB + KV pool); we raise it just enough.
OVERRIDE='{"architectures":["LongcatNextForCausalLM"]}'
DEFAULT_TOKENS=131072
DEFAULT_MEMFRAC=0.72
if [ "${LCN_YARN:-0}" = "1" ]; then
  # Override rope_parameters (NOT rope_scaling): transformers 4.57 rebuilds rope_parameters from a
  # rope_scaling override and drops rope_theta, which the model reads -> KeyError. Setting
  # rope_parameters directly keeps rope_theta AND carries the yarn fields (also aliased to rope_scaling).
  OVERRIDE='{"architectures":["LongcatNextForCausalLM"],"max_position_embeddings":262144,"rope_parameters":{"rope_type":"yarn","rope_theta":10000000,"factor":2.0,"original_max_position_embeddings":131072}}'
  DEFAULT_TOKENS=262144
  DEFAULT_MEMFRAC=0.74
fi

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$INTERNAL" --host 127.0.0.1 \
  --trust-remote-code \
  --json-model-override-args "$OVERRIDE" \
  --mem-fraction-static "${MEM_FRACTION:-$DEFAULT_MEMFRAC}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-$DEFAULT_TOKENS}" \
  --attention-backend flashinfer \
  --disable-cuda-graph --disable-radix-cache --skip-server-warmup \
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
