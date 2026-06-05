#!/bin/bash
# LongCat-Next all-modality server (GB10 / w8a8_int8).
# SGLang runs on an internal port; an OpenAI-compatible gateway (all modalities) serves PORT.
# If EITHER process exits, the container is torn down (don't serve a dead backend) so an
# orchestrator/restart policy can recover it. Env: MODEL_PATH, PORT, MEM_FRACTION, MAX_TOTAL_TOKENS.
INTERNAL="${SGLANG_INTERNAL_PORT:-30000}"
export SGLANG_INTERNAL_PORT="$INTERNAL"
export MODEL_PATH="${MODEL_PATH:-/workspace/model}"

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$INTERNAL" --host 127.0.0.1 \
  --trust-remote-code \
  --json-model-override-args '{"architectures":["LongcatNextForCausalLM"]}' \
  --mem-fraction-static "${MEM_FRACTION:-0.7}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-8192}" \
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
