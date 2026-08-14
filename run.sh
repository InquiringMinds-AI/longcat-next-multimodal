#!/usr/bin/env bash
# Launch the LongCat-Next all-modality server on an NVIDIA DGX Spark (GB10).
#
#   ./run.sh /path/to/longcat-next-gb10-weights
#
# The weights dir is the one downloaded from Hugging Face (see README).
# Generated images/audio land in ./outputs.
#
# SECURITY DEFAULTS: the API is published on 127.0.0.1:8090 only (loopback) and is unauthenticated.
# To expose it on a network, set BOTH:
#   LCN_BIND=0.0.0.0  (or a specific host IP)  -- which interface to publish on
#   LCN_API_KEY=<secret>                        -- required bearer token (clients send
#                                                  `Authorization: Bearer <secret>`)
# e.g.  LCN_BIND=0.0.0.0 LCN_API_KEY=$(openssl rand -hex 24) ./run.sh ./weights
set -euo pipefail
WEIGHTS="${1:?usage: ./run.sh <weights_dir>}"
BIND="${LCN_BIND:-127.0.0.1}"
if [ "$BIND" != "127.0.0.1" ] && [ -z "${LCN_API_KEY:-}" ]; then
  echo "WARNING: publishing on $BIND with no LCN_API_KEY — the API is unauthenticated and reachable" >&2
  echo "         from the network. Set LCN_API_KEY=<secret> unless this network is fully trusted." >&2
fi
mkdir -p outputs
# --shm-size: SGLang moves multimodal pixel tensors between processes via /dev/shm.
# Docker's 64MB default SIGBUS-crashes the server on multi-image requests; tmpfs is
# lazily allocated so the generous ceiling costs nothing until used.
# Forward the whole tuning surface (README: "Tuning (env vars)"). Empty values are
# safe: every consumer applies its default via ${VAR:-default}, which also
# substitutes on set-but-empty. An ARRAY, not a scalar: values containing spaces
# (e.g. LCN_VOICE_DIR="/mnt/custom voices") must stay one docker argument.
TUNING_ENV=()
for v in MEM_FRACTION MAX_TOTAL_TOKENS LCN_AGENT LCN_YARN LCN_RADIX LCN_KV_DTYPE \
         LCN_CUDAGRAPH LCN_CUDAGRAPH_BS LCN_HEAD_GRAPH LCN_INT8_HEADS \
         LCN_NGRAM LCN_NGRAM_DRAFT LCN_NGRAM_EOS LCN_NGRAM_AGENT_COUPLE \
         LCN_REFINER_CFG_RANGE REFINER_STEPS IMAGE_GEN_CFG_SCALE \
         IMAGE_GEN_TEMPERATURE IMAGE_GEN_TOP_K IMAGE_GEN_TOP_P \
         LCN_TTS_STREAM LCN_TTS_MULTI LCN_TTS_SILENCE_FRAMES \
         LCN_TTS_TRIM_LEAD_MS LCN_TTS_TRIM_TAIL_MS \
         AUDIO_GEN_TEMPERATURE AUDIO_GEN_TOP_K \
         LCN_PREWARM LCN_KEEP_ARTIFACTS LCN_VERBOSE LCN_MODEL_NAME LCN_VOICE_DIR; do
  eval "val=\${$v:-}"
  [ -n "$val" ] && TUNING_ENV+=(-e "$v=$val")
done

docker run --rm -it --gpus all --shm-size=32g \
  -v "$(realpath "$WEIGHTS")":/workspace/model:ro \
  -v "$(pwd)/outputs":/workspace/outputs \
  -e LCN_OUTPUT_DIR=/workspace/outputs \
  -e LCN_API_KEY="${LCN_API_KEY:-}" \
  "${TUNING_ENV[@]}" \
  -p "${BIND}:8090:8090" \
  --name longcat-next \
  longcat-next-gb10
