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
docker run --rm -it --gpus all \
  -v "$(realpath "$WEIGHTS")":/workspace/model:ro \
  -v "$(pwd)/outputs":/workspace/outputs \
  -e LCN_OUTPUT_DIR=/workspace/outputs \
  -e LCN_API_KEY="${LCN_API_KEY:-}" \
  -p "${BIND}:8090:8090" \
  --name longcat-next \
  longcat-next-gb10
