#!/bin/bash
# Supervise the GB10 MoE tuning ladder: one batch size per container, resumed
# from checkpoints, until every size is tuned.
#
# WHY one size per process (measured on run 3, 2026-08-05): MemAvailable drifts
# down ~0.75GB/h WITHIN a single batch size, and torch.cuda.empty_cache() at the
# size boundary reclaims none of it — 40GB free at the start of M=4096 was 31GB
# by its end, 12.5h later, with no recovery when the size completed. The stock
# tuner is one long-lived process for the whole ~100h ladder, so it walks into
# Ray's 95% OOM kill somewhere around hour 50 (that is what killed run 1, and
# run 3 was on the same trajectory). A fresh process per size resets the drift,
# and LCN_ONE_SIZE_PER_RUN + the checkpoint-resume in tune_moe_gb10.sh make each
# restart cost nothing already earned.
#
# Run it detached on Spark, NOT through an ssh session that may end:
#   setsid nohup ~/longcat-outputs/tune_moe_supervise.sh > ~/longcat-outputs/supervise.log 2>&1 < /dev/null &
set -uo pipefail

IMAGE=${IMAGE:-longcat-next-gb10:v0516-spec}
NAME=${NAME:-lcn-moe-tune}
OUTDIR=${OUTDIR:-$HOME/longcat-outputs}
CKPT=$OUTDIR/moe_configs/checkpoints
TOTAL=${TOTAL:-18}          # batch sizes in the ladder
MAX_CYCLES=${MAX_CYCLES:-40}

count_ckpts() { ls "$CKPT" 2>/dev/null | grep -c '^ckpt_M[0-9]\+\.json$'; }

echo "=== supervisor start $(date -u +%FT%TZ) — have $(count_ckpts)/$TOTAL sizes"

for cycle in $(seq 1 "$MAX_CYCLES"); do
  have=$(count_ckpts)
  if [ "$have" -ge "$TOTAL" ]; then
    echo "=== all $TOTAL sizes checkpointed; final assembly pass"
    break
  fi

  docker rm -f "$NAME" >/dev/null 2>&1
  echo "--- cycle $cycle: $have/$TOTAL done, launching $(date -u +%FT%TZ)"
  docker run -d --name "$NAME" --gpus all --shm-size=32g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LCN_ONE_SIZE_PER_RUN=1 \
    -v "$HOME/models/LongCat-Next-w8a8int8:/workspace/model:ro" \
    -v "$OUTDIR:/workspace/outputs" \
    --entrypoint bash "$IMAGE" /workspace/outputs/tune_moe_gb10.sh >/dev/null || {
      echo "!!! docker run failed, aborting"; exit 1; }

  rc=$(docker wait "$NAME" 2>/dev/null || echo "wait-failed")
  now=$(count_ckpts)
  echo "--- cycle $cycle: container exit=$rc, checkpoints $have -> $now"

  if [ "$now" -le "$have" ]; then
    echo "!!! no progress this cycle — stopping so this does not spin."
    echo "!!! last 20 log lines:"
    docker logs --tail 200 "$NAME" 2>&1 | tr '\r' '\n' | grep -vE '\|.*1\.92k' | tail -20
    exit 1
  fi
done

# One last run does nothing but the in-script recovery assembly (every size is
# checkpointed, so main() returns early and the recovery block writes the JSONs).
docker rm -f "$NAME" >/dev/null 2>&1
docker run --rm --name "$NAME" --gpus all --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/models/LongCat-Next-w8a8int8:/workspace/model:ro" \
  -v "$OUTDIR:/workspace/outputs" \
  --entrypoint bash "$IMAGE" /workspace/outputs/tune_moe_gb10.sh 2>&1 | tail -20

echo "=== supervisor done $(date -u +%FT%TZ) — $(count_ckpts)/$TOTAL sizes"
ls -la "$OUTDIR/moe_configs/"
