#!/usr/bin/env bash
set -e
docker run --rm --name lc-audiodec --gpus all --ipc=host --shm-size=16g \
  -v /home/magi/models:/models \
  -v /tmp:/tmp \
  -v /home/magi/.cache/pip:/root/.cache/pip \
  -v /home/magi/Projects/LongCat-Next-inference:/home/magi/Projects/LongCat-Next-inference:ro \
  -v /home/magi/Projects/sglang-longcat-next/python/decode_audio_gen.py:/work/decode_audio_gen.py:ro \
  -e MODEL_PATH=/models/output/LongCat-Next-NVFP4-bf16mla \
  -e IN_IDS=/tmp/gen_audio_ids.pt -e OUT_WAV=/tmp/gen_audio.wav \
  --entrypoint bash \
  lmsysorg/sglang:v0.5.12.post1-cu130 \
  -c "pip install -q ujson fire librosa soundfile 2>&1 | tail -1; python3 /work/decode_audio_gen.py"
