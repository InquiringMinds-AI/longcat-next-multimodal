#!/usr/bin/env bash
# Image gen+decode on the multicalib-bf16mla checkpoint, two prompts, sequential, --rm.
# lc-vision MUST be stopped. Backbone (gen) and decoder run as separate model loads
# within the same container process, one after another (never co-resident across containers).
set -e
MP=/models/output/LongCat-Next-NVFP4-multicalib-bf16mla

run_one () {
  local TAG="$1"; local PROMPT="$2"
  docker run --rm --name lc-img-$TAG --gpus all --ipc=host --shm-size=16g \
    -v /home/magi/models:/models \
    -v /tmp:/tmp \
    -v /home/magi/.cache/pip:/root/.cache/pip \
    -v /home/magi/Projects/LongCat-Next-inference:/home/magi/Projects/LongCat-Next-inference:ro \
    -v /home/magi/Projects/sglang-longcat-next/python/gen_image_standalone.py:/work/gen_image_standalone.py:ro \
    -v /home/magi/Projects/sglang-longcat-next/python/decode_phaseB.py:/work/decode_phaseB.py:ro \
    -v /home/magi/lc_overlay/deepseek_v2.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py:ro \
    -v /home/magi/lc_overlay/longcat_flash.py:/sgl-workspace/sglang/python/sglang/srt/models/longcat_flash.py:ro \
    -v /home/magi/lc_overlay/n_gram_embedding.py:/sgl-workspace/sglang/python/sglang/srt/layers/n_gram_embedding.py:ro \
    -v /home/magi/lc_overlay/cutlass_moe.py:/sgl-workspace/sglang/python/sglang/srt/layers/moe/cutlass_moe.py:ro \
    -v /home/magi/lc_overlay/longcat_next_visual.py:/sgl-workspace/sglang/python/sglang/srt/models/longcat_next_visual.py:ro \
    -v /home/magi/lc_overlay/processors_longcat_next.py:/sgl-workspace/sglang/python/sglang/srt/multimodal/processors/longcat_next.py:ro \
    -v /home/magi/lc_overlay/longcat_next_audio.py:/sgl-workspace/sglang/python/sglang/srt/models/longcat_next_audio.py:ro \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e SGLANG_EXTERNAL_MM_MODEL_ARCH=LongcatFlashForCausalLM \
    -e HOME=/home/magi \
    -e MODEL_PATH=$MP \
    -e GEN_PROMPT="$PROMPT" \
    -e CFG_SCALE=5 \
    -e GEN_TAG=$TAG \
    --entrypoint bash \
    lmsysorg/sglang:v0.5.12.post1-cu130 \
    -c "pip install -q ujson fire librosa soundfile 2>&1 | tail -1; \
        echo '=== GEN $TAG ==='; python3 /work/gen_image_standalone.py && \
        echo '=== DECODE $TAG ==='; python3 /work/decode_phaseB.py"
}

run_one flower "A photograph of a single yellow flower on a green stem."
run_one pagoda "A towering pagoda under construction."
echo "ALL IMAGE RUNS DONE"
ls -la /tmp/gen_image_flower.png /tmp/gen_image_pagoda.png
