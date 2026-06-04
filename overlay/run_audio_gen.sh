#!/usr/bin/env bash
# One-shot standalone audio-gen run inside the lc container image (lc-vision must be STOPPED).
set -e
docker run --rm --name lc-audiogen --gpus all --ipc=host --shm-size=16g \
  -v /home/magi/models:/models \
  -v /tmp:/tmp \
  -v /home/magi/.cache/pip:/root/.cache/pip \
  -v /home/magi/Projects/LongCat-Next-inference:/home/magi/Projects/LongCat-Next-inference:ro \
  -v /home/magi/Projects/sglang-longcat-next/python/gen_audio_standalone.py:/work/gen_audio_standalone.py:ro \
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
  -e MODEL_PATH=/models/output/LongCat-Next-NVFP4-bf16mla \
  -e MAX_FRAMES=${MAX_FRAMES:-200} \
  -e SYN_TEXT="${SYN_TEXT:-The quick brown fox jumps over the lazy dog.}" \
  -e A_TEMP=${A_TEMP:-0.2} -e A_TOP_K=${A_TOP_K:-20} -e A_TOP_P=${A_TOP_P:-0.85} -e A_REP=${A_REP:-1.1} \
  -e T_TEMP=${T_TEMP:-0.5} -e T_TOP_K=${T_TOP_K:-5} -e T_TOP_P=${T_TOP_P:-0.85} -e T_REP=${T_REP:-1.3} \
  -e DIAG_NO_STOP=${DIAG_NO_STOP:-0} \
  --entrypoint bash \
  lmsysorg/sglang:v0.5.12.post1-cu130 \
  -c "pip install -q ujson fire librosa soundfile 2>&1 | tail -1; python3 /work/gen_audio_standalone.py"
