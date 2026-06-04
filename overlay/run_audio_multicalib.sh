#!/usr/bin/env bash
# Audio gen + decode on multicalib-bf16mla, English target. lc-vision MUST be stopped.
# gen (backbone) then decode (vocoder) as sequential --rm containers, never co-resident.
set -e
MP=/models/output/LongCat-Next-NVFP4-multicalib-bf16mla

echo "=== AUDIO GEN (multicalib) ==="
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
  -e HOME=/home/magi \
  -e MODEL_PATH=$MP \
  -e MAX_FRAMES=200 \
  -e SYN_TEXT="The quick brown fox jumps over the lazy dog." \
  -e A_TEMP=0.2 -e A_TOP_K=20 -e A_TOP_P=0.85 -e A_REP=1.1 \
  -e T_TEMP=0.5 -e T_TOP_K=5 -e T_TOP_P=0.85 -e T_REP=1.3 \
  -e OUT_IDS=/tmp/gen_audio_ids_multicalib.pt \
  --entrypoint bash \
  lmsysorg/sglang:v0.5.12.post1-cu130 \
  -c "pip install -q --break-system-packages ujson fire librosa soundfile 2>&1 | tail -2; python3 -c 'import ujson,fire,librosa,soundfile' && echo DEPS_OK; python3 /work/gen_audio_standalone.py"

echo "=== AUDIO DECODE (multicalib) ==="
docker run --rm --name lc-audiodec --gpus all --ipc=host --shm-size=16g \
  -v /home/magi/models:/models \
  -v /tmp:/tmp \
  -v /home/magi/.cache/pip:/root/.cache/pip \
  -v /home/magi/Projects/LongCat-Next-inference:/home/magi/Projects/LongCat-Next-inference:ro \
  -v /home/magi/Projects/sglang-longcat-next/python/decode_audio_gen.py:/work/decode_audio_gen.py:ro \
  -e HOME=/home/magi \
  -e MODEL_PATH=$MP \
  -e IN_IDS=/tmp/gen_audio_ids_multicalib.pt -e OUT_WAV=/tmp/gen_multicalib_en.wav \
  --entrypoint bash \
  lmsysorg/sglang:v0.5.12.post1-cu130 \
  -c "pip install -q --break-system-packages ujson fire librosa soundfile 2>&1 | tail -2; python3 /work/decode_audio_gen.py"

echo "=== AUDIO DONE ==="
ls -la /tmp/gen_multicalib_en.wav
