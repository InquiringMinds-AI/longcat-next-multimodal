#!/usr/bin/env bash
set -e
docker rm -f lc-vision 2>/dev/null || true
docker run -d --name lc-vision --gpus all --ipc=host --shm-size=16g -p 8090:8090 \
  -v /home/magi/models:/models \
  -v /tmp:/tmp \
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
  --entrypoint python3 \
  lmsysorg/sglang:v0.5.12.post1-cu130 \
  -m sglang.launch_server --model-path /models/output/LongCat-Next-NVFP4-bf16mla --port 8090 --host 0.0.0.0 \
  --quantization modelopt_fp4 \
  --json-model-override-args "{\"architectures\":[\"LongcatFlashForCausalLM\"],\"use_ngram_embedding\":true,\"ngram_embedding_m\":10223616,\"ngram_embedding_n\":5,\"ngram_embedding_k\":3,\"rope_parameters\":{\"rope_theta\":10000000.0,\"rope_type\":\"default\"},\"disable_quant_module\":[\"self_attn\"]}" \
  --attention-backend flashinfer --mem-fraction-static 0.72 --max-total-tokens 4096 \
  --enable-multimodal --disable-cuda-graph --dtype bfloat16 --skip-server-warmup --disable-radix-cache --watchdog-timeout 600
