# LongCat-Next — all-modality (text + image/audio understanding + image/audio generation)
# served on a single NVIDIA DGX Spark (GB10, sm_121) via SGLang at w8a8_int8.
#
# The cu130 base is the one that compiles + runs Triton for sm_121 (the cu129 image cannot).
# Build:  docker build -t longcat-next-gb10 .
# Run:    see run.sh  (mount the weights dir at /workspace/model)
FROM lmsysorg/sglang:v0.5.12.post1-cu130

ARG SG=/sgl-workspace/sglang/python/sglang/srt

# --- LongCat-Next overlay: model heads/tokenizers, gen loop, decoders, processor ---
# (final GB10-validated versions; the *_audio/_visual/_heads/_processor files are not in base)
COPY new_files/models/longcat_next_mm.py        ${SG}/models/longcat_next_mm.py
COPY new_files/models/longcat_next_audio.py     ${SG}/models/longcat_next_audio.py
COPY new_files/models/longcat_next_visual.py    ${SG}/models/longcat_next_visual.py
COPY new_files/models/longcat_next_heads.py     ${SG}/models/longcat_next_heads.py
COPY new_files/models/longcat_next_processor.py ${SG}/models/longcat_next_processor.py
COPY new_files/models/longcat_flash.py          ${SG}/models/longcat_flash.py
COPY new_files/models/image_refiner.py          ${SG}/models/image_refiner.py
COPY new_files/models/refiner_modules.py        ${SG}/models/refiner_modules.py
COPY new_files/models/cosy24k_vocoder.py        ${SG}/models/cosy24k_vocoder.py
COPY new_files/layers/n_gram_embedding.py       ${SG}/layers/n_gram_embedding.py
COPY new_files/processors/longcat_next.py       ${SG}/multimodal/processors/longcat_next.py
COPY new_files/hf_transformers/processor.py     ${SG}/utils/hf_transformers/processor.py

# --- audio deps (mel extraction + wav I/O for the cosy24k vocoder) ---
RUN pip install --no-cache-dir librosa soundfile

# --- base-config patches: recognize model_type=longcat_next + build the nested
#     visual/audio mm sub-configs the tokenizers need ---
COPY patches/ /tmp/patches/
RUN cd /sgl-workspace/sglang && \
    patch -p1 < /tmp/patches/model_config.patch && \
    patch -p1 < /tmp/patches/configs_longcat_flash.patch

# --- GB10 fix: on an ARM host SGLang routes the int8 MoE to a CPU-only op even on GPU.
#     Require actually-on-CPU so the GB10 GPU/Triton path runs. ---
RUN sed -i 's/use_intel_amx_backend(layer) or _is_cpu_arm64:/use_intel_amx_backend(layer) or (_is_cpu_arm64 and _is_cpu):/' \
    ${SG}/layers/quantization/w8a8_int8.py

# --- build-time smoke test: arch auto-registers ---
RUN python3 -c "from sglang.srt.models.registry import ModelRegistry; \
archs=ModelRegistry.get_supported_archs(); \
assert 'LongcatNextForCausalLM' in archs, sorted(a for a in archs if 'ongcat' in a); \
print('OK registered: LongcatNextForCausalLM')"

# --- bundled client/test scripts + per-language demo reference voices ---
#   voices/en_reference.wav : public-domain LibriVox solo narration (native English)
#   voices/zh_reference.wav : Meituan LongCat example clip spk_syn.wav (MIT, Chinese)
COPY test/ /workspace/scripts/
COPY voices/ /workspace/scripts/voices/
COPY gateway.py /workspace/scripts/gateway.py
COPY longcat_tools.py /workspace/scripts/longcat_tools.py

# default output dir for generated PNG/WAV (override + mount via run.sh)
ENV LCN_OUTPUT_DIR=/tmp

COPY entrypoint.sh /usr/local/bin/lcn-serve
RUN chmod +x /usr/local/bin/lcn-serve
ENTRYPOINT ["/usr/local/bin/lcn-serve"]
