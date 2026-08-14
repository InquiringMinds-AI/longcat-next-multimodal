# LongCat-Next — all-modality (text + image/audio understanding + image/audio generation)
# served on a single NVIDIA DGX Spark (GB10, sm_121) via SGLang at w8a8_int8.
#
# The cu130 base is the one that compiles + runs Triton for sm_121 (the cu129 image cannot).
# Build:  docker build -t longcat-next-gb10 .
# Run:    see run.sh  (mount the weights dir at /workspace/model)
# Rebased v0.5.12.post1 -> v0.5.16 (2026-07-31): overlay three-way-merged; upstream still
# has no multimodal LongCat-Next — this overlay remains the only any-to-any path.
FROM lmsysorg/sglang:v0.5.16-cu130

ARG SG=/sgl-workspace/sglang/python/sglang/srt

# --- LongCat-Next overlay: model heads/tokenizers, gen loop, decoders, processor ---
# (final GB10-validated versions; the *_audio/_visual/_heads/_processor files are not in base)
COPY new_files/models/longcat_next_mm.py        ${SG}/models/longcat_next_mm.py
COPY new_files/models/longcat_next_audio.py     ${SG}/models/longcat_next_audio.py
COPY new_files/models/longcat_next_visual.py    ${SG}/models/longcat_next_visual.py
COPY new_files/models/longcat_next_heads.py     ${SG}/models/longcat_next_heads.py
COPY new_files/models/int8_head_ffn.py          ${SG}/models/int8_head_ffn.py
COPY new_files/models/lcn_head_graph.py         ${SG}/models/lcn_head_graph.py
COPY new_files/models/longcat_next_processor.py ${SG}/models/longcat_next_processor.py
COPY new_files/models/longcat_flash.py          ${SG}/models/longcat_flash.py
COPY new_files/models/image_refiner.py          ${SG}/models/image_refiner.py
COPY new_files/models/refiner_modules.py        ${SG}/models/refiner_modules.py
COPY new_files/models/cosy24k_vocoder.py        ${SG}/models/cosy24k_vocoder.py
COPY new_files/layers/n_gram_embedding.py       ${SG}/layers/n_gram_embedding.py
COPY new_files/model_runner_components/ngram_embedding_manager.py ${SG}/model_executor/model_runner_components/ngram_embedding_manager.py
COPY new_files/processors/longcat_next.py       ${SG}/multimodal/processors/longcat_next.py
COPY new_files/hf_transformers/processor.py     ${SG}/utils/hf_transformers/processor.py
# Neutral module so the scheduler can ask "is generation active?" without
# importing the model (models import schedule_batch — the reverse would cycle).
COPY new_files/lcn_gen_state.py                 ${SG}/lcn_gen_state.py

# --- audio deps (mel extraction + wav I/O for the cosy24k vocoder) ---
# scipy pinned: the vocoder's STFT window comes from scipy.signal.get_window, and the
# scipy 1.18 resolution that rode in with the v0.5.16 base coincided with a garbled
# first-word onset in TTS output (human-ear regression test). Pin to the known-good.
RUN pip install --no-cache-dir librosa soundfile "scipy==1.17.1"

# --- base-config patches: recognize model_type=longcat_next + build the nested
#     visual/audio mm sub-configs the tokenizers need ---
COPY patches/ /tmp/patches/
RUN cd /sgl-workspace/sglang && \
    patch -p1 < /tmp/patches/model_config.patch && \
    patch -p1 < /tmp/patches/configs_longcat_flash.patch && \
    patch -p1 < /tmp/patches/decode_graph_gen_veto.patch && \
    patch -p1 < /tmp/patches/ngram_spec_verify.patch && \
    patch -p1 < /tmp/patches/spec_gen_fallback.patch

# --- GB10 fix: on an ARM host SGLang routes the int8 MoE to a CPU-only op even on GPU.
#     Require actually-on-CPU so the GB10 GPU/Triton path runs. ---
RUN sed -i 's/use_intel_amx_backend(layer) or _is_cpu_arm64:/use_intel_amx_backend(layer) or (_is_cpu_arm64 and _is_cpu):/' \
    ${SG}/layers/quantization/w8a8_int8.py

# --- build-time smoke test: arch auto-registers ---
RUN python3 -c "from sglang.srt.models.registry import ModelRegistry; \
archs=ModelRegistry.get_supported_archs(); \
assert 'LongcatNextForCausalLM' in archs, sorted(a for a in archs if 'ongcat' in a); \
print('OK registered: LongcatNextForCausalLM')"

# --- GB10-tuned fused-MoE Triton kernel configs (roadmap #4) ---
#     Tuned on this checkpoint's real routing distributions across all 18 batch
#     sizes; without them the runtime warns the config is missing and falls back
#     to generic heuristics. Filenames are what get_config_file_name() derives
#     for E=256, N=1024, int8_w8a8, per_channel_quant=True (plus the _down
#     variant for the second projection); triton_3_6_0 matches the image's
#     Triton, and the lookup is keyed by that version directory.
COPY new_files/moe_configs/ ${SG}/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/

# --- bundled client/test scripts + per-language demo reference voices ---
#   voices/en_reference.wav : public-domain LibriVox solo narration (native English)
#   voices/zh_reference.wav : Meituan LongCat example clip spk_syn.wav (MIT, Chinese)
COPY test/ /workspace/scripts/
COPY voices/ /workspace/scripts/voices/
COPY gateway.py /workspace/scripts/gateway.py
COPY longcat_tools.py /workspace/scripts/longcat_tools.py
COPY stream_tools.py /workspace/scripts/stream_tools.py
COPY audio_chat.py /workspace/scripts/audio_chat.py
COPY stream_util.py /workspace/scripts/stream_util.py
COPY anthropic_route.py /workspace/scripts/anthropic_route.py

# default output dir for generated PNG/WAV (override + mount via run.sh)
ENV LCN_OUTPUT_DIR=/tmp

# Build identity, surfaced by GET /status. Pass it at build time:
#   docker build --build-arg LCN_BUILD=$(git rev-parse --short HEAD) -t longcat-next-gb10 .
# Defaults to "unknown" rather than failing, because the build context is rsync'd without
# .git on the serving host and a missing label must not block a rebuild.
ARG LCN_BUILD=unknown
ENV LCN_BUILD=${LCN_BUILD}

COPY entrypoint.sh /usr/local/bin/lcn-serve
RUN chmod +x /usr/local/bin/lcn-serve
ENTRYPOINT ["/usr/local/bin/lcn-serve"]
