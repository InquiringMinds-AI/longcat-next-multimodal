"""LongCat-Next multimodal model wrapper for SGLang.

Extends the LongcatFlash text backbone with:
- Visual tokenizer (Qwen2.5-VL encoder + VQ-RQ) for image understanding
- Audio tokenizer (Whisper + bridge + VQ) for audio understanding
- Visual/Audio generation heads (CasualDepthTransformerHead)

Input flow:
  1. Processor creates placeholder tokens for images/audio
  2. Visual/Audio encoder converts raw media → VQ codebook IDs
  3. Codebook IDs → embed_tokens lookup with offsets → sum over codebooks
  4. Replace placeholder embeddings in the text backbone's input

Output flow (generation):
  1. Text backbone produces hidden states
  2. Mode switch routes to visual_head or audio_head
  3. Depth-wise transformer generates codebook tokens level by level
  4. VQ codes → decoder (image refiner / vocoder) → raw media
"""

import base64
import io
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio generation per-request state
# ---------------------------------------------------------------------------

@dataclass
class AudioGenState:
    """Per-request state for audio generation."""
    mode: str = "transcript"  # "transcript" → "generating" → "done"
    accumulated_ids: list = field(default_factory=list)  # list of [num_codebooks] tensors
    prev_audio_ids: Optional[torch.Tensor] = None  # [seq, num_codebooks] for rep penalty
    step_count: int = 0
    max_audio_steps: int = 1000  # safety limit (~40s of audio at 24kHz)
    transcript_done: bool = False  # whether audio text phase completed
    transcript_steps: int = 0  # count of transcript tokens generated
    max_transcript_steps: int = 100  # force transition after this many transcript tokens
    transcript_tokens: list = field(default_factory=list)  # accumulated transcript token IDs (from lm_head argmax)
    end_run: int = 0  # consecutive level-0 end-flags seen (for END_CONFIRM)
    ended: bool = False  # set when a confirmed end-of-audio cluster is reached
    rid: str = ""  # per-request id → unique output filename (concurrency-safe retrieval)


# Audio generation sampling config (from generation_config.json). Operator-tunable via env
# (e.g. -e AUDIO_GEN_TEMPERATURE=0.7) without rebuilding; defaults are the model-card values.
_envf = lambda k, d: float(os.environ.get(k, d))
_envi = lambda k, d: int(os.environ.get(k, d))
_LCN_VERBOSE = os.environ.get("LCN_VERBOSE", "0") == "1"  # gate per-step debug logging
AUDIO_GEN_TEMPERATURE = _envf("AUDIO_GEN_TEMPERATURE", 0.5)
AUDIO_GEN_TOP_K = _envi("AUDIO_GEN_TOP_K", 5)
AUDIO_GEN_TOP_P = _envf("AUDIO_GEN_TOP_P", 0.85)
AUDIO_GEN_REPETITION_PENALTY = _envf("AUDIO_GEN_REPETITION_PENALTY", 1.3)
# End-of-audio is confirmed by this many CONSECUTIVE level-0 end-flags (canonical guard):
# an isolated/stray end-flag is re-sampled to a real acoustic code so the model speaks for
# exactly as long as its task needs — no arbitrary minimum-length floor.
AUDIO_END_CONFIRM = 2
AUDIO_GEN_SAMPLING_RATE = 24000


@dataclass
class ImageGenState:
    """Per-request state for image generation."""
    accumulated_ids: list = field(default_factory=list)  # list of [num_codebooks] tensors
    current_image_token_num: int = 0  # counter for newline/end logic
    token_h: int = 37  # image height in tokens
    token_w: int = 37  # image width in tokens
    # CFG dual-path state
    uncond_req_pool_idx: int = -1  # req pool index for unconditional KV cache
    uncond_seq_len: int = 0  # current sequence length of unconditional path
    uncond_initialized: bool = False  # whether unconditional prefill has been done
    rid: str = ""  # per-request id → unique output filename (concurrency-safe retrieval)

    @property
    def is_img_newline(self) -> bool:
        return ((self.current_image_token_num + 1) % (self.token_w + 1)) == 0 and not self.is_img_end

    @property
    def is_img_end(self) -> bool:
        return (self.current_image_token_num + 1) / (self.token_w + 1) == self.token_h

    @property
    def total_tokens(self) -> int:
        return self.token_h * (self.token_w + 1)  # h * (w + 1 newline per row)


# Image generation sampling config (from generation_config.json). Operator-tunable via env.
IMAGE_GEN_TEMPERATURE = _envf("IMAGE_GEN_TEMPERATURE", 0.5)
IMAGE_GEN_TOP_K = _envi("IMAGE_GEN_TOP_K", 1024)
IMAGE_GEN_TOP_P = _envf("IMAGE_GEN_TOP_P", 0.75)
IMAGE_GEN_CFG_SCALE = _envf("IMAGE_GEN_CFG_SCALE", 3.0)  # Classifier-Free Guidance scale
AUDIO_GEN_WAVE_OVERLAP = 1200


class DictConfig:
    """Recursively convert a dict to attribute-accessible object."""
    def __init__(self, d):
        for k, v in d.items():
            if not isinstance(k, str):
                continue  # Skip non-string keys
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            elif isinstance(v, list):
                setattr(self, k, [DictConfig(i) if isinstance(i, dict) else i for i in v])
            else:
                setattr(self, k, v)

    def __repr__(self):
        return f"DictConfig({vars(self)})"


def ensure_config_object(cfg):
    """Convert dict to DictConfig if needed."""
    if isinstance(cfg, dict):
        return DictConfig(cfg)
    return cfg


class LongcatNextForCausalLM(LongcatFlashForCausalLM):
    """LongCat-Next with multimodal support.

    Extends the text backbone with visual and audio encoders + generation heads.
    The text backbone is the same LongcatFlash architecture (MLA + MoE + N-gram).
    """

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        print(f"[LCN-INIT] entered cls={type(self).__name__} has_vc={hasattr(config,chr(39)+chr(118)+chr(99)+chr(39))}", flush=True)

        # Visual tokenizer (Qwen2.5-VL encoder + VQ-RQ)
        if hasattr(config, 'visual_config') and config.visual_config is not None:
            try:
                from sglang.srt.models.longcat_next_visual import LongcatNextVisualTokenizer
                # Tokenizer expects the full config with visual_config as sub-attribute
                full_cfg = self._make_full_config(config)
                # Attach to self.model so weight names match checkpoint (model.visual_tokenizer.*)
                self.model.visual_tokenizer = LongcatNextVisualTokenizer(full_cfg)
                logger.info("Visual tokenizer initialized")
            except Exception as e:
                __import__(chr(39)+chr(116)+chr(114)+chr(97)+chr(99)+chr(101)+chr(98)+chr(97)+chr(99)+chr(107)+chr(39)).print_exc(); logger.warning(f"Could not initialize visual tokenizer: {e}")
                self.model.visual_tokenizer = None
        else:
            self.model.visual_tokenizer = None
        self.visual_tokenizer = self.model.visual_tokenizer  # convenience alias

        # Audio tokenizer (Whisper + bridge + VQ)
        if hasattr(config, 'audio_config') and config.audio_config is not None:
            try:
                from sglang.srt.models.longcat_next_audio import LongcatNextAudioTokenizer
                full_cfg = self._make_full_config(config)
                self.model.audio_tokenizer = LongcatNextAudioTokenizer(full_cfg)
                logger.info("Audio tokenizer initialized")
            except Exception as e:
                __import__(chr(39)+chr(116)+chr(114)+chr(97)+chr(99)+chr(101)+chr(98)+chr(97)+chr(99)+chr(107)+chr(39)).print_exc(); logger.warning(f"Could not initialize audio tokenizer: {e}")
                self.model.audio_tokenizer = None
        else:
            self.model.audio_tokenizer = None
        self.audio_tokenizer = self.model.audio_tokenizer  # convenience alias

        # Generation heads
        if hasattr(config, 'visual_config') and config.visual_config is not None:
            try:
                from sglang.srt.models.longcat_next_heads import CasualDepthTransformerHead
                vc = ensure_config_object(config.visual_config)
                self.visual_head = CasualDepthTransformerHead(
                    hidden_size=config.hidden_size,
                    codebook_sizes=vc.vq_config.codebook_sizes,
                    transformer_layer_num=vc.image_head_transformer_layers,
                    transformer_dim=vc.image_head_transformer_dims,
                    transformer_ffn_scale=vc.image_head_transformer_ffn_scale,
                )
                logger.info("Visual generation head initialized")
            except Exception as e:
                __import__(chr(39)+chr(116)+chr(114)+chr(97)+chr(99)+chr(101)+chr(98)+chr(97)+chr(99)+chr(107)+chr(39)).print_exc(); logger.warning(f"Could not initialize visual head: {e}")
                self.visual_head = None
        else:
            self.visual_head = None

        if hasattr(config, 'audio_config') and config.audio_config is not None:
            try:
                from sglang.srt.models.longcat_next_heads import CasualDepthTransformerHead
                ac = ensure_config_object(config.audio_config)
                self.audio_head = CasualDepthTransformerHead(
                    hidden_size=config.hidden_size,
                    codebook_sizes=ac.vq_config.codebook_sizes,
                    transformer_layer_num=ac.audio_head_transformer_layers,
                    transformer_dim=ac.audio_head_transformer_dims,
                    transformer_ffn_scale=ac.audio_head_transformer_ffn_scale,
                )
                logger.info("Audio generation head initialized")
            except Exception as e:
                __import__(chr(39)+chr(116)+chr(114)+chr(97)+chr(99)+chr(101)+chr(98)+chr(97)+chr(99)+chr(107)+chr(39)).print_exc(); logger.warning(f"Could not initialize audio head: {e}")
                self.audio_head = None
        else:
            self.audio_head = None

        # Codebook offset values for visual/audio token embedding
        self._init_codebook_offsets(config)

        # Load separate codebook embeddings for multimodal VQ lookups
        self._codebook_embed = None

        # Audio generation token IDs and state
        ac = getattr(config, 'audio_config', None)
        if ac is not None:
            def _acfg(key, default):
                if isinstance(ac, dict): return ac.get(key, default)
                return getattr(ac, key, default)
            self._audiogen_start_id = _acfg('audiogen_start_token_id', 131123)
            self._audiogen_end_id = _acfg('audiogen_end_token_id', 131124)
            self._audiotext_start_id = _acfg('audiotext_start_token_id', 131120)
            self._audiotext_pad_id = _acfg('audiotext_pad_token_id', 131122)
            self._audio_pad_id = _acfg('audio_pad_token_id', 131105)
            vq = _acfg('vq_config', {})
            if isinstance(vq, dict):
                self._audio_codebook_sizes = vq.get('codebook_sizes', [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024])
            else:
                self._audio_codebook_sizes = getattr(vq, 'codebook_sizes', [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024])
        else:
            self._audiogen_start_id = 131123
            self._audiogen_end_id = 131124
            self._audiotext_start_id = 131120
            self._audiotext_pad_id = 131122
            self._audio_pad_id = 131105
            self._audio_codebook_sizes = [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024]

        # Per-request audio generation state: req_pool_idx → AudioGenState
        self._audio_gen_states: Dict[int, AudioGenState] = {}

        # Image generation token IDs
        vc = getattr(config, 'visual_config', None)
        if vc is not None:
            def _vcfg(key, default):
                if isinstance(vc, dict): return vc.get(key, default)
                return getattr(vc, key, default)
            self._image_start_id = _vcfg('image_start_token_id', 131106)
            self._image_end_id = _vcfg('image_end_token_id', 131107)
            self._image_pad_id = _vcfg('image_pad_token_id', 131108)
            self._image_newline_id = _vcfg('image_newline_token_id', 131109)
            vq = _vcfg('vq_config', {})
            if isinstance(vq, dict):
                self._visual_codebook_sizes = vq.get('codebook_sizes', [16384]*8)
            else:
                self._visual_codebook_sizes = getattr(vq, 'codebook_sizes', [16384]*8)
        else:
            self._image_start_id = 131106
            self._image_end_id = 131107
            self._image_pad_id = 131108
            self._image_newline_id = 131109
            self._visual_codebook_sizes = [16384]*8

        # Per-request image generation state: req_pool_idx → ImageGenState
        self._image_gen_states: Dict[int, ImageGenState] = {}
        self._tokenizer = None  # lazy-loaded for diagnostic logging

        # KV pool references for dual-path CFG (set by model_runner after load)
        self._model_runner = None

    def _setup_kv_pool_refs(self, model_runner):
        """Called by model_runner to provide KV pool access for CFG dual-path."""
        self._model_runner = model_runner
        logger.info("KV pool references registered for CFG dual-path support")

    def _alloc_uncond_kv(self, cond_req_pool_idx: int, cond_seq_len: int,
                         input_ids_for_prefill: torch.Tensor, forward_batch) -> int:
        """Allocate unconditional KV cache and run prefill for CFG.

        Creates a shadow request with zeroed prompt, runs prefill through the
        backbone to build the unconditional KV cache.

        Returns the unconditional req_pool_idx.
        """
        if self._model_runner is None:
            logger.warning("No model_runner reference — cannot allocate uncond KV")
            return -1

        try:
            rtp = self._model_runner.req_to_token_pool
            alloc = self._model_runner.token_to_kv_pool_allocator

            # Allocate a free request slot from the pool directly
            if not rtp.free_slots:
                logger.warning("No free req slots for uncond KV cache")
                return -1
            uncond_idx = rtp.free_slots.pop(0)

            # Allocate token pages for the unconditional sequence
            # Start with just 1 token (we'll extend as we decode)
            n_prefill = len(input_ids_for_prefill)
            token_locs = alloc.alloc(n_prefill)
            if token_locs is None:
                rtp.free_slots.append(uncond_idx)
                logger.warning("No free KV pages for uncond prefill")
                return -1

            rtp.req_to_token[uncond_idx, :n_prefill] = token_locs

            # Build unconditional embeddings matching original's approach:
            # 1. Zero the token IDs for the prompt portion (original line 153/512)
            # 2. Keep anyres_prefix + image_start suffix tokens intact
            # 3. Compute N-gram embeddings on the zeroed IDs (original line 163)
            # 4. Zero the embeddings at prompt positions (original line 164)
            # This ensures the N-gram hash sees zeros for prompt tokens (no leakage)
            # Compute suffix length dynamically: anyres_prefix + image_start
            try:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer
                    model_path = os.environ.get('SGLANG_MODEL_PATH', '/workspace/model')
                    self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                anyres_text = '<longcat_img_token_size>37 37</longcat_img_token_size>'
                ANYRES_SUFFIX_LEN = len(self._tokenizer.encode(anyres_text, add_special_tokens=False)) + 1  # +1 for image_start
            except Exception:
                ANYRES_SUFFIX_LEN = 8  # fallback
            uncond_ids = input_ids_for_prefill.clone()
            if n_prefill > ANYRES_SUFFIX_LEN:
                uncond_ids[:n_prefill - ANYRES_SUFFIX_LEN] = 0  # Zero prompt token IDs
            # Compute N-gram on zeroed IDs (hash sees zeros for prompt)
            if self.model.use_ngram_embedding:
                uncond_embeds = self.model.embed_tokens(uncond_ids, forward_batch)
            else:
                uncond_embeds = self.model.embed_tokens(uncond_ids)
            # Zero the prompt embeddings (original zeros input_embeds at special positions)
            if n_prefill > ANYRES_SUFFIX_LEN:
                uncond_embeds[:n_prefill - ANYRES_SUFFIX_LEN] = 0
            logger.info(f"[ImageGen] Uncond prefill: {n_prefill} tokens, "
                       f"zeroed {max(0, n_prefill - ANYRES_SUFFIX_LEN)} prompt tokens, "
                       f"kept {min(n_prefill, ANYRES_SUFFIX_LEN)} suffix tokens")

            # Create a minimal forward batch for the uncond prefill using copy
            import copy
            from sglang.srt.model_executor.forward_batch_info import ForwardMode
            uncond_fb = copy.copy(forward_batch)
            uncond_fb.batch_size = 1
            uncond_fb.req_pool_indices = torch.tensor([uncond_idx], device=forward_batch.req_pool_indices.device)
            uncond_fb.seq_lens = torch.tensor([n_prefill], dtype=torch.int32, device=forward_batch.seq_lens.device)
            uncond_fb.seq_lens_sum = n_prefill
            uncond_fb.positions = torch.arange(n_prefill, device=forward_batch.positions.device)
            uncond_fb.forward_mode = ForwardMode.EXTEND
            uncond_fb.extend_prefix_lens = torch.tensor([0], dtype=torch.int32, device=forward_batch.seq_lens.device)
            uncond_fb.extend_seq_lens = torch.tensor([n_prefill], dtype=torch.int32, device=forward_batch.seq_lens.device)
            uncond_fb.extend_seq_lens_cpu = [n_prefill]
            uncond_fb.extend_prefix_lens_cpu = [0]
            uncond_fb.out_cache_loc = token_locs
            uncond_fb.mm_inputs = None

            # Init attention backend for uncond prefill
            self._model_runner.attn_backend.init_forward_metadata(uncond_fb)
            uncond_fb.attn_backend = self._model_runner.attn_backend

            # Run backbone with unconditional embeddings
            self.model(input_ids=None, positions=uncond_fb.positions,
                      forward_batch=uncond_fb, input_embeds=uncond_embeds)

            logger.info(f"[ImageGen] Unconditional prefill: {n_prefill} tokens, req_pool_idx={uncond_idx}")
            return uncond_idx

        except Exception as e:
            logger.error(f"[ImageGen] Failed to allocate uncond KV: {e}", exc_info=True)
            return -1

    def _run_uncond_decode(self, state: ImageGenState, position: int,
                          forward_batch, is_newline: bool = False) -> Optional[torch.Tensor]:
        """Run one unconditional decode step and return the hidden state."""
        if state.uncond_req_pool_idx < 0 or self._model_runner is None:
            return None

        try:
            rtp = self._model_runner.req_to_token_pool
            alloc = self._model_runner.token_to_kv_pool_allocator

            # Allocate one more token page for this decode step
            token_loc = alloc.alloc(1)
            if token_loc is None:
                return None

            rtp.req_to_token[state.uncond_req_pool_idx, state.uncond_seq_len] = token_loc

            # Create decode forward batch from copy of current
            import copy
            from sglang.srt.model_executor.forward_batch_info import ForwardMode
            uncond_fb = copy.copy(forward_batch)
            uncond_fb.batch_size = 1
            uncond_fb.req_pool_indices = torch.tensor([state.uncond_req_pool_idx],
                                                       device=forward_batch.req_pool_indices.device)
            uncond_fb.seq_lens = torch.tensor([state.uncond_seq_len + 1], dtype=torch.int32,
                                              device=forward_batch.seq_lens.device)
            uncond_fb.seq_lens_sum = state.uncond_seq_len + 1
            uncond_fb.positions = torch.tensor([position], device=forward_batch.positions.device)
            uncond_fb.forward_mode = ForwardMode.DECODE
            uncond_fb.out_cache_loc = token_loc

            # Init attention backend for uncond decode
            self._model_runner.attn_backend.init_forward_metadata(uncond_fb)
            uncond_fb.attn_backend = self._model_runner.attn_backend

            # For image_pad: zero embedding (like original)
            # For image_newline: use real embedding (structural signal)
            if is_newline:
                newline_id = torch.tensor([self._image_newline_id], device=forward_batch.positions.device)
                if self.model.use_ngram_embedding:
                    embed = self.model.embed_tokens.word_embeder(newline_id)
                else:
                    embed = self.model.embed_tokens(newline_id)
            else:
                # A1/CFG fix: uncond path must also feed back the previously generated
                # codebook tokens (CFG differs only in the text prompt, not the AR history).
                if len(state.accumulated_ids) > 0:
                    _pv = state.accumulated_ids[-1].unsqueeze(0).to(forward_batch.positions.device)
                    embed = self.visual_tokenizer.visual_embedding_layer(
                        self._embed_multimodal_ids(_pv)).to(torch.bfloat16).reshape(1, -1)
                else:
                    embed = torch.zeros(1, self.config.hidden_size,
                                       dtype=torch.bfloat16, device=forward_batch.positions.device)

            # Run backbone
            hidden = self.model(input_ids=None, positions=uncond_fb.positions,
                               forward_batch=uncond_fb, input_embeds=embed)

            state.uncond_seq_len += 1
            return hidden  # [1, hidden_size]

        except Exception as e:
            logger.error(f"[ImageGen] Uncond decode failed: {e}", exc_info=True)
            return None

    def _free_uncond_kv(self, state: ImageGenState):
        """Free the unconditional KV cache."""
        if state.uncond_req_pool_idx >= 0 and self._model_runner is not None:
            try:
                rtp = self._model_runner.req_to_token_pool
                alloc = self._model_runner.token_to_kv_pool_allocator
                token_locs = rtp.req_to_token[state.uncond_req_pool_idx, :state.uncond_seq_len]
                alloc.free(token_locs)
                rtp.free_slots.append(state.uncond_req_pool_idx)
                logger.info(f"[ImageGen] Freed uncond KV: {state.uncond_seq_len} tokens")
            except Exception as e:
                logger.warning(f"[ImageGen] Failed to free uncond KV: {e}")

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text for logging."""
        try:
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                model_path = os.environ.get('SGLANG_MODEL_PATH', '/workspace/model')
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            return self._tokenizer.decode([token_id])
        except Exception:
            return f"<{token_id}>"

    def _make_full_config(self, config):
        """Create a config object compatible with the multimodal tokenizers.

        The visual tokenizer's VisualEncoder needs a Qwen2_5_VLVisionConfig.
        We construct proper HF config objects where needed, and wrap the rest
        as DictConfig for attribute access.
        """
        # Use config's to_dict() if available (HF PretrainedConfig), else vars()
        if hasattr(config, 'to_dict'):
            cfg_dict = config.to_dict()
        else:
            cfg_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}

        # Create the DictConfig base
        full_cfg = DictConfig(cfg_dict)

        # Merge HF-default config fields into visual_config (Qwen2.5-VL vision) and
        # audio_config (Whisper) so the tokenizer submodules find fields like window_size /
        # scale_embedding that the checkpoint config dicts omit. Checkpoint values win.
        import inspect as _insp
        def _merge_defaults(sub_key, cfg_cls):
            sub = cfg_dict.get(sub_key)
            if not isinstance(sub, dict):
                return
            try:
                valid = {k: v for k, v in sub.items() if isinstance(k, str) and k in _insp.signature(cfg_cls.__init__).parameters}
                defaults = cfg_cls(**valid).to_dict()
                merged = {**defaults, **sub}
                for _drop in ("id2label", "label2id", "torch_dtype"):
                    merged.pop(_drop, None)
                setattr(full_cfg, sub_key, DictConfig(merged))
            except Exception as e:
                logger.warning(f"Could not merge defaults for {sub_key}: {e}")
        try:
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
            _merge_defaults("visual_config", Qwen2_5_VLVisionConfig)
        except Exception as e:
            logger.warning(f"visual_config default merge skipped: {e}")
        try:
            from transformers.models.whisper.configuration_whisper import WhisperConfig
            _merge_defaults("audio_config", WhisperConfig)
        except Exception as e:
            logger.warning(f"audio_config default merge skipped: {e}")

        return full_cfg

    def _get_nested(self, obj, key):
        """Access nested config attribute from dict or object."""
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    def _init_codebook_offsets(self, config):
        """Initialize codebook offset values for multimodal token embedding.

        The embedding table layout is: [text | audio | visual]
        - audio starts at config.audio_offset
        - visual starts at config.visual_offset
        Offsets use cumsum like the original model:
          offset_list = [base_offset] + codebook_sizes[:-1]
          offset_vals = cumsum(offset_list)
        """
        vc = getattr(config, 'visual_config', None)
        if vc is not None:
            vq = self._get_nested(vc, 'vq_config')
            codebook_sizes = self._get_nested(vq, 'codebook_sizes')
            visual_offset = getattr(config, 'visual_offset', None)
            if visual_offset is None:
                # Fallback: audio comes before visual
                audio_total = 0
                ac = getattr(config, 'audio_config', None)
                if ac is not None:
                    audio_vq = self._get_nested(ac, 'vq_config')
                    audio_total = sum(self._get_nested(audio_vq, 'codebook_sizes'))
                text_vocab = getattr(config, 'text_vocab_plus_multimodal_special_token_size', 131125)
                visual_offset = text_vocab + audio_total
            offset_list = [visual_offset] + list(codebook_sizes[:-1])
            offsets = torch.cumsum(torch.tensor(offset_list, dtype=torch.long), dim=0)
            self.register_buffer(
                "visual_offset_vals",
                offsets,
                persistent=False,
            )
        else:
            self.visual_offset_vals = None

        ac = getattr(config, 'audio_config', None)
        if ac is not None:
            vq = self._get_nested(ac, 'vq_config')
            codebook_sizes = self._get_nested(vq, 'codebook_sizes')
            audio_offset = getattr(config, 'audio_offset', None)
            if audio_offset is None:
                audio_offset = getattr(config, 'text_vocab_plus_multimodal_special_token_size', 131125)
            offset_list = [audio_offset] + list(codebook_sizes[:-1])
            offsets = torch.cumsum(torch.tensor(offset_list, dtype=torch.long), dim=0)
            self.register_buffer(
                "audio_offset_vals",
                offsets,
                persistent=False,
            )
        else:
            self.audio_offset_vals = None

    def pad_input_ids(self, input_ids: List[int], mm_inputs):
        """Pad input_ids with placeholder tokens for multimodal inputs.

        SGLang calls this during request preprocessing. Uses the standard
        multimodal padding pattern.
        """
        from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def get_image_feature(self, items) -> torch.Tensor:
        """Encode images through the visual tokenizer.

        Flow: pixel_values → visual encoder → VQ-RQ → codebook IDs
              → embed_tokens(IDs + offset) → sum over codebooks → project

        Returns: [total_visual_tokens, hidden_size] tensor
        """
        if self.visual_tokenizer is None:
            return None

        # Extract pixel values and grid info from items
        pixel_values = torch.cat(
            [item.feature for item in items], dim=0
        ).to(dtype=self.visual_tokenizer.visual_model.get_dtype(),
             device=next(self.visual_tokenizer.parameters()).device)

        image_grid_thw = torch.cat(
            [item.image_grid_thw for item in items], dim=0
        )

        # Encode through visual tokenizer: pixels → VQ codes [seq, num_codebooks]
        visual_ids = self.visual_tokenizer.encode(pixel_values, image_grid_thw)

        # Add codebook offsets
        if self.visual_offset_vals is not None:
            visual_ids = visual_ids + self.visual_offset_vals.to(visual_ids.device)

        # Embed codebook IDs using the full codebook embedding table
        visual_embeddings = self._embed_multimodal_ids(visual_ids)  # [seq, hidden_size]

        # Project through visual embedding bridge
        visual_embeddings = self.visual_tokenizer.visual_embedding_layer(visual_embeddings)

        return visual_embeddings

    def _load_codebook_embeddings(self):
        """Load the separate codebook embedding table for VQ token lookups."""
        if self._codebook_embed is not None:
            return
        import os
        from safetensors import safe_open
        model_path = os.environ.get('SGLANG_MODEL_PATH', '')
        # Try to find codebook_embeddings.safetensors in model directory
        for path in [model_path, '/workspace/model']:
            cb_path = os.path.join(path, 'codebook_embeddings.safetensors')
            if os.path.exists(cb_path):
                with safe_open(cb_path, framework='pt') as sf:
                    device = next(self.parameters()).device
                    self._codebook_embed = sf.get_tensor('codebook_embeddings').to(device)
                    logger.info(f"Loaded codebook embeddings: {self._codebook_embed.shape}")
                return
        logger.warning("codebook_embeddings.safetensors not found, multimodal VQ lookups will be clamped")

    def _embed_multimodal_ids(self, ids_with_offset):
        """Embed multimodal VQ IDs using the full codebook embedding table.

        ids_with_offset: [seq, num_codebooks] with codebook offsets applied
        Returns: [seq, hidden_size] summed over codebooks
        """
        self._load_codebook_embeddings()

        if hasattr(self.model.embed_tokens, 'word_embeder'):
            word_embed = self.model.embed_tokens.word_embeder
        else:
            word_embed = self.model.embed_tokens

        text_vocab = word_embed.num_embeddings
        codebook_base = getattr(self.config, 'text_vocab_plus_multimodal_special_token_size', 131125)

        all_embeds = []
        for cb_level in range(ids_with_offset.shape[1]):
            token_ids = ids_with_offset[:, cb_level]
            # IDs < text_vocab → use word_embed, IDs >= codebook_base → use codebook table
            in_text_range = token_ids < text_vocab
            in_codebook_range = token_ids >= codebook_base

            embeds = torch.zeros(len(token_ids), word_embed.embedding_dim,
                                dtype=word_embed.weight.dtype, device=token_ids.device)

            if in_text_range.any():
                embeds[in_text_range] = word_embed(token_ids[in_text_range])

            if in_codebook_range.any() and self._codebook_embed is not None:
                cb_indices = token_ids[in_codebook_range] - codebook_base
                cb_indices = cb_indices.clamp(max=self._codebook_embed.shape[0] - 1)
                embeds[in_codebook_range] = self._codebook_embed[cb_indices].to(embeds.dtype)

            all_embeds.append(embeds)

        return torch.stack(all_embeds, dim=1).sum(dim=1)  # [seq, hidden_size]

    def _codebook_embed_fn(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embedding function for codebook tokens, used by the audio/visual heads.

        Maps token IDs (with codebook offsets applied) to embeddings.
        Works as a drop-in for the original model's embed_tokens.
        token_ids: any shape of integer token IDs
        Returns: (*token_ids.shape, hidden_size) embeddings
        """
        self._load_codebook_embeddings()
        orig_shape = token_ids.shape
        flat_ids = token_ids.reshape(-1)

        if hasattr(self.model.embed_tokens, 'word_embeder'):
            word_embed = self.model.embed_tokens.word_embeder
        else:
            word_embed = self.model.embed_tokens

        codebook_base = getattr(self.config, 'text_vocab_plus_multimodal_special_token_size', 131125)
        text_vocab = word_embed.num_embeddings
        hidden_size = word_embed.embedding_dim

        embeds = torch.zeros(len(flat_ids), hidden_size,
                            dtype=word_embed.weight.dtype, device=flat_ids.device)

        in_text = flat_ids < text_vocab
        in_cb = flat_ids >= codebook_base

        if in_text.any():
            embeds[in_text] = word_embed(flat_ids[in_text])
        if in_cb.any() and self._codebook_embed is not None:
            cb_idx = (flat_ids[in_cb] - codebook_base).clamp(max=self._codebook_embed.shape[0] - 1)
            embeds[in_cb] = self._codebook_embed[cb_idx].to(embeds.dtype)

        return embeds.view(*orig_shape, hidden_size)

    @torch.no_grad()
    def _generate_audio_codebook_step(
        self, hidden_state: torch.Tensor, state: AudioGenState
    ) -> torch.Tensor:
        """Run one step of audio codebook generation via the depth transformer.

        hidden_state: [1, hidden_size] from the backbone
        state: per-request AudioGenState
        Returns: [num_codebooks] tensor of codebook token IDs (with offsets applied)
        """
        device = hidden_state.device
        num_codebooks = len(self._audio_codebook_sizes)

        # Build previous token tensor for conditioning
        # next_token_ids tracks accumulated tokens WITH offsets for embedding
        next_token_ids = torch.zeros(1, num_codebooks, dtype=torch.long, device=device)

        # Build prev_audio_ids context for repetition penalty
        if state.accumulated_ids:
            prev_ids = torch.stack(state.accumulated_ids[-50:], dim=0)  # last 50 frames
        else:
            prev_ids = torch.zeros(0, num_codebooks, dtype=torch.long, device=device)

        for level in range(num_codebooks):
            # Run audio head at this level
            logits = self.audio_head(
                hidden_state,
                next_token_ids,
                self._codebook_embed_fn,
                level,
            )  # [1, codebook_size + 1]

            # One-time diagnostic: check weights and hidden state
            if _LCN_VERBOSE and state.step_count == 0 and level == 0 and not getattr(self, '_audio_head_checked', False):
                self._audio_head_checked = True
                hp_norm = self.audio_head.hidden_proj.weight.float().norm().item()
                h0_norm = self.audio_head.heads[0].weight.float().norm().item()
                q0_norm = self.audio_head.transformer_layers[0].self_attention.q_proj.weight.float().norm().item()
                hs_norm = hidden_state.float().norm().item()
                hs_mean = hidden_state.float().mean().item()
                hs_std = hidden_state.float().std().item()
                hs_nonzero = (hidden_state.abs() > 1e-6).sum().item()
                # Also save hidden state + first 5 frames of generated tokens for offline analysis
                torch.save({
                    'hidden_state': hidden_state.cpu(),
                    'audio_offset_vals': self.audio_offset_vals.cpu(),
                    'codebook_sizes': self._audio_codebook_sizes,
                }, '/tmp/audio_head_debug.pt')
                logger.info(f"[AudioGen] Head weight norms: hidden_proj={hp_norm:.2f} heads.0={h0_norm:.2f} q_proj.0={q0_norm:.2f}")
                logger.info(f"[AudioGen] Hidden state: norm={hs_norm:.2f} mean={hs_mean:.6f} std={hs_std:.4f} nonzero={hs_nonzero}/{hidden_state.numel()}")
                logger.info(f"[AudioGen] Saved debug data to /tmp/audio_head_debug.pt")
                # Expected: ~186, ~464, ~403 from checkpoint. Random init: ~32, ~52, ~32

            # Diagnostic: log logits stats for first few steps (level 0 only)
            if _LCN_VERBOSE and state.step_count == 0 and level == 0:
                end_logit = logits[0, self._audio_codebook_sizes[0]].item()
                top5_vals, top5_idx = logits[0].topk(5)
                logger.info(f"[AudioGen] step={state.step_count} level={level} "
                           f"end_logit={end_logit:.3f} "
                           f"top5_idx={top5_idx.tolist()} top5_vals={[f'{v:.3f}' for v in top5_vals.tolist()]} "
                           f"logits_range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")

            # End-of-audio flag (index == codebook_sizes[level]) is only meaningful at level 0.
            # NO arbitrary minimum-length floor: at level 0 the flag is sampleable freely and
            # adjudicated by END_CONFIRM (below); at levels >0 it is always masked.
            end_token_idx = self._audio_codebook_sizes[level]
            if level == 0:
                tok = self._sample_codebook_logits(logits, level, prev_ids)
                if int(tok) == end_token_idx:
                    state.end_run += 1
                    if state.end_run >= AUDIO_END_CONFIRM:
                        state.ended = True
                        return None  # confirmed genuine end — do NOT store this flag frame
                    # isolated/stray end-flag: re-sample level 0 with the end slot masked so
                    # the frame carries real speech content (the model keeps speaking).
                    logits[0, end_token_idx] = float('-inf')
                    tok = self._sample_codebook_logits(logits, level, prev_ids)
                else:
                    state.end_run = 0
                next_token = tok
            else:
                logits[0, end_token_idx] = float('-inf')
                next_token = self._sample_codebook_logits(logits, level, prev_ids)

            next_token_ids[0, level] = next_token + self.audio_offset_vals[level].item()

        return next_token_ids[0]  # [num_codebooks] with offsets

    def _sample_codebook_logits(self, logits, level, prev_ids):
        """Rep-penalty + temperature + top-k/top-p multinomial sample of one codebook level.
        Clones logits (non-mutating) and returns the raw (pre-offset) token id (0-dim tensor)."""
        logits = logits.clone()
        if prev_ids.shape[0] > 0 and AUDIO_GEN_REPETITION_PENALTY != 1.0:
            prev_level_ids = (prev_ids[:, level] - self.audio_offset_vals[level].item()).clamp(
                min=0, max=logits.shape[-1] - 1)
            for pid in prev_level_ids.unique():
                pid_int = pid.item()
                if logits[0, pid_int] > 0:
                    logits[0, pid_int] /= AUDIO_GEN_REPETITION_PENALTY
                else:
                    logits[0, pid_int] *= AUDIO_GEN_REPETITION_PENALTY
        logits = logits / AUDIO_GEN_TEMPERATURE
        if AUDIO_GEN_TOP_K > 0:
            top_k_vals, _ = logits.topk(min(AUDIO_GEN_TOP_K, logits.shape[-1]), dim=-1)
            logits = logits.masked_fill(logits < top_k_vals[:, -1:], float('-inf'))
        if AUDIO_GEN_TOP_P < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            mask = cumulative_probs - probs > AUDIO_GEN_TOP_P
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _is_audio_end_token(self, audio_ids: torch.Tensor) -> bool:
        """Check if level-0 generated the end-of-audio token.

        The end token is codebook_sizes[0] (=8192), which maps to
        audio_offset_vals[1] when offset is added.
        """
        level0_raw = audio_ids[0].item() - self.audio_offset_vals[0].item()
        return level0_raw == self._audio_codebook_sizes[0]

    @torch.no_grad()
    def _rid_for(self, req_idx, forward_batch) -> str:
        """Return a filesystem-safe per-request id (the SGLang request id) for this req_pool
        index, so generated artifacts are named uniquely per request — concurrency-safe
        retrieval without globbing/locks. Empty string if unavailable (caller falls back)."""
        try:
            rids = getattr(forward_batch, 'rids', None)
            if rids is None:
                return ""
            rpi = forward_batch.req_pool_indices.tolist()
            for bi, rp in enumerate(rpi):
                if rp == req_idx and bi < len(rids) and rids[bi]:
                    return ''.join(c for c in str(rids[bi]) if c.isalnum() or c in '-_')[:64]
        except Exception:
            pass
        return ""

    def _decode_audio_to_wav(self, state: AudioGenState) -> Optional[str]:
        """Decode accumulated codebook tokens to a WAV file.

        Returns the path to the saved WAV file, or None on failure.
        """
        if not state.accumulated_ids or self.audio_tokenizer is None:
            return None

        try:
            # Stack accumulated IDs: [num_frames, num_codebooks]
            audio_ids = torch.stack(state.accumulated_ids, dim=0)

            # Remove offsets to get raw codebook indices
            offsets = self.audio_offset_vals.to(audio_ids.device)
            raw_ids = audio_ids - offsets.unsqueeze(0)

            # NOTE: no end-token truncation here — the gen loop never stores the end-flag frame
            # (see _generate_audio_codebook_step, which returns None on confirmed end), so
            # accumulated_ids can't contain a level-0 end token. The marker is appended below.
            if raw_ids.shape[0] == 0:
                logger.warning("No valid audio frames to decode")
                return None

            # Clamp each level's IDs to valid codebook range [0, codebook_size-1]
            for lvl in range(raw_ids.shape[1]):
                raw_ids[:, lvl] = raw_ids[:, lvl].clamp(min=0, max=self._audio_codebook_sizes[lvl] - 1)

            # No manual end-of-audio marker: lazy_decode_and_save pads a codebook_sizes[0] row
            # itself when the last frame isn't one, and decode_wave_vocoder2 slices it off before
            # vocoding (response[:, :response_len]). Appending here would be redundant.
            logger.info(f"Decoding {raw_ids.shape[0]} audio frames through vocoder pipeline")

            # Ensure vocoder weight path is resolved correctly
            self._ensure_vocoder_path()

            # Use lazy_decode_and_save from the audio tokenizer
            _tag = state.rid or str(int(time.time()))
            save_path = f"{os.environ.get('LCN_OUTPUT_DIR', '/tmp')}/longcat_tts_{_tag}.wav"
            self.audio_tokenizer.lazy_decode_and_save(
                raw_ids,
                sampling_rate=AUDIO_GEN_SAMPLING_RATE,
                wave_concat_overlap=AUDIO_GEN_WAVE_OVERLAP,
                save_path=save_path,
            )
            logger.info(f"Audio saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Audio decode failed: {e}", exc_info=True)
            return None

    def _ensure_vocoder_path(self):
        """Ensure the vocoder weight path is valid, searching model directory."""
        if self.audio_tokenizer is None:
            return
        ac = self.audio_tokenizer.config.audio_config
        voc_cfg = getattr(ac, 'cosy24kvocoder_config', None)
        if voc_cfg is None:
            return
        weight_path = getattr(voc_cfg, 'weight_path', '')
        if weight_path and os.path.exists(weight_path):
            return  # already valid

        # Try to find vocoder in model directory
        model_path = os.environ.get('SGLANG_MODEL_PATH', '/workspace/model')
        candidates = [
            os.path.join(model_path, 'cosy24k_vocoder', 'hift.pt'),
            os.path.join(model_path, 'hift.pt'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                if isinstance(voc_cfg, dict):
                    voc_cfg['weight_path'] = candidate
                else:
                    voc_cfg.weight_path = candidate
                logger.info(f"Vocoder weight path resolved to {candidate}")
                return
        logger.warning(f"Vocoder weights not found (tried {candidates})")

    @torch.no_grad()
    def _generate_image_codebook_step(
        self, cond_hidden: torch.Tensor, uncond_hidden: Optional[torch.Tensor],
        state: ImageGenState,
    ) -> torch.Tensor:
        """Run one step of image codebook generation via the depth transformer.

        Uses Classifier-Free Guidance (CFG) when uncond_hidden is available:
          fused = cfg_scale * (cond - uncond) + uncond

        cond_hidden: [1, hidden_size] — conditional hidden state from backbone
        uncond_hidden: [1, hidden_size] or None — unconditional hidden state
        Returns: [num_codebooks] tensor of codebook token IDs (with offsets applied)
        """
        device = cond_hidden.device
        num_codebooks = len(self._visual_codebook_sizes)
        cfg_scale = IMAGE_GEN_CFG_SCALE

        if cfg_scale != 1.0 and uncond_hidden is not None:
            batched_hidden = torch.cat([cond_hidden, uncond_hidden], dim=0)
        else:
            batched_hidden = cond_hidden

        bs = batched_hidden.shape[0]
        next_token_ids = torch.zeros(bs, num_codebooks, dtype=torch.long, device=device)

        for level in range(num_codebooks):
            logits = self.visual_head(
                batched_hidden, next_token_ids, self._codebook_embed_fn, level,
            )

            # CFG fusion
            if cfg_scale != 1.0 and logits.shape[0] == 2:
                cond_logits, uncond_logits = logits.chunk(2, dim=0)
                logits = cfg_scale * (cond_logits - uncond_logits) + uncond_logits

            # Sampling
            logits = logits / IMAGE_GEN_TEMPERATURE

            if IMAGE_GEN_TOP_K > 0:
                top_k_vals, _ = logits.topk(min(IMAGE_GEN_TOP_K, logits.shape[-1]), dim=-1)
                logits[logits < top_k_vals[:, -1:]] = float('-inf')

            if IMAGE_GEN_TOP_P < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                mask = cumulative_probs - probs > IMAGE_GEN_TOP_P
                sorted_logits[mask] = float('-inf')
                logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token_with_offset = next_token + self.visual_offset_vals[level].item()
            next_token_ids[:, level] = next_token_with_offset

        return next_token_ids[0]

    @torch.no_grad()
    def _decode_image_to_png(self, state: ImageGenState) -> Optional[str]:
        """Decode accumulated visual codebook tokens to a PNG file."""
        if not state.accumulated_ids or self.visual_tokenizer is None:
            return None

        try:
            visual_ids = torch.stack(state.accumulated_ids, dim=0)
            offsets = self.visual_offset_vals.to(visual_ids.device)
            raw_ids = visual_ids - offsets.unsqueeze(0)

            # Clamp to valid range
            for lvl in range(raw_ids.shape[1]):
                raw_ids[:, lvl] = raw_ids[:, lvl].clamp(min=0, max=self._visual_codebook_sizes[lvl] - 1)

            logger.info(f"Decoding {raw_ids.shape[0]} visual tokens ({state.token_h}x{state.token_w}) through image decoder")

            # Resolve decoder weight path
            self._ensure_visual_decoder_path()

            _tag = state.rid or str(int(time.time()))
            save_path = f"{os.environ.get('LCN_OUTPUT_DIR', '/tmp')}/longcat_img_{_tag}.png"
            result = self.visual_tokenizer.lazy_decode_and_save(
                raw_ids, state.token_h, state.token_w, save_path,
            )
            logger.info(f"Image saved to {result}")
            return result[0] if isinstance(result, list) else save_path

        except Exception as e:
            logger.error(f"Image decode failed: {e}", exc_info=True)
            return None

    def _ensure_visual_decoder_path(self):
        """Ensure visual decoder weight path is valid."""
        if self.visual_tokenizer is None:
            return
        vc = getattr(self.visual_tokenizer, 'config', None)
        if vc is None:
            return
        vdc = getattr(getattr(vc, 'visual_config', None), 'visual_decoder_config', None)
        if vdc is None:
            return
        weight_path = getattr(vdc, 'weight_path', '')
        if weight_path and os.path.exists(weight_path):
            return
        model_path = os.environ.get('SGLANG_MODEL_PATH', '/workspace/model')
        candidates = [
            os.path.join(model_path, 'image_decoder', 'image_decoder.safetensors'),
            os.path.join(model_path, 'image_decoder.safetensors'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                vdc.weight_path = candidate
                logger.info(f"Visual decoder weight path resolved to {candidate}")
                return
        logger.warning(f"Visual decoder weights not found")

    @torch.no_grad()
    def get_audio_feature(self, items) -> torch.Tensor:
        """Encode audio through the audio tokenizer.

        Flow: mel spectrogram → whisper encoder → bridge → VQ → codebook IDs
              → embed_tokens(IDs + offset) → sum over codebooks

        Returns: [total_audio_tokens, hidden_size] tensor
        """
        if self.audio_tokenizer is None:
            return None

        all_embeddings = []
        for item in items:
            audio_features = item.feature  # mel spectrogram from processor
            encoder_length = item.model_specific_data.get('encoder_length', None)
            bridge_length = item.model_specific_data.get('bridge_length', None)

            if encoder_length is None or bridge_length is None:
                continue

            device = next(self.audio_tokenizer.parameters()).device

            # Encode through audio tokenizer: mel → VQ codes [seq, num_codebooks]
            audio_tensor = torch.tensor(audio_features, dtype=torch.float32, device=device).unsqueeze(0)
            audio_ids = self.audio_tokenizer.encode(
                audio_tensor,
                torch.tensor([encoder_length], device=device),
                torch.tensor([bridge_length], device=device),
            )

            # Add codebook offsets and embed using full codebook table
            if self.audio_offset_vals is not None:
                offset_ids = audio_ids + self.audio_offset_vals.to(audio_ids.device)
            else:
                offset_ids = audio_ids

            audio_embeddings = self._embed_multimodal_ids(offset_ids)  # [actual_seq, hidden_size]

            # Pad or truncate to match the expected bridge_length from processor
            actual_len = audio_embeddings.shape[0]
            if actual_len < bridge_length:
                pad = torch.zeros(bridge_length - actual_len, audio_embeddings.shape[1],
                                dtype=audio_embeddings.dtype, device=audio_embeddings.device)
                audio_embeddings = torch.cat([audio_embeddings, pad])
            elif actual_len > bridge_length:
                audio_embeddings = audio_embeddings[:bridge_length]
            all_embeddings.append(audio_embeddings)

        if not all_embeddings:
            return None
        return torch.cat(all_embeddings, dim=0)

    def _get_mm_items(self, forward_batch):
        """Extract multimodal items from the forward batch."""
        mm_inputs_list = getattr(forward_batch, 'mm_inputs', None)
        if not mm_inputs_list:
            return [], []
        image_items, audio_items = [], []
        for i, mm_input in enumerate(mm_inputs_list):
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                if item.is_image():
                    image_items.append(item)
                elif item.is_audio():
                    audio_items.append(item)
        return image_items, audio_items

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
    ):
        """Forward pass with multimodal support + audio generation.

        For multimodal input, we need to preserve N-gram embeddings for text
        tokens while replacing multimodal placeholder positions with codebook
        embeddings. The original model:
        1. Zeros out multimodal placeholder positions in input_ids
        2. Computes N-gram embeddings for ALL positions
        3. ADDS multimodal embeddings at placeholder positions

        For audio generation (TTS):
        - When audiogen_start_token_id is detected, enter audio mode
        - Run audio_head on hidden states to generate 8 codebook tokens/step
        - Force audio_pad_token_id as next token (backbone conditioning)
        - When end-of-audio detected, force audiogen_end_token_id
        - Decode accumulated tokens through vocoder pipeline
        """
        # Check for any multimodal inputs (images or audio)
        is_decode = forward_batch.forward_mode.is_decode()
        has_image = not is_decode and forward_batch.contains_image_inputs()
        has_audio = not is_decode and hasattr(forward_batch, 'contains_audio_inputs') and forward_batch.contains_audio_inputs()
        if not has_audio and not is_decode:
            mm_inputs_list = getattr(forward_batch, 'mm_inputs_list', None)
            if mm_inputs_list:
                for mm_inputs in mm_inputs_list:
                    if mm_inputs and any(item.is_audio() for item in mm_inputs.mm_items):
                        has_audio = True
                        break
        has_mm = has_image or has_audio

        # Clamp OOB token IDs to the actual embedding table size.
        max_token_id = getattr(self.config, 'text_vocab_plus_multimodal_special_token_size', self.config.vocab_size) - 1
        input_ids = input_ids.clamp(min=0, max=max_token_id)

        # --- Step 1: Compute embeddings ---
        if has_mm and input_embeds is None:
            input_embeds = self._compute_mm_embeddings(input_ids, forward_batch)
            forward_batch.mm_inputs = None

        # --- Step 1b: Replace audiotext_pad with transcript tokens ---
        # The original model replaces audiotext_pad_token_id in input_ids with
        # the actual transcript token from audio_text_ids BEFORE computing
        # embeddings. This is critical: the backbone must see the transcript
        # text, not the pad token. The pad is just a scheduler-level placeholder.
        # Track which decode positions need embedding zeroing (audio/image gen)
        _gen_zero_mask = None
        _img_feedback = {}  # A1 fix: i -> prev generated visual codebook ids
        _aud_feedback = {}  # audio feedback: i -> prev generated audio codebook frame ids
        if is_decode and (self._audio_gen_states or self._image_gen_states) and input_embeds is None:
            for i in range(forward_batch.batch_size):
                req_idx = forward_batch.req_pool_indices[i].item()
                token = input_ids[i].item()

                # Audio gen: zero embedding at audio_pad positions, then feed back the
                # canonical get_audio_embeddings (Σ codebook_embed of the PREVIOUS frame).
                a_state = self._audio_gen_states.get(req_idx)
                if a_state is not None and a_state.mode == "generating" and token == self._audio_pad_id:
                    input_ids[i] = 0
                    if _gen_zero_mask is None:
                        _gen_zero_mask = torch.zeros(forward_batch.batch_size, dtype=torch.bool, device=input_ids.device)
                    _gen_zero_mask[i] = True
                    if len(a_state.accumulated_ids) > 0:
                        _aud_feedback[i] = a_state.accumulated_ids[-1]

                # Image gen: zero embedding at image_pad positions
                v_state = self._image_gen_states.get(req_idx)
                if v_state is not None and token == self._image_pad_id:
                    input_ids[i] = 0
                    if _gen_zero_mask is None:
                        _gen_zero_mask = torch.zeros(forward_batch.batch_size, dtype=torch.bool, device=input_ids.device)
                    _gen_zero_mask[i] = True
                    if len(v_state.accumulated_ids) > 0:
                        _img_feedback[i] = v_state.accumulated_ids[-1]

        # --- Step 2: Run backbone ---
        # If we have audio gen positions, compute embeddings manually so we
        # can zero them (original model zeros embedding at audio_pad positions).
        if _gen_zero_mask is not None and input_embeds is None:
            if self.model.use_ngram_embedding:
                input_embeds = self.model.embed_tokens(input_ids, forward_batch)
            else:
                input_embeds = self.model.embed_tokens(input_ids)
            input_embeds[_gen_zero_mask] = 0
            for _fi, _prev in _img_feedback.items():
                _pv = _prev.unsqueeze(0).to(input_embeds.device)
                _fb = self.visual_tokenizer.visual_embedding_layer(self._embed_multimodal_ids(_pv))
                input_embeds[_fi] = _fb.to(input_embeds.dtype).reshape(-1)
            # Audio feedback: canonical get_audio_embeddings = embed_tokens(prev_frame).sum(dim=1)
            # — NO visual_embedding_layer projection (audio's get_audio_embeddings only sums).
            for _fi, _prev in _aud_feedback.items():
                _pv = _prev.unsqueeze(0).to(input_embeds.device)
                _fb = self._embed_multimodal_ids(_pv)
                input_embeds[_fi] = _fb.to(input_embeds.dtype).reshape(-1)

        if input_embeds is not None:
            hidden_states = self.model(
                input_ids=None, positions=positions,
                forward_batch=forward_batch, input_embeds=input_embeds,
            )
        else:
            hidden_states = self.model(
                input_ids=input_ids, positions=positions,
                forward_batch=forward_batch,
            )

        # Handle aux_hidden_states from layer capture
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        # --- Step 3: Multimodal generation state machine (decode only) ---
        audio_logit_overrides = {}  # batch_idx → forced_token_id
        if is_decode and self.audio_head is not None:
            audio_logit_overrides = self._audio_gen_decode_step(
                input_ids, hidden_states, forward_batch
            )
        image_logit_overrides = {}
        if is_decode and self.visual_head is not None:
            image_logit_overrides = self._image_gen_decode_step(
                input_ids, hidden_states, forward_batch
            )

        # --- Step 4: Compute logits ---
        # The original model SKIPS lm_head during active visual/audio codebook
        # generation (only the multimodal head runs). We check if ALL positions
        # in the batch are in active generation mode — if so, we can skip
        # the expensive lm_head projection and just return forced logits.
        # Only REAL forced tokens (>=0) let us skip the lm_head. The transcript-phase
        # sentinel (-2) means "the lm_head IS needed this step" (we read its argmax in
        # Step 5), so it must NOT count toward _all_gen — otherwise the lm_head is
        # skipped, logits are all -inf, and the transcript decodes to <unk>.
        _all_gen = False
        if is_decode and (image_logit_overrides or audio_logit_overrides):
            n_forced = (sum(1 for v in audio_logit_overrides.values() if v >= 0)
                        + sum(1 for v in image_logit_overrides.values() if v >= 0))
            _all_gen = n_forced >= forward_batch.batch_size
        if _all_gen:
            # Skip lm_head — create minimal logits output with forced tokens
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput
            vocab = getattr(self.config, 'text_vocab_plus_multimodal_special_token_size', self.config.vocab_size)
            forced_logits = torch.full(
                (forward_batch.batch_size, vocab), float('-inf'),
                device=hidden_states.device, dtype=torch.float32)
            logits_output = LogitsProcessorOutput(next_token_logits=forced_logits)
        else:
            logits_output = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )

        # --- Step 5: Force tokens for audio gen requests ---
        if audio_logit_overrides and logits_output.next_token_logits is not None:
            for batch_idx, forced_token in audio_logit_overrides.items():
                if forced_token == -2:
                    # Transcript phase: check if lm_head naturally wants to end
                    # then force audiotext_pad_token_id as the actual sampled token.
                    req_idx = forward_batch.req_pool_indices[batch_idx].item()
                    state = self._audio_gen_states.get(req_idx)
                    if state is not None:
                        # The transcript phase recites the known input text; we decode it GREEDILY and
                        # force the emitted token to that argmax (one-hot below). That keeps recitation
                        # faithful AND makes the token actually sampled == the token the end-check
                        # tested, so there's no temp>0 decoupling between detection and emission. (Only
                        # the intermediate TEXT is greedy here; the acoustic codebooks are sampled
                        # separately in 'generating' mode.) max_transcript_steps is a runaway backstop,
                        # NOT a task-length floor — the transcript ends whenever the model wants.
                        nl = logits_output.next_token_logits[batch_idx]
                        lm_argmax = nl.argmax().item()
                        transcript_should_end = (
                            lm_argmax in (self._audiotext_pad_id, 2)
                            or state.transcript_steps >= state.max_transcript_steps
                        )
                        if transcript_should_end:
                            if lm_argmax == self._audiotext_pad_id:
                                reason = "natural (audiotext_pad)"
                            elif lm_argmax == 2:
                                reason = "EOS"
                            else:
                                reason = f"max ({state.max_transcript_steps})"
                            logger.info(f"[AudioGen] req={req_idx}: transcript ended ({reason}) "
                                       f"after {state.transcript_steps} steps, forcing audiotext_start")
                            nl[:] = float('-inf')
                            nl[self._audiotext_start_id] = 0.0
                            state.transcript_done = True
                        else:
                            # Continue: emit the argmax of the valid logits (EOS / audiogen_end masked
                            # out) as a one-hot. The emitted token passes through to the scheduler ->
                            # N-gram token table, so the next step's hash context is correct.
                            nl[2] = float('-inf')  # mask EOS
                            nl[self._audiogen_end_id] = float('-inf')  # mask audiogen_end
                            emit_id = nl.argmax().item()
                            if _LCN_VERBOSE and state.transcript_steps <= 12:
                                logger.info(f"[AudioGen] req={req_idx}: transcript step {state.transcript_steps}, "
                                           f"emit={emit_id} ('{self._decode_token(emit_id)}')")
                            nl[:] = float('-inf')
                            nl[emit_id] = 0.0
                elif forced_token >= 0:
                    logits_output.next_token_logits[batch_idx, :] = float('-inf')
                    logits_output.next_token_logits[batch_idx, forced_token] = 0.0

        # --- Step 5b: Force tokens for image gen requests ---
        if image_logit_overrides and logits_output.next_token_logits is not None:
            for batch_idx, forced_token in image_logit_overrides.items():
                if forced_token >= 0:
                    logits_output.next_token_logits[batch_idx, :] = float('-inf')
                    logits_output.next_token_logits[batch_idx, forced_token] = 0.0

        # --- Step 6: Check prefill for generation triggers (extend mode) ---
        if not is_decode and logits_output.next_token_logits is not None:
            self._check_prefill_audio_start(input_ids, logits_output, forward_batch)
            self._check_prefill_image_start(input_ids, logits_output, forward_batch)

        return logits_output

    def _compute_mm_embeddings(self, input_ids, forward_batch):
        """Compute embeddings with multimodal replacement."""
        image_items, audio_items = self._get_mm_items(forward_batch)

        if self.model.use_ngram_embedding:
            input_embeds = self.model.embed_tokens(input_ids, forward_batch)
        else:
            input_embeds = self.model.embed_tokens(input_ids)

        # Zero embeddings at pad positions before replacement
        def _cfg_val(obj, key, default):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)
        ac_cfg = getattr(self.config, 'audio_config', {})
        vc_cfg = getattr(self.config, 'visual_config', {})
        vis_pad_id = _cfg_val(vc_cfg, 'image_pad_token_id', 131108)
        aud_pad_id = _cfg_val(ac_cfg, 'audio_pad_token_id', 131105)
        pad_mask = (input_ids == vis_pad_id) | (input_ids == aud_pad_id)
        input_embeds[pad_mask] = 0

        # Replace audio embeddings
        if audio_items:
            audio_embeds = self.get_audio_feature(audio_items)
            if audio_embeds is not None:
                self._replace_mm_embeddings(input_embeds, audio_items, audio_embeds, forward_batch)

        # Replace image embeddings
        if image_items:
            image_embeds = self.get_image_feature(image_items)
            if image_embeds is not None:
                self._replace_mm_embeddings(input_embeds, image_items, image_embeds, forward_batch)

        return input_embeds

    def _replace_mm_embeddings(self, input_embeds, items, embeds, forward_batch):
        """Replace placeholder embeddings with multimodal embeddings."""
        embed_idx = 0
        for item in items:
            offsets = getattr(item, 'offsets', None)
            if offsets is None:
                continue
            for start, end in offsets:
                n_tokens = end - start
                if embed_idx + n_tokens > embeds.shape[0]:
                    n_tokens = embeds.shape[0] - embed_idx
                if n_tokens <= 0:
                    continue
                prefix_len = 0
                if hasattr(forward_batch, 'extend_prefix_lens_cpu') and forward_batch.extend_prefix_lens_cpu:
                    prefix_len = forward_batch.extend_prefix_lens_cpu[0]
                adj_start = start - prefix_len
                adj_end = adj_start + n_tokens
                if adj_start < 0:
                    embed_idx += n_tokens
                    continue
                if adj_end > input_embeds.shape[0]:
                    adj_end = input_embeds.shape[0]
                    n_tokens = adj_end - adj_start
                if n_tokens > 0:
                    input_embeds[adj_start:adj_end] = embeds[embed_idx:embed_idx+n_tokens].to(input_embeds.dtype)
                embed_idx += n_tokens

    def _audio_gen_decode_step(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Dict[int, int]:
        """Handle audio generation state machine during decode.

        Returns dict of {batch_idx: forced_token_id} for logit manipulation.
        """
        overrides = {}
        batch_size = forward_batch.batch_size

        for i in range(batch_size):
            req_idx = forward_batch.req_pool_indices[i].item()
            token = input_ids[i].item()

            state = self._audio_gen_states.get(req_idx)

            # --- State transitions based on current input token ---
            if token == self._audiogen_start_id:
                # Enter audio mode — start transcript phase
                # Let lm_head generate text normally (transcript of what to speak)
                state = AudioGenState(mode="transcript")
                self._audio_gen_states[req_idx] = state
                logger.info(f"[AudioGen] req={req_idx}: entered audio mode, starting transcript phase")
                # No override — let lm_head generate transcript text
                continue

            if token == self._audiotext_start_id and state is not None:
                # audiotext_start received → now start actual audio codebook generation
                state.mode = "generating"
                logger.info(f"[AudioGen] req={req_idx}: transcript done, audio generation started")

            if token == self._audiogen_end_id and state is not None:
                # Audio generation complete — decode and clean up
                logger.info(f"[AudioGen] req={req_idx}: audio generation ended, "
                           f"{len(state.accumulated_ids)} frames accumulated")
                wav_path = self._decode_audio_to_wav(state)
                if wav_path:
                    logger.info(f"[AudioGen] req={req_idx}: WAV saved to {wav_path}")
                del self._audio_gen_states[req_idx]
                continue  # back to text mode, no override needed

            # --- Transcript phase ---
            # The original model's backbone replaces audiotext_pad positions
            # with actual transcript tokens BEFORE computing N-gram embeddings.
            # The NgramCache then stores the transcript tokens for future hash
            # context. In SGLang, the scheduler writes tokens to the N-gram
            # token table BEFORE the model's forward. So we must let the actual
            # transcript token be the sampled output — this way the scheduler
            # writes it to the token table correctly, and the N-gram hash
            # context matches the original model.
            #
            # Flow: lm_head generates transcript text → passes through as
            # the sampled token → scheduler writes to token table → backbone
            # sees it on next step. Transcript ends when lm_head generates
            # audiotext_pad_token_id or EOS.
            if state is not None and state.mode == "transcript":
                state.transcript_steps += 1
                # Deferred to post-logits: check for transcript end, otherwise
                # let the lm_head's choice pass through (no override)
                overrides[i] = -2  # sentinel for transcript check
                continue

            # --- Active audio generation ---
            if state is not None and state.mode == "generating":
                # Run audio head on this request's hidden state. Returns None when a
                # CONFIRMED end-of-audio cluster is reached (state.ended) — the model
                # decides its own length; no minimum-frame floor. The end-flag frame is
                # NOT stored (it carries no audio).
                hs = hidden_states[i:i+1]  # [1, hidden_size]
                audio_ids = self._generate_audio_codebook_step(hs, state)
                if audio_ids is not None:
                    state.accumulated_ids.append(audio_ids)
                    state.step_count += 1

                # Terminate on confirmed end, or on the safety backstop (NOT a task-length
                # cutoff — just a runaway guard). On end: decode to WAV + EOS.
                if state.ended or state.step_count >= state.max_audio_steps:
                    if state.step_count >= state.max_audio_steps:
                        logger.warning(f"[AudioGen] req={req_idx}: hit safety cap ({state.max_audio_steps} frames)")
                    else:
                        logger.info(f"[AudioGen] req={req_idx}: confirmed end-of-audio, "
                                    f"{len(state.accumulated_ids)} frames")
                    state.rid = self._rid_for(req_idx, forward_batch)
                    wav_path = self._decode_audio_to_wav(state)
                    if wav_path:
                        logger.info(f"[AudioGen] req={req_idx}: WAV saved to {wav_path}")
                    if req_idx in self._audio_gen_states:
                        del self._audio_gen_states[req_idx]
                    overrides[i] = 2  # force EOS to terminate the request cleanly
                else:
                    # Continue generating — feed audio_pad_token_id to backbone
                    overrides[i] = self._audio_pad_id
                    if _LCN_VERBOSE and (state.step_count <= 5 or state.step_count % 50 == 0):
                        logger.info(f"[AudioGen] req={req_idx}: step {state.step_count}, "
                                   f"level0_raw={audio_ids[0].item() - self.audio_offset_vals[0].item()}")

        return overrides

    def _check_prefill_audio_start(
        self, input_ids: torch.Tensor, logits_output, forward_batch: ForwardBatch,
    ):
        """Check if prefill ends with audiogen_start_token_id.

        If so, force the first generated token to be audiotext_start_token_id
        and register the audio gen state for that request.
        """
        if not hasattr(forward_batch, 'extend_seq_lens_cpu') or not forward_batch.extend_seq_lens_cpu:
            return

        # During extend, input_ids is a flat concatenation of all requests.
        # We need to find the last token of each request.
        offset = 0
        for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if seq_len <= 0:
                continue
            last_token_pos = offset + seq_len - 1
            if last_token_pos < len(input_ids):
                last_token = input_ids[last_token_pos].item()
                if last_token == self._audiogen_start_id:
                    req_idx = forward_batch.req_pool_indices[i].item()
                    state = AudioGenState(mode="transcript")
                    self._audio_gen_states[req_idx] = state
                    # Let lm_head's transcript token pass through — the scheduler
                    # writes it to the N-gram token table for correct hash context.
                    # Only mask EOS to prevent early termination.
                    if logits_output.next_token_logits is not None and i < logits_output.next_token_logits.shape[0]:
                        lm_argmax = logits_output.next_token_logits[i].argmax().item()
                        logger.info(f"[AudioGen] req={req_idx}: detected audiogen_start in prefill, "
                                   f"first transcript token: '{self._decode_token(lm_argmax)}' ({lm_argmax})")
                        logits_output.next_token_logits[i, 2] = float('-inf')  # mask EOS
                        logits_output.next_token_logits[i, self._audiogen_end_id] = float('-inf')
            offset += seq_len

    def _image_gen_decode_step(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Dict[int, int]:
        """Handle image generation state machine during decode."""
        overrides = {}
        batch_size = forward_batch.batch_size

        for i in range(batch_size):
            req_idx = forward_batch.req_pool_indices[i].item()
            token = input_ids[i].item()

            state = self._image_gen_states.get(req_idx)

            # Detect image_start_token_id → enter visual mode
            if token == self._image_start_id:
                state = ImageGenState()
                self._image_gen_states[req_idx] = state
                logger.info(f"[ImageGen] req={req_idx}: entered visual mode (37x37 grid)")
                # Force image_pad as first token (first codebook gen position)
                overrides[i] = self._image_pad_id
                continue

            # Detect image_end → clean up and decode
            if token == self._image_end_id and state is not None:
                n_visual = len(state.accumulated_ids)
                logger.info(f"[ImageGen] req={req_idx}: image generation ended, "
                           f"{n_visual} visual tokens accumulated")
                self._free_uncond_kv(state)
                state.rid = self._rid_for(req_idx, forward_batch)
                img_path = self._decode_image_to_png(state)
                if img_path:
                    logger.info(f"[ImageGen] req={req_idx}: image saved to {img_path}")
                del self._image_gen_states[req_idx]
                continue

            # Active image generation
            if state is not None:
                # Check end condition
                if state.is_img_end:
                    logger.info(f"[ImageGen] req={req_idx}: image complete at token {state.current_image_token_num}, "
                               f"forcing image_end")
                    overrides[i] = self._image_end_id
                    continue

                # Check newline condition
                if state.is_img_newline:
                    # Run uncond forward for newline too (to keep KV caches in sync)
                    if state.uncond_initialized:
                        pos = forward_batch.positions[i].item()
                        self._run_uncond_decode(state, pos, forward_batch, is_newline=True)
                    overrides[i] = self._image_newline_id
                    state.current_image_token_num += 1
                    if state.current_image_token_num % (state.token_w + 1) == 0:
                        row = state.current_image_token_num // (state.token_w + 1)
                        if row % 10 == 0 or row == state.token_h - 1:
                            logger.info(f"[ImageGen] req={req_idx}: row {row}/{state.token_h}, "
                                       f"{len(state.accumulated_ids)} visual tokens")
                    continue

                # Generate codebook tokens for this position
                cond_hs = hidden_states[i:i+1]

                # Run unconditional forward for CFG if available
                uncond_hs = None
                if IMAGE_GEN_CFG_SCALE != 1.0 and self._model_runner is not None:
                    # Initialize uncond KV cache on first use (try only once)
                    if not state.uncond_initialized and state.uncond_req_pool_idx == -1 and state.current_image_token_num == 0:
                        # Build uncond prefill with real token IDs (for suffix preservation).
                        # Read the conditional tokens from the N-gram token table.
                        cond_seq_len = forward_batch.seq_lens[i].item()
                        rtp = self._model_runner.req_to_token_pool
                        cond_pool_idx = forward_batch.req_pool_indices[i].item()
                        # Read token IDs from token table (N-gram table stores them)
                        ngram_info = getattr(forward_batch, 'ngram_embedding_info', None)
                        if ngram_info is not None:
                            token_table = ngram_info.token_table
                            prefill_ids = token_table[cond_pool_idx, :cond_seq_len].to(input_ids.device)
                        else:
                            prefill_ids = torch.zeros(cond_seq_len, dtype=input_ids.dtype,
                                                     device=input_ids.device)
                        state.uncond_req_pool_idx = self._alloc_uncond_kv(
                            cond_pool_idx,
                            cond_seq_len, prefill_ids, forward_batch)
                        if state.uncond_req_pool_idx >= 0:
                            state.uncond_seq_len = cond_seq_len
                            state.uncond_initialized = True
                            logger.info(f"[ImageGen] CFG initialized: uncond_req={state.uncond_req_pool_idx}, "
                                       f"seq_len={cond_seq_len}")

                    if state.uncond_initialized:
                        pos = forward_batch.positions[i].item()
                        uncond_hs = self._run_uncond_decode(state, pos, forward_batch)

                visual_ids = self._generate_image_codebook_step(cond_hs, uncond_hs, state)
                state.accumulated_ids.append(visual_ids)
                state.current_image_token_num += 1
                overrides[i] = self._image_pad_id

                if state.current_image_token_num <= 3:
                    logger.info(f"[ImageGen] req={req_idx}: token {state.current_image_token_num}, "
                               f"level0_raw={visual_ids[0].item() - self.visual_offset_vals[0].item()}")

        return overrides

    def _check_prefill_image_start(
        self, input_ids: torch.Tensor, logits_output, forward_batch: ForwardBatch,
    ):
        """Check if prefill ends with image_start_token_id."""
        if not hasattr(forward_batch, 'extend_seq_lens_cpu') or not forward_batch.extend_seq_lens_cpu:
            return

        offset = 0
        for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if seq_len <= 0:
                continue
            last_token_pos = offset + seq_len - 1
            if last_token_pos < len(input_ids):
                last_token = input_ids[last_token_pos].item()
                if last_token == self._image_start_id:
                    req_idx = forward_batch.req_pool_indices[i].item()
                    state = ImageGenState()
                    self._image_gen_states[req_idx] = state
                    logger.info(f"[ImageGen] req={req_idx}: detected image_start in prefill, "
                               f"starting visual generation (37x37)")
                    # Force image_pad as first token
                    if logits_output.next_token_logits is not None and i < logits_output.next_token_logits.shape[0]:
                        logits_output.next_token_logits[i, :] = float('-inf')
                        logits_output.next_token_logits[i, self._image_pad_id] = 0.0
            offset += seq_len

    def _dequant_layer_to_bf16(self, layer_id):
        """Dequantize all FP4 linear weights in a decoder layer to BF16.

        This allows specific layers to run at full precision for better
        multimodal understanding, at the cost of ~2GB extra memory per layer.
        """
        layer = self.model.layers[layer_id]
        fp4_lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.bfloat16, device='cuda')
        dequanted = 0

        for name, module in layer.named_modules():
            if not hasattr(module, 'weight') or not hasattr(module, 'quant_method'):
                continue
            w = getattr(module, 'weight', None)
            if w is None or w.dtype != torch.uint8:
                continue
            # Has FP4 packed weight — dequantize
            # After process_weights_after_loading, weights may be in interleaved format.
            # Try both pre-interleaved (weight_scale/weight_scale_2) and
            # post-interleaved (weight_scale_interleaved/alpha) formats.
            w_scale = getattr(module, 'weight_scale', None)
            w_scale_2 = getattr(module, 'weight_scale_2', None)
            w_scale_il = getattr(module, 'weight_scale_interleaved', None)
            alpha = getattr(module, 'alpha', None)

            N, K_half = w.shape
            low = (w & 0x0F).to(torch.int64)
            high = (w >> 4).to(torch.int64)
            unpacked = torch.empty(N, K_half * 2, dtype=torch.bfloat16, device=w.device)
            unpacked[:, 0::2] = fp4_lut.to(w.device)[low]
            unpacked[:, 1::2] = fp4_lut.to(w.device)[high]
            K = K_half * 2

            if w_scale is not None and w_scale_2 is not None:
                # Pre-interleaved format
                group_size = K // w_scale.shape[1]
                actual_scale = w_scale.float() * w_scale_2.float()
                unpacked_blocked = unpacked.float().view(N, -1, group_size)
                dequant_w = (unpacked_blocked * actual_scale.unsqueeze(-1)).view(N, K).to(torch.bfloat16)
            elif w_scale_il is not None and alpha is not None:
                # Post-interleaved format — deinterleave scales
                # The interleaving groups 2 consecutive scale values for CUTLASS
                # Simplification: use alpha (global scale) with per-block FP8 scales
                n_groups = K // 16
                # weight_scale_interleaved: [N, n_groups] in fp8, interleaved
                raw_scale = w_scale_il.float()
                actual_scale = raw_scale * alpha.float()
                unpacked_blocked = unpacked.float().view(N, n_groups, 16)
                dequant_w = (unpacked_blocked * actual_scale.unsqueeze(-1)).view(N, K).to(torch.bfloat16)
            else:
                continue

            # Replace the weight and switch to unquantized linear method
            module.weight = nn.Parameter(dequant_w, requires_grad=False)
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
            module.quant_method = UnquantizedLinearMethod()
            # Remove FP4-specific attributes that would confuse unquantized path
            for attr in ['weight_scale', 'weight_scale_2', 'weight_scale_interleaved',
                        'input_scale', 'input_scale_inv', 'alpha',
                        'weights_padding_cols', 'logical_widths',
                        'input_size_per_partition', 'output_size_per_partition']:
                if hasattr(module, attr):
                    try: delattr(module, attr)
                    except: pass
            dequanted += 1

        if dequanted > 0:
            logger.info(f"Dequantized layer {layer_id}: {dequanted} linear layers to BF16")

    def load_weights(self, weights):
        """Load weights including multimodal components."""
        # Load text backbone weights (parent class)
        super().load_weights(weights)

        # Dequantize first layer(s) to BF16 for visual embedding processing.
        # FP4 E2M1 (~3.5 effective bits) loses the fine structure in visual
        # embeddings. The first layer sees them directly and needs full precision.
        import os
        n_bf16_layers = int(os.environ.get('BF16_LAYERS', '0'))
        if n_bf16_layers > 0:
            self._dequant_layers_from_checkpoint(n_bf16_layers)
        n_bf16_last = int(os.environ.get('BF16_LAST_LAYERS', '0'))
        if n_bf16_last > 0:
            total = self.config.num_hidden_layers
            self._dequant_layers_from_checkpoint(n_bf16_last, start_layer=total - n_bf16_last)

    def _dequant_layers_from_checkpoint(self, n_layers, start_layer=0):
        """Dequantize N layers starting from start_layer by reloading BF16 weights."""
        import os, glob
        from safetensors.torch import load_file
        model_path = getattr(self, '_model_path', None)
        if model_path is None:
            model_path = os.environ.get('SGLANG_MODEL_PATH', '/workspace/model')

        bf16_path = os.environ.get('BF16_MODEL_PATH', None)
        if bf16_path is None:
            logger.warning("BF16_LAYERS/BF16_LAST_LAYERS set but BF16_MODEL_PATH not set. Skipping dequant.")
            return

        sf_files = sorted(glob.glob(os.path.join(bf16_path, 'model-*.safetensors')))
        if not sf_files:
            logger.warning(f"No safetensors found in {bf16_path}")
            return

        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        layer_prefixes = [f'model.layers.{i}.' for i in range(start_layer, start_layer + n_layers)]
        logger.info(f"Dequanting layers {start_layer}-{start_layer + n_layers - 1} from {bf16_path}")
        replaced = 0

        params_dict = dict(self.named_parameters())

        for sf in sf_files:
            state = load_file(sf, device='cpu')
            for k, v in state.items():
                if not any(k.startswith(p) for p in layer_prefixes):
                    continue
                if k not in params_dict:
                    continue
                param = params_dict[k]
                # Only replace if shapes match (skip FP4 packed weights)
                if param.shape == v.shape:
                    param.data.copy_(v.to(param.dtype))
                    replaced += 1
                elif 'weight' in k and param.dtype == torch.uint8:
                    # This is an FP4 packed weight — need to replace the module
                    # Find the module and replace its weight + quant_method
                    parts = k.rsplit('.', 1)
                    if len(parts) == 2:
                        mod_name, attr_name = parts
                        mod = self
                        for p in mod_name.split('.'):
                            if p.isdigit():
                                mod = mod[int(p)]
                            else:
                                mod = getattr(mod, p)
                        # Replace weight with BF16
                        mod.weight = nn.Parameter(v.to(torch.bfloat16).to(param.device), requires_grad=False)
                        mod.quant_method = UnquantizedLinearMethod()
                        # Clean up FP4 attributes
                        for attr in ['weight_scale', 'weight_scale_2', 'weight_scale_interleaved',
                                    'input_scale', 'input_scale_inv', 'alpha',
                                    'weights_padding_cols']:
                            if hasattr(mod, attr):
                                try: delattr(mod, attr)
                                except: pass
                        replaced += 1
            del state

        if replaced > 0:
            logger.info(f"Replaced {replaced} weights in layers 0-{n_layers-1} with BF16 from {bf16_path}")

        # Multimodal weights are loaded by the parent's load_weights
        # since they share the same checkpoint and naming convention.
        # The visual_tokenizer, audio_tokenizer, visual_head, audio_head
        # weights are included in the NVFP4 checkpoint as BF16.

EntryClass = [LongcatNextForCausalLM]
