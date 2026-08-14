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
    # End-intent telemetry: an isolated end flag is re-sampled into an acoustic frame
    # (see _generate_audio_codebook_step), so a model that signals end INTERMITTENTLY
    # is forced to keep "speaking" near-silence until two flags land consecutively.
    # These make that visible per generation: tail length correlating with resamples
    # after first_end_flag_step is the silent-tail mechanism, measured not assumed.
    end_flag_resamples: int = 0  # isolated end flags converted to acoustic frames
    first_end_flag_step: int = -1  # frame index of the FIRST end flag seen
    # --- multi-round TTS (LCN_TTS_MULTI): transcript→audio rounds, one per sentence ---
    done_segments: list = field(default_factory=list)  # completed segments ([n,cb] tensors)
    rounds: int = 1  # round counter (positive control + runaway bound)
    between_steps: int = 0  # decode steps spent in "between" mode awaiting the next round
    wants_eos: bool = False  # between-mode: the model's masked pick was EOS — close next step
    recitation: bool = False  # the prompt carries the TTS instruction ("synthesize the
    # following content with this voice") → the transcript is a RECITATION CONTRACT and
    # the coverage/repeat stops apply. Without it (voice chat, free speech via raw
    # /generate), the model composing new sentences is the POINT — no content stops;
    # the caller's max_new_tokens, the frame cap, and the model's own EOS intent bound it.
    coverage_pos: int = 0  # char offset into prompt_norm reached by recitation so far:
    # each round must CONTINUE from (about) here — recitation is sequential, so a round
    # found only BEHIND this position is a re-read (measured: fragment re-recitations
    # passed both a repeat check and unpositioned containment) and one found nowhere is
    # invention; both close the request.
    prompt_norm: str = ""  # normalized request text, captured at the prefill that opened
    # audio mode. The coverage stop: a round transcript NOT found in it is the model
    # AUTHORING A CONTINUATION (measured: after reciting all 4 input sentences it wrote
    # 'the device hummed softly, its needle trembling…' — new fiction, round after round,
    # nothing repeating), and closes the request before that round's audio.
    past_transcripts: list = field(default_factory=list)  # per-round transcript token tuples:
    # the loop detector — the model never CHOOSES to stop opening rounds (measured: 22+
    # rounds, same ~30-frame segment, until the frame cap), so a round whose transcript
    # substantially repeats an earlier round's marks the loop and closes the request
    # BEFORE that round's audio is generated.
    prev_segment_tail: Optional[torch.Tensor] = None  # last frame of the banked segment:
    # round N+1's first frame conditions on it (the reference's audio_ids grow globally,
    # so its round N+1 feedback is round N's last frame; ours reset per segment)
    rid: str = ""  # per-request id → unique output filename (concurrency-safe retrieval)
    # --- streaming vocode (LCN_TTS_STREAM) ---
    streamed_frames: int = 0  # frames whose PCM has been emitted to the .part file
    stream_tail: Optional[torch.Tensor] = None  # withheld fade tail [1, FADE] awaiting the next piece
    stream_chunks: int = 0  # emitted chunk count (positive control)
    stream_failed: bool = False  # a chunk emit raised → fall back to the full decode at end


# Audio generation sampling config (from generation_config.json). Operator-tunable via env
# (e.g. -e AUDIO_GEN_TEMPERATURE=0.7) without rebuilding; defaults are the model-card values.
_envf = lambda k, d: float(os.environ.get(k, d))
_envi = lambda k, d: int(os.environ.get(k, d))
_LCN_VERBOSE = os.environ.get("LCN_VERBOSE", "0") == "1"  # gate per-step debug logging
# Cross-request head batching. LCN_HEAD_BATCH=0 forces the per-request path so the
# optimisation can be A/B'd inside ONE build (see _image_gen_flush).
_LCN_HEAD_BATCH = os.environ.get("LCN_HEAD_BATCH", "1").strip() != "0"
# Streaming TTS: vocode sliding windows of the codebook stream DURING generation and
# append PCM to <rid>.pcm.part, so a client can start hearing audio ~2s in instead of
# after the full ~2.6s-per-second-of-audio generation. The final .wav is assembled from
# the SAME streamed chunks, so streaming and non-streaming clients get identical bytes.
# Windowed vocoding is a real math change vs one full-utterance decode (the flow-matching
# decoder sees a window, not the utterance): it was gated on an exactly-paired listening
# test — same generation's ids vocoded full vs prefix vs sliding-window — and the owner's
# verdict was "to my ears, all of these are the same" (2026-08-11). Profile that sized the
# knobs: 12.5 frames/s output, ~177 ms/frame generation (98% of latency), ~0.3 s vocode
# per 2 s chunk (~7% amortized). LCN_TTS_STREAM=0 restores the single full decode.
_LCN_TTS_STREAM = os.environ.get("LCN_TTS_STREAM", "1").strip() != "0"
# Multi-round TTS: the model plans sentence-by-sentence — transcript one sentence,
# render its audio, emit audiogen_end, then (per the reference implementation in the
# model dir, modeling_longcat_next.py ~line 753) generation CONTINUES and the model may
# open the next round with another audiogen_start. This serving loop used to force EOS
# at the first confirmed end-of-audio, decapitating every round after the first: a
# 4-sentence input deterministically rendered only sentence 1 (measured 6/6, transcript
# stopping at sentence 1's token count exactly). With this ON, confirmed end forces
# audiogen_end instead, the segment is banked, and the model decides whether to start
# round N+1; segments are cross-faded into one wav by the SAME multi-segment machinery
# lazy_decode_and_save always had. LCN_TTS_MULTI=0 restores the old single-round stop.
_LCN_TTS_MULTI = os.environ.get("LCN_TTS_MULTI", "1").strip() != "0"
TTS_BETWEEN_BUDGET = int(os.environ.get("LCN_TTS_BETWEEN_BUDGET", "10"))  # tokens to await next round
TTS_MAX_ROUNDS = int(os.environ.get("LCN_TTS_MAX_ROUNDS", "32"))          # runaway bound; frames cap binds first
TTS_STREAM_CHUNK = int(os.environ.get("LCN_TTS_CHUNK_FRAMES", "25"))      # 2 s @ 12.5 fps
TTS_STREAM_LOOKAHEAD = int(os.environ.get("LCN_TTS_LOOKAHEAD", "12"))     # ~1 s right context
TTS_STREAM_LEFT_CTX = int(os.environ.get("LCN_TTS_LEFT_CTX", "50"))       # ~4 s left context
# LCN_DIAG_HS=1: per-row hidden-state fingerprints at the batched head call, for
# localizing cross-request content bleed. Debug only; costs a host sync per row.
_LCN_DIAG_HS = os.environ.get("LCN_DIAG_HS", "0").strip() == "1"
AUDIO_GEN_TEMPERATURE = _envf("AUDIO_GEN_TEMPERATURE", 0.5)
AUDIO_GEN_TOP_K = _envi("AUDIO_GEN_TOP_K", 5)
AUDIO_GEN_TOP_P = _envf("AUDIO_GEN_TOP_P", 0.85)
AUDIO_GEN_REPETITION_PENALTY = _envf("AUDIO_GEN_REPETITION_PENALTY", 1.3)
# Transcript recitation (the audio-text phase the acoustic head conditions on).
# The original samples it with the request's params (TTS recipe: 0.5/5/0.85);
# LCN_TRANSCRIPT_GREEDY=1 reverts to the earlier argmax recitation.
TRANSCRIPT_GREEDY = os.environ.get("LCN_TRANSCRIPT_GREEDY", "0") == "1"
TRANSCRIPT_TEMPERATURE = _envf("LCN_TRANSCRIPT_TEMPERATURE", 0.5)
TRANSCRIPT_TOP_K = _envi("LCN_TRANSCRIPT_TOP_K", 5)
TRANSCRIPT_TOP_P = _envf("LCN_TRANSCRIPT_TOP_P", 0.85)
# First-frame conditioning (owner idea, 2026-07-31): the acoustic head's frame-0
# distribution is its shakiest (onset garble/wrong-content picks). Injecting known
# silence as the first frame(s) gives frame N real in-distribution history before
# the model's first free sample. There is NO canonical single silence code — the
# VQ varies frame-to-frame even on digital silence — so injection replays the
# head of the checkpoint's own encoded-silence sequence
# (quantize/extract_silence_frames.py). LCN_TTS_SILENCE_FRAMES=N injects the
# first N frames (~80ms leading near-silence each; default 0 = off).
_SILENCE_SEQ = [
    [1761, 3757, 234, 692, 64, 144, 483, 504],
    [2490, 1417, 165, 373, 113, 812, 47, 675],
    [1182, 785, 1568, 871, 732, 787, 420, 367],
    [3608, 2959, 105, 491, 387, 336, 420, 675],
]
TTS_SILENCE_FRAMES = _envi("LCN_TTS_SILENCE_FRAMES", 0)
# The model EXTENDS injected silence (momentum: silence history begets silence
# frames — owner-measured 20-40% silent lead at N=2). The wav-side trim cuts the
# rendered lead back to a fixed small beat regardless of how much silence the
# model chose to generate. 0 disables trimming entirely.
TTS_TRIM_LEAD_MS = _envi("LCN_TTS_TRIM_LEAD_MS", 0)
# Trailing-silence trim, the lead trim's twin. Root cause is MODEL behavior, measured
# 2026-08-14 over 21 telemetried generations: the model generates silence AS CONTENT
# before its first end flag (up to ~35 frames on short clips — the owner heard a 2.88s
# tail on a 4.5s clip), and in ~1/4 of clips dithers a further 3-12 frames between
# flags. The confirm gate itself costs ~1 frame. None of that is fixable at the
# sampling layer without a behavioral clamp, so the artifact is trimmed post-hoc:
# everything after (last active audio + this many ms) is cut. 0 disables.
TTS_TRIM_TAIL_MS = _envi("LCN_TTS_TRIM_TAIL_MS", 0)
# End-of-audio is confirmed by this many CONSECUTIVE level-0 end-flags (canonical guard):
# an isolated/stray end-flag is re-sampled to a real acoustic code so the model speaks for
# exactly as long as its task needs — no arbitrary minimum-length floor.
AUDIO_END_CONFIRM = 2
#
# MEASURED AND REJECTED (2026-08-10): requiring the end flag to be the model's TOP choice
# before it counts (an "AUDIO_END_ARGMAX" gate). The idea was that the flag shares the
# sampled distribution with acoustic content, so it can be drawn mid-sound — matching the
# owner-adjudicated defect where renders cut mid-'s' and mid-'l' with an audible click.
# A MATCHED pair on one build, 32 renders per arm, threshold end_jump>=0.06:
#     gate ON  6/32 flagged (max 1.329)
#     gate OFF 5/32 flagged (max 0.763)
# No effect. An earlier 4/16-vs-2/16 reading that appeared to halve the defect compared
# DIFFERENT builds at half the sample size and did not survive matching. The terminal-click
# rate is ~16% either way, so it is a property of audio decoding, not of the end decision.
# The gate was removed rather than left default-off: it has no measured benefit, and it is
# suspected of converting terminal clicks into a SECOND, audibly worse defect (sustained
# mid-utterance babble, e.g. "self test. oooOoOoOooo.. nomina"), which end_jump cannot see
# at all. Two env-only alternatives were also measured and rejected:
# AUDIO_GEN_REPETITION_PENALTY=1.0 cleared the metric (0/14, max 0.048) but produced HTTP
# 500s on 2 of 16 requests — that penalty is also what suppresses runaway repetition, so it
# trades an audible click for a hard failure. AUDIO_GEN_TOP_K=1 halved the flag rate (2/16,
# unmatched) and would flatten the voice by making every codebook greedy.
# Anything attempted here next needs a metric that sees BOTH defect populations; end_jump
# measures only the terminal discontinuity.
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
        # (kind, req_pool_idx) -> consecutive decode steps absent from the batch.
        # Backs the orphaned-generation-state prune in forward(); see the note there.
        self._gen_state_absent: Dict[tuple, int] = {}
        self._tokenizer = None  # lazy-loaded for diagnostic logging

        # --- Gen-trigger latch: sync-free steady-state text decode ---
        # The decode state machines (Step 3) used to run per-element .item()
        # loops on EVERY decode step to spot a gen-entry token. Those host
        # syncs are a per-token latency tax and abort CUDA graph capture.
        # Instead, the post-sample hook (lcn_trigger_scan, called from the
        # ngram manager's update_after_decode — never inside a graph) tests
        # each step's sampled ids for the two gen-ENTRY tokens on-GPU and
        # async-copies a one-byte flag to pinned memory. The NEXT forward —
        # the step where the trigger arrives as *input* — reads the latch and
        # runs the eager state machine. One step "late" is exactly on time.
        # Mid-generation steps are covered by the non-empty state dicts.
        # The latch is STICKY: set at fold-in, cleared only when a state
        # machine observes the trigger token in its input (interleaved batches
        # may read the latch before the trigger-carrying batch forwards), with
        # a decay guard for triggers whose request died before its next step.
        self._trigger_ids_gpu = None  # lazy — needs the runtime device
        self._trigger_host = None
        self._trigger_event = None
        self._trigger_armed = False
        self._trigger_sticky = False
        self._trigger_decay = 0

        # Agent profile (LCN_AGENT=1): generation is disabled at the GATEWAY
        # (403), but raw gen markers in a chat prompt could still create a gen
        # state here — which NGRAM verify rounds would never advance, leaking
        # stale per-req_pool_idx state onto reused slots. Disable the gen
        # machinery in the model too: no state entry, no trigger latch work.
        # (LCN_NGRAM=1 implies LCN_AGENT=1 — see entrypoint.)
        self._lcn_gen_disabled = os.environ.get("LCN_AGENT", "0").strip() == "1"

        # Publish the gen-watch predicate to the scheduler. Under speculative
        # decoding the scheduler must know, at decode-prep time, whether this
        # step needs the Python state machines — verify rounds cannot run them.
        # See lcn_gen_state and patches/spec_gen_fallback.patch. No-op unless
        # a spec algorithm is configured.
        try:
            from sglang.srt.lcn_gen_state import register_gen_probe

            register_gen_probe(self.lcn_gen_watch_active)
        except Exception:
            pass

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
                _suffix_ids = self._tokenizer.encode(anyres_text, add_special_tokens=False) + [self._image_start_id]
                ANYRES_SUFFIX_LEN = len(_suffix_ids)
            except Exception:
                _suffix_ids = None
                ANYRES_SUFFIX_LEN = 8  # fallback
            uncond_ids = input_ids_for_prefill.clone()
            if n_prefill > ANYRES_SUFFIX_LEN:
                uncond_ids[:n_prefill - ANYRES_SUFFIX_LEN] = 0  # Zero prompt token IDs
            # The N-gram token table (our id source) stores special tokens as 0
            # (hash semantics), so the suffix's tag/image_start ids must be
            # rebuilt explicitly — the uncond branch needs them REAL.
            if _suffix_ids is not None and n_prefill >= ANYRES_SUFFIX_LEN:
                uncond_ids[n_prefill - ANYRES_SUFFIX_LEN:n_prefill] = torch.tensor(
                    _suffix_ids, dtype=uncond_ids.dtype, device=uncond_ids.device)
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

            # Run backbone with unconditional embeddings. Keep the LAST hidden:
            # it is the uncond branch's image_start-position state, i.e. the CFG
            # counterpart for visual token 1 (which the cond branch generates from
            # its own image_start hidden in the same forward — original semantics).
            uncond_hidden = self.model(input_ids=None, positions=uncond_fb.positions,
                      forward_batch=uncond_fb, input_embeds=uncond_embeds)
            last_hidden = None
            if uncond_hidden is not None and uncond_hidden.shape[0] >= 1:
                last_hidden = uncond_hidden[-1:].detach()

            logger.info(f"[ImageGen] Unconditional prefill: {n_prefill} tokens, req_pool_idx={uncond_idx}")
            return uncond_idx, last_hidden

        except Exception as e:
            logger.error(f"[ImageGen] Failed to allocate uncond KV: {e}", exc_info=True)
            return -1, None

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

    def _trim_wav_tail(self, path):
        """File-level twin of _trim_tail_pcm for the non-streamed decode path."""
        try:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(path)
            mono = data if data.ndim == 1 else data[:, 0]
            pcm = mono.astype('<i2').tobytes()
            trimmed = self._trim_tail_pcm(pcm)
            if len(trimmed) < len(pcm):
                import numpy as np
                wavfile.write(path, sr, np.frombuffer(trimmed, dtype='<i2'))
        except Exception:
            pass  # trim is cosmetic; never fail the request over it

    def _trim_wav_lead(self, path):
        """Cut the rendered leading silence back to TTS_TRIM_LEAD_MS.

        Energy-based: first 10ms window above 2% of peak RMS marks speech onset;
        everything before (onset - lead) is dropped. No-op on failure or when the
        existing lead is already short."""
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(path)
            mono = data if data.ndim == 1 else data[:, 0]
            x = mono.astype(np.float32)
            win = max(1, int(sr * 0.010))
            n_win = len(x) // win
            if n_win < 3:
                return
            rms = np.sqrt((x[:n_win * win].reshape(n_win, win) ** 2).mean(axis=1))
            thresh = max(rms.max() * 0.02, 1.0)
            above = np.nonzero(rms > thresh)[0]
            if len(above) == 0:
                return
            onset = int(above[0]) * win
            keep_lead = int(sr * TTS_TRIM_LEAD_MS / 1000.0)
            start = max(0, onset - keep_lead)
            if start <= 0:
                return
            wavfile.write(path, sr, mono[start:])
            logger.info(f"[AudioGen] trimmed {start/sr:.2f}s of leading silence from {path}")
        except Exception as e:
            logger.warning(f"[AudioGen] lead trim failed (kept original): {e}")

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

    def _decode_ids(self, ids) -> str:
        """Decode a token-id list to text (transcript coverage checks / logging)."""
        try:
            self._decode_token(0)  # ensure tokenizer loaded via the same lazy path
            return self._tokenizer.decode(list(ids))
        except Exception:
            return ""

    @staticmethod
    def _norm_tts_text(s: str) -> str:
        """Normalize text for transcript-vs-input matching: the model's recitation
        drifts on case, curly quotes and punctuation, never on the words."""
        s = s.lower()
        for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
            s = s.replace(a, b)
        s = "".join(c if (c.isalnum() or c == " ") else " " for c in s)
        return " ".join(s.split())

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

        # BRANCHLESS. `if mask.any()` forces a GPU->host sync just to evaluate a Python `if`,
        # and the masked assignments below it generated nonzero + index_put_ pairs. The heads
        # call this 56x per generated token (7 codebook levels x 8 head levels), and the
        # profiler attributed ~10ms/step plus 168 nonzero and 112 item calls per step to this
        # function alone -- on a loop where 36ms of every 115ms step is ALREADY GPU idle
        # waiting on the host. Computing both lookups and selecting costs one extra gather;
        # gathers are cheap on this box and host syncs are not.
        #
        # Three cases preserved exactly, including the gap: ids below text_vocab embed from
        # the word table, ids at/above codebook_base from the codebook table, and ids BETWEEN
        # them stay zero (which the previous zeros-then-fill form produced implicitly).
        in_text = (flat_ids < text_vocab).unsqueeze(-1)
        emb_text = word_embed(flat_ids.clamp(max=text_vocab - 1))
        if self._codebook_embed is not None:
            in_cb = (flat_ids >= codebook_base).unsqueeze(-1)
            cb_idx = (flat_ids - codebook_base).clamp(min=0, max=self._codebook_embed.shape[0] - 1)
            emb_cb = self._codebook_embed[cb_idx].to(emb_text.dtype)
            embeds = torch.where(in_text, emb_text, torch.where(in_cb, emb_cb, torch.zeros((), dtype=emb_text.dtype, device=emb_text.device)))
        else:
            embeds = torch.where(in_text, emb_text, torch.zeros((), dtype=emb_text.dtype, device=emb_text.device))

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
            logits = self._audio_head_call(
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
            # Still NO arbitrary minimum-length floor — the flag stays sampleable at level 0
            # and is adjudicated by END_CONFIRM (must repeat). At levels >0 it is always
            # masked. See the AUDIO_END_CONFIRM block above for the argmax gate that was
            # measured here and rejected.
            end_token_idx = self._audio_codebook_sizes[level]
            if level == 0:
                tok = self._sample_codebook_logits(logits, level, prev_ids)
                if int(tok) == end_token_idx:
                    if state.first_end_flag_step < 0:
                        state.first_end_flag_step = state.step_count
                    state.end_run += 1
                    if state.end_run >= AUDIO_END_CONFIRM:
                        state.ended = True
                        return None  # confirmed genuine end — do NOT store this flag frame
                    # isolated/stray end-flag: re-sample level 0 with the end slot masked so
                    # the frame carries real speech content (the model keeps speaking).
                    state.end_flag_resamples += 1
                    logits[0, end_token_idx] = float('-inf')
                    tok = self._sample_codebook_logits(logits, level, prev_ids)
                else:
                    state.end_run = 0
                next_token = tok
            else:
                logits[0, end_token_idx] = float('-inf')
                next_token = self._sample_codebook_logits(logits, level, prev_ids)

            next_token_ids[0, level] = next_token + self._audio_offset_host(level)

        return next_token_ids[0]  # [num_codebooks] with offsets

    def _audio_offset_host(self, level):
        """Host-side audio codebook offset, cached.

        audio_offset_vals is a GPU buffer, so reading it with .item() costs a host sync —
        and it was read once per codebook level per generated frame on the hot path. The
        values are fixed at init, so they are cached on first use.
        """
        cache = getattr(self, "_audio_offset_host_cache", None)
        if cache is None:
            cache = self.audio_offset_vals.tolist()
            self._audio_offset_host_cache = cache
        return cache[level]

    def _sample_codebook_logits(self, logits, level, prev_ids):
        """Rep-penalty + temperature + top-k/top-p multinomial sample of one codebook level.
        Clones logits (non-mutating) and returns the raw (pre-offset) token id (0-dim tensor)."""
        logits = logits.clone()
        if prev_ids.shape[0] > 0 and AUDIO_GEN_REPETITION_PENALTY != 1.0:
            # Vectorised, and deliberately so: the previous form looped over every unique
            # prior id calling .item() and then branching on a tensor comparison — two host
            # syncs per unique id, per codebook level, per frame (hundreds per frame, and
            # every one of them stalls the pipeline). The arithmetic below is identical:
            # positive logits divided by the penalty, negative ones multiplied.
            prev_level_ids = (prev_ids[:, level] - self._audio_offset_host(level)).clamp(
                min=0, max=logits.shape[-1] - 1)
            idx = prev_level_ids.unique()
            vals = logits[0, idx]
            logits[0, idx] = torch.where(
                vals > 0, vals / AUDIO_GEN_REPETITION_PENALTY,
                vals * AUDIO_GEN_REPETITION_PENALTY)
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
        if (not state.accumulated_ids and not state.done_segments) or self.audio_tokenizer is None:
            return None

        try:
            # All rounds' segments, offsets removed, clamped, with end-flag markers
            # between segments so lazy_decode_and_save splits and cross-fades them —
            # its multi-segment machinery, fed at last. Single-round requests produce
            # one segment with no markers: byte-identical to the old path.
            raw_ids = self._assemble_raw_segments(state)
            if raw_ids is None or raw_ids.shape[0] == 0:
                logger.warning("No valid audio frames to decode")
                return None

            # No manual end-of-audio marker: lazy_decode_and_save pads a codebook_sizes[0] row
            # itself when the last frame isn't one, and decode_wave_vocoder2 slices it off before
            # vocoding (response[:, :response_len]). Appending here would be redundant.
            logger.info(f"Decoding {raw_ids.shape[0]} audio frames through vocoder pipeline")

            # Ensure vocoder weight path is resolved correctly
            self._ensure_vocoder_path()

            # Use lazy_decode_and_save from the audio tokenizer
            _tag = state.rid or str(int(time.time()))
            save_path = f"{os.environ.get('LCN_OUTPUT_DIR', '/tmp')}/longcat_tts_{_tag}.wav"
            # LCN_TTS_DUMP_IDS=1 (debug): persist the raw codebook ids beside the wav.
            # Exists for the streaming-vocoder work: chunked vocoding changes the audio
            # math (the flow-matching decoder sees a window, not the utterance), so its
            # quality gate needs PERFECTLY PAIRED artifacts — the same generation's ids
            # vocoded full vs chunked, isolating the windowing as the only variable.
            # That pairing is only possible if the ids survive the request.
            if os.environ.get("LCN_TTS_DUMP_IDS", "0").strip() == "1":
                torch.save(raw_ids.cpu(), save_path.replace(".wav", ".ids.pt"))
                logger.info(f"[AudioGen] dumped {raw_ids.shape[0]} frames of codebook ids "
                            f"to {save_path.replace('.wav', '.ids.pt')}")
            self.audio_tokenizer.lazy_decode_and_save(
                raw_ids,
                sampling_rate=AUDIO_GEN_SAMPLING_RATE,
                wave_concat_overlap=AUDIO_GEN_WAVE_OVERLAP,
                save_path=save_path,
            )
            if TTS_TRIM_LEAD_MS > 0:
                self._trim_wav_lead(save_path)
            if TTS_TRIM_TAIL_MS > 0:
                self._trim_wav_tail(save_path)
            logger.info(f"Audio saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Audio decode failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Streaming TTS (LCN_TTS_STREAM) — sliding-window vocode during generation
    # ------------------------------------------------------------------

    def _stream_part_path(self, state) -> str:
        return f"{os.environ.get('LCN_OUTPUT_DIR', '/tmp')}/longcat_tts_{state.rid}.pcm.part"

    def _assemble_raw_segments(self, state) -> Optional[torch.Tensor]:
        """All of a request's segments as ONE raw-ids tensor with end-flag marker rows
        between them — the exact input shape lazy_decode_and_save's multi-segment split
        (audio_end_pos) was built for. Offsets subtracted and levels clamped per
        segment; the marker rows (level0 = codebook_sizes[0], rest 0) are inserted
        AFTER clamping so the clamp cannot destroy them."""
        segments = list(state.done_segments)
        if state.accumulated_ids:
            segments.append(torch.stack(state.accumulated_ids, dim=0))
        if not segments:
            return None
        offsets = self.audio_offset_vals.to(segments[0].device)
        marker = torch.zeros(1, segments[0].shape[1], dtype=torch.long,
                             device=segments[0].device)
        marker[0, 0] = self._audio_codebook_sizes[0]
        parts = []
        for k, seg in enumerate(segments):
            raw = seg - offsets.unsqueeze(0)
            for lvl in range(raw.shape[1]):
                raw[:, lvl] = raw[:, lvl].clamp(min=0, max=self._audio_codebook_sizes[lvl] - 1)
            parts.append(raw)
            if k < len(segments) - 1:
                parts.append(marker)  # lazy_decode pads the final one itself
        return torch.cat(parts, dim=0)

    def _finalize_audio(self, state, req_idx) -> Optional[str]:
        """Decode everything the request produced (all rounds) and write the wav."""
        if _LCN_TTS_STREAM and state.rid and not state.stream_failed:
            try:
                return self._stream_finalize(state)
            except Exception:
                logger.warning(f"[AudioGen] req={req_idx}: stream finalize failed; "
                               f"full decode fallback", exc_info=True)
        return self._decode_audio_to_wav(state)

    @torch.no_grad()
    def _vocode_frames(self, raw_ids: torch.Tensor) -> torch.Tensor:
        """Vocode raw (offset-free, clamped) codebook ids -> wave [1, samples], cpu float.

        The same chain decode_wave_vocoder2 runs, minus its batch handling: bridge decode
        -> audio decoder -> flow matching -> cosy24k vocoder."""
        tok = self.audio_tokenizer
        if tok.cosy24kvocoder is None:
            self._ensure_vocoder_path()
            from sglang.srt.models.cosy24k_vocoder import Cosy24kVocoder
            tok.cosy24kvocoder = Cosy24kVocoder.from_pretrained(
                tok.config.audio_config.cosy24kvocoder_config.weight_path
            ).to(next(tok.parameters()).device)
        device = next(tok.parameters()).device
        ids = raw_ids.to(device)
        ret = tok.decode(ids, bridge_length=torch.tensor([ids.shape[0]], device=device))
        mel = ret.flow_matching_mel[0][: ret.flow_matching_mel_lengths[0], :]
        wave = tok.cosy24kvocoder.decode(
            mel.transpose(0, 1).to(torch.float32).unsqueeze(0))
        return wave.cpu()

    def _stream_raw_window(self, state, start: int, end: int) -> torch.Tensor:
        ids = torch.stack(state.accumulated_ids[start:end], dim=0)
        offsets = self.audio_offset_vals.to(ids.device)
        raw = ids - offsets.unsqueeze(0)
        for lvl in range(raw.shape[1]):
            raw[:, lvl] = raw[:, lvl].clamp(min=0, max=self._audio_codebook_sizes[lvl] - 1)
        return raw

    def _stream_append_pcm(self, state, piece: torch.Tensor, final: bool):
        """Crossfade `piece` against the withheld tail and append int16 PCM to .part.

        Already-emitted bytes cannot be revised, so seams are healed by WITHHOLDING the
        last AUDIO_GEN_WAVE_OVERLAP samples of every piece: the next piece's head is
        faded against the held tail before either is written. `final` flushes the tail."""
        fade = AUDIO_GEN_WAVE_OVERLAP
        out = []
        if state.stream_tail is not None and piece.shape[1] >= fade:
            ramp_d = torch.linspace(1.0, 0.0, fade)[None, :]
            ramp_u = torch.linspace(0.0, 1.0, fade)[None, :]
            out.append(state.stream_tail * ramp_d + piece[:, :fade] * ramp_u)
            piece = piece[:, fade:]
            state.stream_tail = None
        elif state.stream_tail is not None:
            out.append(state.stream_tail)  # piece too short to fade against
            state.stream_tail = None
        if state.stream_chunks == 0 and TTS_TRIM_LEAD_MS > 0 and piece.shape[1] > 0:
            piece = self._trim_lead_tensor(piece)
        if not final and piece.shape[1] > fade:
            state.stream_tail = piece[:, -fade:].clone()
            piece = piece[:, :-fade]
        out.append(piece)
        wave = torch.cat(out, dim=1) if len(out) > 1 else out[0]
        pcm = (wave.squeeze(0).clamp(-1.0, 1.0) * 32767).to(torch.int16).numpy().tobytes()
        with open(self._stream_part_path(state), "ab") as f:
            f.write(pcm)
        return len(pcm)

    def _trim_lead_tensor(self, wave: torch.Tensor) -> torch.Tensor:
        """First-chunk lead-silence trim, mirroring _trim_wav_lead's thresholds
        (10 ms windows, onset = first window above 2% of peak RMS, keep
        TTS_TRIM_LEAD_MS of lead). Streaming cannot trim after the fact, so the
        trim runs on the first emitted piece with peak measured locally."""
        try:
            x = wave.squeeze(0).float()
            sr = AUDIO_GEN_SAMPLING_RATE
            win = max(1, int(sr * 0.010))
            n_win = x.shape[0] // win
            if n_win < 3:
                return wave
            rms = x[: n_win * win].view(n_win, win).pow(2).mean(dim=1).sqrt()
            peak = rms.max()
            if peak <= 0:
                return wave
            above = (rms > 0.02 * peak).nonzero()
            if above.numel() == 0:
                return wave
            onset = int(above[0].item()) * win
            keep_from = max(0, onset - int(sr * TTS_TRIM_LEAD_MS / 1000))
            return wave[:, keep_from:]
        except Exception:
            return wave  # a failed trim must never fail the stream

    def _stream_emit(self, state, final: bool, drain: bool = False):
        """Emit any complete chunk (or, when final/drain, everything left) to .part.

        drain: emit ALL pending frames but keep withholding the fade tail — used at a
        segment boundary in multi-round TTS, so the next round's first piece cross-fades
        against this segment's tail exactly like the offline path's segment joins."""
        n = len(state.accumulated_ids)
        while True:
            pending = n - state.streamed_frames
            if pending <= 0:
                break
            if not (final or drain) and pending < TTS_STREAM_CHUNK:
                break
            emit_end = n if (final or drain) and pending < TTS_STREAM_CHUNK else \
                min(state.streamed_frames + TTS_STREAM_CHUNK, n)
            win_start = max(0, state.streamed_frames - TTS_STREAM_LEFT_CTX)
            win_end = min(emit_end + TTS_STREAM_LOOKAHEAD, n)
            t0 = time.time()
            wave = self._vocode_frames(self._stream_raw_window(state, win_start, win_end))
            spf = wave.shape[1] / (win_end - win_start)
            s0 = int((state.streamed_frames - win_start) * spf)
            s1 = wave.shape[1] if (final and emit_end == n and win_end == n) \
                else min(int((emit_end - win_start) * spf), wave.shape[1])
            nbytes = self._stream_append_pcm(state, wave[:, s0:s1],
                                             final and emit_end == n)
            state.streamed_frames = emit_end
            state.stream_chunks += 1
            # Positive control: a streamed and a non-streamed generation are otherwise
            # indistinguishable in the logs, and an unreached emit path would read as
            # "streaming didn't help" (the failure mode this campaign measured once).
            if state.stream_chunks == 1 or state.stream_chunks % 10 == 0 or final:
                logger.info(f"[AudioGen] STREAM chunk {state.stream_chunks}: frames "
                            f"{win_start}-{win_end} emit {s0/spf + win_start:.0f}-{emit_end} "
                            f"{nbytes}B vocode {time.time()-t0:.2f}s final={final}")

    def _trim_tail_pcm(self, pcm: bytes) -> bytes:
        """Cut trailing silence back to TTS_TRIM_TAIL_MS after the last active audio.

        Same thresholds as the lead trim (10 ms windows, active = >2% of peak RMS).
        Runs on the ASSEMBLED pcm at finalize — a live stream has already sent its
        bytes, so this cleans the .wav every non-streaming client and artifact gets;
        a streaming client simply stops when the stream ends."""
        try:
            import numpy as np
            x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
            sr = AUDIO_GEN_SAMPLING_RATE
            win = max(1, int(sr * 0.010))
            n_win = len(x) // win
            if n_win < 3:
                return pcm
            rms = np.sqrt((x[: n_win * win].reshape(n_win, win) ** 2).mean(axis=1))
            peak = rms.max()
            if peak <= 0:
                return pcm
            active = np.nonzero(rms > 0.02 * peak)[0]
            if len(active) == 0:
                return pcm
            keep_to = min(len(x), (int(active[-1]) + 1) * win + int(sr * TTS_TRIM_TAIL_MS / 1000))
            if keep_to >= len(x):
                return pcm
            logger.info(f"[AudioGen] trimmed {(len(x)-keep_to)/sr:.2f}s of trailing silence")
            return pcm[: keep_to * 2]  # int16 -> 2 bytes/sample
        except Exception:
            return pcm  # a failed trim must never fail the request

    def _stream_finalize(self, state) -> Optional[str]:
        """Emit the remainder, flush the tail, assemble the .wav from the streamed PCM."""
        self._stream_emit(state, final=True)
        if state.stream_tail is not None:  # tail of the very last piece
            self._stream_append_pcm(state, state.stream_tail, final=True)
            state.stream_tail = None
        part = self._stream_part_path(state)
        wav_path = part[: -len(".pcm.part")] + ".wav"
        if os.environ.get("LCN_TTS_DUMP_IDS", "0").strip() == "1":
            _raw = self._assemble_raw_segments(state)
            if _raw is not None:
                torch.save(_raw.cpu(), wav_path.replace(".wav", ".ids.pt"))
        import wave as _wave
        with open(part, "rb") as f:
            pcm = f.read()
        if TTS_TRIM_TAIL_MS > 0:
            pcm = self._trim_tail_pcm(pcm)
        with _wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(AUDIO_GEN_SAMPLING_RATE)
            w.writeframes(pcm)
        # The .wav now holds everything; the .part stays until the GATEWAY finishes
        # tailing it (it deletes both through _discard_artifact). Removing it here
        # would race a streaming client still draining the last bytes.
        logger.info(f"[AudioGen] STREAM finalized: {state.stream_chunks} chunks, "
                    f"{len(pcm)} PCM bytes -> {wav_path}")
        return wav_path

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
        return_all: bool = False,
    ) -> torch.Tensor:
        """Run one step of image codebook generation via the depth transformer.

        Uses Classifier-Free Guidance (CFG) when uncond_hidden is available:
          fused = cfg_scale * (cond - uncond) + uncond

        cond_hidden: [n, hidden_size] — conditional hidden state(s) from backbone
        uncond_hidden: [n, hidden_size] or None — unconditional hidden state
        Returns: [num_codebooks] for the first row, or [n, num_codebooks] when
        return_all (cross-request batching: one head call serves n requests).

        The body is already batch-general — it was written to run cond+uncond as
        a bs=2 CFG pair, and every step (sampling, top-k row slicing, the offset
        write) is per-row. Cross-request batching reuses that generality with
        the rows being DIFFERENT REQUESTS rather than a guidance pair. The two
        cannot be combined in one call: CFG's rows must be fused pairwise, so a
        CFG-active batch still goes one request at a time.

        Takes NO per-request state on purpose. It used to accept an ImageGenState
        that the body never read, which was harmless while every call was batch-1
        and a silent wrong-request bug waiting to happen once one call serves many
        requests (the batched caller would have had to pass SOME row's state, and
        whichever it picked would be wrong for the others). Anything per-request
        belongs to the caller, which has the row->request mapping.
        """
        device = cond_hidden.device
        num_codebooks = len(self._visual_codebook_sizes)
        cfg_scale = IMAGE_GEN_CFG_SCALE

        # Whether the rows of `batched_hidden` are a CFG guidance PAIR. Decided ONCE,
        # here, from the only ground truth there is -- whether an uncond hidden state was
        # actually supplied -- and never re-derived from the batch size downstream.
        #
        # This is the bug that made two concurrent images come out identical: the fusion
        # below used to trigger on `logits.shape[0] == 2`, i.e. it inferred "these two rows
        # are cond+uncond" from the batch merely being 2. That was safe only while every
        # call was batch-1-or-a-CFG-pair. Once cross-request batching put TWO REQUESTS in
        # one call, request A became "cond" and request B became "uncond", they were fused
        # into a single row, and the [1] sampled token broadcast into the [2] slot of
        # next_token_ids -- giving both requests A's tokens for every level from the first
        # batched decode step on. (Token 1 escaped because it is generated on the batch-1
        # prefill path.) cfg_scale is 3.0 by DEFAULT while CFG itself is off, so the guard
        # was live even though CFG never runs -- shape is not a proxy for intent.
        use_cfg = cfg_scale != 1.0 and uncond_hidden is not None

        if use_cfg:
            batched_hidden = torch.cat([cond_hidden, uncond_hidden], dim=0)
        else:
            batched_hidden = cond_hidden

        bs = batched_hidden.shape[0]
        next_token_ids = torch.zeros(bs, num_codebooks, dtype=torch.long, device=device)

        for level in range(num_codebooks):
            logits = self._visual_head_call(
                batched_hidden, next_token_ids, self._codebook_embed_fn, level,
            )

            # CFG fusion. Gated on use_cfg, NOT on logits.shape[0] == 2 -- see above.
            if use_cfg:
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

            # Offset applied ON DEVICE. This previously read the offset back with .item(),
            # which is a blocking host sync once per codebook level -- 8 per generated token,
            # ~11k per image -- on a loop the profiler shows is HOST-BOUND (36ms of every
            # 115ms step is GPU idle waiting on Python). visual_offset_vals is a GPU tensor
            # and next_token is a GPU tensor, so the add never needed the host at all.
            # The audio path already avoids this via _audio_offset_host; the visual path was
            # simply never given the same treatment.
            # SHAPE GUARD. Under CFG the fused sample is [1] and is written to BOTH rows on
            # purpose: cond and uncond must be conditioned on the same tokens at the next
            # level. Without CFG there must be exactly one sample PER ROW -- a [1] into a
            # [bs] slot broadcasts silently and hands every request row 0's token, which is
            # how two concurrent images came out identical. Broadcasting is correct in one
            # case and a correctness bug in the other, so the two cannot share a bare assign.
            if not use_cfg and next_token.shape[0] != bs:
                raise RuntimeError(
                    f"visual codebook sampling produced {next_token.shape[0]} tokens for "
                    f"{bs} rows at level {level} (use_cfg={use_cfg}). Refusing to broadcast: "
                    f"this would give every concurrent request the first row's token."
                )
            next_token_ids[:, level] = next_token + self.visual_offset_vals[level]

        return next_token_ids if return_all else next_token_ids[0]

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
        """Extract multimodal items, PAIRED WITH THE INDEX OF THE REQUEST THEY BELONG TO.

        The request index is load-bearing, not bookkeeping: each item's `offsets` are
        relative to its OWN request's sequence, while the forward tensors are flattened
        across the whole batch. An item cannot be placed without knowing which request it
        came from. This previously returned bare items, discarding that association, which
        silently corrupted every request after the first whenever two multimodal requests
        landed in one batched prefill — measured 3/3 correct sequentially vs 1/3
        concurrently on identical inputs.
        """
        mm_inputs_list = getattr(forward_batch, 'mm_inputs', None)
        if not mm_inputs_list:
            return [], []
        image_items, audio_items = [], []
        for i, mm_input in enumerate(mm_inputs_list):
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                if item.is_image():
                    image_items.append((i, item))
                elif item.is_audio():
                    audio_items.append((i, item))
        return image_items, audio_items

    # --- Gen-trigger latch (see __init__ comment) ---

    def lcn_trigger_scan(self, next_token_ids: torch.Tensor):
        """Post-sample hook: latch whether any just-sampled token is a
        gen-ENTRY trigger (audiogen_start / image_start). Called from the
        ngram manager's update_after_decode with the RAW sampled ids (before
        the hash-table zeroing — the triggers are >= the text vocab and would
        be zeroed away). Never runs inside CUDA graph capture."""
        if self._lcn_gen_disabled:
            return
        dev = next_token_ids.device
        if self._trigger_ids_gpu is None or self._trigger_ids_gpu.device != dev:
            self._trigger_ids_gpu = torch.tensor(
                [self._audiogen_start_id, self._image_start_id],
                dtype=torch.int64, device=dev,
            )
            self._trigger_host = torch.zeros(1, dtype=torch.uint8, pin_memory=True)
            self._trigger_event = torch.cuda.Event()
        hit = torch.isin(
            next_token_ids.to(torch.int64), self._trigger_ids_gpu
        ).any()
        self._trigger_host.copy_(hit.reshape(1).to(torch.uint8), non_blocking=True)
        self._trigger_event.record()
        self._trigger_armed = True

    def _lcn_fold_trigger(self):
        """Fold the last scan's async flag into the sticky latch. The event
        wait covers a 1-byte D2H enqueued right after the previous step's
        sampling — effectively always landed by now (bounded by one decode
        step's GPU tail in deep overlap)."""
        if self._trigger_armed:
            self._trigger_event.synchronize()
            if bool(self._trigger_host[0]):
                self._trigger_sticky = True
                self._trigger_decay = 0
            self._trigger_armed = False

    def lcn_gen_watch_active(self) -> bool:
        """True when the decode state machines must run this forward."""
        if self._lcn_gen_disabled:
            return False
        if torch.cuda.is_current_stream_capturing():
            # CUDA graph capture runs this forward once with a dummy batch:
            # no host waits, and the captured graph must be the sync-free
            # text path (gen batches veto replay and run eager).
            return False
        self._lcn_fold_trigger()
        return bool(
            self._audio_gen_states or self._image_gen_states
        ) or self._trigger_sticky

    def lcn_cuda_graph_veto(self) -> bool:
        """Graph replay veto (consulted by the decode graph runner patch):
        graphed decode skips the Python state machines entirely, so any batch
        that might need them must run eager."""
        return self.lcn_gen_watch_active()

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

        # Prune generation state orphaned by aborted or evicted requests — FIRST,
        # before anything below consumes it. This block originally ran AFTER the
        # state machines, which left a one-step window: on the first decode forward
        # after sglang recycled a dead generation's pool slot to an ordinary text
        # request, the embedding-feedback pass and _image/_audio_gen_decode_step
        # looked the stale state up by req_pool_idx and acted on it — forcing
        # image_pad/image_end overrides into an innocent request's output — and only
        # THEN did the eviction run. The janitor must sweep before the machines eat.
        #
        # _audio_gen_states/_image_gen_states are keyed by req_pool_idx, and sglang
        # REUSES pool slots. Entries are deleted on normal completion only, so a request
        # that dies mid-generation — routine, since image generation runs for minutes and
        # clients time out — pins its state to a slot that will later be handed to someone
        # else.
        # Decay-based, mirroring the trigger latch below: a slot absent from the batch for
        # 64 consecutive decode steps is not coming back. The tolist() is a host sync, but
        # this runs only while generation state exists, and that path is already eager.
        # Decay alone is NOT sufficient, and the gap is what matters most here: it clears a
        # slot only while that slot is ABSENT from the batch. If the aborted request's slot is
        # promptly recycled to ordinary traffic, the slot is present on every step, the absent
        # counter resets every step, and the entry is never cleared. `lcn_gen_watch_active()`
        # reads exactly these dicts, so a single aborted generation whose slot gets reused
        # LATCHES the CUDA-graph and spec-decode veto ON for every request, permanently, until
        # the server restarts — presenting only as "it got slow and stayed slow".
        # So evict on IDENTITY as well: the state records the rid that created it, and a slot
        # now serving a different rid is holding a corpse. rid-based eviction is immediate and
        # needs no absence; the decay below remains for the case where the slot simply goes
        # idle, and for when rids are unavailable (then `_rid_for` returns "" and this is
        # skipped — failing back to the pre-existing behaviour rather than evicting live state).
        if is_decode and (self._audio_gen_states or self._image_gen_states):
            _live = set(forward_batch.req_pool_indices.tolist())
            for _store, _kind in ((self._audio_gen_states, "audio"),
                                  (self._image_gen_states, "image")):
                for _k in list(_store.keys()):
                    _key = (_kind, _k)
                    if _k in _live:
                        _st = _store.get(_k)
                        _now = self._rid_for(_k, forward_batch)
                        if _st is not None and _st.rid and _now and _st.rid != _now:
                            logger.warning(
                                "[GenState] %s generation state for pool slot %d belongs to "
                                "request %s but the slot now serves %s — evicting. The "
                                "original request died mid-generation; keeping this would "
                                "pin the CUDA-graph veto on for every request.",
                                _kind, _k, _st.rid, _now)
                            # Under LCN_CFG=1 an image state may own a shadow request slot
                            # + KV pages for the uncond path, released only on normal
                            # completion. Eviction must release them too or every aborted
                            # CFG image leaks its shadow allocation until restart.
                            # (_free_uncond_kv self-guards: no-op without uncond state.)
                            if _kind == "image":
                                self._free_uncond_kv(_st)
                            # A streaming TTS that dies mid-generation leaves its partial
                            # .pcm.part; only completion assembles + cleans it.
                            if _kind == "audio" and _st.rid:
                                try:
                                    os.unlink(self._stream_part_path(_st))
                                except OSError:
                                    pass
                            _store.pop(_k, None)
                            self._gen_state_absent.pop(_key, None)
                            continue
                        self._gen_state_absent.pop(_key, None)
                        continue
                    _n = self._gen_state_absent.get(_key, 0) + 1
                    self._gen_state_absent[_key] = _n
                    if _n >= 64:
                        logger.warning(
                            "[GenState] %s generation state for pool slot %d absent from "
                            "the batch for 64 decode steps — clearing (request aborted or "
                            "evicted). A later request reusing this slot would otherwise "
                            "inherit it.", _kind, _k)
                        _st = _store.get(_k)
                        if _kind == "image" and _st is not None:
                            self._free_uncond_kv(_st)  # see rid-eviction note above
                        if _kind == "audio" and _st is not None and _st.rid:
                            try:
                                os.unlink(self._stream_part_path(_st))
                            except OSError:
                                pass
                        _store.pop(_k, None)
                        self._gen_state_absent.pop(_key, None)

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
                    elif a_state.prev_segment_tail is not None:
                        # Round N+1's first frame: condition on round N's last frame,
                        # mirroring the reference where audio_ids grow globally.
                        _aud_feedback[i] = a_state.prev_segment_tail

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
        # Gated by the gen-trigger latch: on steady-state text decode the
        # per-element .item() loops are skipped entirely (no host syncs — the
        # path CUDA graphs capture). The latch guarantees the loops run on the
        # step where a gen-entry trigger arrives as input.
        _run_sm = is_decode and self.lcn_gen_watch_active()
        audio_logit_overrides = {}  # batch_idx → forced_token_id
        if _run_sm and self.audio_head is not None:
            audio_logit_overrides = self._audio_gen_decode_step(
                input_ids, hidden_states, forward_batch
            )
        image_logit_overrides = {}
        if _run_sm and self.visual_head is not None:
            image_logit_overrides = self._image_gen_decode_step(
                input_ids, hidden_states, forward_batch
            )
        if (
            _run_sm
            and self._trigger_sticky
            and not self._audio_gen_states
            and not self._image_gen_states
        ):
            # Latch set but no trigger observed and no state entered — the
            # trigger-carrying batch hasn't forwarded yet (interleaving), or
            # its request died first. Decay so a dead trigger can't pin the
            # engine to the eager path forever.
            self._trigger_decay += 1
            if self._trigger_decay >= 64:
                logger.warning(
                    "[GenTrigger] latch decayed after 64 decode steps with no "
                    "trigger observed — clearing (request likely aborted)"
                )
                self._trigger_sticky = False
                self._trigger_decay = 0

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
                        # Pick the transcript token the way the ORIGINAL does: sample
                        # with the TTS recipe's params (detection runs on the SAMPLED
                        # token — original M:725 — never on argmax). We sample here,
                        # pre-scheduler, and emit one-hot so detection == emission.
                        # LCN_TRANSCRIPT_GREEDY=1 restores plain argmax recitation.
                        if TRANSCRIPT_GREEDY:
                            picked = nl.argmax().item()
                        else:
                            work = nl.float().clone()
                            work[self._audiogen_end_id] = float('-inf')  # never a valid pick
                            work = work / max(TRANSCRIPT_TEMPERATURE, 1e-5)
                            if TRANSCRIPT_TOP_K > 0:
                                kth = torch.topk(work, min(TRANSCRIPT_TOP_K, work.shape[-1]))[0][-1]
                                work[work < kth] = float('-inf')
                            if TRANSCRIPT_TOP_P < 1.0:
                                sorted_logits, sorted_idx = torch.sort(work, descending=True)
                                probs = torch.softmax(sorted_logits, dim=-1)
                                cum = torch.cumsum(probs, dim=-1)
                                sorted_logits[cum - probs > TRANSCRIPT_TOP_P] = float('-inf')
                                work = torch.full_like(work, float('-inf')).scatter_(
                                    -1, sorted_idx, sorted_logits)
                            picked = torch.multinomial(
                                torch.softmax(work, dim=-1), 1).item()
                        transcript_should_end = (
                            picked in (self._audiotext_pad_id, 2)
                            or state.transcript_steps >= state.max_transcript_steps
                        )
                        if transcript_should_end:
                            if picked == self._audiotext_pad_id:
                                reason = "natural (audiotext_pad)"
                            elif picked == 2:
                                reason = "EOS"
                            else:
                                reason = f"max ({state.max_transcript_steps})"
                            # Loop detection (multi-round): the model never CHOOSES to
                            # stop opening rounds — measured 22+ rounds of the same
                            # ~30-frame segment until the frame cap — so a transcript
                            # that substantially repeats an earlier round's IS the stop
                            # signal. Fuzzy (not exact) because recitation is sampled at
                            # temp>0 and a looped sentence can vary by a token or two.
                            # Closing here, BEFORE this round's audio, keeps the wav to
                            # one rendition per sentence. Tradeoff accepted + logged: an
                            # input that legitimately repeats a sentence verbatim ends
                            # at the repeat.
                            _cur = list(state.transcript_tokens)
                            _looped = False
                            _reason2 = ""
                            if _LCN_TTS_MULTI and state.recitation and _cur \
                                    and state.past_transcripts:
                                import difflib
                                for _past in state.past_transcripts:
                                    r = difflib.SequenceMatcher(
                                        None, _cur, list(_past)).ratio()
                                    if r >= 0.8:
                                        _looped = True
                                        _reason2 = "repeats an earlier round"
                                        break
                            _txt = "".join(self._decode_token(t) for t in _cur)[:120]
                            # Coverage stop: a transcript not found in the REQUEST TEXT
                            # is the model authoring a continuation, not reciting — the
                            # failure the repeat detector cannot see (nothing repeats;
                            # measured: 20+ rounds of freshly invented story). Fuzzy
                            # (containment, else longest-common-substring ratio) because
                            # recitation drifts on punctuation, never on words.
                            if _LCN_TTS_MULTI and state.recitation and not _looped \
                                    and _cur and state.prompt_norm:
                                _tn = self._norm_tts_text("".join(
                                    self._decode_token(t) for t in _cur))
                                # POSITIONED coverage: recitation is sequential, so the
                                # round must continue from (about) where the last ended.
                                # A small backward slack tolerates overlapping fragment
                                # boundaries; a round found only far behind is a re-read
                                # (fragment re-recitations passed both the repeat check
                                # and unpositioned containment), nowhere is invention,
                                # and a punctuation-only round is degeneracy. All close.
                                _from = max(0, state.coverage_pos - 20)
                                if not _tn:
                                    _looped = True
                                    _reason2 = "is empty/punctuation-only (degenerate round)"
                                else:
                                    _idx = state.prompt_norm.find(_tn, _from)
                                    if _idx >= 0:
                                        state.coverage_pos = _idx + len(_tn)
                                    else:
                                        import difflib
                                        _rem = state.prompt_norm[_from:]
                                        m = difflib.SequenceMatcher(
                                            None, _tn, _rem).find_longest_match(
                                            0, len(_tn), 0, len(_rem))
                                        if m.size / max(len(_tn), 1) >= 0.7:
                                            state.coverage_pos = _from + m.b + m.size
                                        else:
                                            _looped = True
                                            _reason2 = ("is not in the remaining request text "
                                                        "(re-read or authored continuation)")
                            if _looped:
                                logger.info(f"[AudioGen] req={req_idx}: round "
                                            f"{state.rounds} transcript {_reason2} "
                                            f"('{_txt}') — closing after "
                                            f"{len(state.done_segments)} segment(s)")
                                nl[:] = float('-inf')
                                nl[self._audiogen_end_id] = 0.0
                                state.mode = "between"
                                state.wants_eos = True
                                state.between_steps = 0
                            else:
                                state.past_transcripts.append(tuple(_cur))
                                logger.info(f"[AudioGen] req={req_idx}: transcript ended ({reason}) "
                                           f"after {state.transcript_steps} steps "
                                           f"('{_txt}'), forcing audiotext_start")
                                nl[:] = float('-inf')
                                nl[self._audiotext_start_id] = 0.0
                                state.transcript_done = True
                        else:
                            # Emit the picked token as a one-hot. It passes through to the
                            # scheduler -> N-gram token table, so the next step's hash
                            # context is correct.
                            if picked in (2, self._audiogen_end_id):
                                # safety (greedy path): re-pick with both masked
                                masked = nl.clone()
                                masked[2] = float('-inf')
                                masked[self._audiogen_end_id] = float('-inf')
                                picked = masked.argmax().item()
                            if _LCN_VERBOSE and state.transcript_steps <= 12:
                                logger.info(f"[AudioGen] req={req_idx}: transcript step {state.transcript_steps}, "
                                           f"emit={picked} ('{self._decode_token(picked)}')")
                            state.transcript_tokens.append(picked)  # per-round; loop detector input
                            nl[:] = float('-inf')
                            nl[picked] = 0.0
                elif forced_token == -3:
                    # Between rounds (multi-round TTS): the model may open the next
                    # round (audiogen_start) or wind down — but a sampled EOS would
                    # finish the request before the wav is written, so EOS is masked
                    # and the INTENT recorded; the between watcher closes the request
                    # itself next step, after finalizing.
                    req_idx = forward_batch.req_pool_indices[batch_idx].item()
                    state = self._audio_gen_states.get(req_idx)
                    nl = logits_output.next_token_logits[batch_idx]
                    if state is not None and int(nl.argmax().item()) == 2:
                        state.wants_eos = True
                    nl[2] = float('-inf')
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
            self._check_prefill_image_start(input_ids, logits_output, forward_batch,
                                            hidden_states=hidden_states)

        return logits_output

    def _compute_mm_embeddings(self, input_ids, forward_batch):
        """Compute embeddings with multimodal replacement."""
        image_items, audio_items = self._get_mm_items(forward_batch)

        def _cfg_val(obj, key, default):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)
        ac_cfg = getattr(self.config, 'audio_config', {})
        vc_cfg = getattr(self.config, 'visual_config', {})
        vis_pad_id = _cfg_val(vc_cfg, 'image_pad_token_id', 131108)
        aud_pad_id = _cfg_val(ac_cfg, 'audio_pad_token_id', 131105)

        # NOTE on hash-derived multimodal pad ids: sglang gives each mm item a
        # pad_value derived from a CONTENT HASH so that two prompts differing only in
        # their media get different token ids and the radix prefix cache cannot serve one
        # request's media to another (this package previously defeated that by
        # pre-assigning a constant pad_value; see research/FINDINGS.md). Those hashed
        # values are deliberately out of vocab (1e6 + hash%2^30 against vocab 282624).
        #
        # They do NOT need remapping here: forward() clamps input_ids to the embedding
        # table size a few lines before calling this, so the embedding lookup is already
        # safe. A remapping block added here on 2026-08-10 was DEAD CODE — its guard
        # (input_ids >= 1e6) can never be true post-clamp — and its never-firing warning
        # was briefly mistaken for evidence that it was working. Removed rather than kept
        # as a safety net, because a branch that cannot execute is not a safety net.
        #
        # The cache fix is unaffected: the radix key is built from origin_input_ids in the
        # SCHEDULER, before this forward runs, so the processor-side change carries it.

        if self.model.use_ngram_embedding:
            input_embeds = self.model.embed_tokens(input_ids, forward_batch)
        else:
            input_embeds = self.model.embed_tokens(input_ids)

        # Zero embeddings at pad positions before replacement
        pad_mask = (input_ids == vis_pad_id) | (input_ids == aud_pad_id)
        input_embeds[pad_mask] = 0

        # The encoders take bare items; only the scatter needs the request index. Encode
        # order must match scatter order — both walk the same list, so keep it that way.
        if audio_items:
            audio_embeds = self.get_audio_feature([it for _, it in audio_items])
            if audio_embeds is not None:
                self._replace_mm_embeddings(input_embeds, audio_items, audio_embeds, forward_batch)

        if image_items:
            image_embeds = self.get_image_feature([it for _, it in image_items])
            if image_embeds is not None:
                self._replace_mm_embeddings(input_embeds, image_items, image_embeds, forward_batch)

        return input_embeds

    def _replace_mm_embeddings(self, input_embeds, items, embeds, forward_batch):
        """Scatter encoded media embeddings into their placeholder positions.

        `items` is [(req_idx, item)] from _get_mm_items. Each item's offsets are relative
        to its own request's FULL sequence, while `input_embeds` covers the batch-flattened
        EXTEND region only, so every offset needs two corrections:

          - subtract THAT request's cached prefix length (the extend region begins there);
          - add THAT request's start position within the flattened batch.

        Both were wrong before: the prefix was always read from index [0] (the first
        request's), and the per-request base was missing entirely. With a single request in
        flight the base is 0 and index 0 is the right request, so it worked — and every
        prior test issued one request at a time. Under batched prefill it corrupted every
        request after the first, which measured as 3/3 correct sequentially vs 1/3
        concurrently on identical inputs, with the damaged requests collapsing to a
        one-token reply.
        """
        prefix_lens = list(getattr(forward_batch, 'extend_prefix_lens_cpu', None) or [])
        seq_lens = list(getattr(forward_batch, 'extend_seq_lens_cpu', None) or [])
        # Start of each request within the flattened extend region.
        bases, acc = [], 0
        for L in seq_lens:
            bases.append(acc)
            acc += L

        embed_idx = 0
        for req_idx, item in items:
            offsets = getattr(item, 'offsets', None)
            if offsets is None:
                continue
            prefix_len = prefix_lens[req_idx] if req_idx < len(prefix_lens) else 0
            base = bases[req_idx] if req_idx < len(bases) else 0
            for start, end in offsets:
                # offsets are INCLUSIVE on both ends (see the processor): upstream
                # pad_input_tokens writes input_ids[start:end+1], so the span is
                # end - start + 1 tokens.
                n_tokens = end - start + 1
                if embed_idx + n_tokens > embeds.shape[0]:
                    n_tokens = embeds.shape[0] - embed_idx
                if n_tokens <= 0:
                    continue
                rel_start = start - prefix_len
                if rel_start < 0:
                    # Media lies inside this request's cached prefix: its KV is already
                    # resident and must not be re-scattered. Still consume the embeddings.
                    embed_idx += n_tokens
                    continue
                adj_start = base + rel_start
                adj_end = adj_start + n_tokens
                if adj_end > input_embeds.shape[0]:
                    # CANARY, not a known bug. The arithmetic here would lose the tail of
                    # a media item bisected by a chunked-prefill boundary (the remainder is
                    # skipped as cached on the next chunk). That deduction is sound, but the
                    # ANTECEDENT was tested on 2026-08-10 and does not hold: with
                    # chunked_prefill_size=8192, five calibrated requests that placed the
                    # boundary a verified 15%-85% THROUGH a 2512-token image span all
                    # described the image correctly and this branch never fired. sglang keeps
                    # media items whole across chunks. Kept as a cheap canary in case that
                    # ever changes; if it fires, the cross-chunk case is real after all.
                    dropped = adj_end - input_embeds.shape[0]
                    logger.warning(
                        "[MM] media item truncated by the extend window: %d of %d tokens "
                        "not embedded (chunk boundary inside a media item). Output for this "
                        "request may be degraded.", dropped, n_tokens)
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
                self._trigger_sticky = False  # latch consumed (see __init__)
                if state is not None and state.mode == "between":
                    if state.wants_eos or state.rounds >= TTS_MAX_ROUNDS:
                        # We are CLOSING this request (loop/coverage stop or round
                        # bound) — the model asking for another round does not reopen
                        # it. Without this check the close never happened: the stop
                        # forced audiogen_end, the model answered audiogen_start, and
                        # the reuse path reset the state right back to transcript mode
                        # — measured as 150+ reopen/close cycles at ~10/s, the token
                        # budget expiring before any wav was written. Fall THROUGH to
                        # the between watcher, which finalizes and ends the request.
                        state.wants_eos = True  # covers the rounds-bound case too
                    else:
                        # Round N+1: the model chose to keep going after audiogen_end.
                        # REUSE the state — done_segments and the held stream tail are
                        # the request's accumulated audio; a fresh state would orphan
                        # them.
                        state.mode = "transcript"
                        state.rounds += 1
                        state.between_steps = 0
                        state.transcript_done = False
                        state.transcript_steps = 0
                        state.transcript_tokens = []
                        state.end_run = 0
                        state.ended = False
                        state.first_end_flag_step = -1
                        state.end_flag_resamples = 0
                        # Positive control: single-round and multi-round generations
                        # are otherwise indistinguishable until the wav is played.
                        logger.info(f"[AudioGen] req={req_idx}: ROUND {state.rounds} — model "
                                    f"opened another transcript after {len(state.done_segments)} "
                                    f"banked segment(s)")
                        # The round's FIRST transcript token is sampled on THIS step —
                        # mask EOS while it samples (the prefill open does the same; a
                        # sampled EOS finishes the request before any wav exists). The
                        # -3 sentinel masks it and records the intent.
                        overrides[i] = -3
                        continue
                else:
                    # Enter audio mode — start transcript phase
                    # Let lm_head generate text normally (transcript of what to speak)
                    state = AudioGenState(mode="transcript")
                    # Stamp the owning request AT CREATION. rid was previously only set
                    # at completion (for the output filename), so stale-state eviction
                    # had nothing to compare against while the state was live -- which
                    # is exactly when the comparison is needed.
                    state.rid = self._rid_for(req_idx, forward_batch)
                    self._audio_gen_states[req_idx] = state
                    logger.info(f"[AudioGen] req={req_idx}: entered audio mode, starting transcript phase")
                    # lm_head samples the first transcript token this step — mask EOS
                    # like the prefill open does (-3: mask + record intent), otherwise
                    # let it generate freely.
                    overrides[i] = -3
                    continue

            if token == self._audiotext_start_id and state is not None \
                    and state.mode == "transcript":
                # audiotext_start received → now start actual audio codebook generation.
                # Gated on transcript mode: in "between" (multi-round) a stray
                # audiotext_start must not skip the next round's transcript phase.
                state.mode = "generating"
                logger.info(f"[AudioGen] req={req_idx}: transcript done, audio generation started")

            if token == self._audiogen_end_id and state is not None:
                if state.mode == "between":
                    # Our own forced audiogen_end echoing back as input. The model now
                    # picks its next move — and this is exactly the step where it most
                    # wants EOS, so the -3 sentinel must mask it HERE too, not just on
                    # later between steps (see the between watcher below).
                    state.between_steps += 1
                    overrides[i] = -3
                    continue
                # Organic audiogen_end (model ended audio without our end-flag path) —
                # decode everything and clean up.
                logger.info(f"[AudioGen] req={req_idx}: audio generation ended, "
                           f"{len(state.accumulated_ids)} frames accumulated, "
                           f"{len(state.done_segments)} banked segment(s)")
                wav_path = self._finalize_audio(state, req_idx)
                if wav_path:
                    logger.info(f"[AudioGen] req={req_idx}: WAV saved to {wav_path}")
                del self._audio_gen_states[req_idx]
                continue  # back to text mode, no override needed

            # --- Between rounds: audiogen_end is on the wire; the model decides ---
            # what comes next. audiogen_start re-enters above; anything else counts
            # against a small budget. EOS or budget exhaustion closes the request:
            # decode ALL banked segments into one wav (the file must exist BEFORE the
            # request completes — the gateway is polling for it).
            if state is not None and state.mode == "between":
                state.between_steps += 1
                # A sampled EOS would finish the request BEFORE this state machine runs
                # again — no wav would ever be written. So the -3 sentinel masks EOS at
                # the logits and records the model's intent instead; closure is always
                # ours, always after the wav exists.
                if state.wants_eos or state.between_steps > TTS_BETWEEN_BUDGET:
                    why = "model chose EOS" if state.wants_eos \
                        else f"budget ({TTS_BETWEEN_BUDGET} tokens)"
                    logger.info(f"[AudioGen] req={req_idx}: no further round ({why}) — "
                                f"finalizing {len(state.done_segments)} segment(s), "
                                f"{state.rounds} round(s)")
                    state.rid = self._rid_for(req_idx, forward_batch) or state.rid
                    wav_path = self._finalize_audio(state, req_idx)
                    if wav_path:
                        logger.info(f"[AudioGen] req={req_idx}: WAV saved to {wav_path}")
                    del self._audio_gen_states[req_idx]
                    overrides[i] = 2  # close the request; the wav is on disk
                else:
                    overrides[i] = -3  # sentinel: mask EOS, record intent, let it pick
                continue

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
                if state.transcript_steps == 0 and not state.transcript_tokens \
                        and token not in (2, self._audiotext_pad_id,
                                          self._audiotext_start_id,
                                          self._audiogen_start_id,
                                          self._audiogen_end_id):
                    # The round's first transcript token was sampled on the OPEN step
                    # (prefill for round 1, the audiogen_start step for rounds 2+) and
                    # arrives here as the input token — without this append every
                    # transcript is missing its first word. Multi-word inputs survived
                    # the loss on fuzzy stops; a one-word input ("Ready.") degenerated
                    # to bare '.' and closed with NO audio (prewarm failure,
                    # 2026-08-14). Control tokens excluded: a first-sample pad/end is
                    # a genuinely empty transcript and must stay one.
                    state.transcript_tokens.append(token)
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
                if state.step_count < TTS_SILENCE_FRAMES:
                    # First-frame conditioning: replay encoded-silence frames so the
                    # model's first FREE sample sees real audio history.
                    raw = _SILENCE_SEQ[min(state.step_count, len(_SILENCE_SEQ) - 1)]
                    audio_ids = (torch.tensor(raw, dtype=torch.long,
                                              device=self.audio_offset_vals.device)
                                 + self.audio_offset_vals)
                else:
                    audio_ids = self._generate_audio_codebook_step(hs, state)
                if audio_ids is not None:
                    state.accumulated_ids.append(audio_ids)
                    state.step_count += 1
                    # Streaming vocode: emit a windowed chunk once enough new frames
                    # exist. A failed emit falls back to the shipping full decode at
                    # the end rather than losing the request's audio.
                    if _LCN_TTS_STREAM and state.rid and not state.stream_failed:
                        try:
                            self._stream_emit(state, final=False)
                        except Exception:
                            logger.warning(f"[AudioGen] req={req_idx}: stream emit failed; "
                                           f"falling back to full decode at end", exc_info=True)
                            state.stream_failed = True

                # Confirmed end of THIS SEGMENT, or the safety backstop (NOT a
                # task-length cutoff — just a runaway guard).
                if state.ended or state.step_count >= state.max_audio_steps:
                    _capped = state.step_count >= state.max_audio_steps
                    if _capped:
                        logger.warning(f"[AudioGen] req={req_idx}: hit safety cap ({state.max_audio_steps} frames)")
                    else:
                        logger.info(f"[AudioGen] req={req_idx}: confirmed end-of-audio, "
                                    f"round {state.rounds}, "
                                    f"{len(state.accumulated_ids)} frames "
                                    f"(first end flag at frame {state.first_end_flag_step}, "
                                    f"{state.end_flag_resamples} isolated flags resampled)")
                    state.rid = self._rid_for(req_idx, forward_batch) or state.rid
                    if _LCN_TTS_MULTI and not _capped and state.rounds < TTS_MAX_ROUNDS:
                        # Reference semantics (modeling_longcat_next.py ~753): force
                        # audiogen_end and KEEP GENERATING — the model decides whether
                        # the next sentence's round begins. Bank this segment; the
                        # "between" watcher below handles what the model does next.
                        if _LCN_TTS_STREAM and state.rid and not state.stream_failed:
                            try:
                                # Drain the segment's remaining frames to the stream but
                                # HOLD the fade tail: if round N+1 comes, its first piece
                                # cross-fades against it — the same seam treatment the
                                # offline path gives segment joins.
                                self._stream_emit(state, final=False, drain=True)
                            except Exception:
                                logger.warning(f"[AudioGen] req={req_idx}: stream drain failed; "
                                               f"falling back to full decode at end", exc_info=True)
                                state.stream_failed = True
                        if state.accumulated_ids:
                            state.done_segments.append(torch.stack(state.accumulated_ids, dim=0))
                            state.prev_segment_tail = state.accumulated_ids[-1]
                        state.accumulated_ids = []
                        state.streamed_frames = 0
                        state.ended = False
                        state.end_run = 0
                        state.mode = "between"
                        state.between_steps = 0
                        overrides[i] = self._audiogen_end_id
                    else:
                        wav_path = self._finalize_audio(state, req_idx)
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
        if self._lcn_gen_disabled:
            return
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
                    state.rid = self._rid_for(req_idx, forward_batch)  # stamp owner at creation
                    # Capture the request text (this row's prompt tokens) for the
                    # transcript coverage stop — TTS prompts are single-chunk, so the
                    # extend region IS the whole prompt.
                    try:
                        _raw_prompt = self._decode_ids(
                            input_ids[offset:last_token_pos + 1].tolist())
                        state.prompt_norm = self._norm_tts_text(_raw_prompt)
                        # Recitation vs free speech, decided by the request itself: the
                        # TTS instruction (the model card's own phrasing, used by the
                        # gateway, prewarm, and the documented raw-/generate shape)
                        # marks a recitation contract. Voice chat and other open-ended
                        # audio generation lack it and get NO content stops.
                        # Recitation is marked TWO ways, either suffices: the gateway
                        # stamps its TTS requests with an lcntts-prefixed rid (survives
                        # chunked prefill), and the prompt itself may carry the TTS
                        # instruction — but THAT check reads only this extend region,
                        # so a >chunk-size prompt via raw /generate can escape it (the
                        # instruction lands in an earlier chunk). Known residual for
                        # raw callers; the gateway path is covered by the rid.
                        state.recitation = (
                            (state.rid or "").startswith("lcntts")
                            or "用这个声音合成以下内容" in _raw_prompt)
                    except Exception:
                        state.prompt_norm = ""  # coverage stop disabled; budget/frames still bound
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
        # Requests needing a visual codebook token THIS step. Collected rather
        # than served inline so the depth-transformer head runs once for all of
        # them instead of once per request: the head re-reads ~1.5GB of weights
        # per call and costs only 1.22x at bs=8, so N concurrent images were
        # paying N times for traffic that batches almost for free.
        pending = []

        for i in range(batch_size):
            req_idx = forward_batch.req_pool_indices[i].item()
            token = input_ids[i].item()

            state = self._image_gen_states.get(req_idx)

            # Detect image_start_token_id → enter visual mode
            if token == self._image_start_id:
                self._trigger_sticky = False  # latch consumed (see __init__)
                state = ImageGenState()
                state.rid = self._rid_for(req_idx, forward_batch)  # stamp owner at creation
                self._image_gen_states[req_idx] = state
                logger.info(f"[ImageGen] req={req_idx}: entered visual mode (37x37 grid), "
                            f"generating visual token 1 from the image_start hidden state")
                # Original semantics: visual token 1 is generated from THIS
                # step's hidden state (the image_start position); the forced
                # image_pad then carries token-1 feedback into the next step.
                pending.append((i, req_idx, state,
                                forward_batch.positions[i].item(),
                                forward_batch.seq_lens[i].item()))
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
                pending.append((i, req_idx, state,
                                forward_batch.positions[i].item(),
                                forward_batch.seq_lens[i].item()))

        self._image_gen_flush(pending, hidden_states, forward_batch, overrides)
        return overrides

    def _image_gen_flush(self, pending, hidden_states, forward_batch, overrides):
        """Generate one visual token for every request in `pending`, batched.

        One head call serves the whole group. The depth transformer's LEVEL loop
        stays sequential (level L consumes the tokens sampled at 0..L-1), but that
        dependency is entirely within a request — level L for request A never reads
        request B — so the request axis batches at each level while the level axis
        cannot.

        Falls back to the per-request path when CFG is active: CFG needs each
        request's own uncond backbone forward and fuses its rows pairwise, so the
        batch axis is already spoken for. (CFG is off by default — see
        lcn_setup_model_kv_pool_refs — so the batched path is the normal one.)
        """
        if not pending:
            return

        # LCN_HEAD_BATCH=0 forces the per-request path in the SAME build. Exists so
        # batching can be A/B'd without changing anything else: a cross-request defect
        # observed on a batched build is otherwise confounded with every other
        # difference between two images, and swapping builds costs ~8 min of load each
        # way. Default on.
        if not _LCN_HEAD_BATCH or (IMAGE_GEN_CFG_SCALE != 1.0 and self._model_runner is not None):
            for (i, req_idx, state, position, cond_seq_len) in pending:
                overrides[i] = self._image_gen_token_step(
                    i, req_idx, state, hidden_states[i:i+1],
                    forward_batch, position, cond_seq_len,
                )
            return

        rows = torch.cat([hidden_states[i:i+1] for (i, _r, _s, _p, _l) in pending], dim=0)

        # LCN_DIAG_HS=1: fingerprint each row's INPUT hidden state. This localizes a
        # cross-request content bleed to one side of the head: if two concurrent
        # requests arrive here with the SAME fingerprint, their contexts were already
        # identical upstream (backbone/KV/prefix cache) and the head is innocent; if
        # they arrive different and the images still converge, the head path owns it.
        # Off by default -- it costs a host sync per row.
        if _LCN_DIAG_HS and len(pending) > 1:
            # Sample DEEP, not just at the start: the early tokens are near-deterministic
            # and were identical across rows even in a run whose images came out correct,
            # so tokens 1-3 cannot discriminate. Divergence has to be looked for where the
            # images actually differ.
            _t = pending[0][2].current_image_token_num
            if _t in (1, 2, 3, 50, 200, 500, 900, 1300):
                fps = []
                for k, (_i, req_idx, state, _p, _l) in enumerate(pending):
                    r = rows[k].float()
                    fps.append(f"req={req_idx} norm={r.norm().item():.6f} "
                               f"sum={r.sum().item():.6f}")
                same = torch.equal(rows[0], rows[1]) if len(pending) >= 2 else False
                logger.info(f"[ImageGen] HS-DIAG tok={_t} rows_identical={same} :: "
                            + " | ".join(fps))

        visual_ids = self._generate_image_codebook_step(
            rows, None, return_all=True,
        )  # [n, num_codebooks]

        # Paired with HS-DIAG above: same checkpoints, the head's OUTPUT. Together they
        # localize a cross-request bleed to one side of the head call -- identical inputs
        # means the contamination arrived from upstream; different inputs with identical
        # outputs means this call is mixing rows.
        if _LCN_DIAG_HS and len(pending) > 1:
            _t = pending[0][2].current_image_token_num
            if _t in (1, 2, 3, 50, 200, 500, 900, 1300):
                logger.info("[ImageGen] ID-DIAG tok=%d rows=%s identical=%s", _t,
                            visual_ids.tolist(),
                            torch.equal(visual_ids[0], visual_ids[1]))

        # Positive control for the optimization itself, not just for the feature.
        # A batched call and N serial calls produce IDENTICAL logs otherwise, so a
        # silently-unreached flush would look exactly like "batching didn't help" --
        # which is how a previous change in this campaign got measured as a 2% win
        # for a code path that never executed at all. Logged on the first grouped
        # call and every 500th after, so it proves engagement without flooding.
        if len(pending) > 1:
            self._lcn_batched_head_calls = getattr(self, "_lcn_batched_head_calls", 0) + 1
            if self._lcn_batched_head_calls == 1 or self._lcn_batched_head_calls % 500 == 0:
                logger.info(f"[ImageGen] BATCHED head call: {len(pending)} requests in one "
                            f"call (grouped calls so far: {self._lcn_batched_head_calls})")

        if visual_ids.shape[0] != len(pending):
            raise RuntimeError(
                f"batched head returned {visual_ids.shape[0]} rows for {len(pending)} "
                f"requests — refusing to scatter a mismatched result back to per-request state."
            )

        for k, (i, req_idx, state, _position, _cond_seq_len) in enumerate(pending):
            # clone(): the per-request state outlives this step's batch tensor and
            # is later torch.stack'ed, so a row VIEW would pin the whole [n, 8]
            # allocation for the life of the image.
            state.accumulated_ids.append(visual_ids[k].clone())
            state.current_image_token_num += 1
            overrides[i] = self._image_pad_id
            if state.current_image_token_num <= 3:
                logger.info(f"[ImageGen] req={req_idx}: token {state.current_image_token_num}, "
                            f"level0_raw={visual_ids[k][0].item() - self.visual_offset_vals[0].item()}")

    def _image_gen_token_step(self, i, req_idx, state, cond_hs, forward_batch,
                              position, cond_seq_len) -> int:
        """Generate one visual codebook token from cond_hs (CFG init on first use).

        Shared by the decode-step loop AND the image_start trigger steps (decode
        fall-through + prefill detection): the original generates visual token 1
        from the image_start hidden state in the SAME forward — generating it one
        step later (from a zero-embedded pad, with the whole raster shifted +1
        position) measurably degrades global composition.
        Returns the token to force as this step's sampled output (image_pad).
        """
        uncond_hs = None
        first_uncond_hs = None
        if IMAGE_GEN_CFG_SCALE != 1.0 and self._model_runner is not None:
            # Initialize uncond KV cache on first use (try only once)
            if not state.uncond_initialized and state.uncond_req_pool_idx == -1 and state.current_image_token_num == 0:
                # Build uncond prefill with real token IDs (for suffix preservation).
                # Read the conditional tokens from the N-gram token table.
                rtp = self._model_runner.req_to_token_pool
                cond_pool_idx = forward_batch.req_pool_indices[i].item()
                # Read token IDs from token table (N-gram table stores them;
                # specials read back as 0 — _alloc_uncond_kv rebuilds the suffix)
                ngram_info = getattr(forward_batch, 'ngram_embedding_info', None)
                if ngram_info is not None:
                    token_table = ngram_info.token_table
                    prefill_ids = token_table[cond_pool_idx, :cond_seq_len].to(cond_hs.device)
                else:
                    prefill_ids = torch.zeros(cond_seq_len, dtype=torch.int32,
                                             device=cond_hs.device)
                state.uncond_req_pool_idx, first_uncond_hs = self._alloc_uncond_kv(
                    cond_pool_idx,
                    cond_seq_len, prefill_ids, forward_batch)
                if state.uncond_req_pool_idx >= 0:
                    state.uncond_seq_len = cond_seq_len
                    state.uncond_initialized = True
                    logger.info(f"[ImageGen] CFG initialized: uncond_req={state.uncond_req_pool_idx}, "
                               f"seq_len={cond_seq_len}")

            if state.uncond_initialized:
                if first_uncond_hs is not None:
                    # Token 1: the uncond prefill's last hidden IS this position's
                    # uncond state — a decode step here would double-process it.
                    uncond_hs = first_uncond_hs
                else:
                    uncond_hs = self._run_uncond_decode(state, position, forward_batch)

        visual_ids = self._generate_image_codebook_step(cond_hs, uncond_hs)
        state.accumulated_ids.append(visual_ids)
        state.current_image_token_num += 1

        if state.current_image_token_num <= 3:
            logger.info(f"[ImageGen] req={req_idx}: token {state.current_image_token_num}, "
                       f"level0_raw={visual_ids[0].item() - self.visual_offset_vals[0].item()}")
        return self._image_pad_id

    def _check_prefill_image_start(
        self, input_ids: torch.Tensor, logits_output, forward_batch: ForwardBatch,
        hidden_states: torch.Tensor = None,
    ):
        """Check if prefill ends with image_start_token_id.

        Original semantics: visual token 1 is generated from the image_start
        position's hidden state in the SAME forward (here: the prefill), so the
        forced image_pad that follows carries token-1 feedback — never a
        zero-embedded input, and no +1 raster position shift.
        """
        if self._lcn_gen_disabled:
            return
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
                    forced = self._image_pad_id
                    if hidden_states is not None and last_token_pos < hidden_states.shape[0]:
                        forced = self._image_gen_token_step(
                            i, req_idx, state,
                            hidden_states[last_token_pos:last_token_pos+1],
                            forward_batch,
                            forward_batch.positions[last_token_pos].item(),
                            forward_batch.seq_lens[i].item(),
                        )
                    else:
                        logger.warning(f"[ImageGen] req={req_idx}: no hidden state at prefill "
                                       f"trigger — visual token 1 deferred one step (legacy path)")
                    # Force image_pad as first token
                    if logits_output.next_token_logits is not None and i < logits_output.next_token_logits.shape[0]:
                        logits_output.next_token_logits[i, :] = float('-inf')
                        logits_output.next_token_logits[i, forced] = 0.0
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

        # LCN_INT8_HEADS: per-slot int8 for the depth-head FFNs; bf16 originals
        # freed (~1.7GB back across both heads). Scope and measured basis in
        # int8_head_ffn.py (bench: research/int8_heads/bench_depth_head.py).
        # Default OFF — a generation-path change gated on the owner's paired A/B.
        # Per-head selection: '1'/'both', 'audio', 'visual', or '0'/unset.
        # The 2026-08-14 5v5 owner A/B found hard spatial-geometry failures
        # clustering in the int8 VISUAL arm (3/5 vs 1/5, plus a novel
        # cat-embedded-in-wall mode) while audio int8 carries the -34%/frame
        # TTS win with no adverse listen — so the deployment default is
        # 'audio' only.
        _int8_sel = os.environ.get('LCN_INT8_HEADS', '0').strip().lower()
        # 'audio4': int4-g128 trial on the audio head (bench: -18% head time,
        # relerr 15x int8's — ears-gated, not a default candidate until judged).
        _int8_on = {'1': ('visual', 'audio'), 'both': ('visual', 'audio'),
                    'audio': ('audio',), 'visual': ('visual',),
                    'audio4': ('audio:int4',)}.get(_int8_sel, ())
        if _int8_on:
            from sglang.srt.models.int8_head_ffn import attach_int8_ffn, attach_int4_ffn
            for _name, _head in (("visual", self.visual_head), ("audio", self.audio_head)):
                if _head is None:
                    continue
                if _name in _int8_on:
                    _freed = attach_int8_ffn(_head, len(_head.codebook_sizes))
                    logger.info(f"[INT8-HEADS] {_name} head FFN -> per-slot int8 "
                                f"({_freed/(1<<20):.0f}MB bf16 freed)")
                elif f"{_name}:int4" in _int8_on:
                    _freed = attach_int4_ffn(_head, len(_head.codebook_sizes))
                    logger.info(f"[INT8-HEADS] {_name} head FFN -> per-slot int4-g128 "
                                f"({_freed/(1<<20):.0f}MB bf16 freed)")

        # LCN_HEAD_GRAPH: CUDA-graph replay of the per-level head forward — the
        # generation step is launch-latency-bound (~4.4k launches, ~42ms/step
        # distributed idle; see lcn_head_graph.py). The call sites route through
        # these attrs; without the flag they are the bare heads (zero-cost).
        # object.__setattr__: nn.Module.__setattr__ registers a Module value as
        # a child under this name and then REFUSES a non-Module reassignment
        # (crashed the first hgraph launch at load time) — and the routing attr
        # must not enter the module tree / state_dict anyway.
        object.__setattr__(self, '_visual_head_call', self.visual_head)
        object.__setattr__(self, '_audio_head_call', self.audio_head)
        if os.environ.get('LCN_HEAD_GRAPH', '0').strip() == '1':
            from sglang.srt.models.lcn_head_graph import GraphedHeadRunner
            if self.visual_head is not None:
                object.__setattr__(self, '_visual_head_call', GraphedHeadRunner(
                    self.visual_head, len(self.visual_head.codebook_sizes), "visual"))
            if self.audio_head is not None:
                object.__setattr__(self, '_audio_head_call', GraphedHeadRunner(
                    self.audio_head, len(self.audio_head.codebook_sizes), "audio"))
            logger.info("[HeadGraph] enabled — graphs capture lazily per (bsz, level)")

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
