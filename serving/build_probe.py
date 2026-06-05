#!/usr/bin/env python3
"""Fast (CPU, no weights) build harness for LongCat-Next multimodal submodules.
Instantiates the visual/audio tokenizers + gen heads from the real config to surface
every construction bug quickly, instead of 7-min serve cycles."""
import traceback, torch
from sglang.srt.utils.hf_transformers.config import get_config
from sglang.srt.models.longcat_next_mm import LongcatNextForCausalLM, ensure_config_object

c = get_config("/m", trust_remote_code=True,
               model_override_args={"architectures": ["LongcatNextForCausalLM"]})
print("config:", type(c).__name__, "| use_ngram:", getattr(c, "use_ngram_embedding", "?"))

# _make_full_config only uses `config` (no self attrs) -> call unbound with dummy self
full_cfg = LongcatNextForCausalLM._make_full_config(None, c)

def attempt(name, fn):
    try:
        m = fn()
        n = sum(1 for _ in m.named_parameters())
        print(f"[OK] {name}: {n} params")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

from sglang.srt.models.longcat_next_visual import LongcatNextVisualTokenizer
from sglang.srt.models.longcat_next_audio import LongcatNextAudioTokenizer
from sglang.srt.models.longcat_next_heads import CasualDepthTransformerHead

attempt("visual_tokenizer", lambda: LongcatNextVisualTokenizer(full_cfg))
attempt("audio_tokenizer", lambda: LongcatNextAudioTokenizer(full_cfg))
vc = ensure_config_object(c.visual_config); ac = ensure_config_object(c.audio_config)
attempt("visual_head", lambda: CasualDepthTransformerHead(
    hidden_size=c.hidden_size, codebook_sizes=list(vc.vq_config.codebook_sizes),
    transformer_layer_num=vc.image_head_transformer_layers,
    transformer_dim=vc.image_head_transformer_dims,
    transformer_ffn_scale=vc.image_head_transformer_ffn_scale))
attempt("audio_head", lambda: CasualDepthTransformerHead(
    hidden_size=c.hidden_size, codebook_sizes=list(ac.vq_config.codebook_sizes),
    transformer_layer_num=ac.audio_head_transformer_layers,
    transformer_dim=ac.audio_head_transformer_dims,
    transformer_ffn_scale=ac.audio_head_transformer_ffn_scale))
print("PROBE DONE")
