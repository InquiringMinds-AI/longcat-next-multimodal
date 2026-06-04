# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import concurrent.futures
import logging
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs import LongcatFlashConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.kernels import zero_experts_compute_triton
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
from sglang.srt.layers.n_gram_embedding import NgramEmbedding
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_weight_ue8m0_inplace,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.utils import (
    maybe_executor_submit,
    should_async_load,
    should_deepgemm_weight_requant_ue8m0,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    bind_or_assign,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()

if _is_cuda:
    from sgl_kernel import awq_dequantize
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sglang.srt.layers.quantization.awq.awq_triton import (
        awq_dequantize_triton as awq_dequantize,
    )
else:
    pass

logger = logging.getLogger(__name__)


class LongcatFlashMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LongcatFlashRouter(nn.Module):
    def __init__(
        self,
        config,
        zero_expert_num=0,
        rounter_params_dtype=torch.float32,
        prefix: str = "",
    ):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_routed_experts = self.n_routed_experts + zero_expert_num
        self.rounter_params_dtype = rounter_params_dtype
        self.classifier = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=config.router_bias,
            params_dtype=rounter_params_dtype,
            quant_config=None,
            prefix=add_prefix("classifier", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros((self.n_routed_experts), dtype=rounter_params_dtype)
        )

    def forward(self, hidden_states):
        logits, _ = self.classifier(hidden_states.to(self.rounter_params_dtype))
        return logits


class LongcatFlashMoE(nn.Module):

    def __init__(
        self,
        config: LongcatFlashConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_experts = config.n_routed_experts
        self.top_k = config.moe_topk
        self.zero_expert_num = config.zero_expert_num
        self.zero_expert_type = config.zero_expert_type

        if config.rounter_params_dtype == "float32":
            self.rounter_params_dtype = torch.float32
        else:
            self.rounter_params_dtype = torch.bfloat16

        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router = LongcatFlashRouter(
            config=self.config,
            zero_expert_num=self.zero_expert_num,
            rounter_params_dtype=self.rounter_params_dtype,
            prefix=add_prefix("router", prefix),
        )

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=False,
            use_grouped_topk=False,
            correction_bias=self.router.e_score_correction_bias.data,
            layer_id=layer_id,
        )
        self.topk.forward = self.topk.forward_native

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(hidden_states)
        topk_weights, topk_idx, _ = self.topk(
            hidden_states,
            router_logits,
        )
        if self.zero_expert_type is not None:
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_idx,
                expert_scales=topk_weights,
                num_experts=self.num_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )
        # Clamp expert indices to real experts (identity experts set to -1 by zero_experts_compute_triton)
        topk_idx = topk_idx.clamp(min=0, max=self.num_experts - 1)
        topk_output = StandardTopKOutput(topk_weights, topk_idx, _)

        final_hidden_states = self.experts(hidden_states, topk_output)
        final_hidden_states *= self.routed_scaling_factor

        if self.zero_expert_type is not None and hidden_states.shape[0] > 0:
            final_hidden_states += zero_expert_result.to(final_hidden_states.device)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]


class LongcatFlashDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LongcatFlashConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.self_attn = nn.ModuleList(
            [
                DeepseekV2AttentionMLA(
                    config=config,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=config.rope_parameters["rope_theta"],
                    rope_scaling=None,
                    max_position_embeddings=config.max_position_embeddings,
                    quant_config=(
                        None
                        if "self_attn" in getattr(config, "disable_quant_module", [])
                        else quant_config
                    ),
                    layer_id=layer_id * 2 + i,
                    reduce_results=False,
                    prefix=add_prefix(f"self_attn.{i}", prefix),
                    alt_stream=self.alt_stream,
                )
                for i in range(2)
            ]
        )

        self.input_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )

        self.mlps = nn.ModuleList(
            [
                LongcatFlashMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=(
                        None
                        if "mlps" in getattr(config, "disable_quant_module", [])
                        else quant_config
                    ),
                    prefix=add_prefix(f"mlps.{i}", prefix),
                )
                for i in range(2)
            ]
        )

        self.mlp = LongcatFlashMoE(
            layer_id=self.layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.mlp_layer_scatter_modes = [
            LayerScatterModes.init_new(
                layer_id=self.layer_id * 2 + i,
                num_layers=config.num_hidden_layers,
                is_layer_sparse=False,
                is_previous_layer_sparse=False,
                # TODO: Check if the following is correct.
                is_next_layer_sparse=False,
            )
            for i in range(2)
        ]
        self.mlp_layer_communicator = [
            LayerCommunicator(
                layer_scatter_modes=self.mlp_layer_scatter_modes[i],
                input_layernorm=self.input_layernorm[i],
                post_attention_layernorm=self.post_attention_layernorm[i],
                qkv_latent_func=self.self_attn[i].prepare_qkv_latent,
            )
            for i in range(2)
        ]

        self.moe_layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=self.layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=True,
            is_previous_layer_sparse=True,
            # TODO: Check if the following is correct.
            is_next_layer_sparse=True,
        )
        self.moe_layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.moe_layer_scatter_modes,
            input_layernorm=self.input_layernorm[0],
            post_attention_layernorm=self.post_attention_layernorm[0],
            qkv_latent_func=self.self_attn[0].prepare_qkv_latent,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        # first_attn
        hidden_states, residual = self.moe_layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn[0](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        # moe
        hidden_states, residual = self.moe_layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        moe_hidden_states = hidden_states.clone()
        moe_residual = residual.clone()
        moe_hidden_states = self.mlp(moe_hidden_states)
        moe_hidden_states, moe_residual = self.moe_layer_communicator.postprocess_layer(
            moe_hidden_states, moe_residual, forward_batch
        )

        hidden_states, residual = self.forward_mlp(
            hidden_states, positions, residual, forward_batch, zero_allocator
        )

        hidden_states = moe_hidden_states + hidden_states
        return hidden_states, residual

    def forward_mlp(
        self, hidden_states, positions, residual, forward_batch, zero_allocator
    ):
        # first_mlp
        hidden_states = self.mlps[0](hidden_states)
        # TP all_reduce
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        # second_attn
        hidden_states, residual = self.mlp_layer_communicator[1].prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn[1](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        # second_mlp
        hidden_states, residual = self.mlp_layer_communicator[1].prepare_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self.mlps[1](hidden_states)
        # TP all_reduce
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        hidden_states, residual = self.mlp_layer_communicator[1].postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class LongcatFlashModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: LongcatFlashConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size

        if config.use_ngram_embedding:
            self.use_ngram_embedding = True
            self.embed_tokens = NgramEmbedding(
                num_embeddings=getattr(config, "text_vocab_size", config.vocab_size),
                embedding_dim=config.hidden_size,
                over_embedding_m=config.ngram_embedding_m,
                over_embedding_k=config.ngram_embedding_k,
                over_embedding_n=config.ngram_embedding_n,
                word_vocab_size=config.vocab_size,
            )
        else:
            self.use_ngram_embedding = False
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )

        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                LongcatFlashDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        total_num_layers = len(self.layers)
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )
        if input_embeds is None:
            if self.use_ngram_embedding:
                hidden_states = self.embed_tokens(input_ids, forward_batch)
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None

        aux_hidden_states = []
        for i in range(total_num_layers):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual, zero_allocator
                )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class LongcatFlashForCausalLM(nn.Module):
    # for quark model load
    packed_modules_mapping = {}

    def __init__(
        self,
        config: LongcatFlashConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # for quark model load
        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.model = LongcatFlashModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.use_ngram_embedding = config.use_ngram_embedding
        self.lm_head = ParallelLMHead(
            getattr(config, "text_vocab_plus_multimodal_special_token_size", config.vocab_size),
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

        # --- Visual UNDERSTANDING graft (image-in -> text-out only) ---
        # Build the visual tokenizer so its checkpoint weights (model.visual_tokenizer.*)
        # have destination params at load time. Gated on visual_config presence.
        self.visual_tokenizer = None
        self.visual_offset_vals = None
        self._mm_embed_rows = None  # lazy multimodal embedding slice [vocab_mm, hidden]
        if getattr(config, "visual_config", None) is not None:
            try:
                from sglang.srt.models.longcat_next_visual import (
                    LongcatNextVisualTokenizer,
                )

                full_cfg = self._make_visual_full_config(config)
                # Attach to self.model so weight names match ckpt (model.visual_tokenizer.*)
                self.model.visual_tokenizer = LongcatNextVisualTokenizer(full_cfg)
                self.visual_tokenizer = self.model.visual_tokenizer
                self._init_visual_codebook_offsets(config)
                self._init_visual_token_ids(config)
                logger.info("Visual tokenizer (understanding) initialized")
            except Exception as e:
                logger.warning(f"Could not initialize visual tokenizer: {e}")
                self.model.visual_tokenizer = None
                self.visual_tokenizer = None

        # --- Audio UNDERSTANDING graft (audio-in -> text-out only) ---
        # Mirror of the visual graft above. Build the audio tokenizer so its
        # checkpoint weights (model.audio_tokenizer.*) have destination params at
        # load time. Reuses _make_visual_full_config, which now also builds a real
        # WhisperConfig for audio_config (the audio encoder reads Whisper defaults
        # like scale_embedding that the stored config dict omits). Gated on
        # audio_config presence.
        self.audio_tokenizer = None
        self.audio_offset_vals = None
        if getattr(config, "audio_config", None) is not None:
            try:
                from sglang.srt.models.longcat_next_audio import (
                    LongcatNextAudioTokenizer,
                )

                full_cfg = self._make_visual_full_config(config)
                # Attach to self.model so weight names match ckpt (model.audio_tokenizer.*)
                self.model.audio_tokenizer = LongcatNextAudioTokenizer(full_cfg)
                self.audio_tokenizer = self.model.audio_tokenizer
                self._init_audio_codebook_offsets(config)
                self._init_audio_token_ids(config)
                logger.info("Audio tokenizer (understanding) initialized")
            except Exception as e:
                logger.warning(f"Could not initialize audio tokenizer: {e}")
                self.model.audio_tokenizer = None
                self.audio_tokenizer = None

    # ---------------- Visual understanding helpers ----------------
    def _vcfg_get(self, obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _make_visual_full_config(self, config):
        """Wrap config so LongcatNextVisualTokenizer gets visual_config as an
        attribute object plus a real Qwen2_5_VLVisionConfig for the encoder."""
        from sglang.srt.models.longcat_next_visual import LongcatNextVisualTokenizer  # noqa: F401

        class _DictCfg:
            def __init__(self, d):
                for k, v in d.items():
                    if not isinstance(k, str):
                        continue  # skip non-str keys (e.g. HF id2label {0:..,1:..})
                    if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                        v = _DictCfg(v)
                    setattr(self, k, v)

        if hasattr(config, "to_dict"):
            cfg_dict = config.to_dict()
        else:
            cfg_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
        full_cfg = _DictCfg(cfg_dict)
        if isinstance(cfg_dict.get("visual_config"), dict):
            try:
                from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
                    Qwen2_5_VLVisionConfig,
                )
                import inspect

                vc = cfg_dict["visual_config"]
                sig = inspect.signature(Qwen2_5_VLVisionConfig.__init__)
                valid = {k: v for k, v in vc.items() if isinstance(k, str) and k in sig.parameters}
                full_cfg.visual_config._hf_vision_config = Qwen2_5_VLVisionConfig(**valid)
            except Exception as e:
                logger.warning(f"Could not build Qwen2_5_VLVisionConfig: {e}")
        # Audio: the audio tokenizer reads config.audio_config as a real config that
        # carries Whisper defaults (scale_embedding, max_source_positions, ...) NOT
        # present in the stored config dict. Mirror LongcatNextAudioConfig: build a
        # WhisperConfig from the scalar keys, re-attach LongCat-specific scalars, and
        # wrap the nested sub-configs as PretrainedConfig objects. (The _DictCfg form
        # would AttributeError on scale_embedding.)
        if isinstance(cfg_dict.get("audio_config"), dict):
            try:
                from transformers import WhisperConfig, PretrainedConfig

                ac = cfg_dict["audio_config"]
                nested = (
                    "vq_config",
                    "vocoder_config",
                    "flow_matching_config",
                    "cosy24kvocoder_config",
                )
                whisper_kwargs = {
                    k: v
                    for k, v in ac.items()
                    if isinstance(k, str)
                    and k not in nested
                    and not isinstance(v, (dict, list))
                }
                audio_hf = WhisperConfig(**whisper_kwargs)
                # re-attach LongCat-specific scalar keys WhisperConfig dropped
                for k, v in ac.items():
                    if k in nested or isinstance(v, (dict, list)):
                        continue
                    if not hasattr(audio_hf, k):
                        setattr(audio_hf, k, v)
                # nested sub-configs as PretrainedConfig (matches LongcatNextAudioConfig)
                for k in nested:
                    sub = ac.get(k, {}) or {}
                    pc = PretrainedConfig(
                        **{kk: vv for kk, vv in sub.items() if not isinstance(vv, dict)}
                    )
                    if k == "flow_matching_config" and isinstance(
                        sub.get("cfm_params"), dict
                    ):
                        pc.cfm_params = PretrainedConfig(**sub["cfm_params"])
                    setattr(audio_hf, k, pc)
                full_cfg.audio_config = audio_hf
            except Exception as e:
                logger.warning(f"Could not build audio WhisperConfig: {e}")
        return full_cfg

    def _init_visual_codebook_offsets(self, config):
        """visual_offset_vals = cumsum([visual_offset] + codebook_sizes[:-1])."""
        vc = getattr(config, "visual_config", None)
        vq = self._vcfg_get(vc, "vq_config", {})
        codebook_sizes = self._vcfg_get(vq, "codebook_sizes")
        visual_offset = getattr(config, "visual_offset", None)
        if codebook_sizes is None or visual_offset is None:
            self.visual_offset_vals = None
            return
        offset_list = [visual_offset] + list(codebook_sizes[:-1])
        offsets = torch.cumsum(torch.tensor(offset_list, dtype=torch.long), dim=0)
        if "visual_offset_vals" in self.__dict__:
            del self.visual_offset_vals  # drop the placeholder attr so register_buffer can claim the name
        self.register_buffer("visual_offset_vals", offsets, persistent=False)

    def _init_visual_token_ids(self, config):
        vc = getattr(config, "visual_config", None)
        self._image_pad_id = self._vcfg_get(vc, "image_pad_token_id", 131108)
        self._image_start_id = self._vcfg_get(vc, "image_start_token_id", 131106)
        self._image_end_id = self._vcfg_get(vc, "image_end_token_id", 131107)
        self._image_newline_id = self._vcfg_get(vc, "image_newline_token_id", 131109)

    # ---------------- Audio understanding helpers ----------------
    def _init_audio_codebook_offsets(self, config):
        """audio_offset_vals = cumsum([audio_offset] + codebook_sizes[:-1]).
        Mirror of _init_visual_codebook_offsets; audio_offset defaults to
        text_vocab_plus_multimodal_special_token_size (131125)."""
        ac = getattr(config, "audio_config", None)
        vq = self._vcfg_get(ac, "vq_config", {})
        codebook_sizes = self._vcfg_get(vq, "codebook_sizes")
        audio_offset = getattr(config, "audio_offset", None)
        if audio_offset is None:
            audio_offset = getattr(
                config, "text_vocab_plus_multimodal_special_token_size", 131125
            )
        if codebook_sizes is None or audio_offset is None:
            self.audio_offset_vals = None
            return
        offset_list = [audio_offset] + list(codebook_sizes[:-1])
        offsets = torch.cumsum(torch.tensor(offset_list, dtype=torch.long), dim=0)
        if "audio_offset_vals" in self.__dict__:
            del self.audio_offset_vals  # drop placeholder so register_buffer can claim the name
        self.register_buffer("audio_offset_vals", offsets, persistent=False)

    def _init_audio_token_ids(self, config):
        ac = getattr(config, "audio_config", None)
        self._audio_pad_id = self._vcfg_get(ac, "audio_pad_token_id", 131105)
        self._audio_start_id = self._vcfg_get(ac, "audio_start_token_id", 131103)
        self._audio_end_id = self._vcfg_get(ac, "audio_end_token_id", 131104)

    def _load_mm_embed_rows(self):
        """Load the multimodal embedding rows [text_vocab_plus : full_mm_vocab].

        The NVFP4 backbone's model.embed_tokens.weight was truncated to text vocab
        (131125 rows); the original model embeds VQ visual ids with offset using the
        full (282624-row) table. We restore the dropped slice as a side tensor.
        """
        if self._mm_embed_rows is not None:
            return
        import os
        from safetensors import safe_open

        candidates = []
        mp = os.environ.get("SGLANG_MODEL_PATH", "")
        if mp:
            candidates.append(os.path.join(mp, "mm_embed_rows.safetensors"))
            candidates.append(os.path.join(os.path.dirname(mp.rstrip("/")), "lc_mm_embed_rows.safetensors"))
        candidates += [
            "/models/lc_mm_embed_rows.safetensors",
            "/models/output/LongCat-Next-NVFP4-bf16mla/mm_embed_rows.safetensors",
        ]
        device = next(self.model.embed_tokens.parameters()).device
        for cb in candidates:
            if cb and os.path.exists(cb):
                with safe_open(cb, framework="pt", device="cpu") as sf:
                    self._mm_embed_rows = sf.get_tensor("mm_embed_rows").to(device)
                logger.info(f"Loaded mm_embed_rows {tuple(self._mm_embed_rows.shape)} from {cb}")
                return
        logger.warning(
            "lc_mm_embed_rows.safetensors not found; visual tokens will embed as zeros"
        )

    @torch.no_grad()
    def get_image_feature(self, items):
        """pixels -> visual encoder -> VQ ids -> (+offset) embed via full mm table
        -> sum over codebooks -> visual_embedding_layer. Returns [n_vis_tok, hidden]."""
        if self.visual_tokenizer is None:
            return None
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            dtype=self.visual_tokenizer.visual_model.get_dtype(),
            device=next(self.visual_tokenizer.parameters()).device,
        )
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)
        # [n_vis_tok, num_codebooks] raw VQ ids
        visual_ids = self.visual_tokenizer.encode(pixel_values, image_grid_thw)
        if self.visual_offset_vals is not None:
            visual_ids = visual_ids + self.visual_offset_vals.to(visual_ids.device)
        visual_embeddings = self._embed_visual_ids(visual_ids)  # [seq, hidden]
        logger.info(f"[VIS] ids uniq={visual_ids.unique().numel()} pre-vel emb shape={tuple(visual_embeddings.shape)} mean|x|={visual_embeddings.abs().mean().item():.4f} std={visual_embeddings.std().item():.4f}")
        visual_embeddings = self.visual_tokenizer.visual_embedding_layer(visual_embeddings)
        logger.info(f"[VIS] post visual_embedding_layer mean|x|={visual_embeddings.abs().mean().item():.4f} std={visual_embeddings.std().item():.4f}")
        return visual_embeddings

    @torch.no_grad()
    def get_audio_feature(self, items):
        """mel -> whisper encoder -> bridge/VQ ids -> (+offset) embed via full mm
        table -> sum over codebooks. Returns [n_audio_tok, hidden].

        NOTE: unlike get_image_feature there is NO post embedding/projection layer.
        The canonical get_audio_feature (longcat_next_mm.py) goes straight from the
        summed mm-embedding to the backbone, then pads/truncates each item to its
        processor-declared bridge_length. We mirror that exactly. _embed_visual_ids
        is modality-agnostic (subtracts codebook_base=131125) so it is reused."""
        if self.audio_tokenizer is None:
            return None
        all_embeddings = []
        device = next(self.audio_tokenizer.parameters()).device
        for item in items:
            msd = getattr(item, "model_specific_data", None) or {}
            encoder_length = msd.get("encoder_length", None)
            bridge_length = msd.get("bridge_length", None)
            if encoder_length is None or bridge_length is None:
                continue
            # mel features [num_mel_bins, frames] -> [1, num_mel_bins, frames]
            audio_tensor = torch.as_tensor(
                item.feature, dtype=torch.float32, device=device
            ).unsqueeze(0)
            # encode(mel, encoder_length, bridge_length) -> [seq, num_codebooks] VQ ids
            audio_ids = self.audio_tokenizer.encode(
                audio_tensor,
                torch.tensor([encoder_length], device=device),
                torch.tensor([bridge_length], device=device),
            )
            if self.audio_offset_vals is not None:
                audio_ids = audio_ids + self.audio_offset_vals.to(audio_ids.device)
            audio_embeddings = self._embed_visual_ids(audio_ids)  # [actual_seq, hidden]
            # pad/truncate to the processor-declared bridge_length
            actual_len = audio_embeddings.shape[0]
            if actual_len < bridge_length:
                pad = torch.zeros(
                    bridge_length - actual_len,
                    audio_embeddings.shape[1],
                    dtype=audio_embeddings.dtype,
                    device=audio_embeddings.device,
                )
                audio_embeddings = torch.cat([audio_embeddings, pad])
            elif actual_len > bridge_length:
                audio_embeddings = audio_embeddings[:bridge_length]
            logger.info(f"[AUD] ids uniq={audio_ids.unique().numel()} emb shape={tuple(audio_embeddings.shape)} mean|x|={audio_embeddings.abs().mean().item():.4f} std={audio_embeddings.std().item():.4f}")
            all_embeddings.append(audio_embeddings)
        if not all_embeddings:
            return None
        return torch.cat(all_embeddings, dim=0)

    def _embed_visual_ids(self, ids_with_offset):
        """Embed offset visual ids [seq, lev] -> [seq, hidden], summed over codebooks.
        ids < text_vocab use the backbone word embedding; ids >= text_vocab use the
        restored multimodal embedding slice."""
        self._load_mm_embed_rows()
        if hasattr(self.model.embed_tokens, "word_embeder"):
            word_embed = self.model.embed_tokens.word_embeder
        else:
            word_embed = self.model.embed_tokens
        text_vocab = word_embed.num_embeddings
        # The multimodal embedding slice begins at codebook_base (131125), NOT at
        # text_vocab (131072). They differ by the 53 special tokens — subtracting the
        # wrong base shifts every visual lookup by 53 rows. (Matches the proven fork's
        # _embed_multimodal_ids, which kept these as two separate constants.)
        codebook_base = getattr(
            self.config, "text_vocab_plus_multimodal_special_token_size", 131125
        )
        hidden = word_embed.embedding_dim
        all_embeds = []
        for lev in range(ids_with_offset.shape[1]):
            tok = ids_with_offset[:, lev]
            embeds = torch.zeros(
                len(tok), hidden, dtype=word_embed.weight.dtype, device=tok.device
            )
            in_text = tok < text_vocab
            in_mm = tok >= codebook_base
            if in_text.any():
                embeds[in_text] = word_embed(tok[in_text])
            if in_mm.any() and self._mm_embed_rows is not None:
                mm_idx = (tok[in_mm] - codebook_base).clamp(
                    max=self._mm_embed_rows.shape[0] - 1
                )
                embeds[in_mm] = self._mm_embed_rows[mm_idx].to(embeds.dtype)
            all_embeds.append(embeds)
        return torch.stack(all_embeds, dim=1).sum(dim=1)

    def pad_input_ids(self, input_ids, mm_inputs):
        from sglang.srt.managers.mm_utils import (
            MultiModalityDataPaddingPatternMultimodalTokens,
        )

        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def _get_image_items(self, forward_batch):
        mm_inputs_list = getattr(forward_batch, "mm_inputs", None)
        if not mm_inputs_list:
            return []
        image_items = []
        for mm_input in mm_inputs_list:
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                if item.is_image():
                    image_items.append(item)
        return image_items

    def _get_audio_items(self, forward_batch):
        mm_inputs_list = getattr(forward_batch, "mm_inputs", None)
        if not mm_inputs_list:
            return []
        audio_items = []
        for mm_input in mm_inputs_list:
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                if item.is_audio():
                    audio_items.append(item)
        return audio_items

    def _compute_image_embeddings(self, input_ids, forward_batch):
        """Backbone embeddings with image AND audio placeholder positions overwritten."""
        if self.use_ngram_embedding:
            input_embeds = self.model.embed_tokens(input_ids, forward_batch)
        else:
            input_embeds = self.model.embed_tokens(input_ids)
        # zero image-pad AND audio-pad positions before replacement
        pad_mask = input_ids == getattr(self, "_image_pad_id", 131108)
        audio_pad_mask = input_ids == getattr(self, "_audio_pad_id", 131105)
        if pad_mask.any():
            input_embeds[pad_mask] = 0
        if audio_pad_mask.any():
            input_embeds[audio_pad_mask] = 0
        image_items = self._get_image_items(forward_batch)
        audio_items = self._get_audio_items(forward_batch)
        logger.info(f"[VIS] pad_positions={int(pad_mask.sum())} input_embeds={tuple(input_embeds.shape)} n_image_items={len(image_items)}")
        if image_items:
            image_embeds = self.get_image_feature(image_items)
            if image_embeds is not None:
                self._replace_image_embeddings(
                    input_embeds, image_items, image_embeds, forward_batch
                )
                if pad_mask.any():
                    pm = input_embeds[pad_mask]
                    logger.info(f"[VIS] AFTER replace: pad-pos embeds mean|x|={pm.abs().mean().item():.4f} std={pm.std().item():.4f} zero_rows={(pm.abs().sum(-1)==0).sum().item()}/{pm.shape[0]}")
        if audio_items:
            logger.info(f"[AUD] audio_pad_positions={int(audio_pad_mask.sum())} n_audio_items={len(audio_items)}")
            audio_embeds = self.get_audio_feature(audio_items)
            if audio_embeds is not None:
                # _replace_image_embeddings is modality-agnostic (uses item.offsets)
                self._replace_image_embeddings(
                    input_embeds, audio_items, audio_embeds, forward_batch
                )
                if audio_pad_mask.any():
                    am = input_embeds[audio_pad_mask]
                    logger.info(f"[AUD] AFTER replace: pad-pos embeds mean|x|={am.abs().mean().item():.4f} std={am.std().item():.4f} zero_rows={(am.abs().sum(-1)==0).sum().item()}/{am.shape[0]}")
        return input_embeds

    def _replace_image_embeddings(self, input_embeds, items, embeds, forward_batch):
        embed_idx = 0
        for item in items:
            offsets = getattr(item, "offsets", None)
            if offsets is None:
                continue
            for start, end in offsets:
                n_tokens = end - start
                if embed_idx + n_tokens > embeds.shape[0]:
                    n_tokens = embeds.shape[0] - embed_idx
                if n_tokens <= 0:
                    continue
                prefix_len = 0
                if getattr(forward_batch, "extend_prefix_lens_cpu", None):
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
                    input_embeds[adj_start:adj_end] = embeds[
                        embed_idx : embed_idx + n_tokens
                    ].to(input_embeds.dtype)
                logger.info(f"[VIS] replace offsets=({start},{end}) prefix_len={prefix_len} -> wrote input_embeds[{adj_start}:{adj_end}] from embeds[{embed_idx}:{embed_idx+n_tokens}]")
                embed_idx += n_tokens

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        # Visual/Audio understanding: on prefill with image OR audio inputs, compute
        # backbone embeddings and overwrite the media-placeholder positions with the
        # encoded embeds, then run the backbone on the prebuilt embeddings (text-out
        # only). _compute_image_embeddings handles both modalities internally.
        model_input_ids = input_ids
        has_image = (
            self.visual_tokenizer is not None
            and forward_batch.contains_image_inputs()
        )
        has_audio = (
            self.audio_tokenizer is not None
            and forward_batch.contains_audio_inputs()
        )
        if (
            input_embeds is None
            and not forward_batch.forward_mode.is_decode()
            and (has_image or has_audio)
        ):
            # clamp OOB ids (placeholder/special tokens can exceed lm_head vocab)
            max_id = (
                getattr(
                    self.config,
                    "text_vocab_plus_multimodal_special_token_size",
                    self.config.vocab_size,
                )
                - 1
            )
            clamped_ids = input_ids.clamp(min=0, max=max_id)
            input_embeds = self._compute_image_embeddings(clamped_ids, forward_batch)
            forward_batch.mm_inputs = None
            model_input_ids = None  # run backbone on prebuilt embeddings

        hidden_states = self.model(model_input_ids, positions, forward_batch, input_embeds)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def post_load_weights(self, weight_names=None):

        # Perform post-processing after loading weights
        if weight_names is None:
            layer_ids = range(self.config.num_hidden_layers)
        else:
            layer_ids = set()
            for name in weight_names:
                if "kv_b_proj" in name:
                    layer_id = int(name.split(".")[2])
                    if layer_id < self.config.num_hidden_layers:
                        layer_ids.add(layer_id)

        for layer_id in layer_ids:
            for i in range(2):
                self_attn = self.model.layers[layer_id].self_attn[i]
                if hasattr(self_attn.kv_b_proj, "qweight"):
                    # AWQ compatible
                    if _is_cuda or _is_hip:
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                        ).T
                    else:
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                            0,
                            0,
                            0,
                        ).T
                else:
                    w = self_attn.kv_b_proj.weight
                use_deep_gemm_bmm = False

                if w.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    if (
                        hasattr(self.quant_config, "weight_block_size")
                        and self.quant_config.weight_block_size is not None
                    ):
                        weight_block_size = self.quant_config.weight_block_size
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        if _is_fp8_fnuz:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                input_scale=None,
                            )
                        else:
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv

                        if (
                            _is_cuda
                            and weight_block_size[0] == 128
                            and weight_block_size[1] == 128
                        ):
                            if (
                                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                                and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL
                                and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                            ):
                                block_scale = weight_scale
                                use_deep_gemm_bmm = True
                            else:
                                w = block_quant_dequant(
                                    weight,
                                    weight_scale,
                                    weight_block_size,
                                    torch.bfloat16,
                                )
                        else:
                            w, scale = block_quant_to_tensor_quant(
                                weight, weight_scale, weight_block_size
                            )
                            self_attn.w_scale = scale
                    else:
                        if _is_fp8_fnuz:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale,
                                input_scale=None,
                            )
                        else:
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale

                        w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                        self_attn.w_scale = scale

                if w.dtype == torch.int8:
                    if hasattr(self.quant_config, "weight_block_size"):
                        # block-wise int8 need it
                        weight_block_size = self.quant_config.weight_block_size
                        if weight_block_size is not None:
                            assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv
                            w = int8_block_dequant(
                                weight, weight_scale, weight_block_size
                            ).to(torch.bfloat16)
                    else:
                        # channel-wise int8 need it
                        w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                            torch.bfloat16
                        )

                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                if not use_deep_gemm_bmm:
                    self_attn.w_kc = bind_or_assign(
                        self_attn.w_kc,
                        w_kc.transpose(1, 2).contiguous().transpose(1, 2),
                    )
                    self_attn.w_vc = bind_or_assign(
                        self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
                    )
                    if (
                        hasattr(self_attn.kv_b_proj, "weight_scale")
                        and self_attn.w_scale is None
                    ):
                        self_attn.w_scale = bind_or_assign(
                            self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                        )
                        if _is_hip:
                            self_attn.w_scale *= 2.0
                else:
                    num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
                    num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                    ws_kc, ws_vc = block_scale.unflatten(
                        0, (-1, (num_tiles_k + num_tiles_n))
                    ).split([num_tiles_k, num_tiles_n], dim=1)
                    self_attn.w_scale_k = bind_or_assign(
                        self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                    )
                    self_attn.w_scale_v = bind_or_assign(
                        self_attn.w_scale_v, ws_vc.contiguous()
                    )
                    self_attn.w_kc = bind_or_assign(
                        self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                    )
                    self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())
                    self_attn.use_deep_gemm_bmm = True

                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank
                    ) ** 0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.kv_lora_rank
                    ) ** 0.5

        # TODO(linguoyuan) EPMoE not support DEEPGEMM_BLACKWELL, DeepEP needs to be supported in the future
        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 = False

        if should_deepgemm_weight_requant_ue8m0(
            weight_block_size=getattr(self.quant_config, "weight_block_size", None)
        ):
            self._weight_requant_ue8m0()

    def _weight_requant_ue8m0(self):
        weight_block_size = self.quant_config.weight_block_size

        for layer_id in range(self.config.num_hidden_layers):
            layer = self.model.layers[layer_id]
            for i in range(2):
                self_attn = layer.self_attn[i]
                module_list = [
                    self_attn.kv_b_proj,
                    self_attn.o_proj,
                ]

                if self.config.q_lora_rank is not None:
                    module_list.append(self_attn.fused_qkv_a_proj_with_mqa)
                    module_list.append(self_attn.q_b_proj)
                else:
                    module_list.append(self_attn.kv_a_proj_with_mqa)
                    module_list.append(self_attn.q_proj)

                for module in module_list:
                    if hasattr(module, "weight_scale_inv"):
                        requant_weight_ue8m0_inplace(
                            module.weight, module.weight_scale_inv, weight_block_size
                        )

                mlp = layer.mlps[i]
                assert isinstance(mlp, LongcatFlashMLP)
                for module in [
                    mlp.gate_up_proj,
                    mlp.down_proj,
                ]:
                    if hasattr(module, "weight_scale_inv"):
                        requant_weight_ue8m0_inplace(
                            module.weight, module.weight_scale_inv, weight_block_size
                        )

        for layer_id in range(self.config.num_hidden_layers):
            experts = layer.mlp.experts
            if isinstance(experts, DeepEPMoE):
                for w in [
                    (experts.w13_weight, experts.w13_weight_scale_inv),
                    (experts.w2_weight, experts.w2_weight_scale_inv),
                ]:
                    requant_weight_ue8m0_inplace(w[0], w[1], weight_block_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            weight_names = []
            for name, loaded_weight in weights:
                use_async_loading = should_async_load(loaded_weight)
                # visual+audio-understanding overlay: load visual_tokenizer AND
                # audio_tokenizer (encoders + VQ codebooks + bridges); still skip the
                # two generation heads (image-OUT / audio-OUT not supported here)
                if any(p in name for p in ("visual_head", "audio_head")):
                    continue
                if "mtp" in name:
                    continue
                if self.use_ngram_embedding:
                    if ".embed_tokens." in name:
                        name = "model.embed_tokens.word_embeder.weight"
                    if ".ngram_embeddings" in name:
                        self.model.embed_tokens.load_weight(None, name, loaded_weight)
                        continue
                weight_names.append(name)
                if "rotary_emb.inv_freq" in name:
                    continue
                # Visual/audio tokenizer weights (model.visual_tokenizer.* /
                # model.audio_tokenizer.*) are plain nn.Linear/codebooks — route
                # straight to the default loader so the gate_up_proj fusion below does
                # not rewrite their gate/up/down names (the audio bridge has its own
                # gate_proj/up_proj/down_proj that must NOT be fused).
                if "visual_tokenizer" in name or "audio_tokenizer" in name:
                    if name not in params_dict:
                        logger.warning(f"{name} not found in params_dict (mm tokenizer).")
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    maybe_executor_submit(
                        executor=executor,
                        futures=futures,
                        use_async=use_async_loading,
                        func=weight_loader,
                        func_args=(param, loaded_weight),
                    )
                    continue
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    maybe_executor_submit(
                        executor=executor,
                        futures=futures,
                        use_async=use_async_loading,
                        func=weight_loader,
                        func_args=(param, loaded_weight, shard_id),
                    )
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, name),
                            func_kwargs={
                                "shard_id": shard_id,
                                "expert_id": expert_id,
                            },
                        )
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = loaded_weight
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                                cat_dim = 0
                                if self.quant_config is not None and (
                                    self.quant_config.get_name() == "awq"
                                    or self.quant_config.get_name() == "awq_marlin"
                                    or self.quant_config.get_name() in ("moe_wna16", "gptq", "gptq_marlin", "auto-round", "auto_round")
                                ):
                                    cat_dim = 1
                                fused_weight = torch.cat(
                                    [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                )
                                param_name = (
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                param = params_dict[param_name]

                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, fused_weight),
                                )
                                cached_a_proj.pop(q_a_proj_name)
                                cached_a_proj.pop(kv_a_proj_name)
                        else:
                            if (
                                "k_scale" in name or "v_scale" in name
                            ) and name not in params_dict:
                                # modelopt attn kv scale is named differently
                                for scale in ["k_scale", "v_scale"]:
                                    if scale in name:
                                        name = name.replace(
                                            f"{scale[0]}_proj", "attn_mqa"
                                        )
                                        break
                            if name not in params_dict:
                                # modelopt ckpt contains not needed weights for MTP module:
                                # model.decoder.self_attn.attn_mqa.v_scale and
                                # model.decoder.self_attn.attn_mqa.k_scale
                                logger.warning(f"{name} not found in params_dict.")
                                continue
                            param = params_dict[name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(param, loaded_weight),
                            )

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.post_load_weights(weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = [LongcatFlashForCausalLM]
