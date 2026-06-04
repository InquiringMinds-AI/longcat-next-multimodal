import torch
from torch import nn
from torch.nn import Parameter

from sglang.jit_kernel.ngram_embedding import compute_n_gram_ids
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class NgramEmbedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        over_embedding_m: int,
        over_embedding_k: int,
        over_embedding_n: int,
        word_vocab_size: int = None,
    ):
        super().__init__()
        assert (
            over_embedding_n > 1
        ), f"over_embedding_n must be > 1, got {over_embedding_n}"
        self.num_embeddings = num_embeddings  # text vocab for hash function
        self.embedding_dim = embedding_dim
        self.over_embedding_m = over_embedding_m
        self.over_embedding_k = over_embedding_k
        self.over_embedding_n = over_embedding_n

        # word_vocab_size can be larger than num_embeddings (includes multimodal special tokens)
        actual_vocab = word_vocab_size if word_vocab_size is not None else num_embeddings
        # Tokens in range [num_embeddings, actual_vocab) are multimodal special tokens.
        # The original model doesn't divide these by (1+k*(n-1)) in N-gram output.
        if actual_vocab > num_embeddings:
            self._oe_ignored_range = (num_embeddings, actual_vocab)
        else:
            self._oe_ignored_range = None
        self.word_embeder = VocabParallelEmbedding(
            actual_vocab,
            embedding_dim,
            enable_tp=is_dp_attention_enabled(),
        )
        self.n_grams = (over_embedding_n - 1) * over_embedding_k
        oe_hidden_dim = embedding_dim // (over_embedding_k * (over_embedding_n - 1))
        self.exclusive_oe_embedder_size_sums = torch.zeros(
            [over_embedding_k * (over_embedding_n - 1) + 1],
            dtype=torch.int32,
            device="cuda",
        )
        for i in range(over_embedding_k * (over_embedding_n - 1)):
            self.exclusive_oe_embedder_size_sums[i + 1] = (
                self.exclusive_oe_embedder_size_sums[i]
                + int(over_embedding_m + i * 2 + 1)
            )
        # Allocate minimal placeholder — actual storage is int8 (allocated on first load)
        # This avoids allocating 58 GB of BF16 that would OOM on GB10
        self.oe_embeder = VocabParallelEmbedding(
            num_embeddings=1,  # placeholder, replaced by int8 storage during load
            embedding_dim=oe_hidden_dim,
            enable_tp=is_dp_attention_enabled(),
        )
        self._oe_total_rows = int(self.exclusive_oe_embedder_size_sums[-1].item())
        self._oe_hidden_dim = oe_hidden_dim
        # Int8 quantization support for memory reduction
        self.oe_int8_weight = None
        self.oe_int8_scale = None

        self.oe_projection = nn.Parameter(
            torch.empty(
                (over_embedding_n - 1) * over_embedding_k, oe_hidden_dim, embedding_dim
            ),
            requires_grad=False,
        )

        self.oe_mods = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k], dtype=torch.int32
        )
        self.oe_weights = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k, self.over_embedding_n],
            dtype=torch.int32,
        )
        for n in range(2, self.over_embedding_n + 1):
            for k in range(self.over_embedding_k):
                mod = (
                    self.over_embedding_m
                    + 2 * ((n - 2) * self.over_embedding_k + k)
                    + 1
                )
                self.oe_mods[n - 2][k] = mod
                for delta in range(self.over_embedding_n):
                    self.oe_weights[n - 2][k][delta] = pow(num_embeddings, delta, mod)

    def init_buffers(
        self, max_running_requests: int, chunked_prefill_size: int, device: str
    ):
        max_tokens = max(chunked_prefill_size, max_running_requests)
        self.oe_n_gram_ids = torch.zeros(
            [max_tokens, self.n_grams],
            dtype=torch.int32,
            device=device,
        )
        self.exclusive_req_len_sums = torch.zeros(
            max_running_requests + 1, dtype=torch.int32, device=device
        )

    def _ensure_int8_storage(self):
        """Lazily allocate int8 storage on first int8 weight load."""
        if self.oe_int8_weight is None:
            total_rows = self._oe_total_rows
            oe_hidden_dim = self._oe_hidden_dim
            device = self.oe_projection.device
            self.oe_int8_weight = nn.Parameter(
                torch.zeros(total_rows, oe_hidden_dim, dtype=torch.int8, device=device),
                requires_grad=False,
            )
            self.oe_int8_scale = nn.Parameter(
                torch.ones(total_rows, 1, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
            # Free the BF16 embedding since we'll use int8
            self.oe_embeder.weight = nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )

    def load_weight(
        self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor
    ):
        if ".embed_tokens." in weight_name:
            param.weight_loader(param, loaded_weight)
        elif "model.ngram_embeddings.embedders." in weight_name and ".weight_scale" in weight_name:
            # Int8 scale tensor for N-gram embedding
            self._ensure_int8_storage()
            index = int(
                weight_name.replace("model.ngram_embeddings.embedders.", "").replace(
                    ".weight_scale", ""
                )
            )
            oe_weight_start = int(self.exclusive_oe_embedder_size_sums[index].item())
            oe_weight_end = int(self.exclusive_oe_embedder_size_sums[index + 1].item())
            self.oe_int8_scale.data[oe_weight_start:oe_weight_end, 0] = loaded_weight.to(self.oe_int8_scale.dtype)
        elif "model.ngram_embeddings.embedders." in weight_name:
            index = int(
                weight_name.replace("model.ngram_embeddings.embedders.", "").replace(
                    ".weight", ""
                )
            )
            oe_weight_start = int(self.exclusive_oe_embedder_size_sums[index].item())
            oe_weight_end = int(self.exclusive_oe_embedder_size_sums[index + 1].item())
            expected_rows = oe_weight_end - oe_weight_start
            assert (
                expected_rows == loaded_weight.shape[0]
            ), f"{expected_rows=} {loaded_weight.shape[0]=}"

            if loaded_weight.dtype == torch.int8:
                # Int8 pre-quantized N-gram embedding
                self._ensure_int8_storage()
                self.oe_int8_weight.data[oe_weight_start:oe_weight_end] = loaded_weight
            else:
                # BF16 N-gram embedding (legacy path)
                tp_start = self.oe_embeder.shard_indices.org_vocab_start_index
                tp_end = self.oe_embeder.shard_indices.org_vocab_end_index
                to_load_start = max(oe_weight_start, tp_start)
                to_load_end = min(oe_weight_end, tp_end)
                if to_load_start < to_load_end:
                    src_start = to_load_start - oe_weight_start
                    src_end = to_load_end - oe_weight_start
                    dest_start = to_load_start - tp_start
                    dest_end = to_load_end - tp_start
                    self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[
                        src_start:src_end
                    ]
                else:
                    return
        elif "model.ngram_embeddings.post_projs." in weight_name:
            index = int(
                weight_name.replace("model.ngram_embeddings.post_projs.", "").replace(
                    ".weight", ""
                )
            )
            self.oe_projection[index].copy_(loaded_weight.data.t())
        else:
            assert False, f"Unknown ngram embedding weight name: {weight_name}"

    def quantize_embeddings_int8(self):
        """Quantize oe_embeder weights to int8 with per-row scaling. Halves memory."""
        weight = self.oe_embeder.weight.data.float()
        row_max = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = row_max / 127.0
        int8_weight = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        self.oe_int8_weight = nn.Parameter(int8_weight, requires_grad=False)
        self.oe_int8_scale = nn.Parameter(scale.to(torch.bfloat16), requires_grad=False)
        # Free the BF16 weight
        self.oe_embeder.weight = nn.Parameter(
            torch.empty(0, dtype=torch.bfloat16, device=weight.device),
            requires_grad=False,
        )
        import gc; gc.collect()
        if weight.is_cuda:
            torch.cuda.empty_cache()

    def _oe_embed(self, indices):
        """Embedding lookup — uses int8 if quantized, else BF16."""
        if self.oe_int8_weight is not None:
            # Int8 dequant on the fly
            emb = self.oe_int8_weight[indices].to(torch.bfloat16)
            scale = self.oe_int8_scale[indices]
            return emb * scale
        return self.oe_embeder(indices)

    _diag_count = 0

    def forward(self, input_ids: torch.Tensor, forward_batch: ForwardBatch):
        NgramEmbedding._diag_count += 1
        _do_diag = NgramEmbedding._diag_count <= 10

        if (
            forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_decode()
        ):
            ngram_embedding_info = forward_batch.ngram_embedding_info
            torch.cumsum(
                ngram_embedding_info.req_lens,
                dim=0,
                dtype=torch.int32,
                out=self.exclusive_req_len_sums[1 : 1 + forward_batch.batch_size],
            )
            compute_n_gram_ids(
                ne_n=self.over_embedding_n,
                ne_k=self.over_embedding_k,
                ne_weights=self.oe_weights,
                ne_mods=self.oe_mods,
                tokens=input_ids.to(torch.int32),
                exclusive_ne_embedder_size_sums=self.exclusive_oe_embedder_size_sums,
                exclusive_req_len_sums=self.exclusive_req_len_sums[
                    : forward_batch.batch_size + 1
                ],
                ne_token_table=ngram_embedding_info.token_table,
                row_indices=forward_batch.req_pool_indices,
                column_starts=ngram_embedding_info.column_starts,
                n_gram_ids=self.oe_n_gram_ids[: len(input_ids)],
            )

        # [13, seq_len, hidden_dim]
        all_hidden_states = torch.empty(
            [self.n_grams + 1, len(input_ids), self.embedding_dim],
            dtype=self.oe_projection.dtype,
            device=input_ids.device,
        )
        all_hidden_states[0] = self.word_embeder(input_ids)
        # oe_hidden_states: [12, seq_len, hidden_dim / 12]
        ngram_ids = self.oe_n_gram_ids[: len(input_ids)]
        oe_hidden_states = self._oe_embed(
            ngram_ids.permute(1, 0).contiguous()
        )
        torch.bmm(oe_hidden_states, self.oe_projection, out=all_hidden_states[1:])

        if _do_diag:
            import logging
            logger = logging.getLogger("ngram_diag")
            word_norm = all_hidden_states[0].float().norm().item()
            ngram_norm = all_hidden_states[1:].float().norm().item()
            total_norm = all_hidden_states.float().norm().item()
            ngram_ids_nonzero = (ngram_ids != 0).sum().item()
            ngram_ids_total = ngram_ids.numel()
            oe_nonzero = (oe_hidden_states != 0).any(dim=-1).sum().item()
            logger.warning(
                f"[NGRAM] mode={forward_batch.forward_mode} ids={input_ids.shape} "
                f"word_norm={word_norm:.2f} ngram_norm={ngram_norm:.2f} "
                f"ngram_ids_nonzero={ngram_ids_nonzero}/{ngram_ids_total} "
                f"oe_nonzero={oe_nonzero}/{oe_hidden_states.shape[0]*oe_hidden_states.shape[1]}"
            )

        result = all_hidden_states.mean(dim=0)

        # The original model's oe_ignored_token_ids handling: tokens in
        # [text_vocab_size, text_vocab_plus) get the raw sum (not /13).
        # Their hash context is zeroed so N-gram contributions are ~0.
        # Giving them just the word embedding (= sum / 13 * 13 ≈ word_embed)
        # is the cleanest match. Use word_embed directly for these positions.
        if hasattr(self, '_oe_ignored_range') and self._oe_ignored_range is not None:
            lo, hi = self._oe_ignored_range
            ignored_mask = (input_ids >= lo) & (input_ids < hi)
            if ignored_mask.any():
                result[ignored_mask] = all_hidden_states[0, ignored_mask]

        return result
