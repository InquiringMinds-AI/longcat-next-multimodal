"""Utilities for updating LongCat ngram embedding token tables.

OVERLAY (longcat-next-multimodal): multimodal special tokens are written into the
table as ZERO, matching the original NgramCache semantics. The n-gram hash base is
the TEXT vocab (131072); ids >= it were never hashed during training — the original
implementation stores 0 at every special/codebook position (zero contribution, hash
window continues). Writing them raw corrupts the hash context of the up-to-3 tokens
that follow any special token — measured as the TTS first-syllable onset defect.
NOTE: the kernel's own `ignore_tokens` mechanism is NOT equivalent (it BREAKS the
hash window at ignored entries instead of zeroing through them); hence this
zero-at-write approach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

# Text-vocab hash base; ids >= this are stored as 0 (see module docstring).
NGRAM_HASH_TEXT_VOCAB = int(os.environ.get("LCN_NGRAM_HASH_VOCAB", "131072"))

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ForwardMode
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class NgramEmbeddingManager:
    enabled: bool
    table: Optional[torch.Tensor]
    n: int
    k: int
    # LongCat-Next overlay: post-sample gen-trigger scan hook (the model's
    # lcn_trigger_scan). Runs here because update_after_decode is the one
    # per-step host-side point that sees every sampled batch OUTSIDE any CUDA
    # graph capture/replay. None for models without the hook.
    trigger_scan: Optional[object] = None

    @classmethod
    def from_model(
        cls,
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        req_to_token_pool: ReqToTokenPool,
        server_args: ServerArgs,
        max_running_requests: int,
        device: str,
    ):
        token_table = None
        ngram_embedding_n = 0
        ngram_embedding_k = 0
        use_ngram_embedding = model_config.use_ngram_embedding
        if use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            token_table = torch.empty(
                req_to_token_pool.req_to_token.shape[0],
                model_config.context_len,
                dtype=torch.int32,
                device=device,
            )
            chunked_prefill_size = server_args.chunked_prefill_size
            assert (
                chunked_prefill_size is not None and chunked_prefill_size > 0
            ), "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            for module in model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        max_running_requests, chunked_prefill_size, device
                    )
            hf_config = model_config.hf_config
            ngram_embedding_n = hf_config.ngram_embedding_n
            ngram_embedding_k = hf_config.ngram_embedding_k
        return cls(
            enabled=use_ngram_embedding,
            table=token_table,
            n=ngram_embedding_n,
            k=ngram_embedding_k,
            trigger_scan=getattr(model, "lcn_trigger_scan", None),
        )

    def update_after_decode(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Update the ngram embedding token table after sampling."""
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        if self.trigger_scan is not None:
            # Must see the RAW sampled ids — gen-entry triggers are >= the
            # text vocab and are zeroed for the hash table below.
            self.trigger_scan(next_token_ids)
        update_ngram_token_table_after_sampling(
            ngram_embedding_info=ngram_embedding_info,
            next_token_ids=next_token_ids,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            batch_size=forward_batch.batch_size,
        )

    def prepare_for_forward(
        self,
        batch: Optional[ScheduleBatch],
        *,
        chunked_req: Optional[Req],
    ) -> Optional[ScheduleBatch]:
        """Fill the token table for ngram embedding before a forward pass."""
        if batch is None or not self.enabled:
            return batch
        batch.ne_token_table = self.table
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_range.length
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    tokens = fill_ids[start - self.n + 1 : end]
                    column_starts.append(start - self.n + 1)
                # Specials/codebook ids -> 0 in the hash table (original semantics)
                all_tokens.extend(
                    t if t < NGRAM_HASH_TEXT_VOCAB else 0 for t in tokens
                )
                request_lengths.append(len(tokens))
            dtype = self.table.dtype
            device = self.table.device
            update_token_table(
                ne_token_table=self.table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
            # Mark the chunked (not-yet-finished) prefill request so sample()
            # skips writing its pseudo next-token into the ngram token table.
            # Use self.chunked_req identity (not req.is_chunked) to avoid
            # overlap-scheduling timing issues.
            if chunked_req is not None:
                skip_token_table_update = [req is chunked_req for req in batch.reqs]
                batch.ne_skip_token_table_update = (
                    torch.tensor(
                        skip_token_table_update, dtype=torch.bool, device=device
                    )
                    if any(skip_token_table_update)
                    else None
                )
        return batch


def lcn_write_spec_tokens(
    manager: NgramEmbeddingManager,
    *,
    tokens: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    scan_triggers: bool = False,
) -> None:
    """Write speculative-decoding tokens into the ngram hash table.

    Called from the NGRAM spec worker patch (patches/ngram_spec_verify.patch)
    at two points: (a) draft tokens BEFORE the verify forward — the hash
    kernel reads context exclusively from the table, so draft position i needs
    drafts 0..i-1 present (exact for CHAIN drafts; tree branching >1 is not
    supported — the entrypoint pins bfs breadth to 1); (b) accepted tokens
    AFTER verification — the spec path bypasses model_runner.sample(), so the
    normal update_after_decode table write never runs for accepted tokens.
    Rejected drafts left beyond the accepted region are harmless: the kernel
    only ever reads backwards from a live position, and later writes overwrite
    them. Applies the same specials-to-zero hashing rule as every other write.

    scan_triggers: run the model's gen-entry trigger scan on these ids. Set ONLY
    at the accepted-tokens call site (b). model_runner.sample() — the one place
    that normally arms the latch — is bypassed on the spec path, so without this
    a generation entered by a token accepted during a verify round is never
    noticed and the plain-decode fallback never engages. Must NOT be set for
    drafts (a): a rejected draft would arm generation that never happened.
    """
    if not manager.enabled or tokens.numel() == 0:
        return
    if scan_triggers and manager.trigger_scan is not None:
        # RAW ids, before the specials-to-zero rewrite below zeroes the
        # gen-entry triggers (they sit above the text vocab).
        manager.trigger_scan(tokens)
    toks = torch.where(
        tokens < NGRAM_HASH_TEXT_VOCAB, tokens, torch.zeros_like(tokens)
    ).to(torch.int32)
    update_token_table(
        ne_token_table=manager.table,
        tokens=toks,
        row_indices=row_indices,
        column_starts=column_starts.to(torch.int32),
        req_lens=req_lens.to(torch.int32),
        ignore_tokens=None,
    )


def update_ngram_token_table_after_sampling(
    *,
    ngram_embedding_info,
    next_token_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
) -> bool:
    """Update the ngram token table with sampled tokens.

    Returns whether the token table was updated.
    """
    # Specials/codebook ids -> 0 in the hash table (original semantics; see module
    # docstring). Sampled stream tokens during multimodal generation are pads/
    # markers >= the text vocab, which must hash as zero.
    next_token_ids = torch.where(
        next_token_ids < NGRAM_HASH_TEXT_VOCAB,
        next_token_ids,
        torch.zeros_like(next_token_ids),
    )
    skip_token_table_update = ngram_embedding_info.skip_token_table_update
    if skip_token_table_update is not None:
        # Skip chunked (not-yet-finished) prefill requests: their sampled token
        # is a pseudo prediction and must not pollute the token table.
        indices = (~skip_token_table_update).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return False
        update_token_table(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids[indices].to(torch.int32),
            row_indices=req_pool_indices[indices],
            column_starts=seq_lens[indices].to(torch.int32),
            req_lens=torch.ones(
                indices.numel(), dtype=torch.int32, device=next_token_ids.device
            ),
            ignore_tokens=None,
        )
        return True

    ngram_embedding_info.out_column_starts[:batch_size] = seq_lens
    ngram_embedding_info.out_req_lens[:batch_size] = 1
    update_token_table(
        ne_token_table=ngram_embedding_info.token_table,
        tokens=next_token_ids.to(torch.int32),
        row_indices=req_pool_indices,
        column_starts=ngram_embedding_info.out_column_starts,
        req_lens=ngram_embedding_info.out_req_lens,
        ignore_tokens=None,
    )
    return True
