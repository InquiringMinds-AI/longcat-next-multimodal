#!/usr/bin/env python3
"""Cross-request head batching: a row's logits must not depend on its batch-mates.

The optimization replaces N batch-1 calls to CasualDepthTransformerHead with ONE
call of batch N (one per concurrent image, per level). That is only sound if the
head treats rows independently -- no cross-row mixing anywhere in the depth
transformer, the norms, or the output projection.

This cannot be checked by comparing generated images: batching changes the order
torch.multinomial consumes the RNG, so the batched run produces DIFFERENT (equally
valid) samples. The invariant that actually has to hold is upstream of sampling --
the LOGITS -- so that is what this asserts.

A tolerance is required, not equality: batched matmuls use different reduction
orders/kernels than batch-1, so rows agree to floating-point noise, not bitwise.
The threshold is set well below the scale at which top-k/top-p selection could
flip.

    python3 test/test_head_batching.py
"""
import os
import sys

import torch

# Runs both from the dev repo (new_files/models/) and inside the image, where the
# overlay is installed into sglang. Import the SAME file that ships, when present.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "new_files", "models"))
try:
    from sglang.srt.models.longcat_next_heads import CasualDepthTransformerHead  # noqa: E402
except ImportError:
    from longcat_next_heads import CasualDepthTransformerHead  # noqa: E402

HIDDEN = 128
# 8 levels, matching the checkpoint's depth; transformer_dim must be a multiple of
# 128 (asserted by CasualDepthTransformerLayer), so this is the smallest real shape.
CODEBOOKS = [64] * 8
TDIM = 128
LAYERS = 2
FFN_SCALE = 2
N = 4                                  # concurrent "requests"
TOL = 2e-3                             # >> fp noise, << top-k separation


def build():
    torch.manual_seed(0)
    head = CasualDepthTransformerHead(
        hidden_size=HIDDEN,
        codebook_sizes=CODEBOOKS,
        transformer_layer_num=LAYERS,
        transformer_dim=TDIM,
        transformer_ffn_scale=FFN_SCALE,
    ).eval()
    # A stand-in for the model's _codebook_embed_fn: any callable id -> embedding.
    table = torch.randn(max(CODEBOOKS) + 1, HIDDEN)

    def emb(ids):
        return table[ids.clamp(min=0, max=table.shape[0] - 1)]

    return head, emb


@torch.no_grad()
def main():
    head, emb = build()
    torch.manual_seed(1)
    x = torch.randn(N, HIDDEN)
    # Distinct token history per row -- if rows leaked into each other, identical
    # histories would hide it.
    tokens = torch.randint(0, min(CODEBOOKS), (N, len(CODEBOOKS)))

    failed = 0
    for level in range(len(CODEBOOKS)):
        batched = head(x, tokens, emb, level)                       # [N, vocab]
        for i in range(N):
            solo = head(x[i:i + 1], tokens[i:i + 1], emb, level)    # [1, vocab]
            delta = (batched[i] - solo[0]).abs().max().item()
            ok = delta < TOL
            # The selection that actually matters downstream: same argmax.
            same_argmax = batched[i].argmax().item() == solo[0].argmax().item()
            ok = ok and same_argmax
            print(f"[{'PASS' if ok else 'FAIL'}] level {level} row {i}: "
                  f"max|delta|={delta:.2e} argmax_match={same_argmax}")
            if not ok:
                failed += 1

    # A row must also be unaffected by WHO its batch-mates are: same row, different
    # neighbours, same logits. This is the property that makes it safe for requests
    # to join and leave the batch mid-image.
    print("\n=== batch-composition independence ===")
    for level in range(len(CODEBOOKS)):
        full = head(x, tokens, emb, level)
        pair = head(x[:2], tokens[:2], emb, level)
        delta = (full[0] - pair[0]).abs().max().item()
        ok = delta < TOL
        print(f"[{'PASS' if ok else 'FAIL'}] level {level}: row 0 in batch-{N} vs "
              f"batch-2, max|delta|={delta:.2e}")
        if not ok:
            failed += 1

    total = len(CODEBOOKS) * N + len(CODEBOOKS)
    print(f"\n=== {total - failed}/{total} passed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
