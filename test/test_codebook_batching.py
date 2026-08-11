#!/usr/bin/env python3
"""_generate_image_codebook_step must give each BATCHED ROW its own sampled token.

This is the regression test for the bug that made two concurrent images come out
identical. The CFG fusion used to trigger on `logits.shape[0] == 2` -- inferring "these
rows are a cond/uncond guidance pair" from the batch merely being 2. Once cross-request
head batching put two REQUESTS in one call, request A became "cond", request B became
"uncond", the two were fused into a single row, and the resulting [1] sample was
broadcast into the [2] slot of next_token_ids. Both requests got A's tokens.

Note WHY test_head_batching.py could not catch it: that suite tests
CasualDepthTransformerHead, and the head is fine -- it was the CALLER that mixed the
rows. A green test on the component next to the bug is not evidence about the bug.

Sampling is made deterministic (top-k 1 over a stub head whose argmax is a function of
the row index) so this asserts an exact expected value rather than a statistical one.

    python3 test/test_codebook_batching.py
"""
import os
import sys

import torch

VOCAB = 32
LEVELS = 4
HIDDEN = 16


class StubHead:
    """Row k, level l -> logits whose argmax is (k + l) % VOCAB.

    Deliberately makes every row's answer DIFFERENT and PREDICTABLE, so "row 1 got row
    0's token" is an exact, visible failure rather than a suspicious coincidence.
    """

    def __call__(self, x, visual_tokens, embed_fn, level):
        bs = x.shape[0]
        logits = torch.zeros(bs, VOCAB)
        for k in range(bs):
            logits[k, (k + level) % VOCAB] = 100.0
        return logits


class StubModel:
    _visual_codebook_sizes = [VOCAB] * LEVELS
    visual_head = StubHead()
    visual_offset_vals = torch.zeros(LEVELS, dtype=torch.long)

    def _codebook_embed_fn(self, ids):
        return torch.zeros(*ids.shape, HIDDEN)


def main():
    try:
        import sglang.srt.models.longcat_next_mm as M
    except ImportError as e:
        print(f"SKIP: needs the sglang overlay in the image ({e})")
        return 0

    # Deterministic sampling: top-k 1 collapses softmax onto the argmax.
    M.IMAGE_GEN_TOP_K = 1
    M.IMAGE_GEN_TOP_P = 1.0
    M.IMAGE_GEN_TEMPERATURE = 1.0
    # Left at its real default ON PURPOSE. The bug was live precisely because cfg_scale
    # is 3.0 by default while CFG itself never runs, so a test that neutralised it here
    # would pass against the broken code.
    M.IMAGE_GEN_CFG_SCALE = 3.0

    step = M.LongcatNextForCausalLM._generate_image_codebook_step
    stub = StubModel()
    failed = 0

    # --- The regression: 2 concurrent REQUESTS, no CFG (uncond_hidden is None) ---
    rows = torch.randn(2, HIDDEN)
    out = step(stub, rows, None, return_all=True)

    ok = tuple(out.shape) == (2, LEVELS)
    print(f"[{'PASS' if ok else 'FAIL'}] returns one row per request: shape={tuple(out.shape)}")
    failed += not ok

    if ok:
        for k in range(2):
            want = [(k + l) % VOCAB for l in range(LEVELS)]
            got = out[k].tolist()
            good = got == want
            print(f"[{'PASS' if good else 'FAIL'}] row {k} sampled its OWN tokens: "
                  f"got={got} want={want}")
            failed += not good

        distinct = not torch.equal(out[0], out[1])
        print(f"[{'PASS' if distinct else 'FAIL'}] the two rows are not identical "
              f"(identical == the bug: every request gets row 0's tokens)")
        failed += not distinct

    # --- CFG semantics preserved: a real cond/uncond pair still fuses ---
    # Both rows SHOULD carry the same tokens here -- cond and uncond must be conditioned
    # identically at the next level. This is the case the shape test used to serve, and
    # it must keep working now that the gate is `uncond_hidden is not None`.
    cond, uncond = torch.randn(1, HIDDEN), torch.randn(1, HIDDEN)
    out_cfg = step(stub, cond, uncond, return_all=True)
    fused = tuple(out_cfg.shape) == (2, LEVELS) and torch.equal(out_cfg[0], out_cfg[1])
    print(f"[{'PASS' if fused else 'FAIL'}] CFG pair still fuses to a shared token per level: "
          f"shape={tuple(out_cfg.shape)}")
    failed += not fused

    # --- 3 concurrent requests: the old shape guard silently skipped this case ---
    rows3 = torch.randn(3, HIDDEN)
    out3 = step(stub, rows3, None, return_all=True)
    ok3 = tuple(out3.shape) == (3, LEVELS) and all(
        out3[k].tolist() == [(k + l) % VOCAB for l in range(LEVELS)] for k in range(3))
    print(f"[{'PASS' if ok3 else 'FAIL'}] 3 concurrent requests each keep their own tokens")
    failed += not ok3

    total = 7
    print(f"\n=== {total - failed}/{total} passed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
