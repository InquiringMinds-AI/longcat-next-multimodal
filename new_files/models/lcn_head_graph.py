"""CUDA-graph replay for the depth-head forward (LCN_HEAD_GRAPH=1).

Why: the generation decode step is launch-latency-bound — the 2026-08-14
timeline traces show ~4.4k kernel launches/step carrying ~72ms of GPU work
against ~42ms of DISTRIBUTED idle (97% of it in gaps under 500us, invisible
to aggregate profiling). ~3.6k of those launches live inside the 8 per-step
depth-head calls, and the head forward is shape-static per (batch, level):
after the seqlens hoist + dense-SDPA fast path (bit-identical, see
longcat_next_heads.py) it captures cleanly. One replay replaces the whole
interior launch stream, and it frees the HOST too — in the serving loop the
CPU is shared with the scheduler/samplers, so each Python-dispatched launch
costs wall clock twice. The offline bench (research/int8_heads/
bench_head_graph.py) understates the win for exactly that reason: offline the
idle CPU pipelines launches; in the loop it cannot.

Mechanics:
- Graphs are captured lazily per (bsz, level) on first use, replayed after.
  Inputs are copied into static buffers (2 small D2D copies + 1 replay vs
  ~450 launches per level call).
- All graphs share one memory pool (torch.cuda.graph_pool_handle) so
  activation memory does not multiply by graph count.
- Before each capture the exact call runs on a side stream (Triton autotune +
  allocator warmup must happen OUTSIDE capture).
- Any capture failure marks the runner dead and falls back to eager forever
  (logged once) — the feature must never take generation down.
- emb_fn must be a stable callable (an embedding lookup) — data-dependent
  VALUES are fine under graphs, data-dependent SHAPES are not, and the head
  guarantees static shapes per (bsz, level).
"""
import logging

import torch

logger = logging.getLogger(__name__)


class GraphedHeadRunner:
    """Wraps one CasualDepthTransformerHead with per-(bsz, level) graph replay."""

    MAX_GRAPH_BS = 8  # batches beyond this are rare; eager is fine there

    def __init__(self, head, depth, name):
        self.head = head
        self.depth = depth
        self.name = name
        self.graphs = {}  # (bsz, level) -> (graph, sx, stoks, out)
        self.pool = torch.cuda.graph_pool_handle()
        self.dead = False

    def __call__(self, x, visual_tokens, emb_fn, level):
        bsz = x.shape[0]
        if self.dead or bsz > self.MAX_GRAPH_BS or torch.is_grad_enabled():
            return self.head(x, visual_tokens, emb_fn, level)
        key = (bsz, level)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(key, x, visual_tokens, emb_fn, level)
            if entry is None:  # capture failed -> permanent eager
                return self.head(x, visual_tokens, emb_fn, level)
        g, sx, stoks, out = entry
        sx.copy_(x)
        stoks.copy_(visual_tokens)
        g.replay()
        return out

    def _capture(self, key, x, visual_tokens, emb_fn, level):
        try:
            sx = x.clone()
            stoks = visual_tokens.clone()
            # Warm the exact call on a side stream: Triton autotune, cuBLAS
            # workspace, allocator blocks — none of it may happen mid-capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.head(sx, stoks, emb_fn, level)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self.head(sx, stoks, emb_fn, level)
            # Wiring check, once per key: replay and eager run IDENTICAL kernels
            # on identical inputs, so outputs must match exactly — a mismatch
            # means stale buffers or a wrong-graph bug, and we go eager.
            g.replay()
            ref = self.head(sx, stoks, emb_fn, level)
            if not torch.equal(out, ref):
                diff = (out.float() - ref.float()).abs().max().item()
                raise RuntimeError(f"replay/eager mismatch (max|d|={diff:.3e})")
            entry = (g, sx, stoks, out)
            self.graphs[key] = entry
            logger.info(f"[HeadGraph] {self.name}: captured (bsz={key[0]}, "
                        f"level={key[1]}) — {len(self.graphs)} graph(s)")
            return entry
        except Exception as e:  # noqa: BLE001 — capture must never break generation
            self.dead = True
            logger.warning(f"[HeadGraph] {self.name}: capture FAILED, permanent "
                           f"eager fallback — {type(e).__name__}: {str(e)[:200]}")
            return None
