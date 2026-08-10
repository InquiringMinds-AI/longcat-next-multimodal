"""Cross-layer probe for "is a generation state machine active right now?".

The scheduler needs this answer to decide whether a decode batch may take the
speculative path, but `managers/schedule_batch.py` cannot import the model
(circular: models import schedule_batch). So the model registers a callable
here at construction and the scheduler asks through this module.

Deliberately a process-global, matching the granularity of the thing it
reports: image/audio generation state lives on the model instance, is global
to it, and already drives an all-or-nothing CUDA-graph veto. One TP rank per
process, one model per rank, so a module global is the same scope as the
model attribute it forwards to.

Fails CLOSED (returns False -> speculation proceeds as upstream): if no probe
is registered, or the model raised, this must not change behavior for the
text-only/agent deployments that never enter a generation state.
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_GEN_PROBE: Optional[Callable[[], bool]] = None
_FALLBACK_COUNT = 0


def register_gen_probe(probe: Callable[[], bool]) -> None:
    """Called by the model at construction with its gen-watch predicate."""
    global _GEN_PROBE
    _GEN_PROBE = probe


def gen_active() -> bool:
    """True when a generation state machine needs this forward to run the
    model's Python path (plain decode, eager)."""
    probe = _GEN_PROBE
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        # A probe failure must not take the server down; the worst case of
        # answering False is the pre-existing upstream behavior.
        return False


def note_fallback() -> int:
    """Record one spec->plain decode fallback step. Logged sparsely: the first
    engagement (proof the path is live, and the only line most operators will
    ever want) and every 1000th after, so a stuck latch is visible as an
    ever-climbing count without flooding a generation-heavy log."""
    global _FALLBACK_COUNT
    _FALLBACK_COUNT += 1
    if _FALLBACK_COUNT == 1 or _FALLBACK_COUNT % 1000 == 0:
        logger.info(
            "[lcn] spec->plain decode fallback engaged for generation "
            "(steps=%d)",
            _FALLBACK_COUNT,
        )
    return _FALLBACK_COUNT


def fallback_count() -> int:
    return _FALLBACK_COUNT
