"""Incremental streaming with tool-call detection (shared by both API routes).

The model emits tool calls as text markers inside the completion. To stream
tokens live WITHOUT ever leaking a partial marker to the client, ToolStreamFilter
passes text through while withholding a rolling tail just long enough to contain
any marker prefix; the moment a full marker appears in the accumulated text, the
filter goes silent and buffers the remainder for end-of-stream parsing
(longcat_tools.parse_tool_calls handles all three emission dialects — the
TS-style dialect nests inside <longcat_tool_call>, so two top-level markers
suffice).

False-positive cost is graceful: if the tail of a completion merely LOOKS like a
marker start, those characters arrive with the final flush instead of the last
delta — never lost, at most late.
"""

MARKERS = ("<longcat_tool_call>", "<function_calls>")
_HOLD = max(len(m) for m in MARKERS) - 1  # longest prefix that could still become a marker


class ToolStreamFilter:
    """Feed text deltas; get back text that is safe to emit immediately."""

    def __init__(self):
        self._pending = ""     # withheld tail (could be a marker prefix)
        self._raw = []         # everything ever fed (for end-of-stream parsing)
        self._silent = False   # a marker was seen: buffer everything from it on

    def feed(self, delta: str) -> str:
        self._raw.append(delta)
        if self._silent or not delta:
            return ""
        buf = self._pending + delta
        for m in MARKERS:
            i = buf.find(m)
            if i != -1:
                self._silent = True
                self._pending = ""
                return buf[:i]
        # Withhold the longest tail that is a prefix of some marker (bounded by _HOLD)
        keep = 0
        for k in range(min(_HOLD, len(buf)), 0, -1):
            tail = buf[-k:]
            if any(m.startswith(tail) for m in MARKERS):
                keep = k
                break
        self._pending = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep] if keep else buf

    def finish(self):
        """End of stream -> (leftover_text_to_emit, full_raw_text).

        leftover is the withheld tail when NO marker ever appeared ("" after a
        marker: the raw text goes to the parser instead).
        """
        leftover = "" if self._silent else self._pending
        self._pending = ""
        return leftover, "".join(self._raw)

    @property
    def saw_marker(self) -> bool:
        return self._silent
