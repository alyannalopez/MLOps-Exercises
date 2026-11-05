"""Benchmark event timeline (JSONL).

Every turn emits one JSON line with the T0..T8 timestamps defined in
benchmark_variables_spec.md §0. The Step 9 scorer reads these lines and
computes the Family A/B latency variables (A1=L_vad, A2=L_asr, A4=TTFA, ...).

Timestamps are wall-clock seconds from a single time.monotonic() origin, so
differences are exact. A turn is a dict; `complete()` writes it to the log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Timeline:
    """One conversation turn, stamped as events fire."""

    turn_id: int
    t_origin: float  # monotonic() at turn creation (T0 anchor)

    # T0..T8 (None until the event fires)
    T0_user_speech_start: Optional[float] = None
    T1_user_speech_end: Optional[float] = None
    T2_asr_final: Optional[float] = None
    T3_route_decision: Optional[float] = None
    T4_agent_speech_start: Optional[float] = None
    T5_agent_speech_end: Optional[float] = None
    T6_harness_result_ready: Optional[float] = None
    T7_result_spoken_start: Optional[float] = None
    T8_turn_complete: Optional[float] = None

    # payload
    transcript: str = ""
    route: str = ""
    reply: str = ""
    error: Optional[str] = None

    # overlap events (Step 6 fills these; half-duplex leaves them None)
    O0_overlap_onset: Optional[float] = None
    O1_overlap_classified: Optional[float] = None
    O2_agent_reaction: Optional[float] = None

    def stamp(self, name: str) -> float:
        """Record `now` under the given attribute. Returns the timestamp."""
        now = time.monotonic() - self.t_origin
        setattr(self, name, now)
        return now

    def as_dict(self) -> dict:
        d = {
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "route": self.route,
            "reply": self.reply,
        }
        for k in (
            "T0_user_speech_start", "T1_user_speech_end", "T2_asr_final",
            "T3_route_decision", "T4_agent_speech_start", "T5_agent_speech_end",
            "T6_harness_result_ready", "T7_result_spoken_start", "T8_turn_complete",
            "O0_overlap_onset", "O1_overlap_classified", "O2_agent_reaction",
        ):
            d[k] = getattr(self, k)
        if self.error:
            d["error"] = self.error
        return d

    # convenience derived latencies (ms) — mirrors spec Family A
    def latencies_ms(self) -> dict:
        def _diff(a, b):
            return round((b - a) * 1000.0, 1) if (a is not None and b is not None) else None
        return {
            "A1_L_vad": None,  # VAD-internal; not separable in half-duplex loop
            "A2_L_asr": _diff(self.T1_user_speech_end, self.T2_asr_final),
            "A3_L_route": _diff(self.T2_asr_final, self.T3_route_decision),
            "A4_TTFA": _diff(self.T1_user_speech_end, self.T4_agent_speech_start),
            "A5_L_e2e": _diff(self.T1_user_speech_end, self.T4_agent_speech_start),
            "A6_L_turn": _diff(self.T1_user_speech_end, self.T8_turn_complete),
        }


class TimelineLog:
    """Append-only JSONL writer for completed turns."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def record(self, tl: Timeline) -> None:
        line = json.dumps(tl.as_dict(), ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
