"""Conversation loop — wires VAD + ASR + brain + TTS into a working dialogue.

Step 5: half-duplex (strict turn-taking) — HalfDuplexLoop.
Step 6: full-duplex (barge-in + backchannel-vs-interrupt) — FullDuplexLoop.
The timeline/brain/audioio/tts pieces are reused unchanged by both.
"""

from .timeline import Timeline, TimelineLog
from .brain import route, RouteDecision
from .half_duplex import HalfDuplexLoop
from .full_duplex import FullDuplexLoop
from .classifier import classify_overlap

__all__ = [
    "Timeline", "TimelineLog", "route", "RouteDecision",
    "HalfDuplexLoop", "FullDuplexLoop", "classify_overlap",
]
