"""Backchannel-vs-interrupt classifier (Step 6, full-duplex).

When the user starts speaking WHILE the agent is talking, that overlapping
utterance is one of two things:

  backchannel : a short acknowledgment ("yeah", "mm-hm", "uh-huh", "got it").
                The user is following along, NOT trying to take the floor.
                -> we briefly acknowledge and RESUME our interrupted reply.

  interrupt   : a real new request / correction ("stop", "actually...",
                "wait, what about X", a question). The user wants the floor.
                -> we CANCEL our reply and RESPOND to the new utterance.

This is a deterministic, OFFLINE classifier so the full-duplex loop is fully
testable without an LLM. It reuses the backchannel vocabulary from brain.py
and adds a small set of explicit interrupt cues. Anything that is neither a
clean backchannel nor a short filler is treated as an interrupt (the safe
default: give the user the floor).

Interface:  classify_overlap(text, duration_sec) -> "backchannel" | "interrupt"
"""

from __future__ import annotations

import re

# Reuse the exact backchannel vocabulary the brain already trusts, so a "mm-hm"
# is a backchannel everywhere in the system, not just when we're idle.
from .brain import _BACKCHANNEL


# Explicit "take the floor" cues. If any of these appear, it is an interrupt
# regardless of length (e.g. "stop", "wait", "actually, ...").
_INTERRUPT_CUES = re.compile(
    r"\b(stop|wait|hold on|never mind|no no|actually|correction|instead|"
    r"change (that|it)|cancel|shut up|quiet|listen|hey|excuse me|"
    r"what about|how about|but\b|however)\b",
    re.IGNORECASE,
)


def _is_backchannel(text: str) -> bool:
    """True if the whole utterance is a bare acknowledgment."""
    return bool(_BACKCHANNEL.match(text))


def classify_overlap(text: str, duration_sec: float = 0.0) -> str:
    """Classify an utterance that overlaps the agent's speech.

    Returns "backchannel" or "interrupt".

    Decision order (cheap checks first):
      1. Bare backchannel word(s)  -> backchannel
      2. Explicit interrupt cue    -> interrupt
      3. Very short (< ~0.5 s) and a couple of words or fewer -> backchannel
         (a quick "yeah"/"right" that the regex missed)
      4. Everything else           -> interrupt (give the user the floor)
    """
    t = (text or "").strip()
    if not t:
        # No usable transcript: don't yank the floor away over noise.
        return "backchannel"

    if _is_backchannel(t):
        return "backchannel"

    if _INTERRUPT_CUES.search(t):
        return "interrupt"

    words = len(re.findall(r"\w+", t))
    # A very short, wordless-ish blip is almost certainly a backchannel even if
    # the regex didn't catch the exact form.
    if duration_sec < 0.5 and words <= 2:
        return "backchannel"

    return "interrupt"
