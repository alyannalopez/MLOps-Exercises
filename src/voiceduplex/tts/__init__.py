"""TTS — "how do we speak the reply?".

One interface, swappable backends:
  - Kokoro (primary; natural, near-human; isolated 3.12 venv via subprocess)
  - Piper  (backup; lighter, faster, more robotic; in-process)

Config: tts.backend: kokoro  (or pass backend= to make_tts).
This TTS receives each agent reply and returns playable mono PCM.
"""

from __future__ import annotations

from .base import Synthesis, TTS
from .kokoro import KokoroTTS
from .piper import PiperTTS

__all__ = ["TTS", "Synthesis", "KokoroTTS", "PiperTTS", "make_tts"]


def make_tts(backend: str = "kokoro", **kwargs) -> TTS:
    """Build a TTS by name.

    `**kwargs` are passed straight to the chosen backend's constructor, so a
    caller can override any knob (voice, model path, python interpreter, ...)
    without changing the interface.
    """
    backends = {"kokoro": KokoroTTS, "piper": PiperTTS}
    if backend not in backends:
        raise ValueError(f"unknown TTS backend {backend!r}; expected one of {sorted(backends)}")
    return backends[backend](**kwargs)
