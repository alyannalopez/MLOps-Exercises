"""TTS — "how do we speak the reply?".

One interface, swappable backends:
  - Piper  (primary; fast, in-process, slightly robotic)
  - Kokoro (backup; natural, near-human; isolated 3.12 venv via subprocess)
  - fallback (wrapper: try `primary`, retry on `backup` if it fails)

Config: tts.backend: piper  (or pass backend= to make_tts).
This TTS receives each agent reply and returns playable mono PCM.
"""

from __future__ import annotations

from .base import Synthesis, TTS
from .fallback import FallbackTTS
from .kokoro import KokoroTTS
from .piper import PiperTTS

__all__ = ["TTS", "Synthesis", "KokoroTTS", "PiperTTS", "FallbackTTS", "make_tts"]


def make_tts(backend: str = "piper", **kwargs) -> TTS:
    """Build a TTS by name.

    `**kwargs` are passed straight to the chosen backend's constructor, so a
    caller can override any knob (voice, model path, python interpreter, ...)
    without changing the interface.

    ``backend="fallback"`` consumes special kwargs:
      primary_kwargs / backup_kwargs  — per-engine constructor args,
      primary / backup               — engine names (default piper / kokoro).
    """
    backends = {"kokoro": KokoroTTS, "piper": PiperTTS}
    if backend == "fallback":
        p_name = kwargs.pop("primary", "piper")
        b_name = kwargs.pop("backup", "kokoro")
        p_kw = kwargs.pop("primary_kwargs", {})
        b_kw = kwargs.pop("backup_kwargs", {})
        if p_name not in backends or b_name not in backends:
            raise ValueError(f"unknown fallback engine; expected one of {sorted(backends)}")
        return FallbackTTS(backends[p_name](**p_kw), backends[b_name](**b_kw))
    if backend not in backends:
        raise ValueError(f"unknown TTS backend {backend!r}; expected one of {sorted(backends)} or 'fallback'")
    return backends[backend](**kwargs)
