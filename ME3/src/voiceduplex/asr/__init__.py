"""ASR — "what did the human say?".

One interface, swappable backends:
  - WhisperCpp (primary; local/free/offline, Mac + Pi)

Config: asr.backend: whispercpp  (or pass backend= to make_asr).
This ASR later receives each VAD-delimited utterance from the agent loop.
"""

from .base import ASR, Transcript
from .whispercpp import WhisperCpp

__all__ = ["ASR", "Transcript", "WhisperCpp", "make_asr"]


def make_asr(backend: str = "whispercpp", **kwargs) -> ASR:
    return {"whispercpp": WhisperCpp}[backend](**kwargs)
