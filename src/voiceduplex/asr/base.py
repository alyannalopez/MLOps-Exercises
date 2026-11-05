"""Common ASR contract.

An ASR turns 16 kHz mono PCM into text. Two interchangeable backends sit
behind one interface:
  - WhisperCpp  (primary; local, free, offline, runs on Mac AND Pi)
  - (reserved)  a streaming cloud backend could slot in later

Downstream code only ever calls:
  - .transcribe(pcm)  -> Transcript   (batch: whole utterance)
  - .warm()           -> preload the model (call once at startup)

Batch mode is deliberate for Step 3. The full-duplex pipeline (Step 6) will
feed each VAD-delimited utterance to transcribe(); streaming partials are a
later optimization, not a correctness requirement for the first working loop.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class Transcript:
    """Result of one transcription call."""
    text: str                 # the recognized text (may be empty)
    lang: str                 # language tag, e.g. "en"
    duration_audio: float     # seconds of audio fed in
    duration_compute: float   # wall seconds spent computing
    segments: list[tuple[float, float, str]]  # (t0, t1, text) per segment

    @property
    def rtf(self) -> float:
        """Real-time factor: compute_time / audio_time. <1 = faster than real-time."""
        return self.duration_compute / self.duration_audio if self.duration_audio > 0 else 0.0


class ASR(abc.ABC):
    SAMPLE_RATE = 16_000

    @abc.abstractmethod
    def warm(self) -> None:
        """Load the model once. Subsequent transcribe() calls reuse it."""

    @abc.abstractmethod
    def transcribe(self, pcm, lang: str = "en") -> Transcript:
        """Transcribe 16 kHz mono int16 PCM. Returns a Transcript."""
