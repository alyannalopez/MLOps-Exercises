"""Common TTS contract.

A TTS turns text into 24 kHz mono PCM (int16). Two interchangeable backends
sit behind one interface:

  - Kokoro  (primary; natural, near-human; runs in an ISOLATED 3.12 venv via
             a subprocess worker, because kokoro-onnx caps Python <3.14 and the
             rest of the pipeline runs on 3.14)
  - Piper   (backup; lighter, robotic-ish; runs IN-PROCESS in the main env)

Downstream code only ever calls:
  - .warm()           -> preload the model (call once at startup)
  - .synthesize(text) -> Synthesis   (whole reply)

Batch mode is deliberate for Step 4. Streaming / chunked synthesis (so audio
starts playing before the whole reply is rendered) is a Step 6 optimization for
full-duplex; it is NOT a correctness requirement for the first working loop.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Synthesis:
    """Result of one synthesis call."""
    audio: np.ndarray          # int16 mono PCM
    sample_rate: int           # e.g. 24000 (Kokoro) or 22050 (Piper)
    text: str                  # the text that was spoken
    duration_audio: float      # seconds of audio produced
    duration_compute: float    # wall seconds spent synthesizing
    backend: str               # "kokoro" | "piper"

    @property
    def rtf(self) -> float:
        """Real-time factor: compute_time / audio_time. <1 = faster than real-time."""
        return self.duration_compute / self.duration_audio if self.duration_audio > 0 else 0.0

    def to_wav(self, path) -> None:
        """Convenience: write the audio to a 16-bit mono WAV file."""
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.audio.astype(np.int16).tobytes())


class TTS(abc.ABC):
    @abc.abstractmethod
    def warm(self) -> None:
        """Load the model once. Subsequent synthesize() calls reuse it."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> Synthesis:
        """Synthesize text to mono int16 PCM. Returns a Synthesis."""

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Stable identifier, e.g. 'kokoro' or 'piper'."""
