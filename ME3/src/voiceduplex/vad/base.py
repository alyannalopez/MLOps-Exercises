"""Common VAD contract.

A VAD consumes fixed 32 ms (512-sample @ 16 kHz) int16 frames and tracks
state with hysteresis so it doesn't flap on noisy edges. Callers read:
  - .state            -> VADState.SPEAKING / SILENCE
  - .process(frame)   -> advance state, returns (state, event) where event is
                         "speech_start" | "speech_end" | None
  - .reset()          -> clear internal buffers (between utterances)

Both engines wrap a raw per-frame probability/decision in this hysteresis,
so downstream code is engine-agnostic.
"""

from __future__ import annotations

import abc
from enum import Enum


class VADState(Enum):
    SILENCE = 0
    SPEAKING = 1


class VAD(abc.ABC):
    FRAME_SAMPLES = 512  # 32 ms @ 16 kHz

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.state = VADState.SILENCE
        self.threshold = 0.5
        self.min_speech_frames = 4
        self.min_silence_frames = 12
        self._speech_run = 0
        self._silence_run = 0

    @abc.abstractmethod
    def _score(self, frame) -> float:
        """Return a speech-likelihood in [0,1] for one frame."""

    def reset(self) -> None:
        self.state = VADState.SILENCE
        self._speech_run = 0
        self._silence_run = 0

    def process(self, frame) -> tuple[VADState, str | None]:
        """Advance state on one frame. Returns (state, event).

        event is "speech_start", "speech_end", or None. Hysteresis:
          - enter SPEAKING after >= min_speech_frames consecutive voiced frames
          - leave SPEAKING after >= min_silence_frames consecutive unvoiced frames
        """
        voiced = self._score(frame) >= self.threshold

        if self.state == VADState.SILENCE:
            self._speech_run = self._speech_run + 1 if voiced else 0
            if self._speech_run >= self.min_speech_frames:
                self.state = VADState.SPEAKING
                self._silence_run = 0
                return self.state, "speech_start"
        else:  # SPEAKING
            self._silence_run = self._silence_run + 1 if not voiced else 0
            if self._silence_run >= self.min_silence_frames:
                self.state = VADState.SILENCE
                self._speech_run = 0
                return self.state, "speech_end"
        return self.state, None
