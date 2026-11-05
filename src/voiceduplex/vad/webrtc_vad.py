"""WebRTC VAD engine (backup).

Lighter than Silero but coarser. webrtcvad.is_speech() returns a bool per
frame; we expose it as a 0/1 score so the shared hysteresis applies.

webrtcvad only accepts 10/20/30 ms frames at 8/16/32/48 kHz. Our 32 ms
(512-sample) frames don't fit, so we split each into two 16 ms (256-sample)
sub-frames and OR the decisions. Aggressiveness 0..3 (3 = most aggressive).
"""

from __future__ import annotations

import numpy as np
import webrtcvad

from .base import VAD


class WebRtcVAD(VAD):
    def __init__(
        self,
        aggressiveness: int = 2,
        threshold: float = 0.5,  # unused by webrtc (bool), kept for interface parity
        min_speech_frames: int = 4,
        min_silence_frames: int = 12,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__(sample_rate)
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("webrtcvad supports 8/16/32/48 kHz only")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.threshold = threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        # webrtcvad only accepts 10/20/30 ms frames. Use 20 ms (320 samples @16k).
        # A 512-sample (32 ms) frame -> two 20 ms chunks + 112-sample carryover.
        self._sub = int(round(sample_rate * 0.020))
        self._carry = np.zeros(0, dtype=np.int16)

    def reset(self) -> None:
        super().reset()
        self._carry = np.zeros(0, dtype=np.int16)

    def _score(self, frame) -> float:
        pcm = frame.astype(np.int16)
        buf = np.concatenate([self._carry, pcm])
        speech = False
        consumed = 0
        for i in range(0, len(buf), self._sub):
            chunk = buf[i : i + self._sub]
            if chunk.shape[0] != self._sub:
                break  # leftover < 20 ms; carry forward
            consumed = i + self._sub
            if self.vad.is_speech(chunk.tobytes(), self.sample_rate):
                speech = True
                break
        self._carry = buf[consumed:]
        return 1.0 if speech else 0.0
