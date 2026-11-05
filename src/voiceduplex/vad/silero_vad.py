"""Silero VAD engine (primary).

Wraps the official silero_vad.onnx via onnxruntime. Per-frame speech
probability in [0,1]; hysteresis applied in the base class via thresholds.

Model file lives at vad/models/silero_vad.onnx (committed).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .base import VAD

_MODEL = Path(__file__).parent / "models" / "silero_vad.onnx"


class SileroVAD(VAD):
    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        threshold: float = 0.5,
        min_speech_frames: int = 4,  # ~128 ms before declaring speech
        min_silence_frames: int = 12,  # ~384 ms of quiet before declaring end
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__(sample_rate)
        self.threshold = threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames

        path = Path(model_path) if model_path else _MODEL
        if not path.exists():
            raise FileNotFoundError(f"Silero model not found: {path}")

        self._sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        # 16 kHz model: 512-sample frames, 64-sample context prepended to each.
        self._sr = np.array(self.sample_rate, dtype=np.int64)
        self._context_size = 64 if self.sample_rate == 16000 else 32
        self.reset_state()

    def reset_state(self) -> None:
        # Zero-init the recurrent state + audio context buffer.
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self.reset_state()

    def _score(self, frame) -> float:
        # Silero expects float32 mono. Accept int16 or float32.
        if frame.dtype == np.int16:
            x = frame.astype(np.float32) / 32768.0
        else:
            x = frame.astype(np.float32)
        x = x.reshape(1, -1)
        # Prepend the rolling context (last 64 samples of prior audio).
        x = np.concatenate([self._context, x], axis=1)
        out = self._sess.run(
            None,
            {"input": x, "sr": self._sr, "state": self._state},
        )
        prob, self._state = out[0], out[1]
        self._context = x[:, -self._context_size :]
        return float(prob.reshape(-1)[0])
