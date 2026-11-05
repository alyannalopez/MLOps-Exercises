"""Piper TTS backend (BACKUP).

Lighter, faster, more robotic than Kokoro. Runs IN-PROCESS in the main 3.14
env (piper-tts 1.8.0 is already installed there and bundles its own espeak-ng
data, so no external dependency). Use it when Kokoro is unavailable or too slow
on the Pi — flip tts.backend to 'piper' in config/config.yaml.

Uses the NEW piper1-gpl API: piper.voice.PiperVoice.load(model, config).

Config keys (tts: block):
  piper_model : path to the .onnx voice (default: data/voices/en_US-lessac-medium.onnx)
  piper_config: path to the matching .onnx.json (default: alongside the model)
  voice       : unused for Piper (the voice IS the model file); kept for interface parity
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Synthesis, TTS

ROOT = Path(__file__).resolve().parents[3]  # .../ME3
DEFAULT_MODEL = ROOT / "data" / "voices" / "en_US-lessac-medium.onnx"


class PiperTTS(TTS):
    def __init__(
        self,
        model: str = str(DEFAULT_MODEL),
        config: Optional[str] = None,
        speed: float = 1.0,
    ) -> None:
        self.model = model
        self.config = config or (str(Path(model).with_suffix(".onnx.json")))
        self.speed = speed
        self._voice = None
        self._sr = None

    @property
    def backend_name(self) -> str:
        return "piper"

    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return
        from piper.voice import PiperVoice  # lazy: keep import out of module load

        self._voice = PiperVoice.load(self.model, self.config)
        self._sr = int(self._voice.config.sample_rate)

    def warm(self) -> None:
        self._ensure_loaded()

    def synthesize(self, text: str) -> Synthesis:
        self._ensure_loaded()
        text = (text or "").strip()
        if not text:
            return Synthesis(
                audio=np.zeros(0, dtype=np.int16),
                sample_rate=self._sr or 22050,
                text="",
                duration_audio=0.0,
                duration_compute=0.0,
                backend=self.backend_name,
            )

        t0 = time.perf_counter()
        chunks = list(self._voice.synthesize(text))
        audio_f = np.concatenate([np.asarray(c.audio_float_array, dtype=np.float32) for c in chunks])
        audio = (np.clip(audio_f, -1.0, 1.0) * 32767).astype(np.int16)
        wall = time.perf_counter() - t0
        sr = self._sr or int(chunks[0].sample_rate)

        return Synthesis(
            audio=audio,
            sample_rate=sr,
            text=text,
            duration_audio=len(audio) / sr,
            duration_compute=wall,
            backend=self.backend_name,
        )
