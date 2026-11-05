"""Kokoro TTS backend (PRIMARY).

Natural, near-human English speech. Because kokoro-onnx requires Python <3.14
and the rest of the pipeline runs on 3.14, this backend shells out to
kokoro_worker.py in an isolated 3.12 venv (default: .venv-kokoro at the project
root). The interface is identical to Piper, so downstream code never knows the
difference.

Config keys (tts: block in config/config.yaml):
  kokoro_python : path to the 3.12 interpreter (default: .venv-kokoro/bin/python)
  kokoro_model  : path to kokoro-v1.0.onnx
  kokoro_voices : path to voices-v1.0.bin
  voice         : voice id, e.g. af_heart (see get_voices() for the full list)
  speed         : 1.0 = normal
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Synthesis, TTS

ROOT = Path(__file__).resolve().parents[3]  # .../ME3


class KokoroTTS(TTS):
    def __init__(
        self,
        python: str = str(ROOT / ".venv-kokoro" / "bin" / "python"),
        model: str = str(ROOT / "models" / "kokoro-v1.0.onnx"),
        voices: str = str(ROOT / "models" / "voices-v1.0.bin"),
        voice: str = "af_heart",
        speed: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self.python = python
        self.model = model
        self.voices = voices
        self.voice = voice
        self.speed = speed
        self.timeout = timeout
        self.worker = Path(__file__).with_name("kokoro_worker.py")
        self._warmed = False

    @property
    def backend_name(self) -> str:
        return "kokoro"

    def warm(self) -> None:
        """Verify the venv + models are reachable by doing one tiny synthesis."""
        if self._warmed:
            return
        self.synthesize("warmup")
        self._warmed = True

    def synthesize(self, text: str) -> Synthesis:
        text = (text or "").strip()
        if not text:
            return Synthesis(
                audio=np.zeros(0, dtype=np.int16),
                sample_rate=24000,
                text="",
                duration_audio=0.0,
                duration_compute=0.0,
                backend=self.backend_name,
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        req = {
            "text": text,
            "voice": self.voice,
            "speed": self.speed,
            "model": self.model,
            "voices": self.voices,
            "out": out_path,
        }
        t0 = time.perf_counter()
        proc = subprocess.run(
            [self.python, str(self.worker)],
            input=json.dumps(req),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        wall = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"Kokoro worker failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
            )
        meta = json.loads(proc.stdout.strip().splitlines()[-1])
        if not meta.get("ok"):
            raise RuntimeError(f"Kokoro worker reported failure: {meta}")

        audio = np.fromfile(out_path, dtype=np.int16)
        # Read back the exact sample rate the worker reported.
        sr = int(meta["sample_rate"])
        duration_audio = len(audio) / sr if sr else float(meta["duration_audio"])

        return Synthesis(
            audio=audio,
            sample_rate=sr,
            text=text,
            duration_audio=duration_audio,
            duration_compute=wall,
            backend=self.backend_name,
        )
