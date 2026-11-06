"""Lightweight open-source wake-word detector built on openWakeWord.

openWakeWord (Apache-2.0, https://github.com/dscripka/openWakeWord) ships
pretrained models that run on CPU with a mel-spectrogram front-end +
embedding network + small per-wakeword head. Total model footprint is a
few MB; per-frame inference on a Raspberry Pi 4 is a few ms.

Default wake word: **"hey mycroft"** (hey_mycroft_v0.1) — an open-source,
vendor-neutral phrase from the Mycroft project. We deliberately do NOT use
the "alexa" model (trademarked by Amazon).

This wrapper (targets openwakeword **0.4.0**):
  - loads the pretrained model once (ONNX backend, CPU),
  - feeds 16 kHz frames (1280 samples = 80 ms, openWakeWord's recommended
    chunk) to the internal feature extractor,
  - exposes detect(frame) -> probability and a reset() for post-trigger.

No custom training: we use a stock pretrained model, which is what the
project brief asks for ("lightweight, open-source pretrained wake-word
detector").

Version support: works with openwakeword **0.4.0** (the pinned version in
requirements.txt / requirements-pi.txt, matching tiny-alexa) and also with
0.6.x. The APIs differ:
  - 0.4.0: pretrained names resolve via ``openwakeword.models[name]["model_path"]``,
    Model takes ``wakeword_model_paths=[...]``.
  - 0.6.x: the ``openwakeword.models`` dict was replaced by
    ``openwakeword.MODELS`` / ``get_pretrained_model_paths()``, and Model prefers
    ``wakeword_models=[...]`` (still accepts the old kwarg with a deprecation
    warning).
This wrapper detects the installed layout at load time, so the same code runs
on the Mac dev box (0.6.x) and on the Pi bundle (0.4.0).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz, openWakeWord's recommended chunk

DEFAULT_WAKEWORD = "hey_mycroft"


@dataclass
class WakeEvent:
    probability: float
    threshold: float
    triggered: bool


class WakeWordDetector:
    """Thin stateful wrapper around openwakeword.Model (v0.4.0).

    Usage:
        ww = WakeWordDetector(wakeword="hey_mycroft", threshold=0.5)
        for frame in stream:            # int16 or float32, 16 kHz
            ev = ww.detect(frame)
            if ev.triggered:
                ...                      # record the command
                ww.reset()
    """

    def __init__(self, wakeword: str = DEFAULT_WAKEWORD, threshold: float = 0.5):
        import openwakeword
        from openwakeword.model import Model

        path = self._resolve_model_path(openwakeword, wakeword)
        # 0.4.0 keyword; 0.6.x still accepts it (with a deprecation warning).
        self._model = Model(wakeword_model_paths=[path])
        # Both versions name the predict() key after the file stem, e.g.
        # "hey_mycroft_v0.1". Derive it from the basename, stripping whatever
        # extension the file actually has (.onnx on 0.4.0, .tflite on 0.6.x).
        base = os.path.basename(path)
        self._key = base[:base.rfind(".")] if "." in base else base
        self.threshold = float(threshold)
        self.wakeword = wakeword

    @staticmethod
    def _resolve_model_path(openwakeword, wakeword: str) -> str:
        """Find the pretrained ONNX path across openwakeword 0.4.0 and 0.6.x."""
        # 0.4.0: openwakeword.models dict
        models_dict = getattr(openwakeword, "models", None)
        if isinstance(models_dict, dict) and wakeword in models_dict:
            return models_dict[wakeword]["model_path"]
        # 0.6.x: MODELS dict / get_pretrained_model_paths(). Entries may point
        # at .tflite files; prefer the sibling .onnx (this env has no tflite
        # runtime — onnxruntime is the working backend).
        models_dict = getattr(openwakeword, "MODELS", None)
        if isinstance(models_dict, dict) and wakeword in models_dict:
            entry = models_dict[wakeword]
            path = entry.get("model_path") if isinstance(entry, dict) else None
            if path:
                if path.endswith(".tflite"):
                    onnx_alt = path[: -len(".tflite")] + ".onnx"
                    if os.path.exists(onnx_alt):
                        return onnx_alt
                return path
        getter = getattr(openwakeword, "get_pretrained_model_paths", None)
        if callable(getter):
            try:
                paths = getter(model_names=[wakeword])
                if paths:
                    return paths[0]
            except Exception:
                pass
        # Fallback: locate the bundled resource file directly.
        pkg_dir = os.path.dirname(openwakeword.__file__)
        candidate = os.path.join(pkg_dir, "resources", "models",
                                 f"{wakeword}_v0.1.onnx")
        if os.path.exists(candidate):
            return candidate
        available = sorted(set(getattr(models_dict, "keys", lambda: [])()))
        raise ValueError(
            f"unknown wakeword {wakeword!r}; available: {available}")

    def detect(self, frame: np.ndarray) -> WakeEvent:
        """Feed one 16 kHz frame (int16 or float32 in [-1, 1])."""
        frame = np.asarray(frame)
        if frame.dtype == np.float32:
            frame = (frame * 32767).astype(np.int16)
        probs = self._model.predict(frame)
        p = float(probs.get(self._key, 0.0))
        return WakeEvent(probability=p, threshold=self.threshold,
                         triggered=p >= self.threshold)

    def reset(self) -> None:
        """Clear internal feature buffers (call after a trigger)."""
        self._model.reset()
