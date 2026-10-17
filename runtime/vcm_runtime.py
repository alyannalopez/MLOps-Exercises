"""ONNX-runtime voice-command classifier for the Pi runtime.

Loads the exported conformer_s ONNX graph (two-head: intent + slot) and
classifies 1-second command clips.  The feature pipeline is byte-for-byte
the one used in training (vcm.features.log_mel_spectrogram: 16 kHz ->
40-mel log spectrogram, 25 ms window / 10 ms hop, center-crop/pad to
100 frames), so runtime behavior matches the evaluated model exactly.

Slot decoding: the slot head outputs 19-way logits (18 slot values +
NO_SLOT).  For slotted intents, we argmax only over the valid slot
values for that intent (joint decoding, tiny-alexa style).  No ASR.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from vcm.features import SAMPLE_RATE, log_mel_spectrogram, prepare_clip
from vcm.slots import (N_SLOTS, NO_SLOT_IDX, SLOT_VALUES, INTENT_SLOTS,
                       decode_slot)

log = logging.getLogger(__name__)

N_CLASSES = 20
N_MELS = 40
N_FRAMES = 100
CLIP_SECONDS = 1.0  # 16000 samples

# Must match the training label order (scripts/evaluate_vcm_v2.py).
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    "OUT_OF_SCOPE",
]
OOS_IDX = 19


@dataclass
class CommandResult:
    intent: str
    intent_idx: int
    confidence: float
    probabilities: np.ndarray
    latency_ms: float
    slot: str | None = None
    slot_confidence: float | None = None
    audio_ms: float = 1000.0
    in_scope: bool = field(init=False)

    def __post_init__(self):
        self.in_scope = self.intent_idx != OOS_IDX


class VCMClassifier:
    """Real-time command classifier (batch=1, CPU, ONNX Runtime)."""

    def __init__(self, onnx_path: str, *, confidence_threshold: float = 0.35,
                 intra_threads: int | None = None):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = float(confidence_threshold)

    @property
    def intents(self) -> list[str]:
        return INTENTS

    def features(self, waveform: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 -> (1, 1, 40, 100) log-mel, as trained.

        Uses prepare_clip (loudest-1s crop + fixed frame count) so any clip
        length works; this is exactly the transform used in training and
        evaluation.
        """
        w = np.asarray(waveform, dtype=np.float32)
        feat = prepare_clip(w, target_seconds=CLIP_SECONDS, n_mels=N_MELS)
        return feat[None, None]

    def classify(self, waveform: np.ndarray) -> CommandResult:
        """Classify one clip. Accepts any length; features are cropped/padded
        to the fixed 1 s window inside log_mel_spectrogram."""
        t0 = time.perf_counter()
        x = self.features(waveform)
        outputs = self.session.run(None, {self.input_name: x})
        intent_logits = outputs[0][0]
        slot_logits = outputs[1][0] if len(outputs) > 1 else None
        probs = _softmax(intent_logits)
        idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Decode slot (two-head model)
        slot, slot_conf = None, None
        if slot_logits is not None and INTENTS[idx] in INTENT_SLOTS:
            slot = decode_slot(INTENTS[idx], slot_logits.tolist())
            # Slot confidence: softmax over valid slot values only
            valid = INTENT_SLOTS[INTENTS[idx]]
            valid_logits = np.array([slot_logits[i] for i in valid])
            valid_probs = _softmax(valid_logits)
            slot_conf = float(valid_probs.max())

        return CommandResult(
            intent=INTENTS[idx],
            intent_idx=idx,
            confidence=float(probs[idx]),
            probabilities=probs,
            latency_ms=latency_ms,
            slot=slot,
            slot_confidence=slot_conf,
            audio_ms=CLIP_SECONDS * 1000.0,
        )

    def classify_gated(self, waveform: np.ndarray) -> CommandResult | None:
        """Classify with a confidence gate: returns None when the top
        probability is below `confidence_threshold` (treated as no-command).
        This is the operating point used to control false accepts (Stage 5
        showed tau=0.7 gives ~97% in-scope precision for conformer_s)."""
        res = self.classify(waveform)
        if res.confidence < self.confidence_threshold:
            return None
        return res


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
