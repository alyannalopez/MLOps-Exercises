#!/usr/bin/env python3
"""VCM Smart Home — web demo server.

FastAPI + WebSocket app that wires the browser UI to the two-head
Conformer-S voice-command model (intent + slot, no ASR) and a server-side
smart-home state machine.

Flow:
  browser (white/gold/blue UI)
    --WebSocket-->  this server
        1. receives 16 kHz PCM16 frames from the mic (or a recorded take)
        2. on a "take" it runs the EXACT training feature pipeline
           (peak-normalize -> loudest-1s crop -> log-mel 40x100) through the
           two-head conformer_s ONNX graph
        3. decodes intent (20-way) + slot (19-way, joint per-intent decode)
        4. maps the prediction onto a SmartHomeState and streams the result
           + new state back

Run:
  python3 web/app.py                 # http://localhost:8000  (opens browser)
  python3 web/app.py --port 8010     # custom port
  python3 web/app.py --model conformer_s_int8.onnx
"""
from __future__ import annotations
import argparse, asyncio, base64, json, os, random, shutil, subprocess, sys, threading, time, webbrowser
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
MODEL_DIR = BUNDLE / "models"
STATIC = HERE / "static"
MUSIC_DIR = BUNDLE / "assets" / "music"

# Project root (ME2/) holds runtime/ + vcm/. Add it so we can reuse the SAME
# wake-word detector the terminal demo uses (runtime/wakeword.py).
PROJECT_ROOT = BUNDLE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── constants (must match vcm/features.py + vcm_runtime.py) ────────────────
SR = 16000
N_MELS = 40
N_FRAMES = 100
CLIP_SECONDS = 1.0

# 20 intents — MUST match the training label order (vcm_runtime.INTENTS).
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    "OUT_OF_SCOPE",
]
OOS_IDX = 19

# Human-friendly display name per intent (shown in the UI)
INTENT_NAME = {
    "PLAY_MUSIC": "Play Music", "WEATHER": "Weather", "TIME": "Time",
    "LIGHT_ON": "Lights On", "LIGHT_OFF": "Lights Off", "PAUSE": "Pause",
    "STOP": "Stop", "NEXT": "Next Track", "VOLUME_UP": "Volume Up",
    "VOLUME_DOWN": "Volume Down", "CALL": "Call", "MESSAGE": "Message",
    "LIST_REMINDERS": "Reminders", "TIMER": "Timer", "ALARM": "Alarm",
    "TEMPERATURE": "Thermostat", "BRIGHTNESS": "Brightness", "COLOR": "Color",
    "CREATE_REMINDER": "New Reminder", "OUT_OF_SCOPE": "Out of Scope",
}

# ── slot vocabulary (must match vcm/slots.py) ──────────────────────────────
SLOT_VALUES: list[str] = [
    "10 seconds", "30 seconds", "1 minute",          # TIMER
    "6:00 AM", "8:00 AM", "9:00 PM",                  # ALARM
    "18 degrees", "22 degrees", "26 degrees",        # TEMPERATURE
    "20 percent", "60 percent", "100 percent",       # BRIGHTNESS
    "Red", "Blue", "Green",                           # COLOR
    "Drink water", "Study", "Exercise",               # CREATE_REMINDER
    "NO_SLOT",                                        # 18
]
INTENT_SLOTS: dict[str, list[int]] = {
    "TIMER":           [0, 1, 2],
    "ALARM":           [3, 4, 5],
    "TEMPERATURE":     [6, 7, 8],
    "BRIGHTNESS":      [9, 10, 11],
    "COLOR":           [12, 13, 14],
    "CREATE_REMINDER": [15, 16, 17],
}


def decode_slot(intent: str, slot_logits: list[float]) -> str | None:
    """Joint decode: argmax only over the valid slot values for this intent."""
    if intent not in INTENT_SLOTS:
        return None
    valid = INTENT_SLOTS[intent]
    best_idx, best_logit = valid[0], float("-inf")
    for si in valid:
        if slot_logits[si] > best_logit:
            best_logit, best_idx = slot_logits[si], si
    return SLOT_VALUES[best_idx]


# ── feature pipeline (byte-for-byte vcm/features.py: numpy only) ───────────
def _hann(n: int) -> np.ndarray:
    return np.hanning(n)


def _mel_filterbank(sr: int, n_mels: int, n_bins: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min, m_max = hz2mel(fmin), hz2mel(fmax)
    mel_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((n_bins + 1) * hz_pts / (sr / 2.0)).astype(int)
    bins = np.clip(bins, 0, n_bins)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        for b in range(left, center):
            if center > left:
                fb[i - 1, b] = (b - left) / (center - left)
        for b in range(center, right):
            if right > center:
                fb[i - 1, b] = (right - b) / (right - center)
    return fb


def log_mel_spectrogram(waveform: np.ndarray, *, sr: int = SR, n_mels: int = N_MELS,
                        frame_length_ms: float = 25.0, hop_length_ms: float = 10.0,
                        fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    """(n_mels, n_frames) log-mel spectrogram — identical to vcm.features."""
    waveform = np.asarray(waveform, dtype=np.float32)
    n_fft = int(sr * frame_length_ms / 1000.0)
    hop = max(1, int(sr * hop_length_ms / 1000.0))

    pad = n_fft // 2
    x = np.pad(waveform, (pad, pad), mode="reflect") if len(waveform) > 0 else np.zeros(n_fft, dtype=np.float32)
    rem = (n_fft - ((len(x) - n_fft) % hop)) % hop
    x = x[: len(x) - rem]
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, len(x) - 1)
    frames = x[idx]
    frames *= _hann(n_fft)[None, :]

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 / n_fft
    power = np.maximum(power, 1e-10)

    mel = _mel_filterbank(sr, n_mels, n_fft // 2 + 1, fmin, fmax)
    feats = (power @ mel.T).T
    return np.log(feats + 1e-10).astype(np.float32)


def loudest_crop(waveform: np.ndarray, target_samples: int, win: int | None = None) -> np.ndarray:
    """Crop the loudest contiguous `target_samples` window (coarse-to-fine)."""
    waveform = np.asarray(waveform, dtype=np.float32)
    n = len(waveform)
    if n <= target_samples:
        return waveform
    win = win or target_samples // 2
    if win <= 0:
        win = target_samples
    stride = win
    prof = np.array([
        float(np.sqrt(np.mean(waveform[i : i + win] ** 2)))
        for i in range(0, n - target_samples + 1, stride)
    ])
    best_coarse = int(np.argmax(prof)) * stride
    lo = max(0, best_coarse - stride)
    hi = min(n - target_samples, best_coarse + stride)
    best_start, best_rms = lo, -1.0
    for i in range(lo, hi + 1, max(1, win // 4)):
        r = float(np.sqrt(np.mean(waveform[i : i + target_samples] ** 2)))
        if r > best_rms:
            best_rms, best_start = r, i
    if best_rms <= 1e-6:
        best_start = (n - target_samples) // 2
    return waveform[best_start : best_start + target_samples]


def prepare_clip(waveform: np.ndarray, *, target_seconds: float = CLIP_SECONDS,
                 n_mels: int = N_MELS, hop_length_ms: float = 10.0) -> np.ndarray:
    """Peak-normalize, loudest-crop/pad to 1 s, extract log-mel -> (n_mels, n_frames)."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        waveform = np.zeros(SR, dtype=np.float32)
    peak = float(np.abs(waveform).max())
    if peak > 1e-6:
        waveform = waveform / peak * 0.95

    target_samples = int(target_seconds * SR)
    if len(waveform) > target_samples:
        waveform = loudest_crop(waveform, target_samples)
    elif len(waveform) < target_samples:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)))

    return log_mel_spectrogram(waveform, n_mels=n_mels, hop_length_ms=hop_length_ms)


def to_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """16 kHz mono float32 -> (1, 1, n_mels, n_frames), as trained."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        # naive linear resample (browser already sends 16 kHz)
        ratio = SR / sr
        n_out = int(len(y) * ratio)
        xp = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(x, xp, y).astype(np.float32)
    feat = prepare_clip(y)
    return feat[None, None]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ── inference session ──────────────────────────────────────────────────────
def load_session(model_name: str):
    import onnxruntime as ort
    path = MODEL_DIR / model_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available: "
                                f"{[p.name for p in MODEL_DIR.glob('*.onnx')]}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


# ── wake word (openWakeWord "hey mycroft") ─────────────────────────────────
# Reuse the EXACT detector the terminal demo runs (runtime/wakeword.py). It
# resolves the pretrained model across openwakeword 0.4.0 / 0.6.x layouts —
# the old inline loader crashed on the installed version, so the web app
# silently fell back to tap-to-talk (no wake word).
WW_DETECTOR = None
WW_CHUNK = 1280          # 80 ms @ 16 kHz (openWakeWord native frame size)
WW_THRESHOLD = 0.5       # same as the terminal demo (runtime/main.py --ww-threshold)

# ── tiny-alexa detection policy (src/infer.py) ─────────────────────────────
# The old web server fed the detector a single 80 ms chunk and trusted one
# score >= 0.5. openWakeWord's score is a rolling 30-frame average, so a
# single-chunk check fires on partial phrases ("hey...") and re-fires for the
# whole spoken word. tiny-alexa fixes both: require PATIENCE consecutive
# chunks above threshold, warm up WARMUP_CHUNKS before trusting scores, and
# stay refractory REFRACTORY_CHUNKS after a fire.
WW_PATIENCE = 2          # 2 x 80 ms of sustained score before firing
WW_WARMUP_CHUNKS = 10    # ~0.8 s after startup/reset before trusting scores
WW_REFRACTORY_CHUNKS = 15  # ~1.2 s after a detection: score stays high for
                           # the whole spoken word, so without this one "hey
                           # mycroft" fires several times

# ── tiny-alexa command endpointing (src/server.py) ─────────────────────────
# The old web server started recording the command IMMEDIATELY on the wake
# trigger, so the tail of "mycroft" leaked into the command window and the
# 1.2 s auto-stop cut off anything said slightly later. tiny-alexa instead:
#   1. skips WAKE_TAIL_CHUNKS (~0.24 s) so the end of the wake word is gone,
#   2. waits for speech above an adaptive threshold (4x the pre-wake room
#      noise floor) with PREROLL_CHUNKS of audio kept so the quiet onset of
#      the first word ("k" in "call", "s" in "stop") isn't clipped,
#   3. records until END_SILENCE_CHUNKS of trailing silence (or a 4.8 s cap),
#   4. exits early once EARLY_EXIT_SILENCE_CHUNKS of silence follow and the
#      model is >= EARLY_EXIT_PROB sure.
WAKE_TAIL_CHUNKS = 3
NOISE_HISTORY_CHUNKS = 40
NOISE_PERCENTILE = 20
MIN_SPEECH_RMS = 0.006
SPEECH_OVER_NOISE = 4.0
MAX_SPEECH_THRESHOLD = 0.04
PREROLL_CHUNKS = 3
END_SILENCE_CHUNKS = 7      # 0.56 s of silence ends the command
EARLY_EXIT_SILENCE_CHUNKS = 2
EARLY_EXIT_PROB = 0.95
MAX_COMMAND_CHUNKS = 60     # 4.8 s cap
NO_SPEECH_TIMEOUT_S = 6.0   # give up if nothing is said after the wake word


def load_wakeword():
    """Load the pretrained 'hey mycroft' wake-word detector (same as terminal).
    Returns the detector or None if unavailable (server then falls back to
    tap-to-talk)."""
    global WW_DETECTOR
    if WW_DETECTOR is not None:
        return WW_DETECTOR
    try:
        from runtime.wakeword import WakeWordDetector
        det = WakeWordDetector(wakeword="hey_mycroft", threshold=WW_THRESHOLD)
        det.detect(np.zeros(WW_CHUNK, dtype=np.float32))   # warm up
        WW_DETECTOR = det
        print(f"[vcm-web] wake word 'hey mycroft' ready (openWakeWord, "
              f"thr={WW_THRESHOLD}, patience={WW_PATIENCE})", flush=True)
        return det
    except Exception as e:
        print(f"[vcm-web] wake word unavailable ({e}); tap-to-talk only", flush=True)
        return None


class _WWGate:
    """tiny-alexa's detection policy on top of the openWakeWord model.

    Feeding one 80 ms chunk per call and trusting a single score >= threshold
    makes the detector fire on partial phrases and re-fire for the whole
    spoken wake word (openWakeWord's score is a rolling 30-frame average, so
    it stays elevated long after the phrase ends). The gate requires
    PATIENCE consecutive high chunks, ignores scores during a warm-up period,
    and goes refractory after a fire.
    """

    def __init__(self, det, threshold: float = WW_THRESHOLD,
                 patience: int = WW_PATIENCE):
        self.det = det
        self.threshold = threshold
        self.patience = patience
        self.streak = 0
        self.reset()

    def reset(self):
        self.streak = 0
        self.chunks_since_reset = 0
        self.refractory_left = 0
        self.det.reset()

    def process(self, chunk_int16: np.ndarray) -> bool:
        """One 80 ms int16 chunk in -> True if the wake word fired."""
        ev = self.det.detect(chunk_int16)
        score = ev.probability
        self.chunks_since_reset += 1
        if self.refractory_left > 0:
            self.refractory_left -= 1
            return False
        if self.chunks_since_reset <= WW_WARMUP_CHUNKS:
            return False
        if score >= self.threshold:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.patience:
            self.streak = 0
            self.refractory_left = WW_REFRACTORY_CHUNKS
            return True
        return False


def ww_reset_buffers():
    """Reset the detector's internal buffers after a command so the wake word
    doesn't re-trigger on the tail of the spoken command."""
    if WW_DETECTOR is None:
        return
    try:
        WW_DETECTOR.reset()
    except Exception:
        pass


# ── music playback (browser-side, tiny-alexa style) ────────────────────────
# The browser plays the MP3 through an AudioContext GainNode. The server
# never touches the audio — it just tells the browser which track to play.
# Ducking is done in the browser: when the wake word fires, the GainNode
# drops to 25% (smooth 80 ms ramp) so the mic can hear the user's voice.
# This is exactly how tiny-alexa does it (DUCK = .25, setTargetAtTime .08).
_MUSIC_TRACKS = [
    "track1_relaxing",   # Isang Daan – Cup of Joe
    "track2_upbeat",     # Kung Payagan – Cup of Joe
    "track3_lofi",       # Lalayo – Cup of Joe
    "track4_demo_song",  # Pamahiin – Cup of Joe
    "track5_alarm",      # Sandali – Cup of Joe
    "na_para_bang",      # Na Para Bang – Mariah Deborah
]


def _pick_audio_file(stem: str) -> Path | None:
    for ext in (".mp3", ".wav"):
        p = MUSIC_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ── smart home state ───────────────────────────────────────────────────────
@dataclass
class SmartHomeState:
    lights_on: bool = False
    brightness: int = 100
    color: str = "White"
    rgb: tuple = (255, 255, 255)
    music_playing: bool = False
    paused: bool = False
    current_track: str = ""
    volume: int = 60
    timer_active: bool = False
    timer_remaining: int = 0
    timer_total: int = 0
    alarms: list = field(default_factory=list)
    temperature: int = 22
    reminders: list = field(default_factory=list)
    weather: dict = field(default_factory=lambda: {
        "city": "Metro Manila", "temp_c": 31, "condition": "partly cloudy"})

    COLOR_MAP = {"Red": (255, 0, 0), "Blue": (0, 0, 255), "Green": (0, 255, 0),
                 "White": (255, 255, 255), "Yellow": (255, 255, 0),
                 "Purple": (128, 0, 128), "Orange": (255, 165, 0)}

    def _parse_duration(self, s: str) -> int:
        s = (s or "").lower().strip()
        if "minute" in s:
            num = int(''.join(c for c in s if c.isdigit()) or 1)
            return num * 60
        if "second" in s:
            return int(''.join(c for c in s if c.isdigit()) or 0)
        return 0

    def execute(self, command: str, slot_value: str = "") -> str:
        """Apply a command to state; returns a human-readable status message."""
        if command == "PLAY_MUSIC":
            # Pick a random copyright-free demo track (brief: "play randomly").
            stem = random.choice(_MUSIC_TRACKS)
            self.music_playing, self.paused = True, False
            self.current_track = stem
            msg = f"\U0001f3b5 Playing: {stem}"
        elif command == "PAUSE":
            if self.music_playing:
                self.paused = not self.paused
                msg = "\u23f8\ufe0f Paused" if self.paused else "\u25b6\ufe0f Resumed"
            else:
                msg = "\u26a0\ufe0f Nothing playing"
        elif command == "STOP":
            self.music_playing, self.paused = False, False
            self.current_track = ""
            msg = "\u23f9\ufe0f Stopped"
        elif command == "NEXT":
            if self.music_playing:
                pool = [s for s in _MUSIC_TRACKS if s != self.current_track]
                self.current_track = random.choice(pool) if pool else random.choice(_MUSIC_TRACKS)
                msg = f"\U0001f5ed\ufe0f Next track: {self.current_track}"
            else:
                msg = "\u26a0\ufe0f Nothing playing"
        elif command == "VOLUME_UP":
            self.volume = min(100, self.volume + 10)
            msg = f"\U0001f50a Volume \u2192 {self.volume}%"
        elif command == "VOLUME_DOWN":
            self.volume = max(0, self.volume - 10)
            msg = f"\U0001f509 Volume \u2192 {self.volume}%"
        elif command == "LIGHT_ON":
            self.lights_on = True
            msg = f"\u2705 Lights ON (brightness={self.brightness}%, color={self.color})"
        elif command == "LIGHT_OFF":
            self.lights_on = False
            msg = "\U0001f534 Lights OFF"
        elif command == "BRIGHTNESS":
            try:
                pct = int(slot_value.replace("percent", "").strip())
                self.brightness = max(0, min(100, pct))
                msg = f"\U0001f4a1 Brightness \u2192 {self.brightness}%"
            except Exception:
                msg = "\u26a0\ufe0f Could not parse brightness value"
        elif command == "COLOR":
            self.color = (slot_value or "White").title()
            self.rgb = self.COLOR_MAP.get(self.color, (255, 255, 255))
            msg = f"\U0001f3a8 Color \u2192 {self.color} RGB{self.rgb}"
        elif command == "TIMER":
            secs = self._parse_duration(slot_value) or 30
            self.timer_active, self.timer_total, self.timer_remaining = True, secs, secs
            msg = f"\u23f1\ufe0f  Timer started: {secs}s"
        elif command == "ALARM":
            self.alarms.append(slot_value or "6:00 AM")
            msg = f"\u23f0 Alarm set: {self.alarms[-1]} (total: {len(self.alarms)})"
        elif command == "TEMPERATURE":
            try:
                deg = int(slot_value.replace("degrees", "").strip())
                self.temperature = max(16, min(30, deg))
                msg = f"\U0001f321\ufe0f  Thermostat \u2192 {self.temperature}\u00b0C"
            except Exception:
                msg = "\u26a0\ufe0f Could not parse temperature"
        elif command == "CREATE_REMINDER":
            self.reminders.append(slot_value or "New reminder")
            msg = f"\U0001f4dd Reminder: {self.reminders[-1]}"
        elif command == "LIST_REMINDERS":
            items = ", ".join(self.reminders) if self.reminders else "(none yet)"
            msg = f"\U0001f4cb Reminders: {items}"
        elif command == "WEATHER":
            w = self.weather
            msg = f"\U0001f324\ufe0f  {w['city']}: {w['temp_c']}\u00b0C, {w['condition']}"
        elif command == "TIME":
            msg = f"\U0001f550 {time.strftime('%H:%M:%S')}"
        elif command == "CALL":
            msg = f"\U0001f4de Calling {slot_value or 'contact'}\u2026"
        elif command == "MESSAGE":
            msg = f"\u2709\ufe0f  Opening message to {slot_value or 'contact'}\u2026"
        elif command == "OUT_OF_SCOPE":
            msg = "\u2753 Sorry, I didn't understand that command."
        else:
            msg = f"\u2753 Unknown command: {command}"

        # tick timer once per command (simple, matches ui_simulation behavior)
        if self.timer_active and self.timer_remaining > 0:
            self.timer_remaining -= 1
            if self.timer_remaining == 0:
                self.timer_active = False
                msg += "  \u23f0 Timer DONE!"
        return msg

    def to_dict(self) -> dict:
        return {
            "lights_on": self.lights_on, "brightness": self.brightness,
            "color": self.color, "rgb": list(self.rgb),
            "music_playing": self.music_playing, "paused": self.paused,
            "current_track": self.current_track, "volume": self.volume,
            "timer_active": self.timer_active, "timer_remaining": self.timer_remaining,
            "timer_total": self.timer_total, "alarms": list(self.alarms),
            "temperature": self.temperature, "reminders": list(self.reminders),
            "weather": dict(self.weather),
        }


# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="VCM Smart Home")
STATE = SmartHomeState()
SESSION, IN_NAME = None, None
MODEL_NAME = "conformer_s_float.onnx"
WAKE_ENABLED = True


@app.on_event("startup")
async def startup():
    global SESSION, IN_NAME
    SESSION, IN_NAME = load_session(MODEL_NAME)
    print(f"[vcm-web] loaded model: {MODEL_NAME}", flush=True)
    if WAKE_ENABLED:
        await asyncio.to_thread(load_wakeword)
    print(f"[vcm-web] wake word ready: {WW_DETECTOR is not None}", flush=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/api/state")
async def api_state():
    return STATE.to_dict()


@app.post("/api/reset")
async def api_reset():
    global STATE
    STATE = SmartHomeState()
    return STATE.to_dict()


def _classify(y: np.ndarray, sr: int) -> dict:
    """Run the two-head conformer on a waveform.

    Returns intent + slot + probabilities + latency. The slot is decoded
    jointly over the valid slot values for the predicted intent (no ASR).
    """
    x = to_features(y, sr)
    t0 = time.perf_counter()
    outputs = SESSION.run(None, {IN_NAME: x})
    intent_logits = outputs[0][0]
    slot_logits = outputs[1][0] if len(outputs) > 1 else None
    dt_ms = (time.perf_counter() - t0) * 1000.0
    probs = softmax(intent_logits)
    idx = int(probs.argmax())
    intent = INTENTS[idx]

    slot = None
    if slot_logits is not None and intent in INTENT_SLOTS:
        slot = decode_slot(intent, slot_logits.tolist())

    order = np.argsort(probs)[::-1][:3]
    top = [{"label": INTENTS[i], "name": INTENT_NAME[INTENTS[i]],
            "prob": round(float(probs[i]), 4)} for i in order]
    return {
        "intent": intent,
        "name": INTENT_NAME[intent],
        "slot": slot,
        "confidence": round(float(probs.max()), 4),
        "top": top, "latency_ms": round(dt_ms, 1),
    }



@app.get("/music/{filename}")
async def music_file(filename: str):
    """Serve a music file from the assets/music directory."""
    safe = Path(filename).name  # prevent path traversal
    fp = MUSIC_DIR / safe
    if not fp.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    mime = {"mp3": "audio/mpeg", "wav": "audio/wav"}.get(fp.suffix.lstrip("."), "application/octet-stream")
    return FileResponse(fp, media_type=mime, headers={"Cache-Control": "max-age=3600"})


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Browser <-> server voice link (tiny-alexa architecture).

    The browser streams 16 kHz int16 PCM as raw binary WebSocket frames
    (80 ms chunks, exactly what openWakeWord wants). A dedicated worker
    thread owns the wake-word gate and the command state machine so
    inference never blocks the event loop; results come back over an
    asyncio queue.
    """
    await websocket.accept()

    import collections
    import queue

    gate = _WWGate(WW_DETECTOR) if WW_DETECTOR is not None else None
    loop = asyncio.get_running_loop()
    q: "queue.Queue[np.ndarray | None]" = queue.Queue()
    evq: "asyncio.Queue[dict | None]" = asyncio.Queue()

    def emit(ev: dict | None):
        loop.call_soon_threadsafe(evq.put_nowait, ev)

    def _finish_command(y: np.ndarray) -> dict:
        """Classify + act; returns the result payload dict.

        Reads the current global STATE explicitly: assigning to the module
        global elsewhere in this function (the "reset" handler) would turn
        STATE into a local of ws(), and Python forbids reading a local
        before assignment — so reference globals.STATE directly.
        """
        res = _classify(y, SR)
        intent = res["intent"]
        slot = res.get("slot") or ""
        status = globals()["STATE"].execute(intent, slot)
        return {
            "type": "result", "intent": intent, "name": res["name"],
            "slot": slot, "confidence": res["confidence"], "top": res["top"],
            "latency_ms": res["latency_ms"], "status": status,
            "state": globals()["STATE"].to_dict(),
        }

    def worker():
        """tiny-alexa's command state machine, one 80 ms chunk at a time.

        Phases:
          None (idle)  -> feed the wake gate every chunk; keep a rolling
                          history of chunk RMS for the noise floor.
          "tail"       -> skip the end of the wake word (~0.24 s).
          "wait_speech"-> wait for RMS above the adaptive threshold; keep a
                          3-chunk pre-roll so the word's quiet onset survives.
          "recording"  -> accumulate until 0.56 s of trailing silence (or the
                          4.8 s cap); early-exit when the model is >= 0.95
                          sure after 0.16 s of silence.
        """
        rec = None
        levels: "collections.deque[float]" = collections.deque(
            maxlen=NOISE_HISTORY_CHUNKS)
        while True:
            chunk = q.get()
            if chunk is None:
                break

            # Feed the wake gate every chunk, even while recording a command:
            # openWakeWord's reset() only clears scores, not its audio
            # features, so a starved detector would still "hear" the last
            # "hey mycroft" later.
            woke = gate.process(chunk) if gate is not None else False
            f = chunk.astype(np.float32) / 32768.0
            rms = _rms(f)

            if rec is None:
                if woke:
                    nrms = (float(np.percentile(levels, NOISE_PERCENTILE))
                            if levels else 0.0)
                    rec = {"phase": "tail", "tail": 0,
                           "tail_chunks": WAKE_TAIL_CHUNKS,
                           "buf": [], "silent": 0,
                           "preroll": collections.deque(maxlen=PREROLL_CHUNKS),
                           "t0": time.monotonic(),
                           "threshold": min(MAX_SPEECH_THRESHOLD,
                                            max(MIN_SPEECH_RMS,
                                                SPEECH_OVER_NOISE * nrms))}
                    emit({"type": "wake"})
                    print(f"  [wake] noise rms={nrms:.4f} "
                          f"threshold={rec['threshold']:.4f} "
                          f"deaf {WAKE_TAIL_CHUNKS * WW_CHUNK / SR:.2f}s",
                          flush=True)
                else:
                    levels.append(rms)
                continue

            if rec["phase"] == "tail":
                rec["tail"] += 1
                if rec["tail"] >= rec["tail_chunks"]:
                    rec["phase"] = "wait_speech"
                continue

            if rec["phase"] == "wait_speech":
                if rms > rec["threshold"]:
                    rec["phase"] = "recording"
                    rec["buf"] = [*rec["preroll"], f]  # keep the quiet onset
                elif time.monotonic() - rec["t0"] > NO_SPEECH_TIMEOUT_S:
                    rec = None
                    if gate is not None:
                        gate.reset()
                    emit({"type": "idle"})
                else:
                    rec["preroll"].append(f)
                continue

            # phase == "recording"
            rec["buf"].append(f)
            rec["silent"] = rec["silent"] + 1 if rms < rec["threshold"] else 0
            final = (rec["silent"] >= END_SILENCE_CHUNKS
                     or len(rec["buf"]) >= MAX_COMMAND_CHUNKS)
            if not final and rec["silent"] >= EARLY_EXIT_SILENCE_CHUNKS:
                y = np.concatenate(rec["buf"])
                res = _classify(y, SR)
                if res["confidence"] >= EARLY_EXIT_PROB:
                    final = True
            if not final:
                continue

            y = np.concatenate(rec["buf"])
            waited_ms = rec["silent"] * WW_CHUNK / SR * 1000.0
            rec = None
            if gate is not None:
                gate.reset()
            print(f"  [cmd] {len(y) / SR:.2f}s of audio, "
                  f"{waited_ms:.0f} ms of silence", flush=True)
            # Runs in the worker thread (already off the event loop), so no
            # asyncio needed here.
            emit(_finish_command(y))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    async def sender():
        while (ev := await evq.get()) is not None:
            if websocket.client_state.name == "DISCONNECTED":
                break
            await websocket.send_text(json.dumps(ev))

    sender_task = asyncio.create_task(sender())
    print(f"[vcm-web] client connected: {websocket.client}", flush=True)

    # The browser sends 80 ms chunks, but re-chunk anyway: the wake word
    # needs exactly WW_CHUNK samples per call.
    pending = np.zeros(0, dtype=np.int16)

    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            if "bytes" in raw and raw["bytes"] is not None:
                data = raw["bytes"]
                if len(data) % 2:
                    data = data[:-1]
                if not data:
                    continue
                pending = np.concatenate(
                    [pending, np.frombuffer(data, dtype=np.int16)])
                n = len(pending) // WW_CHUNK * WW_CHUNK
                for i in range(0, n, WW_CHUNK):
                    q.put(pending[i:i + WW_CHUNK])
                pending = pending[n:]
            elif "text" in raw and raw["text"] is not None:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "start":
                    if gate is not None:
                        gate.reset()
                    await websocket.send_text(json.dumps(
                        {"type": "ready", "wake": gate is not None}))
                elif kind == "state":
                    await websocket.send_text(json.dumps(
                        {"type": "state", "state": globals()["STATE"].to_dict()}))
                elif kind == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif kind == "reset":
                    globals()["STATE"] = SmartHomeState()
                    await websocket.send_text(json.dumps(
                        {"type": "state",
                         "state": globals()["STATE"].to_dict()}))
    except WebSocketDisconnect:
        pass
    finally:
        q.put(None)
        t.join(2.0)
        emit(None)
        sender_task.cancel()
        print(f"[vcm-web] client disconnected: {websocket.client}", flush=True)


def main():
    global MODEL_NAME, WAKE_ENABLED
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="conformer_s_float.onnx")
    ap.add_argument("--no-wake", action="store_true",
                    help="disable the 'hey mycroft' wake word (tap-to-talk only)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    MODEL_NAME = args.model
    WAKE_ENABLED = not args.no_wake

    import uvicorn
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"[vcm-web] opening {url}  (model: {args.model})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
