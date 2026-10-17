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
from fastapi.responses import HTMLResponse
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
WW_PRIMING = 80          # 6.4 s of silence to prime the internal buffers
WW_MIN_GAP = 1.2         # ignore re-triggers within this many seconds
WW_THRESHOLD = 0.5       # same as the terminal demo (runtime/main.py --ww-threshold)


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
              f"thr={WW_THRESHOLD})", flush=True)
        return det
    except Exception as e:
        print(f"[vcm-web] wake word unavailable ({e}); tap-to-talk only", flush=True)
        return None


def ww_reset_buffers():
    """Reset the detector's internal buffers after a command so the wake word
    doesn't re-trigger on the tail of the spoken command."""
    if WW_DETECTOR is None:
        return
    try:
        WW_DETECTOR.reset()
    except Exception:
        pass


# ── music playback (real songs, brief task 8) ──────────────
_MUSIC_TRACKS = [
    "track1_relaxing",   # Isang Daan – Cup of Joe
    "track2_upbeat",     # Kung Payagan – Cup of Joe
    "track3_lofi",       # Lalayo – Cup of Joe
    "track4_demo_song",  # Pamahiin – Cup of Joe
    "track5_alarm",      # Sandali – Cup of Joe
    "na_para_bang",      # Na Para Bang – Mariah Deborah
]
_PLAYER: subprocess.Popen | None = None


def _pick_audio_file(stem: str) -> Path | None:
    for ext in (".mp3", ".wav"):
        p = MUSIC_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _audio_player_cmd(path: Path) -> list[str] | None:
    if path.suffix == ".wav":
        if shutil.which("afplay"):
            return ["afplay", str(path)]
        if shutil.which("aplay"):
            return ["aplay", "-q", str(path)]
    elif path.suffix == ".mp3":
        for cmd in (["mpg123", "-q", str(path)], ["ffplay", "-nodisp", "-autoexit", str(path)]):
            if shutil.which(cmd[0]):
                return cmd
    return None


def _play_audio_bg(path: Path) -> bool:
    global _PLAYER
    _stop_player()
    cmd = _audio_player_cmd(path)
    if cmd is None:
        return False
    try:
        _PLAYER = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _stop_player() -> None:
    global _PLAYER
    if _PLAYER is not None:
        try:
            _PLAYER.terminate()
        except Exception:
            pass
        _PLAYER = None


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
            if _pick_audio_file(stem) is not None:
                _play_audio_file_bg(stem)
            msg = f"\U0001f3b5 Playing: {stem}"
        elif command == "PAUSE":
            if self.music_playing:
                self.paused = not self.paused
                if self.paused:
                    _stop_player()
                    msg = "\u23f8\ufe0f Paused"
                else:
                    _play_audio_file_bg(self.current_track)
                    msg = "\u25b6\ufe0f Resumed"
            else:
                msg = "\u26a0\ufe0f Nothing playing"
        elif command == "STOP":
            _stop_player()
            self.music_playing, self.paused = False, False
            self.current_track = ""
            msg = "\u23f9\ufe0f Stopped"
        elif command == "NEXT":
            if self.music_playing:
                pool = [s for s in _MUSIC_TRACKS if s != self.current_track]
                self.current_track = random.choice(pool) if pool else random.choice(_MUSIC_TRACKS)
                _play_audio_file_bg(self.current_track)
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


def _play_audio_file_bg(stem: str) -> bool:
    path = _pick_audio_file(stem)
    if path is None:
        return False
    return _play_audio_bg(path)


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
    _stop_player()
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


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    global STATE
    await websocket.accept()
    wake_buf: list[np.ndarray] = []   # rolling window for the wake word
    cmd_buf: list[np.ndarray] = []    # recorded command after a trigger
    phase = "idle"                    # idle -> recording -> busy
    last_trigger = 0.0
    primed = False

    def _feed_wake(pcm: np.ndarray):
        """Feed 16 kHz float samples to the wake-word detector; True on trigger.
        Same detector + threshold as the terminal demo."""
        nonlocal last_trigger
        if WW_DETECTOR is None:
            return False
        triggered = False
        for i in range(0, len(pcm) - WW_CHUNK + 1, WW_CHUNK):
            chunk = pcm[i:i + WW_CHUNK]
            ev = WW_DETECTOR.detect(chunk)
            if ev.triggered:
                if time.time() - last_trigger > WW_MIN_GAP:
                    last_trigger = time.time()
                    triggered = True
        return triggered

    def _finish_command(y: np.ndarray):
        """Classify + act; returns the result payload dict."""
        res = _classify(y, SR)
        intent = res["intent"]
        slot = res.get("slot") or ""
        status = STATE.execute(intent, slot)
        return {
            "type": "result", "intent": intent, "name": res["name"],
            "slot": slot, "confidence": res["confidence"], "top": res["top"],
            "latency_ms": res["latency_ms"], "status": status,
            "state": STATE.to_dict(),
        }

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            if kind == "start":
                wake_buf.clear(); cmd_buf.clear()
                phase = "idle"; primed = False
                ww_reset_buffers()
                await websocket.send_text(json.dumps({"type": "ready",
                                                      "wake": WW_DETECTOR is not None}))

            elif kind == "audio":
                pcm = np.frombuffer(base64.b64decode(msg["data"]),
                                    dtype=np.int16).astype(np.float32) / 32768.0
                if phase == "idle":
                    # prime the wake-word buffers with the first ~6.4 s of audio
                    if not primed:
                        wake_buf.append(pcm)
                        if sum(len(b) for b in wake_buf) >= WW_PRIMING * WW_CHUNK:
                            for b in wake_buf:
                                for i in range(0, len(b) - WW_CHUNK + 1, WW_CHUNK):
                                    WW_DETECTOR.detect(b[i:i + WW_CHUNK])
                            wake_buf.clear()
                            primed = True
                        continue
                    if _feed_wake(pcm):
                        # ---- wake word heard: start capturing the command ----
                        phase = "recording"
                        cmd_buf.clear()
                        await websocket.send_text(json.dumps({"type": "wake"}))
                elif phase == "recording":
                    cmd_buf.append(pcm)
                    # auto-stop after ~1.2 s of command audio
                    if sum(len(b) for b in cmd_buf) >= int(SR * 1.2):
                        phase = "busy"
                        y = np.concatenate(cmd_buf) if cmd_buf else np.zeros(SR, dtype=np.float32)
                        cmd_buf.clear()
                        payload = await asyncio.to_thread(_finish_command, y)
                        await websocket.send_text(json.dumps(payload))
                        ww_reset_buffers()
                        phase = "idle"

            elif kind == "stop":
                # manual stop (tap-to-talk fallback / dev)
                if phase == "recording" or (cmd_buf):
                    phase = "busy"
                    y = np.concatenate(cmd_buf) if cmd_buf else np.zeros(SR, dtype=np.float32)
                    cmd_buf.clear()
                    payload = await asyncio.to_thread(_finish_command, y)
                    await websocket.send_text(json.dumps(payload))
                    ww_reset_buffers()
                    phase = "idle"

            elif kind == "state":
                await websocket.send_text(json.dumps({"type": "state",
                                                      "state": STATE.to_dict()}))

            elif kind == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass


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
