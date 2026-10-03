#!/usr/bin/env python3
"""VCM Smart Home — web demo server.

FastAPI + WebSocket app that wires the browser UI to the CRNN voice-command
model and a server-side smart-home state machine.

Flow:
  browser (white/gold/blue UI)
    --WebSocket-->  this server
        1. receives 16 kHz PCM16 frames from the mic (or a recorded take)
        2. on a "take" it runs the EXACT training feature pipeline
           (loudest-1s crop -> MFCC(40, hop=512) -> 100x40) through the CRNN
        3. maps the predicted intent onto a SmartHomeState (same actions as
           code/ui_simulation.py) and streams the result + new state back

Run:
  python3 web/app.py                 # http://localhost:8000  (opens browser)
  python3 web/app.py --port 8010     # custom port
  python3 web/app.py --model crnn_int8.onnx
"""
from __future__ import annotations
import argparse, asyncio, json, time, threading, webbrowser
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

# ── constants (must match prep_data.py / run_demo.py) ────────────────────────
SR = 16000
N_FRAMES = 100
N_MFCC = 40

LABELS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE", "STOP", "NEXT",
    "VOLUME_UP", "VOLUME_DOWN", "CALL", "MESSAGE", "LIST_REMINDERS", "TIMER", "ALARM",
    "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER", "OUT_OF_SCOPE",
]

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

# ── feature pipeline (verbatim logic from prep_data.py / run_demo.py) ────────
def loudest_crop(y: np.ndarray, sr: int = SR, dur: float = 1.0) -> np.ndarray:
    win = int(sr * dur)
    if len(y) <= win:
        return np.pad(y, (0, win - len(y))) if len(y) < win else y
    hop = win // 4
    best_start, best_amp = 0, -1
    for start in range(0, len(y) - win + 1, hop):
        amp = float(np.max(np.abs(y[start:start + win])))
        if amp > best_amp:
            best_amp, best_start = start, amp
    return y[best_start:best_start + win]


def to_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    import librosa
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = loudest_crop(y)
    mfc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=512)
    xf = mfc.T
    if xf.shape[0] >= N_FRAMES:
        xf = xf[:N_FRAMES]
    else:
        xf = np.pad(xf, ((0, N_FRAMES - xf.shape[0]), (0, 0)))
    return xf.astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ── inference session ────────────────────────────────────────────────────────
def load_session(model_name: str):
    import onnxruntime as ort
    path = MODEL_DIR / model_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available: "
                                f"{[p.name for p in MODEL_DIR.glob('*.onnx')]}")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


# wake word (openWakeWord "hey jarvis")
WW_MODEL = None
WW_THRESHOLD = 0.5
WW_CHUNK = 1280          # 80 ms @ 16 kHz (openWakeWord native frame size)
WW_PRIMING = 80          # 6.4 s of silence to prime the internal buffers
WW_MIN_GAP = 1.2         # ignore re-triggers within this many seconds


def load_wakeword():
    """Load the pretrained 'hey jarvis' openWakeWord model. Returns the model
    or None if unavailable (server then falls back to tap-to-talk)."""
    global WW_MODEL
    if WW_MODEL is not None:
        return WW_MODEL
    try:
        from openwakeword.model import Model
        from openwakeword.utils import download_models
        try:
            download_models(model_names=["hey_jarvis"])
        except TypeError:
            download_models(["hey_jarvis"])
        except Exception as e:
            print(f"[vcm-web] wake-word model download warning: {e}")
        m = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        m.predict(np.zeros(WW_CHUNK, dtype=np.float32))   # warm up
        WW_MODEL = m
        print("[vcm-web] wake word 'hey jarvis' ready (openWakeWord)")
        return m
    except Exception as e:
        print(f"[vcm-web] wake word UNAVAILABLE ({e}); falling back to tap-to-talk")
        return None


def ww_reset_buffers():
    """Re-prime the model's internal buffers after a command so the wake word
    doesn't re-trigger on the tail of the spoken command."""
    if WW_MODEL is None:
        return
    try:
        for _ in range(WW_PRIMING):
            WW_MODEL.predict(np.zeros(WW_CHUNK, dtype=np.float32))
    except Exception:
        pass


# ── smart home state (mirrors code/ui_simulation.py actions) ─────────────────
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
        s = s.lower().strip()
        if "minute" in s:
            num = int(''.join(c for c in s if c.isdigit()) or 1)
            return num * 60
        if "second" in s:
            return int(''.join(c for c in s if c.isdigit()) or 0)
        return 0

    def execute(self, command: str, slot_value: str = "") -> str:
        """Apply a command to state; returns a human-readable status message."""
        if command == "PLAY_MUSIC":
            self.music_playing, self.paused = True, False
            self.current_track = slot_value or "Now Playing"
            msg = f"🎵 Playing: {self.current_track}"
        elif command == "PAUSE":
            if self.music_playing:
                self.paused = not self.paused
                msg = ("⏸️ Paused" if self.paused else "▶️ Resumed")
            else:
                msg = "⚠️ Nothing playing"
        elif command == "STOP":
            self.music_playing, self.paused = False, False
            self.current_track = ""
            msg = "⏹️ Stopped"
        elif command == "NEXT":
            if self.music_playing:
                self.current_track = "Next Track"
                msg = "⏭️ Next track"
            else:
                msg = "⚠️ Nothing playing"
        elif command == "VOLUME_UP":
            self.volume = min(100, self.volume + 10)
            msg = f"🔊 Volume → {self.volume}%"
        elif command == "VOLUME_DOWN":
            self.volume = max(0, self.volume - 10)
            msg = f"🔉 Volume → {self.volume}%"
        elif command == "LIGHT_ON":
            self.lights_on = True
            msg = f"✅ Lights ON (brightness={self.brightness}%, color={self.color})"
        elif command == "LIGHT_OFF":
            self.lights_on = False
            msg = "🔴 Lights OFF"
        elif command == "BRIGHTNESS":
            try:
                pct = int(slot_value.replace("percent", "").strip())
                self.brightness = max(0, min(100, pct))
                msg = f"💡 Brightness → {self.brightness}%"
            except Exception:
                msg = "⚠️ Could not parse brightness value"
        elif command == "COLOR":
            self.color = slot_value.title()
            self.rgb = self.COLOR_MAP.get(self.color, (255, 255, 255))
            msg = f"🎨 Color → {self.color} RGB{self.rgb}"
        elif command == "TIMER":
            secs = self._parse_duration(slot_value) or 30
            self.timer_active, self.timer_total, self.timer_remaining = True, secs, secs
            msg = f"⏱️  Timer started: {secs}s"
        elif command == "ALARM":
            self.alarms.append(slot_value or "6:00 AM")
            msg = f"⏰ Alarm set: {self.alarms[-1]} (total: {len(self.alarms)})"
        elif command == "TEMPERATURE":
            try:
                deg = int(slot_value.replace("degrees", "").strip())
                self.temperature = max(16, min(30, deg))
                msg = f"🌡️  Thermostat → {self.temperature}°C"
            except Exception:
                msg = "⚠️ Could not parse temperature"
        elif command == "CREATE_REMINDER":
            self.reminders.append(slot_value or "New reminder")
            msg = f"📝 Reminder: {self.reminders[-1]}"
        elif command == "LIST_REMINDERS":
            items = ", ".join(self.reminders) if self.reminders else "(none yet)"
            msg = f"📋 Reminders: {items}"
        elif command == "WEATHER":
            w = self.weather
            msg = f"🌤️  {w['city']}: {w['temp_c']}°C, {w['condition']}"
        elif command == "TIME":
            msg = f"🕐 {time.strftime('%H:%M:%S')}"
        elif command == "CALL":
            msg = f"📞 Calling {slot_value or 'contact'}…"
        elif command == "MESSAGE":
            msg = f"✉️  Opening message to {slot_value or 'contact'}…"
        elif command == "OUT_OF_SCOPE":
            msg = "❓ Sorry, I didn't understand that command."
        else:
            msg = f"❓ Unknown command: {command}"

        # tick timer once per command (simple, matches ui_simulation behavior)
        if self.timer_active and self.timer_remaining > 0:
            self.timer_remaining -= 1
            if self.timer_remaining == 0:
                self.timer_active = False
                msg += "  ⏰ Timer DONE!"
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


# ── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="VCM Smart Home")
STATE = SmartHomeState()
SESSION = None
IN_NAME = None
MODEL_NAME = "crnn_float.onnx"
WAKE_ENABLED = True


@app.on_event("startup")
async def _startup():
    global SESSION, IN_NAME
    SESSION, IN_NAME = load_session(MODEL_NAME)
    print(f"[vcm-web] loaded model: {MODEL_NAME}")
    if WAKE_ENABLED:
        await asyncio.to_thread(load_wakeword)


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
    """Run the CRNN on a waveform; returns intent + probabilities + latency."""
    x = to_features(y, sr)[None]
    t0 = time.perf_counter()
    logits = SESSION.run(None, {IN_NAME: x})[0][0]
    dt_ms = (time.perf_counter() - t0) * 1000.0
    probs = softmax(logits)
    order = np.argsort(probs)[::-1][:3]
    top = [{"label": LABELS[i], "name": INTENT_NAME[LABELS[i]],
            "prob": round(float(probs[i]), 4)} for i in order]
    return {
        "intent": LABELS[int(probs.argmax())],
        "name": INTENT_NAME[LABELS[int(probs.argmax())]],
        "confidence": round(float(probs.max()), 4),
        "top": top, "latency_ms": round(dt_ms, 1),
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    global STATE
    await websocket.accept()
    import base64
    wake_buf: list[np.ndarray] = []   # rolling window for the wake word
    cmd_buf: list[np.ndarray] = []    # recorded command after a trigger
    phase = "idle"                    # idle -> recording -> busy
    last_trigger = 0.0
    primed = False

    def _feed_wake(pcm: np.ndarray):
        """Feed 16 kHz float samples to openWakeWord; returns True on trigger."""
        nonlocal last_trigger
        if WW_MODEL is None:
            return False
        triggered = False
        for i in range(0, len(pcm) - WW_CHUNK + 1, WW_CHUNK):
            chunk = pcm[i:i + WW_CHUNK]
            scores = WW_MODEL.predict(chunk)
            if any(v > WW_THRESHOLD for v in scores.values()):
                if time.time() - last_trigger > WW_MIN_GAP:
                    last_trigger = time.time()
                    triggered = True
        return triggered

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
                                                      "wake": WW_MODEL is not None}))

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
                                    WW_MODEL.predict(b[i:i + WW_CHUNK])
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
                        res = await asyncio.to_thread(_classify, y, SR)
                        intent = res["intent"]
                        status = STATE.execute(intent, _guess_slot(intent))
                        await websocket.send_text(json.dumps({
                            "type": "result", "intent": intent, "name": res["name"],
                            "confidence": res["confidence"], "top": res["top"],
                            "latency_ms": res["latency_ms"], "status": status,
                            "state": STATE.to_dict(),
                        }))
                        ww_reset_buffers()
                        phase = "idle"

            elif kind == "stop":
                # manual stop (tap-to-talk fallback / dev)
                if phase == "recording" or (cmd_buf):
                    phase = "busy"
                    y = np.concatenate(cmd_buf) if cmd_buf else np.zeros(SR, dtype=np.float32)
                    cmd_buf.clear()
                    res = await asyncio.to_thread(_classify, y, SR)
                    intent = res["intent"]
                    status = STATE.execute(intent, _guess_slot(intent))
                    await websocket.send_text(json.dumps({
                        "type": "result", "intent": intent, "name": res["name"],
                        "confidence": res["confidence"], "top": res["top"],
                        "latency_ms": res["latency_ms"], "status": status,
                        "state": STATE.to_dict(),
                    }))
                    ww_reset_buffers()
                    phase = "idle"

            elif kind == "reset":
                STATE = SmartHomeState()
                await websocket.send_text(json.dumps({"type": "state",
                                                      "state": STATE.to_dict()}))

            elif kind == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass


def _guess_slot(intent: str) -> str:
    """The CRNN outputs intent only (no slot). Provide sensible defaults so the
    smart-home actions behave sensibly in the live demo."""
    return {
        "BRIGHTNESS": "60", "COLOR": "Blue", "TIMER": "30 seconds",
        "ALARM": "6:00 AM", "TEMPERATURE": "22",
        "CREATE_REMINDER": "Drink water", "PLAY_MUSIC": "SoundHelix-Song-4",
        "CALL": "Mom", "MESSAGE": "Dad",
    }.get(intent, "")


def main():
    global MODEL_NAME, WAKE_ENABLED
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="crnn_float.onnx")
    ap.add_argument("--no-wake", action="store_true",
                    help="disable the 'hey jarvis' wake word (tap-to-talk only)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    MODEL_NAME = args.model
    WAKE_ENABLED = not args.no_wake

    import uvicorn
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"[vcm-web] opening {url}  (model: {args.model})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
