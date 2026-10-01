#!/usr/bin/env python3
"""VCM live demo (Step 3) — Raspberry Pi 4/5.

Streams the USB microphone, detects a spoken command with energy VAD,
classifies it with the INT8 ONNX model, prints the intent + timing, and
loops. This is the on-device real-time proof the exercise asks for.

Run on the Pi:
    pip install onnxruntime numpy
    python live_demo.py --model ../models/crnn_..._int8.onnx --device /dev/snd/pcm0.0

The same file runs on the Mac for a quick sanity check (uses the default mic).
"""
from __future__ import annotations
import argparse, time, sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcm_infer import VCM, TARGET_SR, WINDOW_S, rms_db

FRAME_MS   = 30
MIN_SPEECH = 0.25      # s of voiced audio before we commit
MAX_CMD    = 5.0       # s cap per command
NOISE_DB   = -40.0     # adaptive floor start
TRIGGER_DB = 12.0      # dB above floor to count as speech


def _open_broadcast(url: str | None):
    """Open a fire-and-forget WebSocket to the action handler. Returns a
    send_fn(dict) or None. Non-blocking: a dead handler never crashes the Pi."""
    if not url:
        return None
    try:
        import websocket  # websocket-client (sync, simple, Pi-friendly)
    except ImportError:
        print("[warn] pip install websocket-client  (broadcast disabled)")
        return None
    try:
        ws = websocket.create_connection(url, timeout=3)
        print(f"[broadcast] connected to {url}")
        def send(d):
            try:
                ws.send(json.dumps(d))
            except Exception:
                pass  # never let a dropped handler kill the demo
        return send
    except Exception as e:
        print(f"[warn] broadcast connect failed ({e}); continuing locally")
        return None


_TR = None  # lazy ASR transcriber (loaded on first --numbers use)

_ASR_MODEL = None

def _get_transcriber():
    """Lazily build the whisper.cpp transcriber for slot-filling (--numbers)."""
    global _TR
    if _TR is None:
        from asr import Transcriber, default_model_path
        _TR = Transcriber(_ASR_MODEL or default_model_path())
    return _TR


_SLOT_INTENTS = ("SET_TIMER", "SET_ALARM", "DIM_COLOR_LIGHTS", "THERMOSTAT",
                 "REMINDERS_LISTS")

def _extract_slot(intent: str, pcm: np.ndarray) -> dict | None:
    """For numeric intents, run the small ASR pass and pull out the slot.

    Covers: SET_TIMER, SET_ALARM, DIM_COLOR_LIGHTS (brightness % + color),
    THERMOSTAT, REMINDERS_LISTS (task, e.g. "Remind me to Study").
    Returns a slot dict or None. Only invoked when --numbers is on, so the
    base demo stays 340 KB / instant.
    """
    if intent not in _SLOT_INTENTS:
        return None
    tr = _get_transcriber()
    if not tr.available:
        print("  [numbers] ASR unavailable (whisper-cli/model missing) -> default")
        return None
    text = tr.transcribe_pcm(pcm, TARGET_SR)
    import numparse
    slot = numparse.extract_slot(text, intent)
    if slot.get("transcript"):
        print(f"  [numbers] heard: {slot['transcript']!r}")
    # return slot if it has any actionable key
    if any(k in slot for k in ("duration", "clock", "percent", "direction",
                               "temp", "color", "task")):
        return slot
    return None


def open_stream(device: str | None, sr: int = TARGET_SR, block: int = 480):
    """Open a 16 kHz mono int16 stream. Returns (reader_fn, closer_fn)."""
    try:
        import sounddevice as sd
        q = []
        def cb(indata, frames, t, status):
            # RawInputStream gives a cffi buffer; convert to numpy properly
            arr = np.frombuffer(indata, dtype=np.int16).copy()
            q.append(arr)
        stream = sd.RawInputStream(samplerate=sr, blocksize=block, dtype="int16",
                                   channels=1, callback=cb,
                                   device=device)
        stream.start()
        def read(n_blocks=1):
            out = []
            while len(q) < n_blocks and len(out) < n_blocks:
                time.sleep(0.005)
            while q:
                out.append(q.pop(0))
            if not out:
                return np.zeros(block, dtype=np.int16)
            return np.concatenate(out)[:block * n_blocks]
        return read, lambda: stream.stop() or stream.close()
    except Exception as e:
        print(f"[warn] sounddevice unavailable ({e}); falling back to arecord")
        import subprocess
        proc = subprocess.Popen(
            ["arecord", "-f", "S16_LE", "-r", str(sr), "-c", "1",
             "-d", "0", "-q"] + (["-D", device] if device else []),
            stdout=subprocess.PIPE)
        def read(n_blocks=1):
            chunk = proc.stdout.read(block * 2 * n_blocks)
            if not chunk:
                return np.zeros(block, dtype=np.int16)
            return np.frombuffer(chunk, dtype=np.int16)
        return read, lambda: (proc.terminate(), proc.wait())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None, help="sounddevice index or ALSA name")
    ap.add_argument("--trigger-db", type=float, default=TRIGGER_DB)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--broadcast", default=None,
                    help="ws://host:port to send each intent (e.g. ws://192.168.1.20:8765)")
    ap.add_argument("--simulate", action="store_true",
                    help="run the simulated smart-home hub IN-PROCESS (no websockets needed)")
    ap.add_argument("--numbers", action="store_true",
                    help="enable slot-filling: recognize the NUMBER in timer/alarm "
                         "commands (adds a ~78 MB whisper.cpp ASR pass, ~300 ms)")
    ap.add_argument("--asr-model", default=None,
                    help="path to a GGML whisper model (default: ../asr/ggml-tiny.en.bin)")
    ap.add_argument("--wakeword", action="store_true",
                    help="require a wakeword before listening (openwakeword)")
    ap.add_argument("--wake", default="alexa,hey_jarvis,hey_mycroft,hey_rhasspy,timer",
                    help="comma-separated wakeword models (default: all five)")
    ap.add_argument("--wake-threshold", type=float, default=0.5,
                    help="wakeword detection threshold 0-1 (default: 0.5)")
    a = ap.parse_args()

    global _ASR_MODEL
    _ASR_MODEL = a.asr_model
    vcm = VCM(a.model)
    bc = _open_broadcast(a.broadcast) if a.broadcast else None
    if bc:
        print(f"[broadcast] sending intents to {a.broadcast}\n")

    # --simulate: in-process simulated home (zero extra deps). --broadcast wins
    # if both are given (remote handler takes precedence).
    hub = None
    if a.simulate and not bc:
        from mac_handler import HubSimulator
        hub = HubSimulator()
        print("[simulate] in-process simulated home active. "
              "Say a command and watch the state change.\n")
    read, close = open_stream(a.device)
    block = 480  # 30 ms @16k

    # ---- wakeword gate (optional) ----
    gate = None
    if a.wakeword:
        from wakeword import WakeGate
        wake_names = [w.strip() for w in a.wake.split(",") if w.strip()]
        gate = WakeGate(wakeword_names=wake_names,
                        threshold=a.wake_threshold,
                        quiet=a.quiet)
        print(f"Wakeword active. Say one of: {', '.join(wake_names)}")
        print("Then speak your command. Ctrl-C to quit.\n")
    else:
        print(f"Listening on {a.device or 'default'} mic. Say a command...\n")

    buf = np.array([], dtype=np.int16)
    voiced = 0.0
    floor = NOISE_DB
    t_start = None

    try:
        while True:
            blk = read(1)
            if len(blk) == 0:
                continue

            # ---- wakeword mode: gate controls when we capture ----
            if gate is not None:
                cmd_pcm = gate.feed(blk)
                if cmd_pcm is None:
                    continue  # still idle or still capturing
                # We have a full command — classify it
                buf = cmd_pcm
            else:
                # ---- legacy mode: energy VAD directly ----
                db = rms_db(blk)
                if len(buf) == 0:
                    # Idle: adapt noise floor, wait for speech
                    if db < floor + a.trigger_db * 0.5:
                        floor = 0.95 * floor + 0.05 * db
                    if db > floor + a.trigger_db:
                        buf = blk
                        voiced = FRAME_MS / 1000
                        t_start = time.perf_counter()
                        if not a.quiet:
                            print("\r>> speech detected, capturing...", end="", flush=True)
                    continue
                # Capturing: keep appending until we have >= 1s (model window)
                buf = np.concatenate([buf, blk])
                voiced += FRAME_MS / 1000
                elapsed = time.perf_counter() - t_start
                if len(buf) < int(WINDOW_S * TARGET_SR) and elapsed <= MAX_CMD:
                    continue
                # fall through to classify

            # ---- classify (shared by both modes) ----
            r = vcm.classify_pcm(buf)
            ms = r["timings_ms"]
            slot = _extract_slot(r["intent"], buf) if a.numbers else None
            if bc:
                payload = {"intent": r["intent"], "confidence": round(r["confidence"], 3)}
                if slot:
                    payload["slot"] = slot
                bc(payload)
            # REJECT = model didn't recognise it
            if r["intent"] == "REJECT":
                line = f"  \u26a0\ufe0f UNKNOWN  (didn't recognise  conf={r['confidence']:.2f})"
            else:
                line = (f"  [{r['intent']}]  conf={r['confidence']:.2f}  "
                        f"feat={ms['feature']:.1f}ms  infer={ms['infer']:.2f}ms  "
                        f"TOTAL={ms['total']:.1f}ms")
            print(line)
            if hub:
                hub.handle_intent(r["intent"], r["confidence"], slot)
                if not a.quiet:
                    os.system("clear" if os.name != "nt" else "cls")
                    print(hub.render())
                    print()
            buf = np.array([], dtype=np.int16)
            voiced = 0.0
            t_start = None
    except KeyboardInterrupt:
        print("\nStopping.")
        if hub:
            os.system("clear" if os.name != "nt" else "cls")
            print("Final simulated-home state:\n")
            print(hub.render())
    finally:
        close()


if __name__ == "__main__":
    main()
