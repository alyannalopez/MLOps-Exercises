#!/usr/bin/env python3
"""ME3 localhost web UI server (Step 11 companion).

Runs the REAL full-duplex loop (Step 6/8) and mirrors it to a localhost page:
  - live mic level + VAD score (canvas visualizer)
  - live state machine (listening / thinking / speaking / idle)
  - transcript, route decision, spoken reply, barge-in / backchannel events
  - ME2-style task simulators (lights, music, thermostat, timers, math, ...)
  - a terminal-output mirror (the same lines the CLI prints)

NON-INVASIVE: it wraps the loop with an optional ``event_cb``. The loop's
terminal output is unchanged and still goes to stdout exactly as before --
this server only *observes* it. ``run_full_duplex.py`` is not touched.

Usage:
  PYTHONPATH=src python3 web/server.py
  # then open  http://localhost:8321

The loop runs in a background thread so the HTTP/WS server stays responsive.
Stop the whole thing with Ctrl-C.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from voiceduplex.asr import make_asr            # noqa: E402
from voiceduplex.audioio import make_audio_io   # noqa: E402
from voiceduplex.loop import FullDuplexLoop, TimelineLog  # noqa: E402
from voiceduplex.delegate import build_backend  # noqa: E402
from voiceduplex.tts import make_tts            # noqa: E402
from voiceduplex.vad import make_vad            # noqa: E402
from voiceduplex.smarthome.device import SimulatedSmartHome  # noqa: E402
from voiceduplex.smarthome.weather import weather_response  # noqa: E402

import websockets  # noqa: E402
from websockets.asyncio.server import serve as ws_serve  # noqa: E402


# ---------------------------------------------------------------------------
# Shared state bridging the loop thread -> HTTP/WS server
# ---------------------------------------------------------------------------
class Hub:
    """Holds the live loop state and fans events out to connected clients."""

    def __init__(self) -> None:
        self.state = "loading"            # loading|ready|listening|thinking|speaking|idle|stopped
        self.mic = {"level": 0.0, "score": 0.0, "speaking": False}
        self.last_heard = ""
        self.last_route = ""
        self.last_reply = ""
        self.last_detail = ""
        self.turn = 0
        self.last_ttfa_ms = None
        self.last_asr_ms = None
        self.last_route_ms = None
        self.last_interrupt = ""
        self.last_backchannel = ""
        self.log_lines: list[str] = []    # terminal mirror (bounded)
        self.events: list[dict] = []      # recent structured events (bounded)
        self.clients: set = set()
        self._lock = threading.Lock()
        self.device = SimulatedSmartHome()
        self.started = time.time()

    # -- bounded buffers ----------------------------------------------------
    def _push_log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 400:
                self.log_lines = self.log_lines[-400:]

    def _push_event(self, ev: dict) -> None:
        with self._lock:
            self.events.append(ev)
            if len(self.events) > 200:
                self.events = self.events[-200:]

    # -- the loop observer (runs on the loop thread) ------------------------
    def on_event(self, kind: str, payload) -> None:
        ts = round(time.time() - self.started, 2)
        if kind == "log":
            self._push_log(str(payload))
            return
        ev = {"kind": kind, "t": ts}
        if isinstance(payload, dict):
            ev.update(payload)
        else:
            ev["payload"] = payload

        if kind == "state":
            self.state = payload
        elif kind == "mic":
            self.mic = payload
        elif kind == "heard":
            self.last_heard = payload.get("text", "")
            self.last_asr_ms = payload.get("asr_ms")
        elif kind == "route":
            self.last_route = payload.get("route", "")
            self.last_detail = payload.get("detail", "")
        elif kind == "reply":
            self.last_reply = payload.get("text", "")
        elif kind == "turn_done":
            self.turn = payload.get("turn", self.turn)
            self.last_ttfa_ms = payload.get("ttfa_ms")
            self.last_asr_ms = payload.get("asr_ms", self.last_asr_ms)
            self.last_route_ms = payload.get("route_ms")
        elif kind == "interrupt":
            self.last_interrupt = payload.get("text", "")
        elif kind == "backchannel":
            self.last_backchannel = payload.get("text", "")

        self._push_event(ev)
        self._broadcast({"op": "event", "ev": ev})

    # -- fan-out ------------------------------------------------------------
    def _broadcast(self, msg: dict) -> None:
        data = json.dumps(msg)
        dead = []
        for ws in list(self.clients):
            try:
                # new websockets API: send() is a coroutine; schedule it on the loop
                asyncio.run_coroutine_threadsafe(ws.send(data), self.loop)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "op": "snapshot",
                "state": self.state,
                "mic": self.mic,
                "last_heard": self.last_heard,
                "last_route": self.last_route,
                "last_detail": self.last_detail,
                "last_reply": self.last_reply,
                "turn": self.turn,
                "last_ttfa_ms": self.last_ttfa_ms,
                "last_asr_ms": self.last_asr_ms,
                "last_route_ms": self.last_route_ms,
                "last_interrupt": self.last_interrupt,
                "last_backchannel": self.last_backchannel,
                "log": self.log_lines[-200:],
                "events": self.events[-80:],
                "started": self.started,
            }


HUB = Hub()


# ---------------------------------------------------------------------------
# Model loading (mirrors scripts/run_full_duplex.py exactly)
# ---------------------------------------------------------------------------
def _make_tts(backend: str, tcfg: dict, root: Path):
    if backend == "piper":
        return make_tts(
            "piper",
            model=str(root / tcfg.get("piper_model", "data/voices/en_US-lessac-medium.onnx")),
            speed=float(tcfg.get("speed", 1.0)),
        )
    return make_tts(
        "kokoro",
        python=str(root / tcfg.get("kokoro_python", ".venv-kokoro/bin/python")),
        model=str(root / tcfg.get("kokoro_model", "models/kokoro-v1.0.onnx")),
        voices=str(root / tcfg.get("kokoro_voices", "models/voices-v1.0.bin")),
        voice=tcfg.get("voice", "af_heart"),
        speed=float(tcfg.get("speed", 1.0)),
    )


def build_components(cfg: dict, args):
    a, v, asrcfg, tcfg = cfg["audio"], cfg["vad"], cfg["asr"], cfg["tts"]
    ag = cfg.get("agent", {})
    dx = cfg.get("duplex", {})

    HUB._push_log("Loading models...")
    t0 = time.time()

    audio = make_audio_io(
        a.get("backend"),
        input_device=a.get("input_device"),
        output_device=a.get("output_device"),
        sample_rate=int(a.get("sample_rate", 16000)),
        block=int(a.get("block", 512)),
        playback_rate=int(a.get("playback_rate", 48000)),
    )
    vad = make_vad(
        args.vad or v.get("engine", "silero"),
        threshold=float(v.get("threshold", 0.5)),
        min_speech_frames=int(v.get("min_speech_frames", 4)),
        min_silence_frames=int(v.get("min_silence_frames", 12)),
    )
    asr = make_asr(
        asrcfg.get("backend", "whispercpp"),
        model_path=ROOT / asrcfg.get("model", "models/ggml-base.en.bin"),
        binary=asrcfg.get("binary", "whisper-cli"),
        language=asrcfg.get("language", "en"),
        threads=int(asrcfg.get("threads", 0)),
        beam_size=int(asrcfg.get("beam_size", 5)),
    )
    tts = _make_tts(args.tts or tcfg.get("backend", "kokoro"), tcfg, ROOT)

    HUB._push_log(f"  VAD={vad.__class__.__name__}  ASR={asr.__class__.__name__}  "
                  f"TTS={tts.backend_name}   ({time.time()-t0:.1f}s)")
    HUB._push_log("Warming up...")
    asr.warm()
    tts.warm()
    HUB._push_log("Ready.")

    audio.open()
    log = TimelineLog(args.log)

    barge_in = bool(dx.get("barge_in", True)) and not args.no_barge_in
    delegate = build_backend(cfg)
    dcfg = cfg.get("delegate") or {}
    if delegate.is_available():
        HUB._push_log(f"  delegate: {delegate.name}  (on)")
    else:
        HUB._push_log(f"  delegate: {delegate.name}  (MISSING runner -- will stub)")

    loop = FullDuplexLoop(
        audio, vad, asr, tts, log,
        max_utterance_sec=float(ag.get("max_utterance_sec", 15.0)),
        barge_in=barge_in,
        barge_min_energy=float(dx.get("barge_min_energy", 300.0)),
        backchannel_during_speech=bool(dx.get("backchannel_during_speech", True)),
        play_chunk_sec=float(dx.get("play_chunk_sec", 0.15)),
        delegate=delegate,
        delegate_wait_sec=float(dcfg.get("wait_sec", 120.0)),
        event_cb=HUB.on_event,
    )
    HUB._push_log(f"  barge-in: {'ON' if barge_in else 'OFF'}")
    return loop, audio, log


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------
async def ws_handler(conn):
    HUB.clients.add(conn)
    try:
        await conn.send(json.dumps(HUB.snapshot()))
        async for _ in conn:
            # client->server messages are ignored (UI is read-only live view)
            pass
    finally:
        HUB.clients.discard(conn)


# ---------------------------------------------------------------------------
# HTTP handler (static page + REST for simulators)
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ME3Web/1.0"

    def log_message(self, *a):  # silence default request logging
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (ROOT / "web" / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(HUB.snapshot())
        elif self.path == "/api/device":
            self._json({"state": HUB.device.state, "events": HUB.device.events[-40:]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        action = req.get("action")
        arg = req.get("arg", "")

        dev = HUB.device
        try:
            if action == "light_on":
                resp = dev.light_on()
            elif action == "light_off":
                resp = dev.light_off()
            elif action == "brightness":
                resp = dev.set_brightness(arg)
            elif action == "color":
                resp = dev.set_color(arg)
            elif action == "temp":
                resp = dev.set_temperature(arg)
            elif action == "vol_up":
                resp = dev.volume_up()
            elif action == "vol_down":
                resp = dev.volume_down()
            elif action == "music_play":
                resp = dev.play_music(arg)
            elif action == "music_pause":
                resp = dev.pause()
            elif action == "music_resume":
                resp = dev.resume()
            elif action == "music_stop":
                resp = dev.stop()
            elif action == "music_next":
                resp = dev.next_track()
            elif action == "timer":
                resp = dev.set_timer(arg)
            elif action == "alarm":
                resp = dev.set_alarm(arg)
            elif action == "reminder":
                resp = dev.create_reminder(arg)
            elif action == "list_reminders":
                resp = dev.list_reminders()
            elif action == "call":
                resp = dev.call(arg)
            elif action == "message":
                resp = dev.message(arg)
            elif action == "time":
                resp = dev.tell_time()
            elif action == "date":
                resp = dev.tell_date()
            elif action == "weather":
                resp = weather_response(req.get("city", "Metro Manila"))
            elif action == "math":
                resp = _safe_math(arg)
            else:
                resp = "Unknown action."
        except Exception as e:  # noqa: BLE001
            resp = f"Error: {e}"

        HUB._push_log(f"[web:{action}] {arg!r} -> {resp}")
        self._json({"ok": True, "response": resp,
                    "state": dev.state, "events": dev.events[-40:]})


def _safe_math(expr: str) -> str:
    """Evaluate a simple arithmetic expression safely (no eval of code)."""
    import re
    expr = expr.replace("^", "**").strip()
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return "I can only do basic arithmetic (+ - * /)."
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - sanitized above
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return f"{expr} = {val}."
    except Exception:
        return "I couldn't evaluate that."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="ME3 localhost web UI")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--ws-port", type=int, default=8322)
    ap.add_argument("--vad", choices=["silero", "webrtc"], default=None)
    ap.add_argument("--tts", choices=["kokoro", "piper"], default=None)
    ap.add_argument("--no-barge-in", action="store_true")
    ap.add_argument("--log", default=str(ROOT / "data" / "timeline.jsonl"))
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())

    # --- start the asyncio loop (WS server + broadcast scheduler) ----------
    # The legacy websockets serve() must be *created* inside a thread that has a
    # running event loop, so we build it via a coroutine that runs on the loop
    # thread and signal completion back to the main thread.
    HUB.loop = asyncio.new_event_loop()
    ws_ready = threading.Event()
    ws_holder: dict = {}

    async def _boot_ws():
        ws_holder["server"] = await ws_serve(ws_handler, "127.0.0.1", args.ws_port)
        ws_ready.set()

    def _run_loop_thread():
        asyncio.set_event_loop(HUB.loop)
        HUB.loop.create_task(_boot_ws())
        HUB.loop.run_forever()

    loop_thread = threading.Thread(target=_run_loop_thread, daemon=True)
    loop_thread.start()
    ws_ready.wait(timeout=10)
    ws_server = ws_holder.get("server")

    # --- HTTP server (main thread) ----------------------------------------
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    # --- build models + run the loop in a background thread ---------------
    HUB.state = "loading"
    components = build_components(cfg, args)
    loop_obj, audio, log = components
    HUB.state = "ready"

    def run_loop():
        try:
            loop_obj.run()
        finally:
            HUB.state = "stopped"
            HUB._push_log("[web] loop stopped")

    rt = threading.Thread(target=run_loop, daemon=True)
    rt.start()

    print(f"\n  ME3 web UI  ->  http://localhost:{args.port}")
    print(f"  WebSocket   ->  ws://localhost:{args.ws_port}")
    print(f"  Terminal output is still printed below (unchanged).\n")
    print(f"  Press Ctrl-C to stop.\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] shutting down...")
    finally:
        try:
            loop_obj.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            audio.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            log.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            asyncio.run_coroutine_threadsafe(ws_server.close(), HUB.loop).result(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        HUB.loop.call_soon_threadsafe(HUB.loop.stop)
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
