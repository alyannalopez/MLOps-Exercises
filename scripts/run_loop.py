#!/usr/bin/env python3
"""Step 5 success check — run the half-duplex voice loop.

Usage:
  PYTHONPATH=src python3 scripts/run_loop.py
  PYTHONPATH=src python3 scripts/run_loop.py --tts piper        # backup TTS
  PYTHONPATH=src python3 scripts/run_loop.py --vad webrtc       # backup VAD
  PYTHONPATH=src python3 scripts/run_loop.py --max-turns 3      # auto-stop after N turns

What to do:
  1. Run it. Wait for the banner.
  2. Speak a SHORT sentence, then STOP (silence ends your turn).
  3. Watch the console: it should print  heard -> route -> speaking, then play
     a reply out loud.
  4. Try a few:
       "what time is it"          -> quick_tool  -> "It's ..."
       "what's 12 times 8"        -> quick_tool  -> "That's 96."
       "what date is today"       -> quick_tool  -> "Today is ..."
       "mm-hm"                    -> backchannel -> "Mm-hm."
       "tell me a joke"           -> delegate    -> (stub reply)
  5. Quit: Ctrl-C, or type q + Enter.

Afterwards, inspect the timeline:
  cat data/timeline.jsonl | python3 -m json.tool --compact
  (each line = one turn with T0..T8 + the A-family latencies)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from voiceduplex.asr import make_asr          # noqa: E402
from voiceduplex.audioio import make_audio_io  # noqa: E402
from voiceduplex.loop import HalfDuplexLoop, TimelineLog  # noqa: E402
from voiceduplex.delegate import build_backend  # noqa: E402
from voiceduplex.tts import make_tts          # noqa: E402
from voiceduplex.vad import make_vad          # noqa: E402


def load_cfg() -> dict:
    p = ROOT / "config" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="ME3 half-duplex loop")
    ap.add_argument("--vad", choices=["silero", "webrtc"], default=None)
    ap.add_argument("--tts", choices=["kokoro", "piper"], default=None)
    ap.add_argument("--max-turns", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--log", default=str(ROOT / "data" / "timeline.jsonl"))
    args = ap.parse_args()

    cfg = load_cfg()
    a, v, asrcfg, tcfg = cfg["audio"], cfg["vad"], cfg["asr"], cfg["tts"]

    print("Loading models...", flush=True)
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

    print(f"  VAD={vad.__class__.__name__}  ASR={asr.__class__.__name__}  "
          f"TTS={tts.backend_name}   ({time.time()-t0:.1f}s)\n", flush=True)

    # warm models once (off the mic)
    print("Warming up...", flush=True)
    asr.warm()
    tts.warm()
    print("Ready.\n", flush=True)

    audio.open()
    log = TimelineLog(args.log)

    # Step 8 — delegation backend (OnIt by default; see config `delegate:`).
    delegate = build_backend(cfg)
    dcfg = cfg.get("delegate") or {}
    if delegate.is_available():
        print(f"  delegate: {delegate.name}  (on)", flush=True)
    else:
        print(f"  delegate: {delegate.name}  (MISSING runner — will stub)", flush=True)

    loop = HalfDuplexLoop(
        audio, vad, asr, tts, log,
        max_utterance_sec=float((cfg.get("agent") or {}).get("max_utterance_sec", 15.0)),
        delegate=delegate,
        delegate_wait_sec=float(dcfg.get("wait_sec", 120.0)),
    )

    try:
        if args.max_turns > 0:
            for _ in range(args.max_turns):
                if not loop._running and loop._turn >= args.max_turns:
                    break
                loop._one_turn()
        else:
            loop.run()
    finally:
        audio.close()
        log.close()
        print(f"\nTimeline written to: {args.log}")
    return 0


def _make_tts(backend: str, tcfg: dict, ROOT: Path):
    """Build the right TTS with the right kwargs (avoid cross-backend kwarg leaks)."""
    if backend == "piper":
        return make_tts(
            "piper",
            model=str(ROOT / tcfg.get("piper_model", "data/voices/en_US-lessac-medium.onnx")),
            speed=float(tcfg.get("speed", 1.0)),
        )
    return make_tts(
        "kokoro",
        python=str(ROOT / tcfg.get("kokoro_python", ".venv-kokoro/bin/python")),
        model=str(ROOT / tcfg.get("kokoro_model", "models/kokoro-v1.0.onnx")),
        voices=str(ROOT / tcfg.get("kokoro_voices", "models/voices-v1.0.bin")),
        voice=tcfg.get("voice", "af_heart"),
        speed=float(tcfg.get("speed", 1.0)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
