#!/usr/bin/env python3
"""STEP 4 test — TTS (file-based, no mic/speaker needed).

Synthesizes a sentence with the chosen backend and reports the output sample
rate, audio duration, compute time, and the real-time factor (RTF). Writes the
result to data/audio/ so you can listen to it. Deterministic and repeatable —
ideal for the Pi (headless) and for regression testing.

Usage:
  python scripts/test_tts.py                       # uses tts.backend from config
  python scripts/test_tts.py --backend piper       # force Piper (primary)
  python scripts/test_tts.py --backend kokoro      # force Kokoro (backup)
  python scripts/test_tts.py --backend fallback    # primary->backup (default)
  python scripts/test_tts.py --voice af_sarah      # Kokoro voice id
  python scripts/test_tts.py --text "Your words"   # custom sentence
  python scripts/test_tts.py --no-warm             # skip the warm-up run

Success criteria:
  - A non-empty WAV is produced and is audible when played.
  - RTF < 1.0 on the Mac (faster than real-time). On the Pi expect RTF ~1-3
    for Kokoro; Piper stays well under 1.
  - Both backends produce valid audio (run once per backend to compare).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from voiceduplex.tts import make_tts

DEFAULT_TEXT = "Hello, this is a test of the text to speech module. The voice is ready."


def main() -> int:
    a = argparse.ArgumentParser(description="TTS offline test (Step 4)")
    a.add_argument("--backend", default=None, help="piper | kokoro | fallback (default: from config)")
    a.add_argument("--voice", default=None, help="Kokoro voice id, e.g. af_heart (default: from config)")
    a.add_argument("--text", default=DEFAULT_TEXT, help="sentence to synthesize")
    a.add_argument("--out", default=None, help="output WAV path (default: data/audio/tts_<backend>.wav)")
    a.add_argument("--no-warm", action="store_true", help="skip the warm-up run")
    args = a.parse_args()

    cfg = load_cfg()
    tcfg = cfg.get("tts", {})
    backend = args.backend or tcfg.get("backend", "piper")
    voice = args.voice or tcfg.get("voice", "af_heart")

    # Build kwargs per backend from config so overrides flow through cleanly.
    if backend == "fallback":
        kwargs = dict(
            primary=tcfg.get("primary", "piper"),
            backup=tcfg.get("backup", "kokoro"),
            primary_kwargs=(dict(
                model=tcfg.get("piper_model", str(ROOT / "data" / "voices" / "en_US-lessac-medium.onnx")),
                config=tcfg.get("piper_config"),
                speed=float(tcfg.get("speed", 1.0)),
            ) if tcfg.get("primary", "piper") == "piper" else dict()),
            backup_kwargs=(dict(
                python=tcfg.get("kokoro_python", str(ROOT / ".venv-kokoro" / "bin" / "python")),
                model=tcfg.get("kokoro_model", str(ROOT / "models" / "kokoro-v1.0.onnx")),
                voices=tcfg.get("kokoro_voices", str(ROOT / "models" / "voices-v1.0.bin")),
                voice=voice,
                speed=float(tcfg.get("speed", 1.0)),
            ) if tcfg.get("backup", "kokoro") == "kokoro" else dict()),
        )
    elif backend == "kokoro":
        kwargs = dict(
            python=tcfg.get("kokoro_python", str(ROOT / ".venv-kokoro" / "bin" / "python")),
            model=tcfg.get("kokoro_model", str(ROOT / "models" / "kokoro-v1.0.onnx")),
            voices=tcfg.get("kokoro_voices", str(ROOT / "models" / "voices-v1.0.bin")),
            voice=voice,
            speed=float(tcfg.get("speed", 1.0)),
        )
    elif backend == "piper":
        kwargs = dict(
            model=tcfg.get("piper_model", str(ROOT / "data" / "voices" / "en_US-lessac-medium.onnx")),
            config=tcfg.get("piper_config"),
            speed=float(tcfg.get("speed", 1.0)),
        )
    else:
        print(f"FAIL: unknown backend {backend!r}")
        return 2

    tts = make_tts(backend, **kwargs)

    out = Path(args.out) if args.out else ROOT / "data" / "audio" / f"tts_{backend}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("-" * 52)
    print(f"backend : {backend}" + (f"   voice={voice}" if backend == "kokoro" else ""))
    print(f"text    : \"{args.text}\"")

    if not args.no_warm:
        t0 = time.perf_counter()
        tts.warm()
        print(f"warm-up : model loaded in {time.perf_counter() - t0:.2f}s")

    syn = tts.synthesize(args.text)
    syn.to_wav(out)

    print(f"sample rate : {syn.sample_rate} Hz")
    print(f"audio dur   : {syn.duration_audio:.2f}s")
    print(f"compute     : {syn.duration_compute:.2f}s")
    print(f"RTF         : {syn.rtf:.3f}  (<1 = faster than real-time)")
    print(f"written     : {out.relative_to(ROOT)}  ({out.stat().st_size} bytes)")

    if syn.audio.size == 0:
        print("RESULT    : FAIL  (no audio produced)")
        return 1
    ok_rtf = syn.rtf < 1.0
    print(f"RESULT    : {'PASS' if ok_rtf else 'WARN'}  (audio produced; "
          f"{'RTF < 1.0' if ok_rtf else 'RTF >= 1.0 — fine on Pi, slow on Mac'})")
    return 0


def load_cfg() -> dict:
    p = ROOT / "config" / "config.yaml"
    if not p.exists():
        return {}
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


if __name__ == "__main__":
    raise SystemExit(main())
