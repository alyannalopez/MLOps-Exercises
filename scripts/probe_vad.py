#!/usr/bin/env python3
"""STEP 2 test — VAD live probe.

Listens to the mic and prints speech_start / speech_end events with
timestamps, plus a live meter. This is the Step 2 deliverable: talk, and
you should see events fire within ~300 ms.

Usage:
  python scripts/probe_vad.py                     # Silero, default mic
  python scripts/probe_vad.py --engine webrtc     # backup engine
  python scripts/probe_vad.py --duration 30       # stop after 30 s
  python scripts/probe_vad.py --threshold 0.6     # less sensitive
  python scripts/probe_vad.py --input "iPhone"    # pick a different mic

Success criteria (Step 2):
  1. Silence  -> no events, meter near 0.
  2. Speak    -> one "SPEECH START", then "SPEECH END" after you stop.
  3. No flapping (rapid start/end) on steady speech.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `voiceduplex` importable when run as a script from the ME3 root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from voiceduplex.audioio import make_audio_io
from voiceduplex.vad import make_vad


def load_cfg() -> dict:
    p = ROOT / "config" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    a = argparse.ArgumentParser(description="VAD live probe (Step 2)")
    a.add_argument("--engine", choices=["silero", "webrtc"], default=None)
    a.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    a.add_argument("--threshold", type=float, default=None)
    a.add_argument("--input", dest="input_device", default=None)
    a.add_argument("--backend", choices=["mac", "pi"], default=None)
    return a.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cfg()
    acfg = cfg.get("audio", {})
    vcfg = cfg.get("vad", {})

    engine = args.engine or vcfg.get("engine", "silero")
    threshold = args.threshold if args.threshold is not None else vcfg.get("threshold", 0.5)

    print(f"VAD engine : {engine}")
    print(f"threshold  : {threshold}")
    print(f"min_speech : {vcfg.get('min_speech_frames', 4)} frames (~{vcfg.get('min_speech_frames',4)*32} ms)")
    print(f"min_silence: {vcfg.get('min_silence_frames', 12)} frames (~{vcfg.get('min_silence_frames',12)*32} ms)")
    print("-" * 56)

    audio = make_audio_io(
        kind=args.backend or acfg.get("backend"),
        input_device=args.input_device or acfg.get("input_device"),
        output_device=acfg.get("output_device"),
        sample_rate=acfg.get("sample_rate", 16000),
        block=acfg.get("block", 512),
    )
    vad_kwargs = dict(
        threshold=threshold,
        min_speech_frames=vcfg.get("min_speech_frames", 4),
        min_silence_frames=vcfg.get("min_silence_frames", 12),
        sample_rate=acfg.get("sample_rate", 16000),
    )
    if engine == "webrtc":
        vad_kwargs["aggressiveness"] = vcfg.get("webrtc_aggressiveness", 2)
    vad = make_vad(engine, **vad_kwargs)

    audio.open()
    print(f"Mic open. Talk now. Ctrl-C to stop.\n")

    t_end = time.monotonic() + args.duration if args.duration > 0 else None
    n_events = 0
    last_draw = 0.0

    try:
        for frame in audio.stream():
            now = time.monotonic()
            if t_end and now >= t_end:
                break

            state, event = vad.process(frame.pcm)

            # Live meter (throttled to ~10 fps).
            rms = float(np.sqrt(np.mean(frame.pcm.astype(np.float64) ** 2)))
            if now - last_draw > 0.1:
                bar = "#" * int(min(rms / 800.0, 30))
                tag = "SPEAKING" if state.value else "silence"
                sys.stdout.write(f"\r  [{tag:8}] |{bar:<30}| rms={rms:6.0f}   ")
                sys.stdout.flush()
                last_draw = now

            if event == "speech_start":
                n_events += 1
                sys.stdout.write("\n")
                print(f"  >> t={frame.t:7.2f}s  SPEECH START")
            elif event == "speech_end":
                n_events += 1
                print(f"  << t={frame.t:7.2f}s  SPEECH END")
    except KeyboardInterrupt:
        print("\n(stopped)")
    finally:
        audio.close()

    print("-" * 56)
    print(f"Total events: {n_events}")
    if n_events == 0:
        print("No speech detected. Try raising your volume or lowering --threshold.")
    else:
        print("Step 2 looks GREEN if: silence stayed quiet, speech gave clean start/end, no flapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
