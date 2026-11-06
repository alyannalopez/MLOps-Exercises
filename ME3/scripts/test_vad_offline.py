#!/usr/bin/env python3
"""STEP 2 test — VAD offline (file-based, no mic needed).

Feeds an audio file through the VAD and reports speech segments + timing.
This is the deterministic, repeatable check — ideal for the Pi (headless)
and for regression-testing after you change thresholds.

Usage:
  python scripts/test_vad_offline.py                          # uses data/audio/test_speech.wav
  python scripts/test_vad_offline.py --file path/to.wav
  python scripts/test_vad_offline.py --engine webrtc
  python scripts/test_vad_offline.py --threshold 0.4

Success criteria:
  - A speech file yields >= 1 speech_start and (for Silero) a matching speech_end.
  - Reported segment covers the actual speech, not the padding.
  - A pure-silence file yields zero events.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import wave
import yaml

from voiceduplex.vad import make_vad


def load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr != 16000:
        n_out = int(len(pcm) * 16000 / sr)
        idx = np.linspace(0, len(pcm) - 1, n_out)
        pcm = np.interp(idx, np.arange(len(pcm)), pcm.astype(np.float64)).astype(np.int16)
    return pcm


def main() -> int:
    a = argparse.ArgumentParser(description="VAD offline test (Step 2)")
    a.add_argument("--file", default=str(ROOT / "data" / "audio" / "test_speech.wav"))
    a.add_argument("--engine", choices=["silero", "webrtc"], default=None)
    a.add_argument("--threshold", type=float, default=None)
    a.add_argument("--quiet", action="store_true", help="suppress per-event lines")
    args = a.parse_args()

    cfg = load_cfg()
    vcfg = cfg.get("vad", {})
    engine = args.engine or vcfg.get("engine", "silero")
    threshold = args.threshold if args.threshold is not None else vcfg.get("threshold", 0.5)

    f = Path(args.file)
    if not f.exists():
        print(f"FAIL: audio file not found: {f}")
        return 2
    audio = load_wav_16k(f)
    dur = len(audio) / 16000

    kw = dict(threshold=threshold,
              min_speech_frames=vcfg.get("min_speech_frames", 4),
              min_silence_frames=vcfg.get("min_silence_frames", 12),
              sample_rate=16000)
    if engine == "webrtc":
        kw["aggressiveness"] = vcfg.get("webrtc_aggressiveness", 2)
    vad = make_vad(engine, **kw)

    events = []
    t0 = time.perf_counter()
    for i in range(0, len(audio), 512):
        fr = audio[i:i + 512]
        if fr.shape[0] < 512:
            fr = np.pad(fr, (0, 512 - fr.shape[0]))
        st, ev = vad.process(fr)
        if ev:
            tsec = i / 16000
            events.append((ev, tsec))
            if not args.quiet:
                print(f"  {ev:12} @ {tsec:6.2f}s")
    wall = time.perf_counter() - t0
    rt_factor = wall / dur if dur > 0 else 0.0

    starts = [e for e in events if e[0] == "speech_start"]
    ends = [e for e in events if e[0] == "speech_end"]

    print("-" * 52)
    print(f"file      : {f.name}  ({dur:.2f}s)")
    print(f"engine    : {engine}   threshold={threshold}")
    print(f"events    : {len(starts)} start, {len(ends)} end")
    for ev, t in events:
        print(f"  - {ev:12} {t:6.2f}s")
    print(f"speed     : {rt_factor:.3f}x realtime (wall {wall:.2f}s)")

    # Verdict. speech_end only fires if the file has trailing silence; a raw
    # speech clip ends at EOF so start-without-end is correct, not a failure.
    if len(starts) == 0:
        print("RESULT    : NO SPEECH DETECTED (expected only for a silence file)")
        return 1
    print("RESULT    : PASS  (speech_start fired)")
    if len(ends) == 0:
        print("            note: no speech_end — file ends mid-speech (no trailing silence).")
        print("            To see speech_end, append ~1s of silence to the file.")
    return 0


def load_cfg() -> dict:
    p = ROOT / "config" / "config.yaml"
    if not p.exists():
        return {}
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


if __name__ == "__main__":
    raise SystemExit(main())
