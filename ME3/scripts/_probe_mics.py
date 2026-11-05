#!/usr/bin/env python3
"""Probe all input devices for real signal level."""
from __future__ import annotations
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import sounddevice as sd
from voiceduplex.vad import make_vad

devs = sd.query_devices()
inputs = [(i, d["name"]) for i, d in enumerate(devs) if d.get("max_input_channels", 0) > 0]
print(f"Input devices: {inputs}\n")

for idx, name in inputs:
    try:
        vad = make_vad("silero", threshold=0.5, min_speech_frames=4, min_silence_frames=12, sample_rate=16000)
        with sd.InputStream(samplerate=16000, channels=1, dtype="int16", device=idx, blocksize=512) as s:
            start = time.time(); amp_max = 0; voiced = 0; nframes = 0
            while time.time() - start < 5:
                d, _ = s.read(512)
                nframes += 1
                amp_max = max(amp_max, int(np.max(np.abs(d[:, 0]))))
                state, _ = vad.process(d[:, 0].copy())
                if state.value:
                    voiced += 1
            print(f"{name!r:40} frames={nframes:4} amp_max={amp_max:6} voiced={voiced}")
    except Exception as e:
        print(f"{name!r:40} FAILED {e!r}")
