#!/usr/bin/env python3
"""ME3 room-tone + echo probe (Step 6 diagnostics).

Run this BEFORE a full-duplex session to prove the mic path is healthy:

    cd /Users/yannamii/sandbox/ME3/ME3
    PYTHONPATH=src /opt/miniconda3/bin/python3 scripts/probe_room_tone.py

What it does
------------
1. Lists every input device (so you can confirm which mic is pinned).
2. Samples the pinned mic for N seconds (default 15) and reports RMS stats.
3. Tells you whether the barge-in floor (duplex.barge_min_energy) is safe
   against your room tone, and whether echo is likely.

Interpreting the output
-----------------------
  mean < ~50  and  p95 < ~150   ->  QUIET: full-duplex should work.
  p95 > ~150                    ->  NOISY: raise vad.threshold to 0.55-0.6,
                                    quiet the room, or wear headphones.
  If you run this WHILE the agent is speaking and RMS spikes to hundreds,
  that is the ECHO signature (mic hearing the speakers) -> wear headphones.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
import yaml  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
mic_name = cfg.get("audio", {}).get("input_device", "default")
barge_floor = float(cfg.get("duplex", {}).get("barge_min_energy", 150.0))
vad_thr = float(cfg.get("vad", {}).get("threshold", 0.5))
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

print("INPUT DEVICES:")
idx = None
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] <= 0:
        continue
    mark = ""
    if mic_name != "default" and mic_name.lower() in d["name"].lower():
        idx = i
        mark = "  <-- pinned (config audio.input_device)"
    print(f"  [{i}] {d['name']}  ch={d['max_input_channels']} sr={d['default_samplerate']}{mark}")
if idx is None and mic_name != "default":
    print(f"  NOTE: '{mic_name}' not found by name; falling back to system default.")
print("SYSTEM DEFAULT:", sd.default.device)

dev = idx if idx is not None else None
print(f"\nSampling mic for {secs:.0f}s ... keep the room as quiet as you can.")
with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                    blocksize=512, device=dev) as s:
    n = int(secs * 16000 / 512)
    rms = np.array([float(np.sqrt(np.mean(s.read(512)[0][:, 0].astype(np.float64) ** 2)))
                    for _ in range(n)])

print(f"\nROOM TONE ({secs:.0f}s)")
print(f"  RMS   mean={rms.mean():7.1f}  p50={np.percentile(rms,50):7.1f}"
      f"  p95={np.percentile(rms,95):7.1f}  max={rms.max():7.1f}")
print(f"  barge_min_energy={barge_floor:.0f}  -> frames over floor: "
      f"{(rms > barge_floor).sum()}/{len(rms)}")
print(f"  vad.threshold={vad_thr}")

if rms.mean() < 50 and np.percentile(rms, 95) < barge_floor:
    print("\n  VERDICT: QUIET — mic path healthy. Full-duplex should work.")
elif np.percentile(rms, 95) >= barge_floor:
    print("\n  VERDICT: NOISY — ambient noise crosses the barge-in floor.")
    print("            Fix: quiet the room, or raise vad.threshold to 0.55-0.6.")
else:
    print("\n  VERDICT: borderline — acceptable, but watch for phantom turns.")

print("\n  ECHO CHECK: run this AGAIN while the agent is speaking (headphones OFF).")
print("  If RMS jumps to hundreds during its speech, the mic is hearing the")
print("  speakers -> wear headphones (output_device already follows your BT/headset).")
