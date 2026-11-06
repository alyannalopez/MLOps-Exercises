"""Benchmark the openWakeWord 'hey_mycroft' detector on real dataset audio.

Positive: a synthetic TTS clip saying "Hey Microft, please turn on the
lights" (generated with macOS `say`; see scripts/make_tp_clip.py).
Negatives: the full neg_test split (non-command speech/noise).
Reports per-frame latency, peak probability, and false-trigger count at the
operating threshold. Saves results/wakeword_bench.json.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from runtime.wakeword import FRAME_SAMPLES, WakeWordDetector

AUDIO = ROOT / "data" / "me2-dataset" / "audio"
POSITIVE = Path("/tmp/hey_mycroft.wav")  # synthetic TTS "Hey Microft, ..." (make_tp_clip.py)
THRESHOLD = 0.5


def run_wav(ww: WakeWordDetector, path: Path):
    w, _ = sf.read(path, dtype="float32")
    w = w.flatten()
    n = len(w) // FRAME_SAMPLES
    lats, peak = [], 0.0
    t0 = time.perf_counter()
    for i in range(n):
        fr = w[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        pt = time.perf_counter()
        ev = ww.detect(fr)
        lats.append((time.perf_counter() - pt) * 1000)
        peak = max(peak, ev.probability)
    wall = (time.perf_counter() - t0) * 1000
    ww.reset()
    return peak, wall, float(np.median(lats)), float(np.percentile(lats, 95))


def main() -> None:
    ww = WakeWordDetector(wakeword="hey_mycroft", threshold=THRESHOLD)
    pos_peak, pos_wall, pos_p50, pos_p95 = run_wav(ww, POSITIVE)
    print(f"positive: peak_prob={pos_peak:.3f} wall={pos_wall:.0f}ms "
          f"per-frame p50={pos_p50:.2f}ms p95={pos_p95:.2f}ms "
          f"rt_factor={pos_wall / (sf.info(str(POSITIVE)).duration * 1000):.2f}")

    rows = list(csv.DictReader(open(ROOT / "data" / "me2-dataset" / "manifest.csv")))
    neg = [r for r in rows if r["split"] == "neg_test"]
    peaks, walls, meds, p95s = [], [], [], []
    for r in neg:
        p, w, m, q = run_wav(ww, AUDIO / "neg_test" / Path(r["file"]).name)
        peaks.append(p); walls.append(w); meds.append(m); p95s.append(q)
    peaks = np.array(peaks); walls = np.array(walls)
    res = {
        "wakeword_model": "openwakeword hey_mycroft_v0.1 (onnx, CPU, openwakeword 0.4.0)",
        "threshold": THRESHOLD,
        "positive_clip": {
            "file": "synthetic TTS (macOS say) 'Hey Microft, please turn on the lights'",
            "transcript": "hey microft please turn on the lights",
            "peak_prob": float(pos_peak),
            "wall_ms": float(pos_wall),
            "per_frame_ms_p50": pos_p50, "per_frame_ms_p95": pos_p95,
            "rt_factor": float(pos_wall / (sf.info(str(POSITIVE)).duration * 1000)),
        },
        "neg_test": {
            "n_clips": len(neg),
            "peak_prob_max": float(peaks.max()),
            "peak_prob_mean": float(peaks.mean()),
            "false_triggers_at_threshold": int((peaks >= THRESHOLD).sum()),
            "per_frame_ms_p50": float(np.median(meds)),
            "per_frame_ms_p95": float(np.percentile(p95s, 95)),
            "per_clip_wall_ms_p50": float(np.median(walls)),
            "per_clip_wall_ms_max": float(walls.max()),
        },
    }
    out = ROOT / "results" / "wakeword_bench.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["neg_test"], indent=1))


if __name__ == "__main__":
    main()
