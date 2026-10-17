"""Identify which neg_test clips cross the wake-word threshold."""
import csv
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.wakeword import FRAME_SAMPLES, WakeWordDetector

AUDIO = ROOT / "data" / "me2-dataset" / "audio"
rows = list(csv.DictReader(open(ROOT / "data" / "me2-dataset" / "manifest.csv")))
neg = [r for r in rows if r["split"] == "neg_test"]
ww = WakeWordDetector(wakeword="hey_mycroft", threshold=0.5)
offenders = []
for r in neg:
    p = AUDIO / "neg_test" / Path(r["file"]).name
    w, _ = sf.read(p, dtype="float32")
    w = w.flatten()
    n = len(w) // FRAME_SAMPLES
    peak = 0.0
    for i in range(n):
        ev = ww.detect(w[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
        peak = max(peak, ev.probability)
    ww.reset()
    if peak >= 0.5:
        offenders.append({"file": r["file"], "peak": round(float(peak), 4)})
print("offenders at 0.5:", offenders)
