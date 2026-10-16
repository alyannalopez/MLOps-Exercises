"""Check the hey_mycroft detector on a synthetic true-positive clip."""
import sys

import soundfile as sf

sys.path.insert(0, ".")
from runtime.wakeword import FRAME_SAMPLES, WakeWordDetector

w, sr = sf.read("/tmp/hey_mycroft.wav", dtype="float32")
print("dur", len(w) / sr, "sr", sr)
ww = WakeWordDetector(wakeword="hey_mycroft", threshold=0.5)
n = len(w) // FRAME_SAMPLES
peak, trig_frames = 0.0, 0
for i in range(n):
    ev = ww.detect(w[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
    peak = max(peak, ev.probability)
    trig_frames += ev.triggered
print(f"TRUE POSITIVE 'hey microft': peak={peak:.4f} triggered_frames={trig_frames}")
