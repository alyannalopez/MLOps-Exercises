"""Generate 5 copyright-free demo audio tracks for the music skills (brief tasks 1 & 8).

The brief says: "download 5 basic copyright mp3 (like demo songs/alarm) to play
randomly when commanded." To keep the repo fully offline and unambiguously
copyright-free we *synthesize* five short tracks (arpeggios, a lo-fi drone, a
melody, and an alarm chirp) with numpy and write them as 16-bit mono WAV.

Why WAV and not MP3: this machine has no MP3 encoder (no ffmpeg/lame; macOS
afconvert cannot encode MP3). WAV is universally playable — `afplay` on macOS,
`aplay` on the Pi — and the files are small (~200 KB each). The runtime plays
whichever of .mp3/.wav exists, so dropping in real MP3s later is a no-op.

Output: assets/music/{track1_relaxing..track5_alarm}.wav  (~4-6 s each)
"""
from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

import numpy as np

SR = 22050
OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "music"


def _tone(freq: float, dur: float, vol: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    env = np.ones_like(t)
    fade = int(0.02 * SR)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return vol * env * np.sin(2 * np.pi * freq * t)


def _arpeggio(notes: list[float], note_dur: float, vol: float = 0.45) -> np.ndarray:
    return np.concatenate([_tone(f, note_dur, vol=vol) for f in notes])


def track1_relaxing() -> np.ndarray:
    """Slow major arpeggio, C4-E4-G4-C5, looped twice."""
    notes = [261.63, 329.63, 392.00, 523.25]
    return _arpeggio(notes * 2, 0.55)


def track2_upbeat() -> np.ndarray:
    """Faster A-minor-pentatonic run."""
    notes = [440.00, 523.25, 587.33, 659.25, 783.99,
             659.25, 587.33, 523.25]
    return _arpeggio(notes, 0.28)


def track3_lofi() -> np.ndarray:
    """Low C3/G3 drone with a slow 0.8 Hz tremolo — 'lo-fi' feel."""
    t = np.linspace(0, 5.0, int(SR * 5.0), endpoint=False)
    base = 0.4 * np.sin(2 * np.pi * 130.80 * t)
    harm = 0.15 * np.sin(2 * np.pi * 196.00 * t)
    trem = 0.7 + 0.3 * np.sin(2 * np.pi * 0.8 * t)
    return base * trem + harm * trem


def track4_demo_song() -> np.ndarray:
    """Simple 7-note melody (Twinkle-ish contour)."""
    seq = [523.25, 523.25, 783.99, 783.99, 880.00, 880.00, 783.99]
    return np.concatenate([_tone(f, 0.4, vol=0.4) for f in seq])


def track5_alarm() -> np.ndarray:
    """Classic alarm chirp: alternating 880/660 Hz bursts."""
    hi = _tone(880.0, 0.25, vol=0.6)
    lo = _tone(660.0, 0.25, vol=0.6)
    gap = np.zeros(int(0.05 * SR))
    burst = np.concatenate([hi, gap, lo, gap])
    return np.tile(burst, 5)


TRACKS = {
    "track1_relaxing.wav": track1_relaxing,
    "track2_upbeat.wav": track2_upbeat,
    "track3_lofi.wav": track3_lofi,
    "track4_demo_song.wav": track4_demo_song,
    "track5_alarm.wav": track5_alarm,
}


def _write_wav(path: Path, wav: np.ndarray) -> None:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in TRACKS.items():
        wav = fn()
        out = OUT_DIR / name
        _write_wav(out, wav)
        print(f"  wrote {out.name}  ({out.stat().st_size / 1024:.0f} KB, "
              f"{len(wav) / SR:.1f} s)")
    print(f"\n5 copyright-free demo tracks in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
