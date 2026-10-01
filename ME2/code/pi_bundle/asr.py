#!/usr/bin/env python3
"""Tiny local ASR for slot-filling (whisper.cpp CLI wrapper).

The 340 KB INT8 intent model tells us WHICH intent; this module transcribes the
SAME utterance with a small English-only Whisper model so we can pull out the
NUMBER ("5" in "set a timer for 5 minutes").

It shells out to the `whisper-cli` binary (brew install whisper-cpp) with a
GGML model, so it adds ZERO Python dependencies -- just the ~78 MB model file.
On the Pi you'd build whisper.cpp (see SETUP_GUIDE.md) or run the ASR pass on
the host that has it.

Usage:
    from asr import Transcriber
    tr = Transcriber("/path/to/ggml-tiny.en.bin")
    text = tr.transcribe_wav("/tmp/cmd.wav")      # "set a timer for 5 minutes"

If whisper-cli or the model is missing, Transcriber.available is False and
transcribe_wav() returns "" so callers can fall back gracefully.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional


def find_whisper_cli() -> Optional[str]:
    """Locate the whisper-cli binary. Env var WHISPER_CLI wins, then PATH,
    then common Homebrew locations."""
    env = os.environ.get("WHISPER_CLI")
    if env and os.path.exists(env):
        return env
    found = shutil.which("whisper-cli")
    if found:
        return found
    for cand in ("/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"):
        if os.path.exists(cand):
            return cand
    return None


class Transcriber:
    def __init__(self, model_path: str, cli: Optional[str] = None,
                 threads: int = 0, language: str = "en"):
        self.model_path = model_path
        self.cli = cli or find_whisper_cli()
        # whisper.cpp crashes with -t 0 (tries to spawn 0 threads); default to 4.
        self.threads = threads if threads > 0 else 4
        self.language = language
        self.available = bool(self.cli) and os.path.exists(model_path)
        self._last_ms = 0.0

    def transcribe_wav(self, wav_path: str) -> str:
        """Transcribe a 16 kHz mono WAV. Returns the text ("" on failure)."""
        if not self.available:
            return ""
        # whisper.cpp APPENDS the extension to -of, so pass a base name WITHOUT
        # ".txt" -- otherwise it writes "<base>.txt.txt".
        tmp_base = tempfile.mktemp(prefix="vcm_asr_", suffix="")
        cmd = [self.cli, "-m", self.model_path, "-f", wav_path,
               "-l", self.language, "-nt", "-otxt", "-np",
               "-t", str(self.threads), "-of", tmp_base]
        tmp_txt = tmp_base + ".txt"
        try:
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            self._last_ms = (time.perf_counter() - t0) * 1000
            # Read BEFORE cleanup below removes the temp files.
            text = ""
            if os.path.exists(tmp_txt):
                with open(tmp_txt, "r", encoding="utf-8", errors="ignore") as f:
                    text = " ".join(line.strip() for line in f if line.strip())
            return text.strip()
        except Exception:
            return ""
        finally:
            for p in (tmp_txt, tmp_txt + ".json", tmp_txt + ".srt",
                      tmp_txt + ".vtt", tmp_txt + ".csv", tmp_txt + ".lrc"):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

    def transcribe_pcm(self, pcm: "np.ndarray", sr: int = 16000) -> str:
        """Transcribe an in-memory int16 PCM array by writing a temp WAV."""
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            with wave.open(tf.name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(pcm.astype("<i2").tobytes())
            name = tf.name
        try:
            return self.transcribe_wav(name)
        finally:
            try:
                os.remove(name)
            except OSError:
                pass


def default_model_path() -> str:
    """Where we keep the GGML model next to this file (asr/ggml-tiny.en.bin)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "asr", "ggml-tiny.en.bin")


if __name__ == "__main__":
    import sys
    mp = default_model_path()
    tr = Transcriber(mp)
    print(f"cli:       {tr.cli}")
    print(f"model:     {mp}")
    print(f"available: {tr.available}")
    if len(sys.argv) > 1 and tr.available:
        print(f"text:      {tr.transcribe_wav(sys.argv[1])!r}")
        print(f"latency:   {tr._last_ms:.0f} ms")
