"""Whisper.cpp backend — local, free, offline.

Wraps the `whisper-cli` binary (installed via Homebrew on Mac, built for
aarch64 on the Pi) as a subprocess. We deliberately use the CLI rather than a
Python binding because:
  - it's already installed and proven on this machine,
  - it's the *same binary* on the Pi (just rebuilt for ARM), so the Mac->Pi
    transfer is zero-code,
  - no fragile pip wheels to reconcile across platforms.

The CLI prints per-segment lines like:
  [00:00:00.000 --> 00:00:04.340]   Hello, this is a test...
to stdout; all the ggml/metal init chatter goes to stderr (we discard it).
We parse stdout for segments and fall back to the raw text if the format
shifts between versions.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from .base import ASR, Transcript

# Matches "[mm:ss:SS.mmm --> mm:ss:SS.mmm]   text"
_SEG_RE = re.compile(r"\[(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*(\d+):(\d+):(\d+)\.(\d+)\]\s*(.*)")


class WhisperCpp(ASR):
    def __init__(
        self,
        model_path: str | Path,
        binary: str = "whisper-cli",
        language: str = "en",
        threads: int = 0,          # 0 = let whisper pick (cores)
        beam_size: int = 5,
        no_timestamps: bool = True,
        sample_rate: int = 16_000,
    ) -> None:
        self.model_path = Path(model_path)
        self.binary = binary
        self.language = language
        self.threads = threads
        self.beam_size = beam_size
        self.no_timestamps = no_timestamps
        self.sample_rate = sample_rate
        self._warmed = False
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"whisper model not found: {self.model_path}\n"
                f"Download it (see scripts/download_model.sh) or set asr.model in config."
            )

    # -- lifecycle -----------------------------------------------------------
    def warm(self) -> None:
        """Verify the binary resolves and the model loads (cheap smoke run)."""
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"'{self.binary}' not on PATH. Install whisper.cpp "
                f"(mac: brew install whisper-cpp; pi: build from source)."
            )
        # A 0-frame run forces model load without doing real work.
        self._run(empty=True)
        self._warmed = True

    # -- public API ----------------------------------------------------------
    def transcribe(self, pcm, lang: str | None = None) -> Transcript:
        pcm = np.asarray(pcm, dtype=np.int16).ravel()
        audio_dur = len(pcm) / self.sample_rate
        t0 = time.perf_counter()
        segments, raw_text = self._run(pcm)
        compute = time.perf_counter() - t0
        text = raw_text.strip() if raw_text else " ".join(s[2].strip() for s in segments).strip()
        return Transcript(
            text=text,
            lang=lang or self.language,
            duration_audio=audio_dur,
            duration_compute=compute,
            segments=segments,
        )

    # -- internals -----------------------------------------------------------
    def _run(self, pcm: np.ndarray | None = None, *, empty: bool = False) -> tuple[list, str]:
        """Run whisper-cli over a temp wav. Returns (segments, raw_stdout_text)."""
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "in.wav"
            self._write_wav(wav, pcm if pcm is not None else np.zeros(0, dtype=np.int16))
            cmd = [
                self.binary,
                "-m", str(self.model_path),
                "-f", str(wav),
                "-l", self.language,
                "--beam-size", str(self.beam_size),
                "-np",  # no print progress
            ]
            if self.threads > 0:
                cmd += ["-t", str(self.threads)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                # surface the tail of stderr for debugging, not the whole metal dump
                err_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
                raise RuntimeError(f"whisper-cli exited {proc.returncode}: {err_tail}")
            return self._parse_segments(proc.stdout)

    @staticmethod
    def _write_wav(path: Path, pcm: np.ndarray) -> None:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())

    @staticmethod
    def _parse_segments(stdout: str) -> tuple[list[tuple[float, float, str]], str]:
        """Parse whisper-cli stdout into [(t0,t1,text)] plus the concatenated text."""
        segments: list[tuple[float, float, str]] = []
        texts: list[str] = []
        for line in stdout.splitlines():
            m = _SEG_RE.search(line)
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2, txt = m.groups()
                t0 = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
                t1 = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
                segments.append((t0, t1, txt.strip()))
                texts.append(txt.strip())
        return segments, "\n".join(texts)
