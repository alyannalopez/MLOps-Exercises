"""Hardware-agnostic audio interface.

The rest of the pipeline only ever talks to this contract:
  - open()/close()
  - read()  -> a fixed-size int16 frame at sample_rate (16 kHz mono)
  - play()  -> render int16 audio (blocking)
  - stream()-> generator yielding frames (convenience for the VAD loop)

Mac and Pi both implement this. Nothing outside audioio/ imports sounddevice
directly, so porting = writing pi.py against this same base.
"""

from __future__ import annotations

import abc
import queue
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """One chunk of 16-bit PCM, mono, at the device sample rate."""

    pcm: np.ndarray  # shape (n,), dtype int16
    t: float  # wall-clock seconds when captured (time.monotonic())

    @property
    def n(self) -> int:
        return int(self.pcm.shape[0])


class AudioIO(abc.ABC):
    SAMPLE_RATE = 16_000
    BLOCK = 512  # 32 ms at 16 kHz — the unit the VAD wants

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> Frame:
        """Block until one BLOCK-sized frame is available."""

    @abc.abstractmethod
    def play(self, pcm: np.ndarray) -> None:
        """Render int16 mono audio at SAMPLE_RATE. Blocking."""

    def play_async(self, pcm: np.ndarray) -> None:
        """Queue int16 mono audio for NON-BLOCKING playback.

        Used by the full-duplex loop so the mic keeps streaming while we talk.
        Default falls back to blocking play(); backends with a real output
        stream override this to queue into it.
        """
        self.play(pcm)

    def stream(self):
        """Yield Frames forever. Caller closes the device when done."""
        while True:
            yield self.read()
