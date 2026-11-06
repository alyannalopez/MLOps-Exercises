"""Raspberry Pi audio backend (ALSA/PulseAudio via sounddevice).

Written against the SAME base.AudioIO contract as mac.py, so porting is
"fill in the device resolution + open/read/play" — nothing upstream changes.

On the Pi you will typically:
  - Use a USB mic / ReSpeaker HAT. Resolve by name (e.g. "USB PnP Audio") or
    pass an explicit ALSA card index.
  - Keep sample_rate=16000 and block=512 to match the VAD.
  - Expect higher capture latency than macOS; tune barge-in thresholds in
    config, not here.

Left intentionally minimal — finalize device names once the Pi is provisioned
(Step 1). The interface is already correct.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import sounddevice as sd

from .base import AudioIO, Frame


class PiAudioIO(AudioIO):
    def __init__(
        self,
        input_device: str | int | None = None,  # None = system default; set to your mic
        output_device: str | int | None = None,
        sample_rate: int = 16_000,
        block: int = 512,
        playback_rate: int = 48_000,
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.block = block
        self.playback_rate = int(playback_rate)
        self._q: "queue.Queue[np.ndarray]" = None  # type: ignore
        self._stream = None
        self._t0 = 0.0

    def open(self) -> None:
        self._q = queue.Queue()
        sr, block = self.sample_rate, self.block

        def _cb(indata, frames, t, status):
            self._q.put(indata[:, 0].copy())

        self._t0 = time.monotonic()
        self._stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="int16",
            blocksize=block,
            device=self.input_device,
            callback=_cb,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self) -> Frame:
        pcm = self._q.get()
        return Frame(pcm=pcm.astype(np.int16), t=time.monotonic() - self._t0)

    def play(self, pcm: np.ndarray) -> None:
        # pcm is int16 mono, already resampled to self.playback_rate by the loop.
        # Pass it as int16 to match the stream's dtype — the old float32/32768
        # cast scaled samples into ~[-1,1] which an int16 stream truncates to
        # silence (same bug that was fixed in mac.py).
        sd.play(pcm.astype(np.int16), self.playback_rate, device=self.output_device)
        sd.wait()

    def play_async(self, pcm: np.ndarray) -> None:
        """Non-blocking playback (full-duplex). Mirrors mac.py's output-stream
        approach so the mic keeps streaming while we talk on the Pi too."""
        pcm = pcm.astype(np.int16)
        if getattr(self, "_out_stream", None) is None:
            self._out_q: "queue.Queue[np.ndarray]" = queue.Queue()
            self._out_cur = None
            self._out_pos = 0

            def _out_cb(outdata, frames, t, status):
                written = 0
                while written < frames:
                    if self._out_cur is None or self._out_pos >= len(self._out_cur):
                        try:
                            self._out_cur = self._out_q.get_nowait()
                            self._out_pos = 0
                        except queue.Empty:
                            break
                    if self._out_cur is None:
                        break
                    take = min(frames - written, len(self._out_cur) - self._out_pos)
                    outdata[written:written + take, 0] = self._out_cur[self._out_pos:self._out_pos + take]
                    written += take
                if written < frames:
                    outdata[written:, 0] = 0

            self._out_stream = sd.OutputStream(
                samplerate=self.playback_rate,
                channels=1,
                dtype="int16",
                blocksize=min(1024, self.playback_rate // 10),
                device=self.output_device,
                callback=_out_cb,
            )
            self._out_stream.start()
        self._out_q.put(pcm.copy())
