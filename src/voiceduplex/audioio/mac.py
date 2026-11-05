"""macOS audio backend (CoreAudio via sounddevice).

Default input is the MacBook Air Microphone; default output is the Speakers.
Device indices can change between boots, so we resolve by name when possible
and fall back to the system default otherwise.
"""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from .base import AudioIO, Frame


class MacAudioIO(AudioIO):
    def __init__(
        self,
        input_device: str | int | None = "MacBook Air Microphone",
        output_device: str | int | None = "MacBook Air Speakers",
        sample_rate: int = 16_000,
        block: int = 512,
        playback_rate: int = 48_000,
    ) -> None:
        self.input_device = self._resolve(input_device, kind="input")
        self.output_device = self._resolve(output_device, kind="output")
        # Playback sample rate. TTS comes in at 24k/22.05k; the loop resamples to
        # this before play(). 48k is what Bluetooth A2DP headsets negotiate —
        # feeding 16k can go silent on a BT sink while built-in speakers cope.
        self.playback_rate = int(playback_rate)
        self.sample_rate = int(sample_rate)
        self.block = int(block)
        self._stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        # Ring buffer fed by play_async() and drained by the output callback.
        # The callback lives on the CoreAudio thread, so the producer side
        # (play_async / stop_playback) must hold _out_lock when touching it.
        self._out_buf: list[np.ndarray] = []
        self._out_pos = 0  # offset into _out_buf[0]
        self._out_lock = __import__("threading").Lock()
        self._out_open_warned = False

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(name: str | int | None, kind: str) -> int | None:
        """Resolve a device by name (case-insensitive substring) or index."""
        if name is None:
            return None
        if isinstance(name, int):
            return name
        try:
            devs = sd.query_devices()
        except Exception:
            return None
        for idx, d in enumerate(devs):
            if kind == "input" and d.get("max_input_channels", 0) <= 0:
                continue
            if kind == "output" and d.get("max_output_channels", 0) <= 0:
                continue
            if name.lower() in str(d.get("name", "")).lower():
                return idx
        # No match: fall back to the system default for that direction.
        try:
            default = sd.default.device
            return default[0] if kind == "input" else default[1]
        except Exception:
            return None

    # ------------------------------------------------------------------
    def open(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block,
            device=self.input_device,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.stop_playback()

    def read(self) -> Frame:
        assert self._stream is not None, "call open() first"
        data, _overflowed = self._stream.read(self.block)
        return Frame(pcm=data[:, 0].copy(), t=time.monotonic())

    # ------------------------------------------------------------------
    def play(self, pcm: np.ndarray) -> None:
        # Blocking one-shot (Step 5 path). Also works for Step 6 callers that
        # don't need barge-in.
        self._ensure_out_stream()
        with self._out_lock:
            self._out_buf = [np.ascontiguousarray(pcm, dtype=np.int16)]
            self._out_pos = 0
        # Block until the callback has consumed everything.
        deadline = time.monotonic() + len(pcm) / self.playback_rate + 2.0
        while time.monotonic() < deadline:
            with self._out_lock:
                empty = not self._out_buf
            if empty:
                break
            time.sleep(0.01)

    def play_async(self, pcm: np.ndarray) -> None:
        """Enqueue audio for playback WITHOUT blocking the caller.

        Uses ONE persistent callback-driven OutputStream. The CoreAudio
        callback drains our ring buffer in real time, so there is no
        per-chunk stream open/close (which on this Mac costs ~300ms per
        150ms chunk and caused the "i-i-it-s" stutter) and no GIL-bound
        Python thread pumping samples (which caused the static).

        The mic InputStream stays open the whole time, so barge-in still
        works: stop_playback() flushes the ring buffer and the callback
        goes silent within one CoreAudio block (~10ms).
        """
        self._ensure_out_stream()
        chunk = np.ascontiguousarray(pcm, dtype=np.int16)
        with self._out_lock:
            self._out_buf.append(chunk)

    def stop_playback(self) -> None:
        """Flush queued audio so barge-in cuts off within ~10ms."""
        with self._out_lock:
            self._out_buf = []
            self._out_pos = 0

    def _ensure_out_stream(self) -> None:
        if self._out_stream is not None:
            return
        rate = self.playback_rate

        def _callback(outdata, frames, time_info, status):  # CoreAudio thread
            # Fill outdata[:, 0] from the ring buffer; silence on underrun.
            filled = 0
            with self._out_lock:
                buf = self._out_buf
                pos = self._out_pos
                while filled < frames and buf:
                    seg = buf[0]
                    take = min(len(seg) - pos, frames - filled)
                    if take <= 0:
                        buf.pop(0)
                        pos = 0
                        continue
                    outdata[filled : filled + take, 0] = seg[pos : pos + take]
                    filled += take
                    pos += take
                    if pos >= len(seg):
                        buf.pop(0)
                        pos = 0
                self._out_pos = pos
            if filled < frames:
                outdata[filled:, 0] = 0

        self._out_stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            blocksize=max(64, rate // 100),  # 10ms blocks: tight + low CPU
            device=self.output_device,
            callback=_callback,
        )
        self._out_stream.start()
