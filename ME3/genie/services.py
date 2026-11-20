"""Pipecat FrameProcessor services that call the Genie model server over HTTP.

WhisperHTTPSTTService  -> POST /transcribe  (mlx-whisper on Mac)
KokoroHTTPTTSService   -> POST /speak       (Kokoro KPipeline on Mac)

Both are subclasses of pipecat's SegmentedSTTService / TTSService base classes
so they drop straight into a Pipeline.
"""
from __future__ import annotations

import io
import wave
from typing import Any

import httpx
import numpy as np

from pipecat.frames.frames import (
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService


class WhisperHTTPSTTService(SegmentedSTTService):
    """STT service that sends VAD-segmented WAV to the Genie /transcribe endpoint.

    Extends SegmentedSTTService so that:
    - Audio is buffered between VAD start/stop events.
    - run_stt() is called once per utterance with a WAV container.
    - Trailing silence (default 0.5s) is appended to prevent Whisper dropping
      the last word.
    - wants_wav_segments=True (default) — the server expects WAV.
    """

    def __init__(self, *, base_url: str, sample_rate: int = 16000,
                 language: str = "en", timeout: float = 30.0,
                 trailing_silence_secs: float = 0.5, **kwargs):
        super().__init__(
            name="WhisperHTTP",
            sample_rate=sample_rate,
            trailing_silence_secs=trailing_silence_secs,
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        # Satisfy STTSettings validation
        self._stt_settings.model = "mlx-whisper-large-v3-turbo"
        self._stt_settings.language = language

    async def run_stt(self, audio: bytes):
        """audio: WAV-encoded bytes (16-bit signed little-endian, mono).

        SegmentedSTTService already wraps the buffered PCM in a WAV container
        before calling run_stt() (wants_wav_segments=True by default).
        Send it directly to the /transcribe endpoint.
        """
        try:
            # Skip very short segments (< 0.2s of audio at sample_rate)
            # WAV header is 44 bytes; audio data is len(audio) - 44
            sr = self._sample_rate or 16000
            audio_data_len = len(audio) - 44  # subtract WAV header
            duration_secs = audio_data_len / (sr * 2)  # 2 bytes per sample
            if duration_secs < 0.2:
                print(f"[genie-stt] skipping short segment: {duration_secs:.2f}s", flush=True)
                return

            resp = await self._client.post(
                f"{self._base_url}/transcribe",
                content=audio,
                headers={"Content-Type": "audio/wav"},
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("text") or "").strip()
            if text:
                yield TranscriptionFrame(
                    text=text,
                    user_id="user",
                    timestamp="",
                    language=self._language,
                    finalized=True,
                )
        except Exception as exc:
            print(f"[genie-stt] error: {exc}", flush=True)


class KokoroHTTPTTSService(TTSService):
    """TTS service that POSTs text to the Genie /speak endpoint and yields
    TTSAudioRawFrame chunks. Uses Kokoro KPipeline on the server side."""

    def __init__(self, *, base_url: str, sample_rate: int = 24000,
                 voice: str | None = None, timeout: float = 30.0, **kwargs):
        super().__init__(name="KokoroHTTP", sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._voice = voice or "af_bella"
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        # Satisfy TTSSettings validation
        self._tts_settings.model = "kokoro-0.9.4"
        self._tts_settings.voice = self._voice
        self._tts_settings.language = "en"

    async def run_tts(self, text: str, context_id: str):
        """Yield TTSAudioRawFrame(s) for the given text."""
        try:
            payload: dict[str, Any] = {"text": text}
            if self._voice:
                payload["voice"] = self._voice
            resp = await self._client.post(
                f"{self._base_url}/speak",
                json=payload,
            )
            resp.raise_for_status()
            wav_bytes = resp.content

            # Parse WAV header to get sample rate / channels.
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                sr = w.getframerate()
                nch = w.getnchannels()
                frames = w.getnframes()
                raw = w.readframes(frames)

            yield TTSAudioRawFrame(
                audio=raw,
                sample_rate=sr,
                num_channels=nch,
                context_id=context_id,
            )
        except Exception as exc:
            print(f"[genie-tts] error: {exc}", flush=True)


# Keep backward compat alias
PiperHTTPTTSService = KokoroHTTPTTSService
