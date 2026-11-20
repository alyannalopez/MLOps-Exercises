"""Build the Genie Pipecat FrameProcessor pipeline.

Pipeline order (client-side on the Pi, or server-side on the Mac):

    InputAudioRawFrame  (mic audio from the Pi)
        -> SileroVAD  (voice activity)
        -> WhisperHTTPSTTService  (POST /transcribe -> TranscriptionFrame)
        -> GenieBrainProcessor  (LLM tool-calling -> LLMTextFrame / tool results)
        -> PiperHTTPTTSService  (POST /speak -> TTSAudioRawFrame)
        -> OutputAudioRawFrame  (play on Pi speaker)

Barge-in: when the VAD detects user speech while TTS is playing, the pipeline
pushes an InterruptionFrame downstream, which clears the TTS queue and lets the
user be heard immediately.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from .brain import GENIE_SYSTEM_PROMPT, tool_json_schemas, dispatch_tool
from .services import WhisperHTTPSTTService, PiperHTTPTTSService
from .config import Config


class GenieBrainProcessor(FrameProcessor):
    """LLM brain FrameProcessor.

    Receives TranscriptionFrame, calls the LLM (OpenAI-compatible) with the
    Genie system prompt + tool schemas, handles tool calls, and emits
    LLMTextFrame for TTS. On user speech during playback it pushes
    InterruptionFrame.
    """

    def __init__(self, *, config: Config, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._client = None  # httpx.AsyncClient, lazily created
        self._speaking = False  # TTS is currently playing
        self._system_prompt = GENIE_SYSTEM_PROMPT

    async def _get_client(self) -> "httpx.AsyncClient":
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _process_frame(self, frame: Any, direction: FrameDirection):
        await super()._process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking — if TTS is playing, barge in.
            self._speaking = True
            if direction == FrameDirection.DOWNSTREAM:
                await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._speaking = False
            return

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return
            # Run the LLM turn.
            await self._llm_turn(text)
            return

        if isinstance(frame, LLMTextFrame):
            # Forward TTS request downstream.
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)
            return

        # Pass everything else through.
        await self.push_frame(frame, direction)

    async def _llm_turn(self, user_text: str):
        """Call the LLM, handle tool calls, emit LLMTextFrame."""
        try:
            client = await self._get_client()
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_text},
            ]
            # First pass: ask the model, with tool schemas available.
            resp = await client.post(
                f"{self._config.llm_base_url}/chat/completions",
                json={
                    "model": self._config.llm_model,
                    "messages": messages,
                    "tools": tool_json_schemas(),
                    "tool_choice": "required",
                    "max_tokens": self._config.llm_max_tokens,
                    "temperature": self._config.llm_temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Execute tools and feed results back.
                tool_results = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = json.loads(fn.get("arguments", "{}") or "{}")
                    result = dispatch_tool(name, args)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    })
                messages.append(msg)
                messages.extend(tool_results)
                resp2 = await client.post(
                    f"{self._config.llm_base_url}/chat/completions",
                    json={
                        "model": self._config.llm_model,
                        "messages": messages,
                        "max_tokens": self._config.llm_max_tokens,
                        "temperature": self._config.llm_temperature,
                    },
                )
                resp2.raise_for_status()
                data2 = resp2.json()
                final_text = data2["choices"][0].get("message", {}).get("content", "")
            else:
                final_text = msg.get("content", "")

            if final_text:
                await self.push_frame(
                    LLMTextFrame(text=final_text),
                    FrameDirection.DOWNSTREAM,
                )
        except Exception as exc:  # pragma: no cover - network diagnostics
            print(f"[genie-brain] error: {exc}", flush=True)


def build_pipeline(config: Config | None = None) -> Pipeline:
    """Build the Genie Pipecat Pipeline."""
    cfg = config or Config()

    stt = WhisperHTTPSTTService(
        base_url=cfg.server_url,
        sample_rate=cfg.sample_rate,
        language=cfg.asr_lang,
    )
    tts = PiperHTTPTTSService(
        base_url=cfg.server_url,
        sample_rate=cfg.sample_rate,
    )
    brain = GenieBrainProcessor(config=cfg)

    pipeline = Pipeline([
        stt,
        brain,
        tts,
    ])
    return pipeline


def build_task(config: Config | None = None) -> PipelineTask:
    """Build a PipelineTask ready to run."""
    cfg = config or Config()
    pipeline = build_pipeline(cfg)
    params = PipelineParams(
        allow_interruptions=True,
        interruption_modes=["immediate"],
    )
    task = PipelineTask(pipeline, params=params)
    return task
