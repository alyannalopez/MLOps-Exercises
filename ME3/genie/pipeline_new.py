"""Genie pipeline — Pipecat-native full-duplex voice assistant.

Architecture (each stage is a separate Pipecat component):

  transport.input()   — mic capture → InputAudioRawFrame
    → WhisperHTTPSTTService — SegmentedSTTService, VAD-gated, calls Mac /transcribe, yields TranscriptionFrame
      → LLMUserAggregator   — collects TranscriptionFrame → LLMContext (VAD-driven turn detection)
        → OpenRouterLLMService — LLM reasoning + tool calling
          → LLMAssistantAggregator — collects LLM text → TTS
            → KokoroHTTPTTSService — HTTP POST to Mac /speak
              → transport.output() — speaker playback, 24kHz mono

Key design decisions:
  • Native Pipecat Pipeline/PipelineWorker/WorkerRunner — no hand-rolled loop.
  • Each stage is a distinct FrameProcessor — no monolithic GenieAgent.
  • VAD is passed to LLMUserAggregatorParams (broadcasts upstream to STT).
  • SegmentedSTTService receives VAD events via broadcast from LLMUserAggregator.
  • LLMContext manages conversational history with user/assistant aggregators.
  • Tools registered via llm.register_function(name, handler).
  • Interruptions: VAD detects speech during assistant turn → aggregator
    cancels the current turn → new transcription begins.

Configuration:
  Edit .env in the project root (copy from .env.example).
  Environment variables override .env values.

Run:
  PYTHONPATH=. .venv-genie/bin/python -m genie.pipeline_new
"""
from __future__ import annotations

import asyncio
import logging
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.aggregators.llm_context import (
    FunctionSchema,
    LLMContext,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from genie.brain import GENIE_SYSTEM_PROMPT, TOOL_SCHEMAS, dispatch_tool
from genie.config import Config
from genie.services import KokoroHTTPTTSService, WhisperHTTPSTTService

log = logging.getLogger("genie")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def make_handler(tool_name: str):
    """Create an async tool handler that dispatches to dispatch_tool().

    The handler receives FunctionCallParams and calls result_callback()
    with the tool's string result.
    """
    async def handler(params):
        try:
            args = dict(params.arguments)
            result = dispatch_tool(tool_name, args)
            await params.result_callback(result)
        except Exception as e:
            log.error("Tool %s(%s) failed: %s", tool_name, params.arguments, e)
            await params.result_callback(f"Error executing {tool_name}: {e}")
    return handler


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(cfg: Config) -> Pipeline:
    """Build the full-duplex voice assistant pipeline with native Pipecat stages."""
    log.info("Model server: %s", cfg.server_url)

    # --- Validate API key ---
    if not cfg.openrouter_api_key:
        log.warning(
            "OPENROUTER_API_KEY not set in .env or environment. "
            "LLM calls will fail. Add your key to .env or set the env var."
        )

    # --- 1. Audio transport (mic in → speaker out) ---
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            audio_out_channels=1,
            input_device_index=cfg.mic_device,
            output_device_index=cfg.speaker_device,
        ),
    )

    # --- 2. VAD (Silero) — passed to user aggregator, NOT a pipeline stage ---
    vad = SileroVADAnalyzer(
        sample_rate=16000,
        params=VADParams(
            confidence=cfg.vad_threshold,
            start_secs=cfg.min_speech_ms / 1000,
            stop_secs=cfg.endpointing_ms / 1000,
        ),
    )

    # --- 3. STT (HTTP → Mac /transcribe) ---
    stt = WhisperHTTPSTTService(
        base_url=cfg.server_url,
        sample_rate=16000,
        language="en",
        timeout=30.0,
    )

    # --- 4. LLM (OpenRouter) ---
    llm = OpenRouterLLMService(
        api_key=cfg.openrouter_api_key,
        settings=OpenRouterLLMService.Settings(
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        ),
    )

    # --- 5. TTS (HTTP → Mac /speak) ---
    tts = KokoroHTTPTTSService(
        base_url=cfg.server_url,
        sample_rate=24000,
        timeout=30.0,
    )

    # --- 6. Register tool handlers on the LLM ---
    for schema in TOOL_SCHEMAS:
        handler = make_handler(schema.name)
        llm.register_function(schema.name, handler)
        log.info("Registered tool: %s — %s", schema.name, schema.description)

    # --- 7. LLM Context with system prompt ---
    lmc = LLMContext(
        messages=[
            {"participant": "system", "text": GENIE_SYSTEM_PROMPT},
        ],
    )

    # Register tool schemas on the context
    tool_schemas: list[FunctionSchema] = []
    for schema in TOOL_SCHEMAS:
        fs = FunctionSchema(
            name=schema.name,
            description=schema.description,
            properties=schema.parameters.get("properties", {}),
            required=schema.parameters.get("required", []),
        )
        tool_schemas.append(fs)
    lmc.set_tools(tool_schemas)

    # --- 8. Aggregator pair (VAD drives user turn detection) ---
    pair = LLMContextAggregatorPair(
        lmc,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    # --- 9. Assemble pipeline ---
    # Correct cascade order (verified against Pipecat 1.15.0):
    #   transport.input() → stt → user agg → llm → assistant agg → tts → transport.output()
    #
    # Frame flow:
    #   InputAudioRawFrame: transport → stt (buffers) → user (passes) → llm (ignores)
    #   VADUserStartedSpeakingFrame: user broadcasts → stt (upstream) + llm (downstream)
    #   TranscriptionFrame: stt → user (aggregates, pushes LLMContextFrame) → llm (consumes)
    #   LLMFullResponseStartFrame: llm → assistant (starts collecting) → tts (ignores)
    #   LLMTextFrame: llm → assistant (collects, converts to TextFrame) → tts (synthesizes)
    #   TTSAudioRawFrame: tts → transport.output() (plays)
    pipeline = Pipeline([
        transport.input(),
        stt,
        pair.user(),
        llm,
        pair.assistant(),
        tts,
        transport.output(),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    """Run the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load configuration from .env + environment variables
    cfg = Config.load()
    log.info("Loaded config: server=%s, model=%s", cfg.server_url, cfg.llm_model)

    if not cfg.openrouter_api_key:
        log.error(
            "OPENROUTER_API_KEY is empty. Please add your key to .env or "
            "set the OPENROUTER_API_KEY environment variable.\n"
            "  Example: echo 'OPENROUTER_API_KEY=sk-or-...' >> .env"
        )
        sys.exit(1)

    pipeline = build_pipeline(cfg)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
        idle_timeout_secs=60,
    )

    runner = WorkerRunner(handle_sigint=True)

    log.info("Starting Genie pipeline...")
    log.info("Say something to begin the conversation.")
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
