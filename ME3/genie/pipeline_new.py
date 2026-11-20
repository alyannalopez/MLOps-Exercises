"""Genie pipeline — Pipecat-native full-duplex voice assistant.

Architecture (each stage is a separate Pipecat component):

  transport.input()   — mic capture → InputAudioRawFrame
    → WhisperHTTPSTTService — SegmentedSTTService, VAD-gated, calls Mac /transcribe, yields TranscriptionFrame
      → LLMUserAggregator   — collects TranscriptionFrame → LLMContext (VAD-driven turn detection)
        → OpenRouterLLMService — LLM reasoning + tool calling
          → KokoroHTTPTTSService — HTTP POST to Mac /speak, yields TTSAudioRawFrame
            → transport.output() — speaker playback, 24kHz mono
              → LLMAssistantAggregator — collects LLM text for context (after TTS, doesn't block audio)

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
from pipecat.processors.aggregators.llm_response_universal import EmptyUserTurnConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.mcp_service import MCPClient, StreamableHttpParameters
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from genie.brain import build_system_prompt
from genie.config import Config
from genie.mcp_server import server as mcp_server
from genie.services import KokoroHTTPTTSService, WhisperHTTPSTTService
from genie.task_manager import get_task_manager, NotificationCoordinator
# Dashboard imports kept for future re-enablement (M4.76):
# from genie.terminal_dashboard import TerminalDashboard
# from genie.dashboard_observer import DashboardObserver

log = logging.getLogger("genie")


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

async def build_pipeline(cfg: Config, **kwargs) -> Pipeline:
    """Build the full-duplex voice assistant pipeline with native Pipecat stages."""
    log.info("Model server: %s", cfg.server_url)

    # --- Validate API key ---
    if not cfg.openrouter_api_key:
        log.warning(
            "OPENROUTER_API_KEY not set in .env or environment. "
            "LLM calls will fail. Add your key to .env or set the env var."
        )

    # --- 1. Audio transport (mic in → speaker out) ---
    mic_idx = cfg.mic_device
    if mic_idx is None:
        # Auto-detect: prefer USB wireless mic, fall back to built-in
        import pyaudio
        pa = pyaudio.PyAudio()
        mic_idx = -1
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev["maxInputChannels"] < 1:
                continue
            name = dev["name"].lower()
            # Prefer USB wireless mic (TH30), then built-in
            if "th30" in name or "usb" in name or "wireless" in name:
                mic_idx = i
                break
            elif "microphone" in name or "air" in name:
                if mic_idx == -1:
                    mic_idx = i  # fallback candidate
        dev_name = pa.get_device_info_by_index(mic_idx)["name"]
        pa.terminate()
        if mic_idx == -1:
            raise RuntimeError(
                "No microphone found. Set GENIE_MIC_DEVICE in .env "
                "to the index of your desired input device."
            )
        log.info("Auto-detected mic: %s (device %d)", dev_name, mic_idx)
    else:
        log.info("Using configured mic device: %d", mic_idx)

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            audio_out_channels=1,
            input_device_index=mic_idx,
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
            min_volume=0.01,  # Lowered: built-in mics produce low-amplitude signals
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

    # --- 6. MCP Client — discovers and registers tools from MCP server ---
    mcp = MCPClient(
        server_params=StreamableHttpParameters(
            url=cfg.mcp_url,
            timeout=30.0,
        )
    )
    mcp_tools = await mcp.tools()
    log.info("MCP tools discovered: %d", len(mcp_tools.standard_tools))
    for t in mcp_tools.standard_tools:
        log.info("  - %s: %s", t.name, t.description[:70])

    # --- 7. LLM Context with dynamic system prompt + MCP tools ---
    system_prompt = build_system_prompt(cfg)
    lmc = LLMContext(
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        tools=mcp_tools,  # MCP tools auto-register handlers with LLM
    )

    # --- 8. Aggregator pair (VAD drives user turn detection) ---
    pair = LLMContextAggregatorPair(
        lmc,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            empty_user_turn=EmptyUserTurnConfig(
                interrupted_prompt=None,
                idle_prompt=None,
            ),
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    # --- 9. Assemble pipeline ---
    # Correct cascade order (verified against Pipecat 1.15.0):
    #   transport.input() → stt → user agg → llm → tts → transport.output()
    #     → dashboard observer → assistant agg
    # The DashboardObserver is transparent — it forwards all frames unchanged
    # while notifying the terminal dashboard of state changes.
    dashboard_observer: FrameProcessor | None = kwargs.get("dashboard_observer")
    stages: list[FrameProcessor] = [
        transport.input(),
        stt,
        pair.user(),
        llm,
        tts,
        transport.output(),
    ]
    if dashboard_observer is not None:
        stages.append(dashboard_observer)
    stages.append(pair.assistant())
    pipeline = Pipeline(stages)

    return pipeline, tts


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    """Run the pipeline with MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load configuration from .env + environment variables
    cfg = Config.load()
    log.info("Loaded config: server=%s, model=%s, mcp=%s",
             cfg.server_url, cfg.llm_model, cfg.mcp_url)

    if not cfg.openrouter_api_key:
        log.error(
            "OPENROUTER_API_KEY is empty. Please add your key to .env or "
            "set the OPENROUTER_API_KEY environment variable.\n"
            "  Example: echo 'OPENROUTER_API_KEY=sk-or-...' >> .env"
        )
        sys.exit(1)

    # --- Start MCP server in background ---
    log.info("Starting MCP server on %s:%d...", cfg.mcp_host, cfg.mcp_port)
    mcp_task = asyncio.create_task(
        mcp_server.run_streamable_http_async(
            host=cfg.mcp_host,
            port=cfg.mcp_port,
            streamable_http_path="/mcp",
        )
    )
    # Give the MCP server a moment to bind its port
    await asyncio.sleep(1)

    # Verify MCP server is alive before continuing
    if mcp_task.done():
        exc = mcp_task.exception()
        if exc:
            if "Address already in use" in str(exc) or "OSError" in str(type(exc).__name__):
                log.error(
                    "Port %d is already in use. Another MCP server may be running. "
                    "Kill it or change MCP_PORT in .env.",
                    cfg.mcp_port,
                )
            else:
                log.error("MCP server failed to start: %s", exc)
            sys.exit(1)
        log.error("MCP server task finished unexpectedly.")
        sys.exit(1)

    log.info("MCP server started on %s", cfg.mcp_url)

    # --- Build pipeline (includes MCP client connection) ---
    # --- M4.5: Task Manager ---
    tm = get_task_manager()

    # --- M4.76: Terminal Dashboard — DISABLED for demo stability ---
    # dashboard = TerminalDashboard()
    # await dashboard.start()
    # obs = DashboardObserver(dashboard=dashboard, task_manager=tm)
    # pipeline, tts_service = await build_pipeline(cfg, dashboard_observer=obs)
    # --- End dashboard disable ---
    pipeline, tts_service = await build_pipeline(cfg, dashboard_observer=None)
    notifier = NotificationCoordinator(tts_service, tm)
    await notifier.start()
    log.info("M4.5: Notification coordinator started")

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

    # Cleanup: stop notification coordinator and MCP server
    await notifier.stop()
    log.info("M4.5: Notification coordinator stopped")
    # --- M4.76: Dashboard cleanup — DISABLED ---
    # await obs.shutdown()
    # log.info("M4.76: Dashboard observer stopped")
    # await dashboard.shutdown()
    # log.info("M4.76: Dashboard stopped")
    # --- End dashboard cleanup disable ---
    mcp_task.cancel()
    try:
        await mcp_task
    except asyncio.CancelledError:
        pass
    log.info("MCP server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
