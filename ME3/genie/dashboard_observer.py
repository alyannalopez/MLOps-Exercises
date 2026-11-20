"""M4.76: Dashboard Observer — transparent FrameProcessor that bridges pipeline
events to the TerminalDashboard.

This observer:
  - Sits at the END of the pipeline (after transport.output(), before pair.assistant()).
  - Observes all downstream frames that have traversed the full pipeline.
  - Also observes UserStartedSpeakingFrame / UserStoppedSpeakingFrame which
    are emitted by LLMUserAggregator via broadcast_frame() in BOTH directions
    (upstream to STT and downstream through the rest of the pipeline). Because
    the observer sits downstream of LLMUserAggregator, it receives the
    DOWNSTREAM copies of these VAD frames.
  - Forwards every frame unchanged and immediately — never blocks frame flow.
  - Subscribes to TaskManager events for background task visibility.
  - Uses non-blocking queue delivery to the dashboard.
  - Tracks all spawned dashboard-update tasks for safe shutdown.

Frame coverage:
  DOWNSTREAM (through this observer):
    - TranscriptionFrame: user speech recognized (from STT → LLM → TTS → transport)
    - UserStartedSpeakingFrame: VAD detected user started speaking (broadcast)
    - UserStoppedSpeakingFrame: VAD detected user stopped speaking (broadcast)
    - LLMTextFrame: Genie's response text
    - LLMFullResponseStartFrame / LLMFullResponseEndFrame: LLM thinking boundaries
    - TTSAudioRawFrame: TTS audio sent to output device (not confirmed speaker playback)
    - TTSSpeakFrame: TTS speak request
    - InterruptionFrame: user interrupted
    - FunctionCallsStartedFrame: tool call initiated
    - FunctionCallResultFrame: tool call completed
    - ErrorFrame: pipeline errors
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)

from genie.terminal_dashboard import GenieState

if TYPE_CHECKING:
    from genie.task_manager import TaskManager, TaskEvent


log = logging.getLogger(__name__)


class DashboardObserver(FrameProcessor):
    """Transparent frame processor that observes pipeline events and forwards
    them to the TerminalDashboard.

    Frames are forwarded unchanged via push_frame(). Dashboard updates are
    scheduled as tracked async tasks that do not block frame processing.
    """

    def __init__(
        self,
        dashboard: Any,
        task_manager: TaskManager | None = None,
    ) -> None:
        super().__init__()
        self._dashboard = dashboard
        self._tm = task_manager

        # Tracked dashboard-update tasks for safe shutdown
        self._pending_tasks: set[asyncio.Task] = set()

        # Internal state tracking
        self._llm_active = False
        self._current_genie_text = ""
        self._tts_audio_seen = False

        # Subscribe to TaskManager events if available
        if task_manager:
            task_manager.on_event(self._on_task_event)

    # ---- Frame processing (Pipecat interface) ------------------------------

    async def process_frame(self, frame, direction: FrameDirection):
        """Process a frame: observe it for the dashboard, then forward unchanged.

        Returns None — frames are forwarded via push_frame() as required by
        Pipecat's FrameProcessor contract.
        """
        self._observe_frame(frame, direction)
        await self.push_frame(frame, direction)

    # ---- Observation -------------------------------------------------------

    def _observe_frame(self, frame, direction: FrameDirection):
        """Observe a frame and schedule dashboard updates.

        Only DOWNSTREAM frames are observed. UserStartedSpeakingFrame and
        UserStoppedSpeakingFrame reach this observer via the DOWNSTREAM copies
        emitted by LLMUserAggregator.broadcast_frame().
        """
        if direction != FrameDirection.DOWNSTREAM:
            return

        if isinstance(frame, TranscriptionFrame):
            text = frame.text if hasattr(frame, "text") else ""
            if text:
                self._schedule(self._dashboard.add_user_text(text))
            self._schedule(self._dashboard.set_state(GenieState.IDLE))

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._schedule(self._dashboard.set_state(GenieState.LISTENING))

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._schedule(self._dashboard.set_state(GenieState.ASR_PROCESSING))

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_active = True
            self._current_genie_text = ""
            self._tts_audio_seen = False
            self._schedule(self._dashboard.set_state(GenieState.THINKING))

        elif isinstance(frame, LLMTextFrame):
            if self._llm_active:
                text = frame.text if hasattr(frame, "text") else ""
                self._current_genie_text += text
                self._schedule(self._dashboard.add_genie_text(text))

        elif isinstance(frame, TTSSpeakFrame):
            self._schedule(self._dashboard.set_state(GenieState.TTS_GENERATING))

        elif isinstance(frame, TTSAudioRawFrame):
            if not self._tts_audio_seen:
                self._tts_audio_seen = True
                # First audio chunk — audio is being sent to the output device.
                # This is not confirmed speaker playback; the label AUDIO_OUTPUT
                # reflects what was actually observed.
                self._schedule(self._dashboard.set_state(GenieState.AUDIO_OUTPUT))

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_active = False
            self._current_genie_text = ""
            self._tts_audio_seen = False

        elif isinstance(frame, InterruptionFrame):
            self._llm_active = False
            self._current_genie_text = ""
            self._tts_audio_seen = False
            self._schedule(self._dashboard.set_state(GenieState.INTERRUPTED))
            self._schedule(self._reset_after_interrupt())

        elif isinstance(frame, FunctionCallsStartedFrame):
            calls = getattr(frame, "function_calls", [])
            for call in calls:
                name = getattr(call, "function_name", "unknown")
                args = getattr(call, "arguments", {})
                args_str = str(args)[:80] if args else ""
                self._schedule(
                    self._dashboard.add_tool_start(name, args_str)
                )

        elif isinstance(frame, FunctionCallResultFrame):
            # FunctionCallResultFrame carries a single result (flat structure)
            name = getattr(frame, "function_name", "unknown")
            result = getattr(frame, "result", "")
            status = "success" if not isinstance(result, Exception) else "error"
            summary = str(result)[:100] if result else ""
            self._schedule(
                self._dashboard.add_tool_result(name, status, summary)
            )

        elif isinstance(frame, ErrorFrame):
            error = frame.error if hasattr(frame, "error") else str(frame)
            self._schedule(self._dashboard.add_error(str(error)))

    # ---- Task tracking -----------------------------------------------------

    def _schedule(self, coro) -> None:
        """Schedule a dashboard update task, tracking it for safe shutdown."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _reset_after_interrupt(self) -> None:
        """Reset to IDLE after a brief delay post-interruption."""
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return
        await self._dashboard.set_state(GenieState.IDLE)

    # ---- TaskManager event handler -----------------------------------------

    async def _on_task_event(self, event: TaskEvent) -> None:
        """Handle TaskManager events and forward to dashboard."""
        etype = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        desc = event.message or "Unknown task"
        result = str(event.result) if hasattr(event, "result") and event.result else ""
        self._schedule(
            self._dashboard.add_task_event(
                event.task_id, desc, etype, result
            )
        )

    # ---- Shutdown ----------------------------------------------------------

    async def shutdown(self) -> None:
        """Await all pending dashboard-update tasks."""
        if self._pending_tasks:
            log.debug(
                "DashboardObserver: awaiting %d pending tasks",
                len(self._pending_tasks),
            )
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
