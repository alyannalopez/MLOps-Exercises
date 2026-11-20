"""M4.76: Terminal Dashboard — Rich-based real-time status display for Genie.

Observes the running Genie pipeline and presents:
  - Startup / service status
  - Current state (Idle, Listening, Thinking, Audio Output, Interrupted)
  - User transcripts and Genie responses
  - MCP tool execution
  - Background tasks (state, progress)
  - Errors and runtime info

Design principles:
  - Non-blocking: all dashboard updates are queued via asyncio.Queue.
  - Failure-isolated: Rich rendering errors never crash the pipeline.
  - Bounded memory: conversation log and tool log are capped.
  - Graceful shutdown: terminal is restored on exit.
  - Priority queue: conversation, task, and error events are preserved
    over routine status updates.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dashboard event types
# ---------------------------------------------------------------------------

class DashboardEventKind(str, Enum):
    STATE_CHANGE = "state"
    CONVERSATION = "conversation"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    TASK_EVENT = "task_event"
    ERROR = "error"
    LOG = "log"


_IMPORTANT_KINDS = frozenset({
    DashboardEventKind.CONVERSATION,
    DashboardEventKind.TASK_EVENT,
    DashboardEventKind.ERROR,
})


@dataclass
class DashboardEvent:
    kind: DashboardEventKind
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Genie states
# ---------------------------------------------------------------------------

class GenieState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    ASR_PROCESSING = "asr_processing"
    THINKING = "thinking"
    TTS_GENERATING = "tts_generating"
    AUDIO_OUTPUT = "audio_output"
    INTERRUPTED = "interrupted"


STATE_COLORS = {
    GenieState.IDLE: "dim",
    GenieState.LISTENING: "bold yellow",
    GenieState.ASR_PROCESSING: "yellow",
    GenieState.THINKING: "bold blue",
    GenieState.TTS_GENERATING: "blue",
    GenieState.AUDIO_OUTPUT: "bold green",
    GenieState.INTERRUPTED: "bold red",
}

STATE_EMOJI = {
    GenieState.IDLE: "💤",
    GenieState.LISTENING: "👂",
    GenieState.ASR_PROCESSING: "🎙️",
    GenieState.THINKING: "🧠",
    GenieState.TTS_GENERATING: "📝",
    GenieState.AUDIO_OUTPUT: "🔊",
    GenieState.INTERRUPTED: "✋",
}


# ---------------------------------------------------------------------------
# Terminal Dashboard
# ---------------------------------------------------------------------------

class TerminalDashboard:
    """Real-time terminal display for Genie pipeline status.

    Receives events via a bounded asyncio.Queue. A background task
    (_run_loop) drains the queue and updates a Rich Live display.
    """

    MAX_CONVERSATION = 20
    MAX_TOOLS = 15
    MAX_LOG = 30
    QUEUE_SIZE = 100

    def __init__(self) -> None:
        self._console = Console(force_terminal=True, soft_wrap=False)
        self._live: Live | None = None
        self._run_task: asyncio.Task | None = None
        self._running = False

        self._queue: asyncio.Queue[DashboardEvent] = asyncio.Queue(
            maxsize=self.QUEUE_SIZE
        )

        self._state = GenieState.IDLE
        self._started_at = time.time()

        # Conversation log: [(role, text), ...]
        self._conversation: list[tuple[str, str]] = []

        # Tool log: [{"name", "status", "summary"}, ...]
        self._tool_log: list[dict[str, str]] = []

        # Background tasks: {task_id: {"description", "state", "detail"}}
        self._bg_tasks: dict[str, dict[str, str]] = {}

        # System log: [(level, message), ...]
        self._system_log: list[tuple[str, str]] = []

    # ---- Properties --------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_state(self) -> GenieState:
        return self._state

    # ---- Public API --------------------------------------------------------

    async def start(self) -> None:
        """Start the dashboard display and background event loop."""
        self._running = True
        self._started_at = time.time()

        panel = Panel(
            self._build_display(),
            title="[bold cyan]Genie[/] Terminal Dashboard",
            border_style="cyan",
        )
        self._live = Live(panel, console=self._console, refresh_per_second=4)
        self._live.start()

        self._run_task = asyncio.create_task(self._run_loop())

    async def shutdown(self) -> None:
        """Stop the dashboard display and restore the terminal."""
        self._running = False

        # Cancel the run loop — it catches CancelledError internally
        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        if self._live:
            self._live.stop()
            self._live = None

        if self._console:
            self._console.print(
                "\n[bold green]Genie stopped.[/]", justify="center"
            )

    # ---- Event submission (called from observer / task manager) ------------

    def put_event(self, event: DashboardEvent) -> bool:
        """Non-blocking event enqueue with priority for important events.

        Important events (conversation, task, error) evict a routine event
        from the front of the queue if the queue is full. If the queue is
        full of important events, the new event is logged and dropped.

        Returns True if enqueued, False if dropped.
        """
        important = event.kind in _IMPORTANT_KINDS
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            if important:
                # Try to evict one routine event from the front
                evicted = False
                while not self._queue.empty():
                    candidate = self._queue.get_nowait()
                    if candidate.kind in _IMPORTANT_KINDS:
                        # Put it back — queue is full of important events
                        try:
                            self._queue.put_nowait(candidate)
                        except asyncio.QueueFull:
                            pass
                        break
                    else:
                        # Evicted a routine event — make room
                        evicted = True
                        self._queue.put_nowait(event)
                        # Restore any remaining important events we pulled
                        break
                if not evicted:
                    log.warning(
                        "Dashboard queue full of important events, "
                        "dropping: %s",
                        event.kind.value,
                    )
                    return False
            else:
                # Routine event dropped — acceptable, status coalesces
                return False
        return True

    async def set_state(self, state: GenieState) -> None:
        self.put_event(DashboardEvent(DashboardEventKind.STATE_CHANGE, {"state": state}))

    async def add_user_text(self, text: str) -> None:
        if not text.strip():
            return
        self.put_event(DashboardEvent(
            DashboardEventKind.CONVERSATION,
            {"role": "user", "text": text.strip()},
        ))

    async def add_genie_text(self, text: str) -> None:
        if not text.strip():
            return
        self.put_event(DashboardEvent(
            DashboardEventKind.CONVERSATION,
            {"role": "genie", "text": text.strip()},
        ))

    async def add_tool_start(self, name: str, args: str = "") -> None:
        self.put_event(DashboardEvent(
            DashboardEventKind.TOOL_START,
            {"name": name, "args": args},
        ))

    async def add_tool_result(
        self, name: str, status: str = "success", summary: str = ""
    ) -> None:
        self.put_event(DashboardEvent(
            DashboardEventKind.TOOL_RESULT,
            {"name": name, "status": status, "summary": summary},
        ))

    async def add_task_event(
        self, task_id: str, description: str, event_type: str, detail: str = "",
    ) -> None:
        self.put_event(DashboardEvent(
            DashboardEventKind.TASK_EVENT,
            {"task_id": task_id, "description": description,
             "event_type": event_type, "detail": detail},
        ))

    async def add_error(self, message: str) -> None:
        self.put_event(DashboardEvent(
            DashboardEventKind.ERROR, {"message": message}
        ))

    async def add_log(self, level: str, message: str) -> None:
        self.put_event(DashboardEvent(
            DashboardEventKind.LOG, {"level": level, "message": message}
        ))

    # ---- Internal event loop -----------------------------------------------

    async def _run_loop(self) -> None:
        try:
            while self._running:
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=0.05
                    )
                except asyncio.TimeoutError:
                    continue

                try:
                    await self._handle_event(event)
                except Exception:
                    log.exception("Dashboard event handler error")

                if self._live:
                    panel = Panel(
                        self._build_display(),
                        title="[bold cyan]Genie[/] Terminal Dashboard",
                        border_style="cyan",
                    )
                    self._live.update(panel)
        except asyncio.CancelledError:
            pass

    async def _handle_event(self, event: DashboardEvent) -> None:
        if event.kind == DashboardEventKind.STATE_CHANGE:
            self._state = event.data.get("state", self._state)

        elif event.kind == DashboardEventKind.CONVERSATION:
            role = event.data.get("role", "unknown")
            text = event.data.get("text", "")
            self._conversation.append((role, text))
            if len(self._conversation) > self.MAX_CONVERSATION:
                self._conversation = self._conversation[-self.MAX_CONVERSATION:]

        elif event.kind == DashboardEventKind.TOOL_START:
            self._tool_log.append({
                "name": event.data.get("name", "unknown"),
                "status": "running",
                "summary": "",
            })
            if len(self._tool_log) > self.MAX_TOOLS:
                self._tool_log = self._tool_log[-self.MAX_TOOLS:]

        elif event.kind == DashboardEventKind.TOOL_RESULT:
            name = event.data.get("name", "unknown")
            status = event.data.get("status", "unknown")
            for entry in reversed(self._tool_log):
                if entry["name"] == name and entry["status"] == "running":
                    entry["status"] = status
                    entry["summary"] = event.data.get("summary", "")
                    break

        elif event.kind == DashboardEventKind.TASK_EVENT:
            tid = event.data.get("task_id", "")
            desc = event.data.get("description", "")
            etype = event.data.get("event_type", "")
            detail = event.data.get("detail", "")
            if tid:
                state_map = {
                    "created": "created", "running": "running",
                    "completed": "completed", "failed": "failed",
                    "cancelled": "cancelled",
                }
                self._bg_tasks[tid] = {
                    "description": desc,
                    "state": state_map.get(etype, etype),
                    "detail": detail,
                }

        elif event.kind == DashboardEventKind.ERROR:
            self._system_log.append(("error", event.data.get("message", "")))
            if len(self._system_log) > self.MAX_LOG:
                self._system_log = self._system_log[-self.MAX_LOG:]

        elif event.kind == DashboardEventKind.LOG:
            self._system_log.append((
                event.data.get("level", "info"),
                event.data.get("message", ""),
            ))
            if len(self._system_log) > self.MAX_LOG:
                self._system_log = self._system_log[-self.MAX_LOG:]

    # ---- Display rendering -------------------------------------------------

    def _build_display(self) -> str:
        lines: list[str] = []

        # State
        color = STATE_COLORS.get(self._state, "white")
        emoji = STATE_EMOJI.get(self._state, "❓")
        lines.append(f"[{color}]State: {emoji} {self._state.value}[/]")
        lines.append("")

        # Conversation
        if self._conversation:
            lines.append("[bold]Conversation:[/]")
            for role, text in self._conversation[-8:]:
                icon = "👤" if role == "user" else "🤖"
                preview = text[:100] + ("…" if len(text) > 100 else "")
                lines.append(f"  {icon} {preview}")
            lines.append("")

        # Tools
        active_tools = [t for t in self._tool_log if t["status"] == "running"]
        if active_tools:
            lines.append("[bold yellow]Active tools:[/]")
            for t in active_tools[-5:]:
                lines.append(f"  ⏳ {t['name']}")
            lines.append("")

        recent_tools = [t for t in self._tool_log if t["status"] != "running"]
        if recent_tools:
            lines.append("[dim]Recent tools:[/]")
            for t in recent_tools[-3:]:
                status_icon = "✅" if t["status"] == "success" else "❌"
                lines.append(f"  {status_icon} {t['name']}")
            lines.append("")

        # Background tasks
        active_tasks = [
            (tid, t) for tid, t in self._bg_tasks.items()
            if t["state"] in ("created", "running")
        ]
        if active_tasks:
            lines.append("[bold]Tasks:[/]")
            for tid, t in active_tasks:
                lines.append(
                    f"  ⏳ {t['description']} "
                    f"({t['state']})"
                )
            lines.append("")

        # Errors
        errors = [m for lvl, m in self._system_log if lvl == "error"]
        if errors:
            lines.append("[bold red]Errors:[/]")
            for msg in errors[-5:]:
                lines.append(f"  ❌ {msg[:80]}")
            lines.append("")

        # Uptime
        lines.append(f"[dim]Uptime: {self._format_uptime()}[/]")

        return "\n".join(lines)

    def _format_uptime(self) -> str:
        elapsed = int(time.time() - self._started_at)
        mins, secs = divmod(elapsed, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}h {mins}m {secs}s"
        if mins:
            return f"{mins}m {secs}s"
        return f"{secs}s"
