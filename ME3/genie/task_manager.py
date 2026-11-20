"""M4.5: Background Task Manager — manages long-running operations independently.

Provides:
  * TaskManager — in-process singleton managing task lifecycle (PENDING → RUNNING →
    CANCELLED/COMPLETED/FAILED).
  * TaskEvent — immutable event published on state transitions.
  * NotificationCoordinator — consumes TaskEvents, coalesces progress, and queues
    TTSSpeakFrame announcements to the TTS service.
  * ResearchRunner — multi-step web-research runner using the existing web_search
    capability, with isolated LLM context for summarization.

Design principles:
  - Tasks never block the Pipecat voice pipeline.
  - Background LLM summarization uses a fresh, independent context (never mutates
    the active conversation).
  - Cancellation is honest: CANCELLATION_REQUESTED is a distinct state from CANCELLED.
  - Blocking DDGS threads cannot be killed; their results are suppressed on cancel.
  - Task results are always stored in TaskManager, even if the spoken announcement
    is interrupted.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from loguru import logger

log = logger.bind(name="genie.task_manager")


# ---------------------------------------------------------------------------
# Task event types
# ---------------------------------------------------------------------------
class TaskEventType(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Task state machine
# ---------------------------------------------------------------------------
class TaskState(Enum):
    PENDING = auto()               # Created, not yet started
    RUNNING = auto()               # Actively executing
    CANCELLATION_REQUESTED = auto()  # Cancel requested, task still running
    CANCELLED = auto()             # Task has actually stopped
    COMPLETED = auto()             # Finished successfully
    FAILED = auto()                # Finished with error

    @property
    def is_terminal(self) -> bool:
        return self in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)


# ---------------------------------------------------------------------------
# Task event
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskEvent:
    """Immutable event published on task state transitions."""
    task_id: str
    event_type: TaskEventType
    message: str = ""
    result: str = ""
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def started(cls, task_id: str, message: str = "") -> "TaskEvent":
        return cls(task_id=task_id, event_type=TaskEventType.STARTED, message=message)

    @classmethod
    def progress(cls, task_id: str, message: str) -> "TaskEvent":
        return cls(task_id=task_id, event_type=TaskEventType.PROGRESS, message=message)

    @classmethod
    def completed(cls, task_id: str, result: str) -> "TaskEvent":
        return cls(task_id=task_id, event_type=TaskEventType.COMPLETED, result=result)

    @classmethod
    def failed(cls, task_id: str, error: str) -> "TaskEvent":
        return cls(task_id=task_id, event_type=TaskEventType.FAILED, error=error)

    @classmethod
    def cancelled(cls, task_id: str, message: str = "") -> "TaskEvent":
        return cls(task_id=task_id, event_type=TaskEventType.CANCELLED, message=message)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class Task:
    """Represents a single background task."""

    def __init__(self, task_id: str, description: str):
        self.id = task_id
        self.description = description
        self.state = TaskState.PENDING
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.progress: list[str] = []
        self.result: str | None = None
        self.error: str | None = None
        self._cancelled: bool = False

    def check_cancellation(self) -> bool:
        """Raise TaskCancelledError if cancellation was requested."""
        if self._cancelled:
            raise TaskCancelledError(f"Task {self.id} was cancelled")
        return False

    def request_cancellation(self) -> bool:
        """Request cancellation. Returns True if the request was accepted."""
        if self.state.is_terminal:
            return False
        if self.state == TaskState.PENDING:
            # Cancel before it started
            self.state = TaskState.CANCELLED
            self._cancelled = True
            return True
        if self.state == TaskState.CANCELLATION_REQUESTED:
            return False  # Already requested
        self.state = TaskState.CANCELLATION_REQUESTED
        self._cancelled = True
        return True


class TaskCancelledError(Exception):
    """Raised when a cooperative task detects its cancellation flag."""
    pass


# ---------------------------------------------------------------------------
# Task Manager
# ---------------------------------------------------------------------------
class TaskManager:
    """Manages background tasks. Designed as an in-process singleton.

    All tasks run within the same asyncio event loop as the Pipecat pipeline.
    Blocking operations (e.g., DDGS) are dispatched to ThreadPoolExecutor workers.
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._event_handlers: list[Callable[[TaskEvent], Coroutine]] = []

    # ---- Public API ----

    async def create_task(
        self,
        description: str,
        coro: Coroutine[Any, Any, str],
        on_task_created: Callable[[Task], None] | None = None,
    ) -> str:
        """Create and start a background task.

        Args:
            description: Human-readable description of the task.
            coro: Async coroutine that performs the work. Must periodically call
                  task.check_cancellation() or listen for cancellation.

        Returns:
            task_id: The unique identifier for the task.
        """
        task_id = str(uuid.uuid4())[:8]
        task = Task(task_id, description)
        self._tasks[task_id] = task

        # Notify caller of the task reference (for closures that need it)
        if on_task_created:
            on_task_created(task)

        # Wrap the coroutine with lifecycle management
        asyncio.create_task(self._run_task(task, coro))
        return task_id

    async def cancel_task(self, task_id: str) -> str:
        """Request cancellation of a running task.

        Returns:
            Status message describing what happened.
        """
        task = self._tasks.get(task_id)
        if not task:
            return f"Task {task_id} not found."

        if task.state.is_terminal:
            return f"Task {task_id} is already {task.state.name.lower()}."

        if task.request_cancellation():
            if task.state == TaskState.CANCELLED:
                # Cancelled before it started — publish immediately
                await self._publish(TaskEvent.cancelled(task_id, "Cancelled before it started."))
                return f"Task {task_id} cancelled (had not started)."
            return f"Task {task_id} cancellation requested. It will stop when safe."

        return f"Task {task_id} cancellation already requested."

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict[str, Any]]:
        """List all tasks with their current status."""
        result = []
        for tid, task in self._tasks.items():
            entry: dict[str, Any] = {
                "id": task.id,
                "description": task.description,
                "state": task.state.name,
                "created_at": task.created_at,
                "progress": task.progress[-3:] if task.progress else [],  # Last 3 steps
            }
            if task.result:
                entry["result"] = task.result[:200]  # Truncated preview
            if task.error:
                entry["error"] = task.error
            result.append(entry)
        return result

    def get_task_result(self, task_id: str) -> str:
        """Get the full result or error of a task.

        Returns:
            The result string, error, or a status message if not complete.
        """
        task = self._tasks.get(task_id)
        if not task:
            return f"Task {task_id} not found."
        if task.state == TaskState.COMPLETED and task.result:
            return task.result
        if task.state == TaskState.FAILED and task.error:
            return f"Task failed: {task.error}"
        if task.state == TaskState.CANCELLED:
            return f"Task was cancelled."
        return f"Task {task_id} is {task.state.name.lower()}. No result yet."

    def on_event(self, handler: Callable[[TaskEvent], Coroutine]):
        """Register an event handler (e.g., NotificationCoordinator)."""
        self._event_handlers.append(handler)

    async def _publish(self, event: TaskEvent):
        """Publish an event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception as e:
                log.warning("Event handler error: %s", e)

    # ---- Internal task runner ----

    async def _run_task(self, task: Task, coro: Coroutine[Any, Any, str]):
        """Execute a task coroutine with lifecycle management."""
        # Transition PENDING → RUNNING
        task.state = TaskState.RUNNING
        await self._publish(TaskEvent.started(task.id, task.description))

        try:
            result = await coro
        except TaskCancelledError:
            # Cooperative cancellation — task noticed and exited
            task.state = TaskState.CANCELLED
            await self._publish(TaskEvent.cancelled(task.id, "Task stopped cooperatively."))
            return
        except asyncio.CancelledError:
            # asyncio-level cancellation (shouldn't happen for our tasks,
            # but handle defensively)
            if task.state == TaskState.CANCELLATION_REQUESTED:
                task.state = TaskState.CANCELLED
                await self._publish(TaskEvent.cancelled(task.id, "Task cancelled."))
            else:
                task.state = TaskState.FAILED
                task.error = "Task was externally cancelled."
                await self._publish(TaskEvent.failed(task.id, "Task was externally cancelled."))
            return
        except Exception as e:
            # Unexpected error
            if task.state == TaskState.CANCELLATION_REQUESTED:
                # Error occurred during cancellation — treat as cancelled
                task.state = TaskState.CANCELLED
                task.error = f"Task failed during cancellation: {e}"
                await self._publish(TaskEvent.cancelled(task.id, f"Failed during cancellation: {e}"))
            else:
                task.state = TaskState.FAILED
                task.error = str(e)
                await self._publish(TaskEvent.failed(task.id, str(e)))
            return

        # Check if cancellation was requested AFTER the work completed
        if task.state == TaskState.CANCELLATION_REQUESTED:
            # Work finished but cancellation was pending — suppress the result
            task.state = TaskState.CANCELLED
            task.result = None
            await self._publish(
                TaskEvent.cancelled(
                    task.id,
                    "Blocking operation completed after cancellation request. Result suppressed.",
                )
            )
            return

        # Normal completion
        task.state = TaskState.COMPLETED
        task.result = result
        await self._publish(TaskEvent.completed(task.id, result))


# ---------------------------------------------------------------------------
# Notification Coordinator
# ---------------------------------------------------------------------------
class NotificationCoordinator:
    """Consumes TaskEvents and produces TTSSpeakFrame announcements.

    Design:
    - Progress events are coalesced (never spoken, available via get_task_status).
    - Only completed/failed/cancelled events trigger announcements.
    - Announcements are queued as TTSSpeakFrame(append_to_context=False) to the
      TTS service. The TTS service's internal audio context management naturally
      serializes playback — _hold_frames_while_audio_contexts_open() blocks new
      frames while audio is playing.
    - If an announcement is interrupted by the user, the full result remains
      available via get_task_result.
    - Duplicate announcements for the same task are prevented.
    """

    def __init__(self, tts_service, task_manager: TaskManager):
        self._tts_service = tts_service
        self._task_manager = task_manager
        self._announced_tasks: set[str] = set()
        self._pending: dict[str, TaskEvent] = {}  # latest event per task
        self._running = False

    async def start(self):
        """Start consuming events from the task manager."""
        self._running = True
        self._task_manager.on_event(self._handle_event)

    async def stop(self):
        """Stop the coordinator."""
        self._running = False

    async def _handle_event(self, event: TaskEvent):
        """Handle a task event. Coalesces progress, queues announcements."""
        if not self._running:
            return

        if event.event_type == TaskEventType.PROGRESS:
            # Progress events are never announced — just stored in the task
            return

        if event.event_type == TaskEventType.STARTED:
            # Task started — no announcement needed (already acknowledged at creation)
            return

        if event.event_type in (TaskEventType.COMPLETED, TaskEventType.FAILED, TaskEventType.CANCELLED):
            # Store the latest terminal event for this task
            self._pending[event.task_id] = event

            # Attempt to speak the announcement
            await self._try_announce(event.task_id)

    async def _try_announce(self, task_id: str):
        """Queue a TTSSpeakFrame announcement for a task.

        The TTS service's _hold_frames_while_audio_contexts_open() ensures this
        doesn't overlap with active playback. If the user interrupts the
        announcement, the full result is still available via get_task_result.
        """
        event = self._pending.pop(task_id, None)
        if event is None:
            return

        # Prevent duplicate announcements for the same terminal state
        if task_id in self._announced_tasks:
            return
        self._announced_tasks.add(task_id)

        text = self._format_announcement(event)
        if not text:
            return

        try:
            from pipecat.frames.frames import TTSSpeakFrame

            frame = TTSSpeakFrame(
                text=text,
                append_to_context=False,  # Does NOT mutate active LLM conversation
            )
            await self._tts_service.queue_frame(frame)
            log.info("Announced task %s: %s", event.event_type.value, text[:80])
        except Exception as e:
            log.warning("Failed to announce task %s: %s", task_id, e)

    def _format_announcement(self, event: TaskEvent) -> str:
        """Format a task event into a spoken announcement."""
        if event.event_type == TaskEventType.COMPLETED:
            result_preview = event.result[:300] if event.result else ""
            return f"I've finished your task. {result_preview}"
        elif event.event_type == TaskEventType.FAILED:
            return f"Your task encountered an error: {event.error}"
        elif event.event_type == TaskEventType.CANCELLED:
            return f"Your task has been cancelled. {event.message}"
        return ""


# ---------------------------------------------------------------------------
# Research Runner — multi-step web research with isolated LLM context
# ---------------------------------------------------------------------------
async def research_runner(
    task: Task,
    query: str,
    search_func: Callable[[str], Coroutine[Any, Any, str]],
    summarize_func: Callable[[str, str], Coroutine[Any, Any, str]],
) -> str:
    """Execute a multi-step research task.

    Steps:
    1. Search for the query
    2. Check cancellation
    3. Summarize results using an isolated LLM context
    4. Check cancellation
    5. Return consolidated summary

    Args:
        task: The Task object (for cancellation checking).
        query: The research query.
        search_func: Async function that performs web search (e.g., web_search).
        summarize_func: Async function that summarizes search results with
                       isolated LLM context. Signature: (instructions, search_text) -> str.

    Returns:
        Consolidated research summary.

    Raises:
        TaskCancelledError: If cancellation is detected.
    """
    # Step 1: Search
    task.check_cancellation()
    task.progress.append("Searching the web...")
    search_results = await search_func(query)

    # Step 2: Check cancellation after search
    task.check_cancellation()
    task.progress.append("Search complete. Analyzing results...")

    # Step 3: Summarize with isolated LLM context
    task.check_cancellation()
    task.progress.append("Preparing summary...")
    summary = await summarize_func(
        f"Summarize the following research findings about: {query}. "
        "Provide a concise but informative summary of the key points.",
        search_results,
    )

    # Step 4: Final cancellation check
    task.check_cancellation()

    task.progress.append("Summary complete.")
    return summary


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_task_manager_instance: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get or create the global TaskManager singleton."""
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager()
    return _task_manager_instance


def reset_task_manager():
    """Reset the singleton (for testing)."""
    global _task_manager_instance
    _task_manager_instance = None
