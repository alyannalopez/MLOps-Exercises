"""M4.76: Terminal Dashboard and Observer Tests.

Tests the dashboard and observer without requiring a running Pipecat pipeline,
audio hardware, or network services.

Run standalone:
    python genie/test_m476_dashboard.py

Run via pytest:
    python -m pytest genie/test_m476_dashboard.py -v
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for_queue(dashboard, expected_min: int = 1, timeout: float = 3.0):
    """Wait until the dashboard queue has at least *expected_min* items."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if dashboard._queue.qsize() >= expected_min:
            return True
        await asyncio.sleep(0.05)
    return dashboard._queue.qsize() >= expected_min


async def _drain_events(dashboard):
    """Drain all pending events from the dashboard queue and process them."""
    while not dashboard._queue.empty():
        try:
            event = dashboard._queue.get_nowait()
            await dashboard._handle_event(event)
        except Exception:
            break


async def _cleanup(dashboard):
    """Ensure dashboard is stopped."""
    if dashboard.is_running:
        await dashboard.shutdown()


def _run_async(coro):
    """Run an async coroutine in a fresh event loop (for pytest compat)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Dashboard tests
# ---------------------------------------------------------------------------

def test_dashboard_initializes():
    """Dashboard creates with correct initial state."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard, GenieState
        d = TerminalDashboard()
        assert not d.is_running
        assert d.current_state == GenieState.IDLE
    _run_async(_run())
    return True


def test_dashboard_start_and_stop():
    """Dashboard starts and stops without crashing."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        await d.start()
        assert d.is_running
        await d.shutdown()
        assert not d.is_running
    _run_async(_run())
    return True


def test_dashboard_state_transitions():
    """All GenieState values can be set and retrieved."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard, GenieState
        d = TerminalDashboard()
        for state in GenieState:
            await d.set_state(state)
        await _drain_events(d)
        assert d.current_state == GenieState.INTERRUPTED
    _run_async(_run())
    return True


def test_dashboard_conversation_messages():
    """Conversation messages are appended and capped at MAX_CONVERSATION."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        for i in range(25):
            await d.add_user_text(f"Message {i}")
        await _drain_events(d)
        assert len(d._conversation) == d.MAX_CONVERSATION
        assert d._conversation[0][1] == "Message 5"
    _run_async(_run())
    return True


def test_dashboard_conversation_cap():
    """Empty conversation messages are skipped."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        await d.add_user_text("")
        await d.add_user_text("  ")
        await _drain_events(d)
        assert len(d._conversation) == 0
    _run_async(_run())
    return True


def test_dashboard_tool_tracking():
    """Tool starts and results are tracked in the tool log."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        await d.add_tool_start("weather", '{"city":"Tokyo"}')
        await d.add_tool_result("weather", "success", "22°C sunny")
        await _drain_events(d)
        assert len(d._tool_log) == 1
        assert d._tool_log[0]["name"] == "weather"
        assert d._tool_log[0]["status"] == "success"
    _run_async(_run())
    return True


def test_dashboard_tool_error():
    """Tool errors are recorded with error status."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        await d.add_tool_start("broken_tool")
        await d.add_tool_result("broken_tool", "error", "Connection refused")
        await _drain_events(d)
        assert d._tool_log[0]["status"] == "error"
    _run_async(_run())
    return True


def test_dashboard_task_events():
    """Background task events populate the bg_tasks dict."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        await d.add_task_event("t-001", "Research quokkas", "created")
        await d.add_task_event("t-001", "Research quokkas", "running", "Searching...")
        await d.add_task_event("t-001", "Research quokkas", "completed", "Found info")
        await _drain_events(d)
        assert "t-001" in d._bg_tasks
        task = d._bg_tasks["t-001"]
        assert task["description"] == "Research quokkas"
        assert task["state"] == "completed"
    _run_async(_run())
    return True


def test_dashboard_error_logging():
    """Errors are appended to the system log."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        await d.add_error("Something went wrong")
        await d.add_log("warning", "Low battery")
        await _drain_events(d)
        assert len(d._system_log) == 2
        assert d._system_log[0] == ("error", "Something went wrong")
        assert d._system_log[1] == ("warning", "Low battery")
    _run_async(_run())
    return True


def test_dashboard_render_no_crash():
    """Rendering an empty dashboard does not crash."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        output = d._build_display()
        assert isinstance(output, str)
        assert "idle" in output.lower()
    _run_async(_run())
    return True


def test_dashboard_render_with_data():
    """Rendering with data shows conversation and state."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard, GenieState
        d = TerminalDashboard()
        await d.set_state(GenieState.LISTENING)
        await d.add_user_text("Hello Genie")
        await _drain_events(d)
        output = d._build_display()
        assert "listening" in output.lower()
        assert "Hello Genie" in output
    _run_async(_run())
    return True


def test_dashboard_uptime_formatting():
    """Uptime formats correctly for seconds, minutes, and hours."""
    from genie.terminal_dashboard import TerminalDashboard
    d = TerminalDashboard()
    d._started_at = time.time() - 45
    assert d._format_uptime() == "45s"

    d._started_at = time.time() - 135
    assert "2m" in d._format_uptime()

    d._started_at = time.time() - 3725
    assert "1h" in d._format_uptime()
    return True


def test_dashboard_high_volume_queue():
    """Dashboard handles high event volume without crashing."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard, DashboardEventKind
        d = TerminalDashboard()
        for i in range(200):
            await d.add_user_text(f"Msg {i}")
        await _drain_events(d)
        assert len(d._conversation) == d.MAX_CONVERSATION
    _run_async(_run())
    return True


def test_dashboard_failure_isolation():
    """Malformed events do not crash the dashboard."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard, DashboardEvent, DashboardEventKind
        d = TerminalDashboard()
        # Malformed state — dashboard stores whatever it receives without crashing
        await d._handle_event(DashboardEvent(DashboardEventKind.STATE_CHANGE, {"state": "bogus"}))
        # Should not crash; the raw value is stored
        assert d.current_state == "bogus"
        # Empty conversation event — should not crash
        await d._handle_event(DashboardEvent(DashboardEventKind.CONVERSATION, {}))
    _run_async(_run())
    return True


def test_dashboard_concurrent_updates():
    """Concurrent event submissions do not cause race conditions."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        tasks = []
        for i in range(5):
            tasks.append(asyncio.create_task(d.add_user_text(f"Concurrent {i}")))
        await asyncio.gather(*tasks)
        await _drain_events(d)
        assert len(d._conversation) == 5
    _run_async(_run())
    return True


# ---------------------------------------------------------------------------
# Priority queue tests
# ---------------------------------------------------------------------------

def test_priority_queue_evicts_routine():
    """Important events evict routine events when queue is full."""
    from genie.terminal_dashboard import TerminalDashboard, DashboardEvent, DashboardEventKind
    d = TerminalDashboard()

    # Fill the queue with routine events
    for _ in range(d.QUEUE_SIZE):
        d.put_event(DashboardEvent(DashboardEventKind.STATE_CHANGE, {"state": "idle"}))

    # Queue is full — an important event should evict a routine one
    result = d.put_event(DashboardEvent(DashboardEventKind.CONVERSATION, {"role": "user", "text": "important"}))
    assert result is True, "Important event should have been enqueued via eviction"
    return True


def test_priority_queue_drops_when_full_of_important():
    """If queue is full of important events, new important events are dropped."""
    from genie.terminal_dashboard import TerminalDashboard, DashboardEvent, DashboardEventKind
    d = TerminalDashboard()

    # Fill queue entirely with important events
    for _ in range(d.QUEUE_SIZE):
        d.put_event(DashboardEvent(DashboardEventKind.ERROR, {"message": "err"}))

    # Another important event should be dropped (logged, not silently lost)
    result = d.put_event(DashboardEvent(DashboardEventKind.ERROR, {"message": "overflow"}))
    assert result is False, "Should drop when queue is full of important events"
    return True


def test_priority_queue_drops_routine():
    """Routine events are dropped when queue is full."""
    from genie.terminal_dashboard import TerminalDashboard, DashboardEvent, DashboardEventKind
    d = TerminalDashboard()

    for _ in range(d.QUEUE_SIZE):
        d.put_event(DashboardEvent(DashboardEventKind.STATE_CHANGE, {"state": "idle"}))

    result = d.put_event(DashboardEvent(DashboardEventKind.STATE_CHANGE, {"state": "thinking"}))
    assert result is False, "Routine event should be dropped when queue is full"
    return True


# ---------------------------------------------------------------------------
# Observer tests
# ---------------------------------------------------------------------------

def test_observer_creation():
    """Observer creates with or without TaskManager."""
    from genie.terminal_dashboard import TerminalDashboard
    from genie.dashboard_observer import DashboardObserver
    d = TerminalDashboard()
    obs = DashboardObserver(dashboard=d)
    assert obs._dashboard is d
    assert obs._pending_tasks == set()
    return True


def test_observer_with_task_manager():
    """Observer subscribes to TaskManager events when provided."""
    from genie.terminal_dashboard import TerminalDashboard
    from genie.dashboard_observer import DashboardObserver
    from genie.task_manager import TaskManager
    d = TerminalDashboard()
    tm = TaskManager()
    obs = DashboardObserver(dashboard=d, task_manager=tm)
    assert obs._tm is tm
    return True


def test_observer_transcription_state():
    """TranscriptionFrame sets user text and transitions to IDLE."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)
        tf = TranscriptionFrame(
            text="Hello there", user_id="user", timestamp=str(int(time.time()))
        )
        await obs.process_frame(tf, FrameDirection.DOWNSTREAM)
        await _wait_for_queue(d)
        await _drain_events(d)
        assert len(d._conversation) == 1
        assert d._conversation[0][1] == "Hello there"
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_llm_state_tracking():
    """LLM start/text/end frames track thinking state and accumulate text."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import LLMTextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        await obs.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        assert obs._llm_active is True
        assert obs._current_genie_text == ""

        await obs.process_frame(LLMTextFrame(text="Hello"), FrameDirection.DOWNSTREAM)
        assert obs._current_genie_text == "Hello"

        await obs.process_frame(LLMTextFrame(text=" world"), FrameDirection.DOWNSTREAM)
        assert obs._current_genie_text == "Hello world"

        await obs.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        assert obs._llm_active is False
        assert obs._current_genie_text == ""
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_tts_state_tracking():
    """TTSAudioRawFrame sets tts_audio_seen flag."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import TTSAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        assert obs._tts_audio_seen is False

        audio_frame = TTSAudioRawFrame(
            audio=b"\x00\x01\x02", sample_rate=24000, num_channels=1
        )
        await obs.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
        assert obs._tts_audio_seen is True
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_interruption_reset():
    """InterruptionFrame resets LLM and TTS state flags."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import (
            LLMFullResponseStartFrame, LLMTextFrame,
            TTSAudioRawFrame, InterruptionFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        await obs.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        await obs.process_frame(LLMTextFrame(text="test"), FrameDirection.DOWNSTREAM)
        await obs.process_frame(
            TTSAudioRawFrame(audio=b"\x00", sample_rate=24000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        assert obs._llm_active is True
        assert obs._tts_audio_seen is True

        await obs.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        assert obs._llm_active is False
        assert obs._current_genie_text == ""
        assert obs._tts_audio_seen is False
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_vad_frames():
    """UserStartedSpeakingFrame and UserStoppedSpeakingFrame trigger states."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        await obs.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await _wait_for_queue(d)
        await _drain_events(d)
        assert d.current_state.value == "listening"

        await obs.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await _wait_for_queue(d)
        await _drain_events(d)
        assert d.current_state.value == "asr_processing"
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_tool_events():
    """FunctionCallsStartedFrame and FunctionCallResultFrame trigger tool events."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import (
            FunctionCallsStartedFrame, FunctionCallResultFrame,
            FunctionCallFromLLM,
        )
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        fc = FunctionCallFromLLM(
            function_name="test_tool",
            tool_call_id="call-1",
            arguments={"key": "value"},
            context=None,
        )
        await obs.process_frame(
            FunctionCallsStartedFrame(function_calls=[fc]),
            FrameDirection.DOWNSTREAM,
        )
        await _wait_for_queue(d)
        await _drain_events(d)
        assert len(d._tool_log) == 1
        assert d._tool_log[0]["name"] == "test_tool"

        await obs.process_frame(
            FunctionCallResultFrame(
                function_name="test_tool",
                tool_call_id="call-1",
                arguments={},
                result="OK",
            ),
            FrameDirection.DOWNSTREAM,
        )
        await _wait_for_queue(d)
        await _drain_events(d)
        assert d._tool_log[0]["status"] == "success"
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_error_frame():
    """ErrorFrame triggers error event on dashboard."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import ErrorFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        await obs.process_frame(ErrorFrame(error="ASR connection lost"), FrameDirection.DOWNSTREAM)
        await _wait_for_queue(d)
        await _drain_events(d)
        assert len(d._system_log) >= 1
        assert any("ASR connection lost" in msg for _, msg in d._system_log)
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_frame_forwarding():
    """Observer forwards frames unchanged and preserves direction."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import LLMTextFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        forwarded_frames = []

        async def capture_push(frame, direction):
            forwarded_frames.append((frame, direction))

        obs.push_frame = capture_push

        test_frame = LLMTextFrame(text="Forward me")
        await obs.process_frame(test_frame, FrameDirection.DOWNSTREAM)

        assert len(forwarded_frames) == 1
        fwd_frame, fwd_dir = forwarded_frames[0]
        assert fwd_frame is test_frame
        assert fwd_dir == FrameDirection.DOWNSTREAM
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_upstream_frames_passed_through():
    """UPSTREAM frames are forwarded but not observed (no dashboard events)."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import LLMTextFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        forwarded = []

        async def capture_push(frame, direction):
            forwarded.append((frame, direction))

        obs.push_frame = capture_push

        test_frame = LLMTextFrame(text="Upstream text")
        await obs.process_frame(test_frame, FrameDirection.UPSTREAM)

        assert len(forwarded) == 1
        assert forwarded[0][1] == FrameDirection.UPSTREAM
        # No dashboard events scheduled for upstream frames
        assert d._queue.empty()
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_task_manager_integration():
    """Observer receives TaskManager events and forwards them to dashboard."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from genie.task_manager import TaskManager, TaskEvent, TaskEventType
        d = TerminalDashboard()
        tm = TaskManager()
        obs = DashboardObserver(dashboard=d, task_manager=tm)

        event = TaskEvent(
            task_id="t-integration",
            event_type=TaskEventType.COMPLETED,
            message="Integration test task",
            result="Success",
        )
        await obs._on_task_event(event)
        await _wait_for_queue(d, expected_min=1)
        await _drain_events(d)

        assert "t-integration" in d._bg_tasks
        task = d._bg_tasks["t-integration"]
        assert task["description"] == "Integration test task"
        assert task["state"] == "completed"
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_task_manager_failure_event():
    """Observer handles task failure events correctly."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from genie.task_manager import TaskManager, TaskEvent, TaskEventType
        d = TerminalDashboard()
        tm = TaskManager()
        obs = DashboardObserver(dashboard=d, task_manager=tm)

        event = TaskEvent(
            task_id="t-fail",
            event_type=TaskEventType.FAILED,
            message="Failed task",
            result="Error occurred",
        )
        await obs._on_task_event(event)
        await _wait_for_queue(d, expected_min=1)
        await _drain_events(d)

        assert "t-fail" in d._bg_tasks
        assert d._bg_tasks["t-fail"]["state"] == "failed"
        await _cleanup(d)
    _run_async(_run())
    return True


def test_observer_task_tracking():
    """Observer tracks pending tasks and cleans them up after completion."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import LLMTextFrame, LLMFullResponseStartFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        obs = DashboardObserver(dashboard=d)

        await obs.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        await obs.process_frame(LLMTextFrame(text="hello"), FrameDirection.DOWNSTREAM)

        # Tasks were scheduled
        assert len(obs._pending_tasks) >= 1

        # Wait for tasks to complete and auto-remove themselves
        await asyncio.sleep(0.3)
        assert len(obs._pending_tasks) == 0
    _run_async(_run())
    return True


def test_observer_shutdown_awaits_tasks():
    """Observer shutdown awaits pending tasks."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        from genie.dashboard_observer import DashboardObserver
        from pipecat.frames.frames import LLMFullResponseStartFrame
        from pipecat.processors.frame_processor import FrameDirection
        d = TerminalDashboard()
        # No TaskManager to avoid off_event issue
        obs = DashboardObserver(dashboard=d)

        # Schedule some tasks
        await obs.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        assert len(obs._pending_tasks) >= 1

        # Shutdown should await them
        await obs.shutdown()
        assert len(obs._pending_tasks) == 0
    _run_async(_run())
    return True


def test_dashboard_console_none_guard():
    """Dashboard shutdown does not crash when _console is None."""
    async def _run():
        from genie.terminal_dashboard import TerminalDashboard
        d = TerminalDashboard()
        d._console = None  # Simulate initialization failure
        await d.shutdown()  # Should not raise
    _run_async(_run())
    return True


def test_audio_output_state_exists():
    """AUDIO_OUTPUT state exists and is used (not SPEAKING)."""
    from genie.terminal_dashboard import GenieState, STATE_COLORS, STATE_EMOJI
    assert hasattr(GenieState, "AUDIO_OUTPUT")
    assert GenieState.AUDIO_OUTPUT.value == "audio_output"
    assert GenieState.AUDIO_OUTPUT in STATE_COLORS
    assert GenieState.AUDIO_OUTPUT in STATE_EMOJI
    # SPEAKING should NOT exist
    assert not hasattr(GenieState, "SPEAKING")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # Dashboard core
    ("Dashboard initializes", test_dashboard_initializes),
    ("Dashboard start and stop", test_dashboard_start_and_stop),
    ("Dashboard state transitions", test_dashboard_state_transitions),
    ("Dashboard conversation messages", test_dashboard_conversation_messages),
    ("Dashboard conversation cap", test_dashboard_conversation_cap),
    ("Dashboard tool tracking", test_dashboard_tool_tracking),
    ("Dashboard tool error", test_dashboard_tool_error),
    ("Dashboard task events", test_dashboard_task_events),
    ("Dashboard error logging", test_dashboard_error_logging),
    ("Dashboard render no crash", test_dashboard_render_no_crash),
    ("Dashboard render with data", test_dashboard_render_with_data),
    ("Dashboard uptime formatting", test_dashboard_uptime_formatting),
    ("Dashboard high volume queue", test_dashboard_high_volume_queue),
    ("Dashboard failure isolation", test_dashboard_failure_isolation),
    ("Dashboard concurrent updates", test_dashboard_concurrent_updates),
    # Priority queue
    ("Priority queue evicts routine", test_priority_queue_evicts_routine),
    ("Priority queue drops when full of important", test_priority_queue_drops_when_full_of_important),
    ("Priority queue drops routine", test_priority_queue_drops_routine),
    # Observer core
    ("Observer creation", test_observer_creation),
    ("Observer with task manager", test_observer_with_task_manager),
    ("Observer transcription state", test_observer_transcription_state),
    ("Observer LLM state tracking", test_observer_llm_state_tracking),
    ("Observer TTS state tracking", test_observer_tts_state_tracking),
    ("Observer interruption reset", test_observer_interruption_reset),
    ("Observer VAD frames", test_observer_vad_frames),
    ("Observer tool events", test_observer_tool_events),
    ("Observer error frame", test_observer_error_frame),
    ("Observer frame forwarding", test_observer_frame_forwarding),
    ("Observer upstream frames passthrough", test_observer_upstream_frames_passed_through),
    ("Observer task manager integration", test_observer_task_manager_integration),
    ("Observer task manager failure", test_observer_task_manager_failure_event),
    ("Observer task tracking", test_observer_task_tracking),
    ("Observer shutdown awaits tasks", test_observer_shutdown_awaits_tasks),
    # Robustness
    ("Dashboard console None guard", test_dashboard_console_none_guard),
    ("AUDIO_OUTPUT state exists", test_audio_output_state_exists),
]


def run_tests():
    passed = 0
    failed = 0
    errors = []
    for name, fn in ALL_TESTS:
        try:
            result = fn()
            if result is not False:
                passed += 1
                print(f"  ✅ {name}")
            else:
                failed += 1
                errors.append((name, "Assertion returned False"))
                print(f"  ❌ {name}")
        except Exception as exc:
            failed += 1
            errors.append((name, str(exc)))
            print(f"  ❌ {name}: {exc}")
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
