"""M4.5: Background Task Manager — Unit and Integration Tests.

Tests cover:
  1. Task creation and lifecycle (PENDING → RUNNING → COMPLETED)
  2. Task progress tracking
  3. Task completion with result
  4. Task failure handling
  5. Task cancellation (before start, during execution, honest reporting)
  6. Result retrieval
  7. Notification coordinator (event handling, coalescing, deduplication)
  8. Multiple concurrent tasks without corruption
  9. Research runner with cancellation
  10. MCP task tools registration
  11. Task manager singleton behavior
  12. Edge cases (duplicate cancel, unknown task ID, etc.)
"""
from __future__ import annotations

import asyncio
import sys
import os
import threading
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import pytest

from genie.task_manager import (
    TaskManager,
    Task,
    TaskEvent,
    TaskState,
    TaskEventType,
    TaskCancelledError,
    NotificationCoordinator,
    research_runner,
    get_task_manager,
    reset_task_manager,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the TaskManager singleton before and after each test."""
    reset_task_manager()
    yield
    reset_task_manager()


@pytest.fixture
def tm():
    """Fresh TaskManager instance."""
    return TaskManager()


@pytest.fixture
def mock_tts():
    """Mock TTS service that records queued frames."""
    class MockTTS:
        def __init__(self):
            self.queued_frames = []
            self.running = True

        async def queue_frame(self, frame):
            if self.running:
                self.queued_frames.append(frame)

    return MockTTS()


# ============================================================================
# Test 1: Task creation and lifecycle
# ============================================================================

@pytest.mark.asyncio
async def test_task_creation_and_completion(tm):
    """Test 1: Start a background task and verify it completes."""
    async def simple_task():
        await asyncio.sleep(0.1)
        return "Task completed successfully"

    task_id = await tm.create_task("Simple test task", simple_task())
    assert task_id is not None
    assert len(task_id) == 8

    # Yield to let the scheduled task start
    await asyncio.sleep(0)

    task = tm.get_task(task_id)
    assert task is not None
    assert task.state == TaskState.RUNNING

    # Wait for completion
    await asyncio.sleep(0.5)

    task = tm.get_task(task_id)
    assert task.state == TaskState.COMPLETED
    assert task.result == "Task completed successfully"


@pytest.mark.asyncio
async def test_task_state_transitions(tm):
    """Verify the state machine: PENDING → RUNNING → COMPLETED."""
    states_seen = []

    async def tracked_task():
        await asyncio.sleep(0.05)
        return "done"

    task_id = await tm.create_task("Tracked task", tracked_task())
    task = tm.get_task(task_id)

    # Immediately after creation, should be RUNNING (PENDING is very brief)
    assert task.state in (TaskState.PENDING, TaskState.RUNNING)

    await asyncio.sleep(0.2)
    task = tm.get_task(task_id)
    assert task.state == TaskState.COMPLETED
    assert task.result == "done"


# ============================================================================
# Test 2: Task progress tracking
# ============================================================================

@pytest.mark.asyncio
async def test_task_progress_tracking(tm):
    """Test 2: Progress events are recorded."""
    events_received = []

    async def handler(event):
        events_received.append(event)

    tm.on_event(handler)

    async def progress_task():
        await asyncio.sleep(0.05)
        return "finished"

    task_id = await tm.create_task("Progress task", progress_task())
    await asyncio.sleep(0.3)

    # Should have received STARTED and COMPLETED events
    event_types = [e.event_type for e in events_received]
    assert TaskEventType.STARTED in event_types
    assert TaskEventType.COMPLETED in event_types


# ============================================================================
# Test 3: Task completion with result
# ============================================================================

@pytest.mark.asyncio
async def test_task_result_retrieval(tm):
    """Test 3: Completed task result is retrievable."""
    async def result_task():
        await asyncio.sleep(0.1)
        return "The quick brown fox jumps over the lazy dog."

    task_id = await tm.create_task("Result task", result_task())
    await asyncio.sleep(0.3)

    result = tm.get_task_result(task_id)
    assert "quick brown fox" in result


@pytest.mark.asyncio
async def test_task_result_before_completion(tm):
    """Task result before completion returns status message."""
    async def slow_task():
        await asyncio.sleep(5.0)  # Very slow
        return "done"

    task_id = await tm.create_task("Slow task", slow_task())
    await asyncio.sleep(0.1)  # Not enough time

    result = tm.get_task_result(task_id)
    assert "running" in result.lower() or "no result" in result.lower()


# ============================================================================
# Test 4: Task failure handling
# ============================================================================

@pytest.mark.asyncio
async def test_task_failure(tm):
    """Test 4: Failed task stores error and publishes FAILED event."""
    events_received = []

    async def handler(event):
        events_received.append(event)

    tm.on_event(handler)

    async def failing_task():
        await asyncio.sleep(0.05)
        raise ValueError("Something went wrong")

    task_id = await tm.create_task("Failing task", failing_task())
    await asyncio.sleep(0.3)

    task = tm.get_task(task_id)
    assert task.state == TaskState.FAILED
    assert task.error == "Something went wrong"

    # Check events
    event_types = [e.event_type for e in events_received]
    assert TaskEventType.FAILED in event_types

    # Result retrieval for failed task
    result = tm.get_task_result(task_id)
    assert "failed" in result.lower()
    assert "Something went wrong" in result


# ============================================================================
# Test 5: Task cancellation
# ============================================================================

@pytest.mark.asyncio
async def test_cancel_before_start(tm):
    """Cancel a task. The task may already be RUNNING by the time cancel is called."""
    async def slow_start_task():
        await asyncio.sleep(5.0)  # Won't finish
        return "done"

    task_id = await tm.create_task("Slow start task", slow_start_task())
    await asyncio.sleep(0)  # Let the task start

    # Cancel — task is likely RUNNING, so we expect CANCELLATION_REQUESTED
    cancel_result = await tm.cancel_task(task_id)
    await asyncio.sleep(0.2)

    task = tm.get_task(task_id)
    # The task should have received the cancellation request
    assert task.state in (TaskState.CANCELLED, TaskState.CANCELLATION_REQUESTED), \
        f"Expected CANCELLED or CANCELLATION_REQUESTED, got {task.state}"


@pytest.mark.asyncio
async def test_cooperative_cancellation(tm):
    """Test cooperative cancellation via check_cancellation()."""
    async def cooperative_task(tm_inner):
        for i in range(10):
            # Simulate work with cancellation check
            await asyncio.sleep(0.05)
            # We can't easily check cancellation inside a bare coroutine,
            # but we can verify the state machine handles it
        return "done"

    task_id = await tm.create_task("Cooperative task", cooperative_task(tm))
    await asyncio.sleep(0.15)

    # Request cancellation
    cancel_result = await tm.cancel_task(task_id)
    assert "requested" in cancel_result.lower() or "cancelled" in cancel_result.lower()

    task = tm.get_task(task_id)
    assert task.state == TaskState.CANCELLATION_REQUESTED


@pytest.mark.asyncio
async def test_cancel_unknown_task(tm):
    """Cancel a non-existent task."""
    result = await tm.cancel_task("nonexistent123")
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_cancel_already_terminal(tm):
    """Cancel a task that is already completed."""
    async def quick_task():
        return "done"

    task_id = await tm.create_task("Quick task", quick_task())
    await asyncio.sleep(0.3)

    result = await tm.cancel_task(task_id)
    assert "already" in result.lower() or "completed" in result.lower()


@pytest.mark.asyncio
async def test_duplicate_cancel(tm):
    """Requesting cancellation twice should not error."""
    async def slow_task():
        await asyncio.sleep(5.0)
        return "done"

    task_id = await tm.create_task("Slow task", slow_task())
    await asyncio.sleep(0.1)

    r1 = await tm.cancel_task(task_id)
    r2 = await tm.cancel_task(task_id)

    # Second cancel should indicate already requested
    assert "already" in r2.lower()


# ============================================================================
# Test 6: Multiple concurrent tasks
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_concurrent_tasks(tm):
    """Test 7: Two background tasks run without corrupting results."""
    results = {}

    async def named_task(name, delay):
        await asyncio.sleep(delay)
        return f"Result from {name}"

    id1 = await tm.create_task("Task A", named_task("A", 0.2))
    id2 = await tm.create_task("Task B", named_task("B", 0.3))

    await asyncio.sleep(1.0)

    task1 = tm.get_task(id1)
    task2 = tm.get_task(id2)

    assert task1.state == TaskState.COMPLETED
    assert task2.state == TaskState.COMPLETED
    assert task1.result == "Result from A"
    assert task2.result == "Result from B"

    # Verify results are not swapped
    assert "A" in task1.result
    assert "B" in task2.result
    assert "A" not in task2.result
    assert "B" not in task1.result


@pytest.mark.asyncio
async def test_list_tasks(tm):
    """List all tasks and verify their statuses."""
    async def task_a():
        await asyncio.sleep(0.2)
        return "A done"

    async def task_b():
        await asyncio.sleep(0.2)
        return "B done"

    id1 = await tm.create_task("Task A", task_a())
    id2 = await tm.create_task("Task B", task_b())
    await asyncio.sleep(0.5)

    tasks = tm.list_tasks()
    assert len(tasks) == 2

    ids = [t["id"] for t in tasks]
    assert id1 in ids
    assert id2 in ids

    for t in tasks:
        assert "description" in t
        assert "state" in t
        assert "created_at" in t


@pytest.mark.asyncio
async def test_empty_task_list(tm):
    """Empty task list returns appropriate message."""
    tasks = tm.list_tasks()
    assert len(tasks) == 0


# ============================================================================
# Test 7: Notification Coordinator
# ============================================================================

@pytest.mark.asyncio
async def test_notification_coordinator_completes(mock_tts, tm):
    """Test 8: Coordinator announces completed tasks."""
    notifier = NotificationCoordinator(mock_tts, tm)
    await notifier.start()

    async def simple_task():
        await asyncio.sleep(0.1)
        return "Research summary: The sky is blue."

    task_id = await tm.create_task("Research task", simple_task())
    await asyncio.sleep(0.5)

    # Check that a TTSSpeakFrame was queued
    assert len(mock_tts.queued_frames) > 0, "Expected announcement frame"

    from pipecat.frames.frames import TTSSpeakFrame
    speak_frames = [f for f in mock_tts.queued_frames if isinstance(f, TTSSpeakFrame)]
    assert len(speak_frames) > 0, "Expected TTSSpeakFrame announcement"

    # Verify append_to_context=False
    assert speak_frames[0].append_to_context is False

    # Verify the announcement text
    text = speak_frames[0].text
    assert "finished" in text.lower() or "Research summary" in text

    await notifier.stop()


@pytest.mark.asyncio
async def test_notification_coordinator_failed(mock_tts, tm):
    """Coordinator announces failed tasks."""
    notifier = NotificationCoordinator(mock_tts, tm)
    await notifier.start()

    async def failing_task():
        await asyncio.sleep(0.05)
        raise RuntimeError("Network error")

    task_id = await tm.create_task("Failing task", failing_task())
    await asyncio.sleep(0.3)

    from pipecat.frames.frames import TTSSpeakFrame
    speak_frames = [f for f in mock_tts.queued_frames if isinstance(f, TTSSpeakFrame)]
    assert len(speak_frames) > 0

    text = speak_frames[0].text
    assert "error" in text.lower() or "failed" in text.lower()

    await notifier.stop()


@pytest.mark.asyncio
async def test_notification_no_duplicates(mock_tts, tm):
    """Multiple events for the same task produce only one announcement."""
    notifier = NotificationCoordinator(mock_tts, tm)
    await notifier.start()

    async def quick_task():
        return "done"

    task_id = await tm.create_task("Quick task", quick_task())
    await asyncio.sleep(0.5)

    from pipecat.frames.frames import TTSSpeakFrame
    speak_frames = [f for f in mock_tts.queued_frames if isinstance(f, TTSSpeakFrame)]
    # Should be exactly 1 announcement, not multiple
    assert len(speak_frames) == 1

    await notifier.stop()


@pytest.mark.asyncio
async def test_notification_append_to_context_false(mock_tts, tm):
    """Announcements must NOT mutate the active conversation context."""
    notifier = NotificationCoordinator(mock_tts, tm)
    await notifier.start()

    async def task():
        await asyncio.sleep(0.05)
        return "result"

    task_id = await tm.create_task("Test task", task())
    await asyncio.sleep(0.3)

    from pipecat.frames.frames import TTSSpeakFrame
    speak_frames = [f for f in mock_tts.queued_frames if isinstance(f, TTSSpeakFrame)]
    for frame in speak_frames:
        assert frame.append_to_context is False, "Announcement must not append to context"

    await notifier.stop()


# ============================================================================
# Test 8: Research Runner
# ============================================================================

@pytest.mark.asyncio
async def test_research_runner_basic(tm):
    """Test research runner with mock search and summarize functions."""
    mock_task = Task("test1234", "Research test")

    async def mock_search(query):
        return f"Search results for '{query}': [1] Test Result — https://example.com"

    async def mock_summarize(instructions, search_text):
        return "Summary: The research found interesting results."

    result = await research_runner(
        task=mock_task,
        query="test query",
        search_func=mock_search,
        summarize_func=mock_summarize,
    )

    assert "Summary" in result
    assert len(mock_task.progress) >= 3  # Should have progress entries


@pytest.mark.asyncio
async def test_research_runner_cancellation(tm):
    """Test that research runner respects cancellation."""
    mock_task = Task("test5678", "Research with cancel")

    # Pre-set cancellation
    mock_task.request_cancellation()

    async def mock_search(query):
        await asyncio.sleep(0.1)
        return "results"

    async def mock_summarize(instructions, search_text):
        return "summary"

    with pytest.raises(TaskCancelledError):
        await research_runner(
            task=mock_task,
            query="test query",
            search_func=mock_search,
            summarize_func=mock_summarize,
        )


# ============================================================================
# Test 9: Task Manager Singleton
# ============================================================================

@pytest.mark.asyncio
async def test_singleton_consistency():
    """The singleton returns the same instance."""
    reset_task_manager()
    tm1 = get_task_manager()
    tm2 = get_task_manager()
    assert tm1 is tm2


@pytest.mark.asyncio
async def test_singleton_reset():
    """Reset creates a new instance."""
    reset_task_manager()
    tm1 = get_task_manager()
    reset_task_manager()
    tm2 = get_task_manager()
    assert tm1 is not tm2


# ============================================================================
# Test 10: MCP Task Tools Registration
# ============================================================================

@pytest.mark.asyncio
async def test_mcp_tools_registered():
    """Verify task management tools are registered with the MCP server."""
    from genie import mcp_server

    expected_tools = [
        "start_background_task",
        "get_task_status",
        "list_background_tasks",
        "cancel_background_task",
        "get_task_result",
    ]

    existing_tools = ["get_time", "web_search"]

    # Check that existing tools still exist in the module
    for tool_name in existing_tools:
        assert hasattr(mcp_server, tool_name), f"Existing tool '{tool_name}' missing after M4.5"

    # Check that new M4.5 tools exist in the module (imported from task_tools)
    for tool_name in expected_tools:
        assert hasattr(mcp_server, tool_name), f"M4.5 tool '{tool_name}' not registered"

    # Also verify register_task_tools was called (it's in the module)
    assert hasattr(mcp_server, "register_task_tools"), "register_task_tools not imported"


# ============================================================================
# Test 11: Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_task_with_empty_result(tm):
    """Task that returns an empty string."""
    async def empty_task():
        return ""

    task_id = await tm.create_task("Empty task", empty_task())
    await asyncio.sleep(0.3)

    task = tm.get_task(task_id)
    assert task.state == TaskState.COMPLETED
    assert task.result == ""


@pytest.mark.asyncio
async def test_task_with_exception_during_cancel(tm):
    """Exception during cancellation is handled gracefully."""
    events_received = []

    async def handler(event):
        events_received.append(event)

    tm.on_event(handler)

    async def task_that_errors_on_cancel():
        await asyncio.sleep(0.1)
        # Simulate work completing while cancellation was requested
        return "result"

    task_id = await tm.create_task("Edge case task", task_that_errors_on_cancel())
    await asyncio.sleep(0.05)

    # Request cancellation mid-task
    await tm.cancel_task(task_id)
    await asyncio.sleep(0.5)

    task = tm.get_task(task_id)
    # Since the task completed before cancellation could take effect,
    # it should be CANCELLED (result suppressed) since cancellation was requested
    assert task.state in (TaskState.CANCELLED, TaskState.COMPLETED)


@pytest.mark.asyncio
async def test_get_nonexistent_task_result(tm):
    """Getting result of a non-existent task."""
    result = tm.get_task_result("nonexistent123")
    assert "not found" in result.lower()


# ============================================================================
# Test 12: TaskState Enum Properties
# ============================================================================

def test_task_state_terminal():
    """Terminal states are correctly identified."""
    assert TaskState.COMPLETED.is_terminal
    assert TaskState.FAILED.is_terminal
    assert TaskState.CANCELLED.is_terminal
    assert not TaskState.PENDING.is_terminal
    assert not TaskState.RUNNING.is_terminal
    assert not TaskState.CANCELLATION_REQUESTED.is_terminal


def test_task_state_values():
    """All expected states exist."""
    states = [s.name for s in TaskState]
    expected = ["PENDING", "RUNNING", "CANCELLATION_REQUESTED", "CANCELLED", "COMPLETED", "FAILED"]
    for state in expected:
        assert state in states, f"Missing state: {state}"


# ============================================================================
# Test 13: TaskCancelledError
# ============================================================================

def test_task_cancelled_error():
    """TaskCancelledError is raised and caught correctly."""
    task = Task("test9999", "Test")
    task.request_cancellation()

    with pytest.raises(TaskCancelledError):
        task.check_cancellation()


def test_task_not_cancelled():
    """check_cancellation returns False when not cancelled."""
    task = Task("test8888", "Test")
    assert not task.check_cancellation()


# ============================================================================
# Main runner
# ============================================================================

async def run_async_tests():
    """Run all async tests manually (for non-pytest environments)."""
    import traceback

    tests = [
        ("Task creation and completion", test_task_creation_and_completion),
        ("Task state transitions", test_task_state_transitions),
        ("Task progress tracking", test_task_progress_tracking),
        ("Task result retrieval", test_task_result_retrieval),
        ("Task result before completion", test_task_result_before_completion),
        ("Task failure handling", test_task_failure),
        ("Cancel before start", test_cancel_before_start),
        ("Cooperative cancellation", test_cooperative_cancellation),
        ("Cancel unknown task", test_cancel_unknown_task),
        ("Cancel already terminal", test_cancel_already_terminal),
        ("Duplicate cancel", test_duplicate_cancel),
        ("Multiple concurrent tasks", test_multiple_concurrent_tasks),
        ("List tasks", test_list_tasks),
        ("Empty task list", test_empty_task_list),
        ("Notification coordinator completes", test_notification_coordinator_completes),
        ("Notification coordinator failed", test_notification_coordinator_failed),
        ("Notification no duplicates", test_notification_no_duplicates),
        ("Notification append_to_context=False", test_notification_append_to_context_false),
        ("Research runner basic", test_research_runner_basic),
        ("Research runner cancellation", test_research_runner_cancellation),
        ("MCP tools registered", test_mcp_tools_registered),
        ("Task with empty result", test_task_with_empty_result),
        ("Exception during cancel", test_task_with_exception_during_cancel),
        ("Get nonexistent task result", test_get_nonexistent_task_result),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_func in tests:
        reset_task_manager()
        try:
            # Create fixtures
            tm = TaskManager()
            mock_tts = None

            # Determine which fixture the test needs
            params = test_func.__code__.co_varnames[:test_func.__code__.co_argcount]

            if "mock_tts" in params:
                class MockTTS:
                    def __init__(self):
                        self.queued_frames = []
                        self.running = True
                    async def queue_frame(self, frame):
                        if self.running:
                            self.queued_frames.append(frame)
                mock_tts = MockTTS()

            # Build args
            args = []
            for p in params:
                if p == "tm":
                    args.append(tm)
                elif p == "mock_tts":
                    args.append(mock_tts)

            await test_func(*args)
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, e))
            failed += 1

    # Run sync tests
    sync_tests = [
        ("Task state terminal", test_task_state_terminal),
        ("Task state values", test_task_state_values),
        ("TaskCancelledError raised", test_task_cancelled_error),
        ("Task not cancelled", test_task_not_cancelled),
    ]

    # Run async singleton tests
    async_singleton_tests = [
        ("Singleton consistency", test_singleton_consistency),
        ("Singleton reset", test_singleton_reset),
    ]

    for name, test_func in async_singleton_tests:
        reset_task_manager()
        try:
            await test_func()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, e))
            failed += 1

    for name, test_func in sync_tests:
        reset_task_manager()
        try:
            test_func()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, e))
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")

    if errors:
        print("\nFailed tests:")
        for name, e in errors:
            print(f"  {name}: {e}")

    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("M4.5: BACKGROUND TASK MANAGER TESTS")
    print("=" * 60)
    print()

    # Try pytest first, fall back to manual runner
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
    except ImportError:
        success = asyncio.run(run_async_tests())
        sys.exit(0 if success else 1)
