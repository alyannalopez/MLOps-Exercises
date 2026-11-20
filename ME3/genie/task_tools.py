"""M4.5: MCP Task Management Tools — background task tools for the MCP server.

These tools are registered with the MCP server alongside the existing tools
(get_time, web_search, etc.). They allow the LLM to manage background tasks:

  - start_background_task: Start a long-running research task.
  - get_task_status: Check the status of a specific task.
  - list_background_tasks: List all tasks.
  - cancel_background_task: Request cancellation of a running task.
  - get_task_result: Get the result of a completed task.

The functions are defined here at module level so they can be imported
into mcp_server.py and registered with @server.tool().
"""
from __future__ import annotations

import httpx

from genie.config import Config
from genie.task_manager import get_task_manager, research_runner, Task


# ---------------------------------------------------------------------------
# start_background_task
# ---------------------------------------------------------------------------
async def start_background_task(query: str) -> str:
    """Start a background research task.

    Args:
        query: The research query or topic to investigate.

    Returns:
        Task ID and confirmation message.
    """
    tm = get_task_manager()

    # Import web_search from the MCP server module
    from genie.mcp_server import web_search

    # Create an isolated summarization function
    async def _summarize(instructions: str, search_text: str) -> str:
        """Summarize search results using an isolated LLM context.

        This uses the same OpenRouter API as the main pipeline but with
        a completely independent conversation context.
        """
        cfg = Config.load()
        if not cfg.openrouter_api_key:
            return "(Error: OpenRouter API key not configured)"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {cfg.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": cfg.llm_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a research summarizer. "
                                    "Provide clear, factual summaries based on the provided search results. "
                                    "Cite sources when available. Be concise but thorough."
                                ),
                            },
                            {"role": "user", "content": instructions},
                            {"role": "user", "content": f"\n\nSearch results:\n{search_text}"},
                        ],
                        "max_tokens": 500,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"(Summarization error: {e})"

    # Create the research coroutine with task reference via closure
    task_ref: list[Task | None] = [None]

    async def _run_research():
        task = task_ref[0]
        if task is None:
            raise RuntimeError("Task reference not set")
        return await research_runner(
            task=task,
            query=query,
            search_func=web_search,
            summarize_func=_summarize,
        )

    task_id = await tm.create_task(
        description=f"Research: {query}",
        coro=_run_research(),
        on_task_created=lambda t: task_ref.__setitem__(0, t),
    )

    return (
        f"Background task started (ID: {task_id}). "
        f"I'm researching '{query}'. You can ask me other questions while I work. "
        f"Ask about the task status anytime."
    )


# ---------------------------------------------------------------------------
# get_task_status
# ---------------------------------------------------------------------------
async def get_task_status(task_id: str) -> str:
    """Get the status of a specific task.

    Args:
        task_id: The task ID (e.g., 'a1b2c3d4').

    Returns:
        Current status, progress, and result if available.
    """
    tm = get_task_manager()
    task = tm.get_task(task_id)

    if not task:
        return f"Task {task_id} not found."

    lines = [
        f"Task: {task.description}",
        f"Status: {task.state.name.lower()}",
    ]

    if task.progress:
        lines.append(f"Progress: {' → '.join(task.progress[-3:])}")

    if task.state.name == "COMPLETED" and task.result:
        lines.append(f"Result: {task.result[:500]}")
    elif task.state.name == "FAILED" and task.error:
        lines.append(f"Error: {task.error}")
    elif task.state.name == "CANCELLED":
        lines.append("This task was cancelled.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# list_background_tasks
# ---------------------------------------------------------------------------
async def list_background_tasks() -> str:
    """List all background tasks.

    Returns:
        Summary of all tasks with their IDs and statuses.
    """
    tm = get_task_manager()
    tasks = tm.list_tasks()

    if not tasks:
        return "No background tasks."

    lines = ["Current background tasks:"]
    for t in tasks:
        lines.append(
            f"  - [{t['id']}] {t['description']} — {t['state'].lower()}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cancel_background_task
# ---------------------------------------------------------------------------
async def cancel_background_task(task_id: str) -> str:
    """Cancel a running background task.

    Args:
        task_id: The task ID to cancel.

    Returns:
        Cancellation status message.
    """
    tm = get_task_manager()
    return await tm.cancel_task(task_id)


# ---------------------------------------------------------------------------
# get_task_result
# ---------------------------------------------------------------------------
async def get_task_result(task_id: str) -> str:
    """Get the full result of a task.

    Args:
        task_id: The task ID.

    Returns:
        The complete result, error, or status message.
    """
    tm = get_task_manager()
    return tm.get_task_result(task_id)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------
def register_task_tools(server):
    """Register background task management tools with an MCP server.

    Args:
        server: The MCPServer instance from mcp_server.py.
    """
    from mcp.server.mcpserver import MCPServer

    # Register each tool with the MCP server
    server.tool(
        name="start_background_task",
        description=(
            "Start a long-running background research task. The task runs independently "
            "while you continue the conversation. Use this for multi-step research, "
            "summarization, or any task that takes more than a few seconds. "
            "The user will be notified when the task completes."
        ),
    )(start_background_task)

    server.tool(
        name="get_task_status",
        description="Check the current status of a background task by its ID.",
    )(get_task_status)

    server.tool(
        name="list_background_tasks",
        description="List all background tasks and their current statuses.",
    )(list_background_tasks)

    server.tool(
        name="cancel_background_task",
        description="Request cancellation of a running background task.",
    )(cancel_background_task)

    server.tool(
        name="get_task_result",
        description="Get the full result of a completed background task.",
    )(get_task_result)
