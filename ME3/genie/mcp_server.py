"""Genie MCP Server — exposes tools via Streamable HTTP transport.

Tools exposed:
  - get_time          (real: returns actual local time)
  - web_search        (real: DuckDuckGo search via ddgs)
  - set_timer         (simulated: no actual countdown)
  - set_alarm         (simulated: no actual alarm)
  - lights_control    (simulated: no physical device)
  - thermostat_set    (simulated: no physical device)
  - add_reminder      (simulated: not persisted)
  - list_items        (simulated: always empty)
  - contact_message   (simulated: no message sent)
  - start_background_task  (M4.5: starts async research task)
  - get_task_status          (M4.5: check task status)
  - list_background_tasks    (M4.5: list all tasks)
  - cancel_background_task   (M4.5: cancel a task)
  - get_task_result          (M4.5: get task result)
  - create_website_project   (M4.75: create a website project)
  - create_website_file      (M4.75: create a file in a project)
  - read_website_file        (M4.75: read a project file)
  - edit_website_file        (M4.75: edit a project file)
  - list_website_files       (M4.75: list project files)
  - preview_website          (M4.75: start local preview)
  - get_website_project_status (M4.75: project info)
  - check_github_configuration (M4.75: check GitHub setup)
  - check_github_repository    (M4.75: check repo status)
  - publish_website_to_github  (M4.75: publish to Pages)
  - get_github_pages_status    (M4.75: deployment status)

Run standalone:
    python -m genie.mcp_server

The server listens on port 8301 by default (configurable via MCP_PORT env var).
Final deployment: this server runs on the Raspberry Pi.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from mcp.server.mcpserver import MCPServer

from genie.config import Config

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
server = MCPServer(
    name="genie-tools",
    title="Genie Assistant Tools",
    description="Smart-home and utility tools for the Genie voice assistant.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Real tools
# ---------------------------------------------------------------------------

@server.tool(
    name="get_time",
    description="Get the current local time and date.",
)
def get_time() -> str:
    """Return the current local time as a human-readable string."""
    return time.strftime("%I:%M %p on %A, %B %d, %Y")


@server.tool(
    name="web_search",
    description="Search the web for current information. Returns real search results with titles, URLs, and snippets. Use this for questions about current events, news, or facts that may have changed recently.",
)
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return formatted results.

    Args:
        query: The search query string.

    Returns:
        Formatted search results with titles, URLs, snippets, and dates.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    def _do_search(q):
        """Run the synchronous ddgs call in a thread."""
        ddgs = DDGS()
        # Use auto backend (tries multiple engines) for reliability
        return ddgs.text(q, max_results=5)

    try:
        # Run search with a 10-second timeout so the voice assistant doesn't hang
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            results = await asyncio.wait_for(
                loop.run_in_executor(pool, _do_search, query),
                timeout=15.0,
            )

        if not results:
            return (
                f"No search results found for '{query}'. "
                "The search returned empty results — try rephrasing your question."
            )

        lines = [f"Search results for '{query}':"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", "")
            body = r.get("body", "")
            snippet = body[:250].strip() if body else ""
            lines.append(f"\n[{i}] {title}")
            if url:
                lines.append(f"    URL: {url}")
            if snippet:
                lines.append(f"    {snippet}")

        return "\n".join(lines)

    except asyncio.TimeoutError:
        return (
            f"Search timed out for '{query}'. "
            "The search service was too slow to respond. Please try again."
        )
    except DDGSException as e:
        return (
            f"Search temporarily unavailable: {e}. "
            "I could not retrieve web results at this time. "
            "Please try again in a moment."
        )
    except Exception as e:
        return (
            f"Search temporarily unavailable: {e}. "
            "I could not retrieve web results at this time. "
            "Please try again in a moment."
        )


# ---------------------------------------------------------------------------
# Simulated tools — clearly marked as simulated in their responses
# ---------------------------------------------------------------------------

@server.tool(
    name="set_timer",
    description="Set a countdown timer. NOTE: This is simulated — no actual countdown or notification occurs.",
)
def set_timer(seconds: int) -> str:
    """Set a timer (simulated — no actual countdown or alert).

    Args:
        seconds: Number of seconds for the timer.
    """
    return (
        f"I've noted a timer for {seconds} seconds, but please be aware "
        "this is simulated — no actual countdown or notification will occur."
    )


@server.tool(
    name="set_alarm",
    description="Set an alarm for a specific time. NOTE: This is simulated — no actual alarm will sound.",
)
def set_alarm(time_str: str) -> str:
    """Set an alarm (simulated — no actual alarm sounds).

    Args:
        time_str: Alarm time in HH:MM format.
    """
    return (
        f"I've noted an alarm for {time_str}, but please be aware "
        "this is simulated — no actual alarm will sound."
    )


@server.tool(
    name="lights_control",
    description="Control lights in a room. NOTE: This is simulated — no physical lights are controlled.",
)
def lights_control(room: str, state: str) -> str:
    """Control lights (simulated — no physical device connected).

    Args:
        room: Name of the room.
        state: 'on' or 'off'.
    """
    valid_states = ["on", "off"]
    if state.lower() not in valid_states:
        return f"Invalid state '{state}'. Use 'on' or 'off'."
    return (
        f"I've noted turning the lights {state.lower()} in {room}, "
        "but this is simulated — no physical lights are connected."
    )


@server.tool(
    name="thermostat_set",
    description="Set the thermostat temperature. NOTE: This is simulated — no physical thermostat is controlled.",
)
def thermostat_set(temperature: float) -> str:
    """Set thermostat temperature (simulated — no physical device).

    Args:
        temperature: Target temperature in Celsius.
    """
    return (
        f"I've noted setting the thermostat to {temperature}°C, "
        "but this is simulated — no physical thermostat is connected."
    )


@server.tool(
    name="add_reminder",
    description="Add a reminder. NOTE: This is simulated — the reminder is not persisted or notified.",
)
def add_reminder(message: str, due: str | None = None) -> str:
    """Add a reminder (simulated — not persisted).

    Args:
        message: Reminder text.
        due: Optional due time.
    """
    due_str = f" due {due}" if due else ""
    return (
        f"I've noted the reminder: '{message}'{due_str}, "
        "but this is simulated — it won't persist or notify you."
    )


@server.tool(
    name="list_items",
    description="List items in a category. NOTE: This is simulated — no items are stored.",
)
def list_items(category: str) -> str:
    """List items (simulated — always empty, no storage).

    Args:
        category: Category to list (e.g. groceries, tasks).
    """
    return (
        f"No items in '{category}'. "
        "This is simulated — no items are actually stored."
    )


@server.tool(
    name="contact_message",
    description="Send a message to a contact. NOTE: This is simulated — no message is actually sent.",
)
def contact_message(contact: str, message: str) -> str:
    """Send a message (simulated — no actual messaging).

    Args:
        contact: Contact name.
        message: Message text.
    """
    return (
        f"I've noted a message to {contact}: '{message}', "
        "but this is simulated — no message was actually sent."
    )


# ---------------------------------------------------------------------------
# M4.5: Background task management tools
# ---------------------------------------------------------------------------
from genie.task_tools import (
    register_task_tools,
    start_background_task,
    get_task_status,
    list_background_tasks,
    cancel_background_task,
    get_task_result,
)

register_task_tools(server)

# M4.75: Website creation and GitHub Pages publishing tools
from genie.website_tools import register_website_tools
register_website_tools(server)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def run_server(port: int = 8301, host: str = "0.0.0.0"):
    """Run the MCP server on the given port using Streamable HTTP transport."""
    print(f"[mcp] Starting Genie MCP server on {host}:{port}")
    print(f"[mcp] Tools registered: {[t['name'] for t in await server.list_tools()]}")
    await server.run_streamable_http_async(
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )


def main():
    """CLI entry point."""
    cfg = Config.load()
    port = int(cfg.server_port) + 1  # Default: model_server_port + 1 = 8301
    # Allow override via env var
    import os
    env_port = os.environ.get("MCP_PORT")
    if env_port:
        port = int(env_port)
    asyncio.run(run_server(port=port))


if __name__ == "__main__":
    main()
