"""Genie brain — LLM tool-calling assistant with MCP integration.

Contains:
  * build_system_prompt(cfg) → system prompt with identity baked in
  * Tool schemas are now provided by the MCP server, not this file.
  * dispatch_tool(name, args) -> str  (kept for backward compat / non-MCP use)

The brain is called by the Pipecat pipeline over an OpenAI-compatible endpoint.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Dynamic system prompt — built from Config identity fields
# ---------------------------------------------------------------------------
def build_system_prompt(cfg: Any) -> str:
    """Build the Genie system prompt from config identity fields.

    Identity fields (project_creator, project_owner) are EMPTY by default.
    Genie must NEVER guess or invent these — it says "not configured" instead.

    The prompt is ~230 tokens and includes:
      - Assistant name, creator, owner (or "not configured")
      - Available capabilities (real vs. simulated)
      - Model provider info (gateway vs. underlying developer)
    """
    name = cfg.assistant_name
    creator = cfg.project_creator.strip()
    owner = cfg.project_owner.strip()
    gateway = cfg.llm_gateway
    model = cfg.llm_model

    # Social media links (only included if creator is configured)
    social_github = getattr(cfg, "social_github", "").strip()
    social_linkedin = getattr(cfg, "social_linkedin", "").strip()
    social_website = getattr(cfg, "social_website", "").strip()
    social_twitter = getattr(cfg, "social_twitter", "").strip()

    # Build identity section — never invent names
    identity_parts = [f"You are {name}, a voice assistant running on a Raspberry Pi."]
    if creator:
        identity_parts.append(f"You were developed by {creator}.")
    else:
        identity_parts.append("Your project creator is not configured.")
    if owner:
        identity_parts.append(f"You are owned by {owner}.")
    else:
        identity_parts.append("Your project owner is not configured.")
    identity_parts.append(
        f"You use the {model} model via {gateway} (the gateway) for reasoning. "
        f"The underlying model is developed by the provider of {model}, not by {gateway}."
    )
    identity_text = " ".join(identity_parts)

    # Identity response instructions
    if creator:
        creator_instr = f"If asked who made or created you, say you were developed by {creator}."
        # Add social media info if available
        social_links = []
        if social_github:
            social_links.append(f"GitHub: {social_github}")
        if social_linkedin:
            social_links.append(f"LinkedIn: {social_linkedin}")
        if social_website:
            social_links.append(f"Website: {social_website}")
        if social_twitter:
            social_links.append(f"Twitter/X: {social_twitter}")
        if social_links:
            creator_instr += f" You can share the following links: {'; '.join(social_links)}."
    else:
        creator_instr = "If asked who made or created you, say that information is not configured."
    if owner:
        owner_instr = f"If asked who owns you, say {owner}."
    else:
        owner_instr = "If asked who owns you, say that information is not configured."

    return (
        f"{identity_text}\n\n"
        "You speak naturally, briefly, and helpfully — one or two sentences at most.\n\n"
        "Available capabilities:\n"
        "  • Check the current time (real)\n"
        "  • Search the web for current information (real)\n"
        "  • Start background research tasks (real — runs while you chat)\n"
        "  • Set timers and alarms (simulated — no actual countdown or alert)\n"
        "  • Control lights and thermostat (simulated — no physical devices)\n"
        "  • Add reminders (simulated — not persisted)\n"
        "  • List items (simulated — no storage)\n"
        "  • Send messages (simulated — not actually sent)\n\n"
        "Background tasks:\n"
        "  • For complex research or multi-step queries, use start_background_task.\n"
        "  • Tell the user you'll work on it and they can ask other questions.\n"
        "  • Use get_task_status, list_background_tasks, cancel_background_task,\n"
        "    and get_task_result to manage tasks.\n"
        "  • If a task completes, share the result when the user asks or naturally.\n\n"
"Website creation:\n"
"  • You can create, edit, and preview websites using website tools.\n"
"  • Use create_website_project to start a new project. This automatically creates index.html.\n"
"  • Use edit_website_file to modify index.html or any existing file.\n"
"  • Use create_website_file only for NEW files that don't exist yet (e.g., css/style.css).\n"
"  • Keep generated HTML simple and concise — a single page with inline styles.\n"
"  • Do not generate large multi-page sites. One clean index.html is sufficient.\n"
"  • If the user wants a complex site, use start_background_task for the work.\n"
"  • Use preview_website to start a local preview server.\n"
"  • Projects are stored in ~/GenieWorkspace/websites/<project-name>/.\n"
"  • For complex website generation, use start_background_task so the\n"
"    user can continue talking while you work.\n\n"
"GitHub Pages publishing:\n"
"  • You can publish websites to GitHub Pages when the user explicitly asks.\n"
"  • First check configuration with check_github_configuration.\n"
"  • ALWAYS show the target repository and ask for explicit confirmation\n"
"    before publishing. Never publish without user consent.\n"
"  • Set confirm_publish=True only after the user has explicitly agreed.\n"
"  • Explain what will be uploaded and any effects on existing content.\n"
"  • Report the actual deployment result honestly.\n\n"
        "Rules:\n"
        "  • Always use the available tools when they match the request.\n"
        "  • Be honest about what is simulated vs. real. Never claim a simulated "
        "action actually occurred.\n"
        f"  • {creator_instr}\n"
        f"  • {owner_instr}\n"
        "  • If asked about your underlying model, say you use "
        f"{model} via {gateway}. Distinguish the gateway from the model developer.\n"
        "  • You may discuss Spotify and music topics; use web_search for current info.\n"
        "  • If a user asks for something you cannot do, say so politely.\n"
        "  • Keep your answers short and conversational for voice.\n"
        "  • For web search results, cite the source title and summarize briefly.\n"
        "  • Do not invent facts, sources, publication dates, or personal details.\n"
        "  • When answering current-events questions, rely on web search results, "
        "not on your training data cutoff.\n"
    )


# Count tokens (rough: 4 chars/token). Used for validation only.
def _token_count(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool schemas — smart home + utilities. Provided by MCP server.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[ToolSpec] = [
    ToolSpec(
        "get_time",
        "Get the current local time.",
        {
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    ToolSpec(
        "set_timer",
        "Set a timer for a number of seconds.",
        {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Seconds to count down."},
            },
            "required": ["seconds"],
        },
    ),
    ToolSpec(
        "set_alarm",
        "Set an alarm for a specific time (HH:MM).",
        {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "Alarm time in HH:MM format."},
            },
            "required": ["time"],
        },
    ),
    ToolSpec(
        "lights_control",
        "Turn lights on or off in a room.",
        {
            "type": "object",
            "properties": {
                "room": {"type": "string", "description": "Room name."},
                "state": {"type": "string", "enum": ["on", "off"], "description": "On or off."},
            },
            "required": ["room", "state"],
        },
    ),
    ToolSpec(
        "thermostat_set",
        "Set the thermostat to a target temperature in Celsius.",
        {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "description": "Target temperature in Celsius."},
            },
            "required": ["temperature"],
        },
    ),
    ToolSpec(
        "add_reminder",
        "Add a reminder with a message and optional due time.",
        {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Reminder text."},
                "due": {"type": "string", "description": "Optional due time."},
            },
            "required": ["message"],
        },
    ),
    ToolSpec(
        "list_items",
        "List items in a category (e.g. groceries, tasks).",
        {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category to list."},
            },
            "required": ["category"],
        },
    ),
    ToolSpec(
        "contact_message",
        "Send a message to a contact.",
        {
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "Contact name."},
                "message": {"type": "string", "description": "Message text."},
            },
            "required": ["contact", "message"],
        },
    ),
    ToolSpec(
        "web_search",
        "Search the web for a query.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool dispatch — returns human-readable strings.
# Kept for backward compat / non-MCP use. MCP server is primary now.
# ---------------------------------------------------------------------------
def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name and return a string result.

    Real tools: get_time, web_search.
    Simulated tools: everything else (responses are honest about simulation).
    """
    try:
        if name == "get_time":
            return time.strftime("%I:%M %p on %A, %B %d, %Y")
        if name == "web_search":
            from ddgs import DDGS
            from ddgs.exceptions import DDGSException
            query = args.get("query", "")
            try:
                ddgs = DDGS()
                results = ddgs.text(query, max_results=5, backend="html")
                if not results:
                    return f"No search results found for '{query}'."
                lines = [f"Search results for '{query}':"]
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    url = r.get("href", "")
                    body = r.get("body", "")[:200]
                    lines.append(f"[{i}] {title} — {url}")
                    if body:
                        lines.append(f"    {body}")
                return "\n".join(lines)
            except DDGSException as e:
                return f"Search error: {e}."
            except Exception as e:
                return f"Search error: {e}."
        if name == "set_timer":
            secs = float(args.get("seconds", 0))
            return (
                f"I've noted a timer for {int(secs)} seconds, "
                "but this is simulated — no actual countdown or alert will occur."
            )
        if name == "set_alarm":
            return (
                f"I've noted an alarm for {args.get('time')}, "
                "but this is simulated — no actual alarm will sound."
            )
        if name == "lights_control":
            room = args.get("room", "the room")
            state = args.get("state", "").lower()
            return (
                f"I've noted turning the lights {state} in {room}, "
                "but this is simulated — no physical lights are connected."
            )
        if name == "thermostat_set":
            temp = args.get("temperature", 0)
            return (
                f"I've noted setting the thermostat to {temp}°C, "
                "but this is simulated — no physical thermostat is connected."
            )
        if name == "add_reminder":
            msg = args.get("message", "")
            due = args.get("due")
            due_str = f" due {due}" if due else ""
            return (
                f"I've noted the reminder: '{msg}'{due_str}, "
                "but this is simulated — it won't persist or notify you."
            )
        if name == "list_items":
            cat = args.get("category", "unknown")
            return (
                f"No items in '{cat}'. "
                "This is simulated — no items are actually stored."
            )
        if name == "contact_message":
            contact = args.get("contact", "")
            msg = args.get("message", "")
            return (
                f"I've noted a message to {contact}, "
                "but this is simulated — no message was actually sent."
            )
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error ({name}): {e}"


def tool_json_schemas() -> list[dict[str, Any]]:
    """Return OpenAI-style function schemas for all tools (backward compat)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in TOOL_SCHEMAS
    ]


if __name__ == "__main__":
    from genie.config import Config
    cfg = Config.load()
    prompt = build_system_prompt(cfg)
    n = _token_count(prompt)
    print(f"System prompt tokens (approx): {n}")
    print(f"Target: ~250  {'OK' if 200 <= n <= 300 else 'OUT OF RANGE'}")
    print(f"Tools: {len(TOOL_SCHEMAS)}")
    print(f"\n--- System Prompt ---\n{prompt}")
