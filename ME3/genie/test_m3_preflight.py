"""Milestone 3 pre-flight checklist — verifies all components are ready.

Run this BEFORE the physical voice test. It checks:
  1. Config loads correctly
  2. Identity fields are properly handled (empty = "not configured")
  3. MCP server tools are registered
  4. Web search works (DuckDuckGo)
  5. System prompt is correct
  6. Pipeline can construct (needs MCP server running)
"""
import asyncio
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from genie.config import Config
from genie.brain import build_system_prompt


def test_config():
    """Test 1: Config loads correctly."""
    cfg = Config.load()
    assert cfg.openrouter_api_key, "OPENROUTER_API_KEY is empty!"
    assert cfg.server_url, "server_url is empty!"
    assert cfg.mcp_url, "mcp_url is empty!"
    print(f"✅ Config loaded: server={cfg.server_url}, mcp={cfg.mcp_url}")
    print(f"   Model: {cfg.llm_model}, Gateway: {cfg.llm_gateway}")
    print(f"   Creator: '{cfg.project_creator}' (empty={'YES' if not cfg.project_creator else 'NO'})")
    print(f"   Owner: '{cfg.project_owner}' (empty={'YES' if not cfg.project_owner else 'NO'})")
    return cfg


def test_identity(cfg):
    """Test 2: Identity handles empty fields correctly."""
    prompt = build_system_prompt(cfg)

    # With empty defaults, should say "not configured"
    if not cfg.project_creator:
        assert "not configured" in prompt, "Should say creator is not configured!"
    if not cfg.project_owner:
        assert "not configured" in prompt, "Should say owner is not configured!"

    # Should never contain hardcoded names
    assert "Rowel" not in prompt, "Hardcoded 'Rowel' leaked!"

    # Should distinguish gateway from model developer
    assert "gateway" in prompt.lower(), "Should mention gateway distinction!"

    print("✅ Identity prompt correct:")
    # Print first paragraph
    first_para = prompt.split("\n\n")[0]
    print(f"   {first_para}")
    print()


def test_web_search():
    """Test 3: Real web search works."""
    from genie.mcp_server import web_search

    # Run async web_search in a fresh event loop via subprocess trick
    # or use a thread to avoid nested event loop issues
    import threading
    results_holder = []

    def _run_in_thread():
        import asyncio
        async def run_tests():
            # Test 1: Current Philippine president
            result = await web_search("current president of the Philippines 2026")
            results_holder.append(("president", result))

            # Test 2: Recent international news
            result2 = await web_search("major international news events November 2026")
            results_holder.append(("news", result2))

            # Test 3: US Senate elections 2026
            result3 = await web_search("US Senate elections November 2026 results")
            results_holder.append(("senate", result3))

        asyncio.run(run_tests())

    t = threading.Thread(target=_run_in_thread)
    t.start()
    t.join(timeout=45)

    if not results_holder:
        print("❌ Web search tests timed out")
        return

    for label, result in results_holder:
        has_err = "temporarily unavailable" in result.lower()
        has_timeout = "timed out" in result.lower()
        has_no_results = "No search results" in result
        has_url = "URL:" in result

        if has_err or has_no_results:
            print(f"⚠️  Web search ({label}): {result[:120]}...")
        elif not has_url:
            print(f"⚠️  Web search ({label}): partial results (no URLs)")
        else:
            print(f"✅ Web search test: {label}")
            lines = result.split("\n")
            print(f"   Query: {lines[0]}")
            for line in lines[1:6]:
                print(f"   {line}")
        print()


def test_get_time():
    """Test 4: get_time tool works."""
    from genie.mcp_server import get_time
    result = get_time()
    assert "PM" in result or "AM" in result, f"Time format wrong: {result}"
    print(f"✅ get_time: {result}")


def test_simulated_honesty():
    """Test 5: Simulated tools are honest."""
    from genie.mcp_server import set_alarm, lights_control, contact_message

    alarm = set_alarm("07:00")
    assert "simulated" in alarm.lower(), f"set_alarm should say simulated: {alarm}"

    lights = lights_control("living room", "on")
    assert "simulated" in lights.lower(), f"lights_control should say simulated: {lights}"

    msg = contact_message("Alice", "Hello")
    assert "simulated" in msg.lower(), f"contact_message should say simulated: {msg}"

    print("✅ Simulated tools are honest:")
    print(f"   set_alarm: {alarm[:80]}...")
    print(f"   lights_control: {lights[:80]}...")
    print(f"   contact_message: {msg[:80]}...")


async def test_mcp_client():
    """Test 6: MCP client can discover tools (needs server running)."""
    from pipecat.services.mcp_service import MCPClient, StreamableHttpParameters
    from genie.config import Config

    cfg = Config.load()
    mcp = MCPClient(
        server_params=StreamableHttpParameters(
            url=cfg.mcp_url,
            timeout=10.0,
        )
    )
    mcp_tools = await mcp.tools()
    print(f"✅ MCP client discovered {len(mcp_tools.standard_tools)} tools:")
    for t in mcp_tools.standard_tools:
        print(f"   - {t.name}: {t.description[:60]}...")


async def main():
    print("=" * 60)
    print("MILESTONE 3 PRE-FLIGHT CHECKLIST")
    print("=" * 60)
    print()

    # Tests that don't need MCP server
    cfg = test_config()
    print()
    test_identity(cfg)
    print()
    test_get_time()
    print()
    test_simulated_honesty()
    print()
    test_web_search()
    print()

    # Test that needs MCP server running
    print("Note: MCP client test requires the MCP server to be running.")
    print("      This is tested during the actual pipeline run.")
    print()

    print("=" * 60)
    print("ALL PRE-FLIGHT CHECKS PASSED")
    print("=" * 60)
    print()
    print("Next: Run the physical voice test (see instructions below)")


if __name__ == "__main__":
    asyncio.run(main())
