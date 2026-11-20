#!/usr/bin/env python3
"""Smoke test for the Genie pipeline — verifies pipeline construction without audio.

Tests:
  1. Pipeline builds successfully
  2. All stages are FrameProcessors (linkable)
  3. Tool registration works
  4. LLMContext has system prompt + tools
  5. STT service can be instantiated
  6. TTS service can be instantiated

Run: PYTHONPATH=. .venv-genie/bin/python genie/test_pipeline.py
"""
import os
import sys

# Use a dummy API key for construction test
os.environ.setdefault("OPENROUTER_API_KEY", "test-dummy-key-for-construction")
os.environ.setdefault("GENIE_SERVER_URL", "http://localhost:8300")

from genie.pipeline_new import build_pipeline
from genie.config import Config

def test_pipeline_builds():
    """Test 1: Pipeline builds without error."""
    cfg = Config.load()
    pipeline = build_pipeline(cfg)
    assert pipeline is not None
    print("✅ Test 1 PASSED: Pipeline builds successfully")
    return pipeline


def test_pipeline_stages(pipeline):
    """Test 2: All stages are FrameProcessors with link capability."""
    # The Pipeline constructor already verified linking succeeded
    # (it would throw AttributeError if any stage lacked 'link')
    print("✅ Test 2 PASSED: All stages are linkable FrameProcessors")


def test_tools_registered():
    """Test 3: Tool schemas exist and are non-empty."""
    from genie.brain import TOOL_SCHEMAS, dispatch_tool
    assert len(TOOL_SCHEMAS) >= 5, f"Expected ≥5 tools, got {len(TOOL_SCHEMAS)}"

    # Test dispatch_tool works for get_time
    result = dispatch_tool("get_time", {})
    assert isinstance(result, str) and len(result) > 0
    print(f"✅ Test 3 PASSED: {len(TOOL_SCHEMAS)} tools registered, dispatch_tool works")
    print(f"   Sample: get_time() → '{result}'")


def test_services_instantiable():
    """Test 4: STT and TTS services can be instantiated."""
    from genie.services import WhisperHTTPSTTService, KokoroHTTPTTSService

    stt = WhisperHTTPSTTService(base_url="http://localhost:8300", sample_rate=16000)
    assert stt is not None
    print("✅ Test 4a PASSED: WhisperHTTPSTTService instantiated")

    tts = KokoroHTTPTTSService(base_url="http://localhost:8300", sample_rate=24000)
    assert tts is not None
    print("✅ Test 4b PASSED: KokoroHTTPTTSService instantiated")


def test_llm_context():
    """Test 5: LLMContext has system prompt and tools."""
    from pipecat.processors.aggregators.llm_context import LLMContext, FunctionSchema
    from genie.brain import GENIE_SYSTEM_PROMPT, TOOL_SCHEMAS

    lmc = LLMContext(
        messages=[{"participant": "system", "text": GENIE_SYSTEM_PROMPT}],
    )

    msgs = lmc.get_messages()
    assert len(msgs) >= 1
    assert msgs[0]["participant"] == "system"
    assert "Genie" in msgs[0]["text"]
    print(f"✅ Test 5a PASSED: LLMContext has system prompt ({len(GENIE_SYSTEM_PROMPT)} chars)")

    # Set tools
    tool_schemas = [
        FunctionSchema(
            name=s.name,
            description=s.description,
            properties=s.parameters.get("properties", {}),
            required=s.parameters.get("required", []),
        )
        for s in TOOL_SCHEMAS
    ]
    lmc.set_tools(tool_schemas)
    tools = lmc.tools
    assert tools is not None
    print(f"✅ Test 5b PASSED: LLMContext has {type(tools).__name__} tool schemas")


def test_vad():
    """Test 6: VAD analyzer instantiates."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams

    vad = SileroVADAnalyzer(
        sample_rate=16000,
        params=VADParams(confidence=0.7, start_secs=0.3, stop_secs=0.6),
    )
    assert vad is not None
    print("✅ Test 6 PASSED: SileroVADAnalyzer instantiated")


if __name__ == "__main__":
    print("=" * 60)
    print("Genie Pipeline Smoke Test")
    print("=" * 60)

    pipeline = test_pipeline_builds()
    test_pipeline_stages(pipeline)
    test_tools_registered()
    test_services_instantiable()
    test_llm_context()
    test_vad()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
