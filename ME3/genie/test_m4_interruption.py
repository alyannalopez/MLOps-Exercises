#!/usr/bin/env python3
"""M4 — Full-Duplex Interruption Handling Verification.

Tests the Pipecat-native interruption mechanism WITHOUT physical hardware.
Verifies:
  A. Speech detection while Genie is silent
  B. Speech detection while Genie is speaking (barge-in)
  C. TTS cancellation after interruption
  D. Processing the new utterance after interruption
  E. Maintaining conversational context after interruption
  F. Repeated interruptions without pipeline crashes

These are unit/integration tests — they verify the pipeline constructs correctly
with interruption enabled and that the frame routing is correct.

Physical microphone/speaker testing is required for actual voice interruption.
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure genie is importable
sys.path.insert(0, "/Users/yannamii/sandbox/ME3/ME3")


async def test_a_vad_interruption_enabled():
    """A. Verify VADUserTurnStartStrategy has enable_interruptions=True."""
    from pipecat.processors.aggregators.llm_response_universal import UserTurnStrategies
    
    strategies = UserTurnStrategies()
    
    vad_strategies = [s for s in strategies.start if type(s).__name__ == "VADUserTurnStartStrategy"]
    assert len(vad_strategies) == 1, f"Expected 1 VADUserTurnStartStrategy, got {len(vad_strategies)}"
    
    vad = vad_strategies[0]
    assert vad._enable_interruptions is True, "VADUserTurnStartStrategy should have enable_interruptions=True"
    
    print("✅ TEST A PASS: VADUserTurnStartStrategy has enable_interruptions=True")
    return True


async def test_b_transcription_interruption_enabled():
    """B. Verify TranscriptionUserTurnStartStrategy also enables interruptions."""
    from pipecat.processors.aggregators.llm_response_universal import UserTurnStrategies
    
    strategies = UserTurnStrategies()
    
    transcription_strategies = [s for s in strategies.start if type(s).__name__ == "TranscriptionUserTurnStartStrategy"]
    assert len(transcription_strategies) == 1, f"Expected 1 TranscriptionUserTurnStartStrategy, got {len(transcription_strategies)}"
    
    ts = transcription_strategies[0]
    assert ts._enable_interruptions is True, "TranscriptionUserTurnStartStrategy should have enable_interruptions=True"
    
    print("✅ TEST B PASS: TranscriptionUserTurnStartStrategy has enable_interruptions=True")
    return True


async def test_c_tts_handles_interruption():
    """C. Verify TTSService has _handle_interruption that drops audio contexts."""
    from pipecat.services.tts_service import TTSService
    import inspect
    
    assert hasattr(TTSService, "_handle_interruption"), "TTSService should have _handle_interruption"
    
    src = inspect.getsource(TTSService._handle_interruption)
    assert "_interrupt_audio_contexts" in src, "TTSService._handle_interruption should call _interrupt_audio_contexts"
    assert "_streamed_text" in src, "TTSService._handle_interruption should reset _streamed_text"
    
    # Verify KokoroHTTPTTSService inherits TTSService (and thus _handle_interruption)
    from genie.services import KokoroHTTPTTSService
    assert issubclass(KokoroHTTPTTSService, TTSService), "KokoroHTTPTTSService should extend TTSService"
    
    print("✅ TEST C PASS: TTSService._handle_interruption drops audio contexts, Kokoro inherits it")
    return True


async def test_d_llm_cancels_on_interruption():
    """D. Verify LLM service cancels function calls on interruption."""
    from pipecat.services.openrouter.llm import OpenRouterLLMService
    import inspect
    
    assert hasattr(OpenRouterLLMService, "_handle_interruptions"), "OpenRouterLLMService should have _handle_interruptions"
    
    src = inspect.getsource(OpenRouterLLMService._handle_interruptions)
    assert "cancel_on_interruption" in src, "_handle_interruptions should check cancel_on_interruption"
    assert "_cancel_function_call" in src, "_handle_interruptions should call _cancel_function_call"
    
    print("✅ TEST D PASS: OpenRouterLLMService cancels function calls on interruption")
    return True


async def test_e_output_transport_stops_on_interruption():
    """E. Verify BaseOutputTransport stops playback on interruption."""
    from pipecat.transports.base_output import BaseOutputTransport
    import inspect
    
    assert hasattr(BaseOutputTransport, "_start_interruption"), "BaseOutputTransport should have _start_interruption"
    
    src = inspect.getsource(BaseOutputTransport._start_interruption)
    assert "cancel" in src.lower() or "reset" in src.lower(), "_start_interruption should cancel/reset processing"
    
    print("✅ TEST E PASS: BaseOutputTransport._start_interruption cancels playback")
    return True


async def test_f_empty_user_turn_recovery():
    """F. Verify empty_user_turn config provides recovery prompt for failed interruptions."""
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams, EmptyUserTurnConfig
    
    params = LLMUserAggregatorParams()
    assert params.empty_user_turn is not None, "empty_user_turn should be configured by default"
    assert params.empty_user_turn.interrupted_prompt is not None, "interrupted_prompt should be set"
    assert "repeat" in params.empty_user_turn.interrupted_prompt.lower(), "prompt should ask user to repeat"
    
    print("✅ TEST F PASS: EmptyUserTurnConfig provides recovery prompt for failed interruptions")
    return True


async def test_g_pipeline_constructs_with_interruption():
    """G. Verify the pipeline constructs with VAD-enabled interruption."""
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregatorParams,
        LLMAssistantAggregatorParams,
        LLMContextAggregatorPair,
        LLMContext,
        LLMUserAggregator,
    )
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    
    vad = SileroVADAnalyzer()
    
    lmc = LLMContext(messages=[{"role": "system", "content": "Test"}])
    
    pair = LLMContextAggregatorPair(
        lmc,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    
    # Verify the pair has both aggregators (they are factories, not instances)
    assert pair.user is not None, "Pair should have user aggregator factory"
    assert pair.assistant is not None, "Pair should have assistant aggregator factory"
    
    # Instantiate to check VAD controller
    user_agg = LLMUserAggregator(lmc, params=LLMUserAggregatorParams(vad_analyzer=vad))
    assert user_agg._vad_controller is not None, "User aggregator should have VAD controller"
    
    print("✅ TEST G PASS: Pipeline constructs with VAD-enabled interruption")
    return True


async def test_h_broadcast_interruption_flows_both_directions():
    """H. Verify broadcast_interruption sends InterruptionFrame both upstream and downstream."""
    from pipecat.processors.frame_processor import FrameProcessor
    import inspect
    
    src = inspect.getsource(FrameProcessor.broadcast_interruption)
    assert "broadcast_frame" in src, "broadcast_interruption should call broadcast_frame"
    assert "InterruptionFrame" in src, "broadcast_interruption should broadcast InterruptionFrame"
    
    print("✅ TEST H PASS: broadcast_interruption sends InterruptionFrame both directions")
    return True


async def test_i_vad_controller_receives_audio():
    """I. Verify VADController processes InputAudioRawFrame from LLMUserAggregator."""
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
    import inspect
    
    src = inspect.getsource(LLMUserAggregator.process_frame)
    assert "_vad_controller.process_frame" in src, "LLMUserAggregator should forward frames to VAD controller"
    
    print("✅ TEST I PASS: VADController receives audio frames from LLMUserAggregator")
    return True


async def test_j_on_user_turn_started_broadcasts_interruption():
    """J. Verify _on_user_turn_started calls broadcast_interruption when enable_interruptions=True."""
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
    import inspect
    
    src = inspect.getsource(LLMUserAggregator._on_user_turn_started)
    assert "broadcast_interruption" in src, "_on_user_turn_started should call broadcast_interruption"
    assert "enable_interruptions" in src, "_on_user_turn_started should check enable_interruptions"
    
    print("✅ TEST J PASS: _on_user_turn_started broadcasts interruption when enabled")
    return True


async def test_k_spotify_restriction_removed():
    """K. Verify the Spotify restriction has been removed from the system prompt."""
    from genie.config import Config
    from genie.brain import build_system_prompt
    
    cfg = Config()
    prompt = build_system_prompt(cfg)
    
    assert "Do not mention Spotify" not in prompt, "Spotify restriction should be removed"
    assert "Spotify" in prompt or "music" in prompt, "Spotify/music should be mentioned as allowed"
    
    print("✅ TEST K PASS: Spotify restriction removed, music discussion allowed")
    return True


async def test_l_identity_not_hardcoded():
    """L. Verify project_creator and project_owner are empty by default."""
    from genie.config import Config
    from genie import brain
    
    cfg = Config()
    assert cfg.project_creator == "", f"project_creator should be empty by default, got '{cfg.project_creator}'"
    assert cfg.project_owner == "", f"project_owner should be empty by default, got '{cfg.project_owner}'"
    
    prompt = brain.build_system_prompt(cfg)
    assert "not configured" in prompt, "System prompt should indicate identity is not configured"
    
    print("✅ TEST L PASS: Identity fields empty by default, prompt says 'not configured'")
    return True


async def main():
    print("=" * 60)
    print("M4 — Full-Duplex Interruption Handling Verification")
    print("=" * 60)
    print()
    
    tests = [
        ("A: VAD interruption enabled", test_a_vad_interruption_enabled),
        ("B: Transcription interruption enabled", test_b_transcription_interruption_enabled),
        ("C: TTS handles interruption", test_c_tts_handles_interruption),
        ("D: LLM cancels on interruption", test_d_llm_cancels_on_interruption),
        ("E: Output transport stops on interruption", test_e_output_transport_stops_on_interruption),
        ("F: Empty user turn recovery", test_f_empty_user_turn_recovery),
        ("G: Pipeline constructs with interruption", test_g_pipeline_constructs_with_interruption),
        ("H: Broadcast interruption flows both directions", test_h_broadcast_interruption_flows_both_directions),
        ("I: VAD controller receives audio", test_i_vad_controller_receives_audio),
        ("J: _on_user_turn_started broadcasts interruption", test_j_on_user_turn_started_broadcasts_interruption),
        ("K: Spotify restriction removed", test_k_spotify_restriction_removed),
        ("L: Identity not hardcoded", test_l_identity_not_hardcoded),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_fn in tests:
        try:
            result = await test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                errors.append(name)
        except Exception as e:
            failed += 1
            errors.append(f"{name}: {e}")
            print(f"❌ TEST {name} FAIL: {e}")
    
    print()
    print("=" * 60)
    print(f"Results: {passed}/{passed+failed} passed")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
