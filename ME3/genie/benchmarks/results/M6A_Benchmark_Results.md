# M6-A Component Benchmark Results

## Environment

| Property | Value |
|----------|-------|
| Date | 2026-11-20T14:44:21.958284+00:00 |
| Host | Yannas-MacBook-Air.local |
| Platform | macOS-27.0.1-arm64-arm-64bit-Mach-O |
| Python | 3.14.7 |
| Server | http://127.0.0.1:8300 |
| Health | {"status": "ok", "asr": true, "tts": true, "asr_model": "mlx-community/whisper-large-v3-turbo", "tts_voice": "af_bella", "sample_rate": 24000} |
| Warmup trials (discarded) | 3 |
| Benchmark trials per input | 20 |

## A1: Whisper ASR Latency

| Input | Duration | Size | Mean (ms) | Median (ms) | StdDev (ms) | P95 (ms) | Min (ms) | Max (ms) |
|-------|----------|------|-----------|-------------|-------------|----------|----------|----------|
| Synthetic 1s (16kHz) | 1.0s | 32044B | 1427.5 | 1429.0 | 9.2 | 1439.0 | 1415.0 | 1456.0 |
| Synthetic 3s (16kHz) | 3.0s | 96044B | 1505.8 | 1513.0 | 37.2 | 1555.0 | 1452.0 | 1563.0 |
| Synthetic 5s (16kHz) | 5.0s | 160044B | 1701.2 | 1582.5 | 247.3 | 2208.0 | 1522.0 | 2439.0 |
| Real smooth_turn_taking (10.4s, 16kHz) | 10.4s | 332844B | 1856.3 | 1754.5 | 236.3 | 2330.0 | 1676.0 | 2550.0 |
| Real pause_handling (4.7s, 48kHz⚠️) | 4.72s | 453164B | 8061.0 | 7375.5 | 1832.8 | 12209.0 | 6298.0 | 12631.0 |

## A2: Kokoro TTS Latency

| Input | Words | Chars | Mean (ms) | Median (ms) | StdDev (ms) | P95 (ms) | ms/char | ms/word |
|-------|-------|-------|-----------|-------------|-------------|----------|---------|---------|
| Short 5W | 5 | 25 | 579.8 | 527.8 | 96.2 | 751.6 | 23.19 | 115.96 |
| Medium 16W | 16 | 84 | 1453.3 | 1442.2 | 181.3 | 1726.4 | 17.30 | 90.83 |
| Long 40W | 40 | 307 | 4623.9 | 4555.0 | 581.1 | 5702.7 | 15.06 | 115.60 |
| XLong 101W | 101 | 729 | 11583.8 | 11172.2 | 1837.1 | 12909.4 | 15.89 | 114.69 |

## A3: OpenRouter LLM Response Latency

| Prompt | TTFB Mean (ms) | TTFB P95 (ms) | Total Mean (ms) | Total P95 (ms) |
|--------|----------------|---------------|-----------------|----------------|
| What is the capital of France?... | 940.7 | 1359.3 | 946.1 | 1359.3 |
| Explain the difference between supervise... | 2932.9 | 3105.6 | 3077.4 | 3739.0 |
| Write a short haiku about artificial int... | 898.7 | 1336.1 | 903.3 | 1366.8 |

## A4: Network RTT

### Local Loopback (TCP, 127.0.0.1:8300)

| Metric | Value |
|--------|-------|
| Count | 50 |
| Min | 0.91ms |
| Avg | 1.60ms |
| Max | 17.82ms |
| StdDev | 2.33ms |
| P95 | 1.71ms |

**Note**: TCP RTT to local server (includes HTTP request/response overhead). Actual UDP/IP ping would be lower. Pi-to-Mac LAN RTT not yet measured (requires Pi availability).

## Limitations

1. **ASR inputs are synthetic or pre-recorded** — real-time VAD-segmented audio may differ.
2. **TTS latency includes network round-trip** (local loopback) — negligible but present.
3. **LLM latency includes HTTP overhead** — actual TTFB from OpenRouter may be slightly less.
4. **No Pi-to-Mac network measurement** — requires Pi availability.
5. **Cold-start vs warmed-up**: warmup trials were discarded; all reported results are steady-state.
6. **Concurrent load not tested** — benchmarks run sequentially, not under concurrent user load.
7. **48kHz audio anomaly**: The `pause_handling` real WAV (48kHz) showed 4-6× higher ASR latency vs. 16kHz inputs. This is caused by server-side resampling from 48kHz to 16kHz. In production, Pi sends 16kHz audio, so this outlier does not reflect real-world behavior.
8. **TTS wall-clock vs. server-side**: TTS latency measured as full HTTP round-trip (client-side `perf_counter()`), not the server-reported `X-Synthesis-Latency` header. This includes network + serialization overhead.

## Instrumentation Inspection Findings

### TaskEvent Timestamps

- `TaskEvent.timestamp` is an ISO-format string (UTC) set at event creation.
- `Task.created_at` is an ISO-format string set at task construction.
- **No `Task.completed_at` field exists** — task duration must be computed from `TaskEvent` pairs (STARTED → COMPLETED timestamps).
- **Actionable**: Background task duration CAN be derived from existing `TaskEvent` timestamps without code changes.

### Pipecat 1.15.0 Metrics API

- `StartFrame(enable_metrics=True)` enables built-in metrics emission.
- Metrics emitted as `MetricsFrame(list[MetricsData])` downstream.
- Available types: `TTFBMetricsData` (LLM TTFB), `TTFAMetricsData` (TTS TTFA),
  `ProcessingMetricsData`, `TurnMetricsData`, `SmartTurnMetricsData`.
- **Currently disabled** in `pipeline_new.py` (`enable_metrics=False` by default).
- `OpenRouterLLMService` emits `TTFBMetricsData` when metrics are enabled.

### Generated Audio vs. Physical Playback

- `TTSAudioRawFrame` carries generated audio bytes — NOT confirmation of physical playback.
- Physical playback occurs when `DuplexAudioTransport` writes to the speaker device.
- **Interruption-to-playback-stop latency** requires measuring from
  `UserStartedSpeakingFrame` to the transport's audio buffer flush/stop event.
- The transport does not emit a 'playback stopped' frame; this must be inferred from
  `InterruptionFrame` timing or transport callback hooks.
