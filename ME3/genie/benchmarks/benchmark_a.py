#!/usr/bin/env python3
"""M6-A Component Benchmarks — Mac-only, no pipeline modification.

Benchmarks:
  A1: Whisper ASR latency (pre-recorded WAV samples, varying durations)
  A2: Kokoro TTS latency (fixed text inputs, varying lengths)
  A3: OpenRouter LLM response latency (controlled prompts)
  A4: Network RTT (curl-based, Pi-to-Mac equivalent)

Cold-start vs warmed-up are measured separately.
Results saved as JSON + Markdown to genie/benchmarks/.

Usage:
  PYTHONPATH=. .venv-genie/bin/python genie/benchmarks/benchmark_a.py
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ME3/ME3/
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERVER_URL = "http://127.0.0.1:8300"  # Local for Mac benchmarks

N_WARMUP_TRIALS = 3  # Discarded from stats
N_TRIALS = 20         # Per input variant

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stats(values: list[float]) -> dict:
    """Compute mean, median, stddev, P95."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)
    p95_idx = max(0, min(n - 1, int(math.ceil(0.95 * n)) - 1))
    p95 = s[p95_idx]
    return {
        "count": n,
        "mean_ms": round(mean * 1000, 1),
        "median_ms": round(med * 1000, 1),
        "stddev_ms": round(std * 1000, 1),
        "p95_ms": round(p95 * 1000, 1),
        "min_ms": round(s[0] * 1000, 1),
        "max_ms": round(s[-1] * 1000, 1),
    }


def generate_sine_wav(duration_s: float, sr: int = 16000, freq: float = 440.0) -> bytes:
    """Generate a sine-wave WAV file as bytes (simulated speech-like audio)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def load_existing_wav(path: str) -> tuple[bytes, float, int]:
    """Load a WAV file, return (bytes, duration_s, sample_rate)."""
    wav_path = BASE_DIR / path
    if not wav_path.exists():
        return None, 0, 0
    with open(wav_path, "rb") as f:
        data = f.read()
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        frames = w.getnframes()
        dur = frames / sr
    return data, dur, sr


# ---------------------------------------------------------------------------
# A1: Whisper ASR Latency
# ---------------------------------------------------------------------------
async def benchmark_asr(session, label: str, wav_bytes: bytes, duration_s: float, n_trials: int):
    """Send WAV to /transcribe, measure latency. Returns list of latency_s."""
    latencies = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        async with session.post(f"{SERVER_URL}/transcribe", data=wav_bytes,
                                headers={"Content-Type": "audio/wav"}) as resp:
            resp.raise_for_status()
            result = await resp.json()
        elapsed = time.perf_counter() - t0
        server_latency = result.get("latency_s", elapsed)
        latencies.append(server_latency)
        if (i + 1) % 5 == 0:
            print(f"  A1 [{label}] trial {i+1}/{n_trials}: {server_latency*1000:.0f}ms (server-reported)")
    return latencies


# ---------------------------------------------------------------------------
# A2: Kokoro TTS Latency
# ---------------------------------------------------------------------------
async def benchmark_tts(session, label: str, text: str, n_trials: int):
    """Send text to /speak, measure latency. Returns list of latency_s."""
    latencies = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        async with session.post(f"{SERVER_URL}/speak", json={"text": text}) as resp:
            resp.raise_for_status()
            await resp.read()
        elapsed = time.perf_counter() - t0
        # Server also reports X-Synthesis-Latency header, but we measure wall-clock
        latencies.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"  A2 [{label}] trial {i+1}/{n_trials}: {elapsed*1000:.0f}ms")
    return latencies


# ---------------------------------------------------------------------------
# A3: OpenRouter LLM Latency
# ---------------------------------------------------------------------------
async def benchmark_llm(session, label: str, prompt: str, n_trials: int):
    """Call OpenRouter API directly, measure TTFB and total latency."""
    from genie.config import Config
    cfg = Config.load()
    api_key = cfg.openrouter_api_key
    model = cfg.llm_model

    if not api_key:
        print(f"  A3 [{label}] SKIPPED: no OPENROUTER_API_KEY configured")
        return [], []

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8300",
        "X-Title": "Genie M6 Benchmark",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 200,
        "stream": False,
    }

    ttfs = []  # time-to-first-byte
    totals = []

    for i in range(n_trials):
        t0 = time.perf_counter()
        async with session.post(url, headers=headers, json=body) as resp:
            resp.raise_for_status()
            ttfb = time.perf_counter() - t0
            await resp.read()
            total = time.perf_counter() - t0
        ttfs.append(ttfb)
        totals.append(total)
        if (i + 1) % 5 == 0:
            print(f"  A3 [{label}] trial {i+1}/{n_trials}: TTFB={ttfb*1000:.0f}ms, total={total*1000:.0f}ms")
    return ttfs, totals


# ---------------------------------------------------------------------------
# A4: Network RTT (local only — Pi-to-Mac measured later)
# ---------------------------------------------------------------------------
def benchmark_network_rtt(host: str = "127.0.0.1", count: int = 50) -> dict:
    """Ping-based RTT measurement."""
    try:
        result = subprocess.run(
            ["ping", "-c", str(count), host],
            capture_output=True, text=True, timeout=60
        )
        # Parse "rtt min/avg/max/mdev = X/Y/Z/W ms"
        for line in result.stdout.split("\n"):
            if "rtt " in line and "/avg/" in line:
                parts = line.split("=")[1].strip().split("/")
                return {
                    "min_ms": round(float(parts[0]), 1),
                    "avg_ms": round(float(parts[1]), 1),
                    "max_ms": round(float(parts[2]), 1),
                    "mdev_ms": round(float(parts[3]), 1),
                }
    except Exception as e:
        return {"error": str(e)}
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_benchmarks():
    import aiohttp

    print("=" * 70)
    print("M6-A Component Benchmarks")
    print("=" * 70)
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print(f"Host: {platform.node()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Server: {SERVER_URL}")
    print()

    # Verify server health
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{SERVER_URL}/health") as resp:
            health = await resp.json()
        print(f"Health: {health}")
        print()

        all_results = {
            "metadata": {
                "date": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "server_url": SERVER_URL,
                "health": health,
                "n_warmup": N_WARMUP_TRIALS,
                "n_trials": N_TRIALS,
            },
            "A1_asr": {},
            "A2_tts": {},
            "A3_llm": {},
            "A4_network": {},
        }

        # ---------------------------------------------------------------
        # A1: ASR Benchmarks — varying durations
        # ---------------------------------------------------------------
        print("-" * 60)
        print("A1: Whisper ASR Latency")
        print("-" * 60)

        # Prepare test audio: synthetic sine waves + existing WAV files
        asr_inputs = {}

        # Synthetic: short (1s), medium (3s), long (5s)
        for dur_label, dur in [("short_1s", 1.0), ("medium_3s", 3.0), ("long_5s", 5.0)]:
            wav = generate_sine_wav(dur, sr=16000)
            asr_inputs[dur_label] = (wav, dur)

        # Existing WAV files
        existing_wavs = [
            "ai231-me3-full-duplex-benchmarking/tests/example_data/smooth_turn_taking_example/1/input.wav",
            "ai231-me3-full-duplex-benchmarking/tests/example_data/pause_handling_example/1/input.wav",
        ]
        for wpath in existing_wavs:
            data, dur, sr = load_existing_wav(wpath)
            if data:
                label = wpath.replace("/", "_").replace(".wav", "")
                asr_inputs[f"real_{label}"] = (data, dur)

        for label, (wav_bytes, dur) in asr_inputs.items():
            print(f"\n  [{label}] duration={dur:.2f}s, size={len(wav_bytes)} bytes")

            # Warmup trials (discarded)
            await benchmark_asr(session, f"{label}-warm", wav_bytes, dur, N_WARMUP_TRIALS)
            print(f"    (warmup complete)")

            # Real trials
            lats = await benchmark_asr(session, label, wav_bytes, dur, N_TRIALS)
            s = stats(lats)
            s["input_duration_s"] = round(dur, 2)
            s["input_size_bytes"] = len(wav_bytes)
            s["raw_latencies_s"] = [round(x, 4) for x in lats]
            all_results["A1_asr"][label] = s
            print(f"    Stats: mean={s['mean_ms']}ms, median={s['median_ms']}ms, "
                  f"std={s['stddev_ms']}ms, P95={s['p95_ms']}ms")

        # ---------------------------------------------------------------
        # A2: TTS Benchmarks — varying text lengths
        # ---------------------------------------------------------------
        print("\n" + "-" * 60)
        print("A2: Kokoro TTS Latency")
        print("-" * 60)

        tts_inputs = {
            "short_5w": "Hello, how are you today?",
            "medium_20w": "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon.",
            "long_50w": "Artificial intelligence has transformed many aspects of modern life, from healthcare diagnostics to autonomous vehicles. Researchers continue to push the boundaries of what machines can learn and accomplish, raising both excitement and ethical considerations about the future of human-machine collaboration.",
            "xlong_100w": "The history of computing spans thousands of years, from the ancient abacus to modern quantum processors. Each generation of technology has brought exponential improvements in speed, capacity, and accessibility. Today's artificial intelligence systems represent the culmination of decades of research in mathematics, neuroscience, and computer science. As we continue to develop more sophisticated models, the implications for society become increasingly profound, touching everything from education and entertainment to scientific discovery and national security. The challenge ahead is not just technical but also deeply philosophical, requiring us to consider what kind of future we want to build with these powerful new tools.",
        }

        for label, text in tts_inputs.items():
            word_count = len(text.split())
            char_count = len(text)
            print(f"\n  [{label}] words={word_count}, chars={char_count}")

            # Warmup
            await benchmark_tts(session, f"{label}-warm", text, N_WARMUP_TRIALS)
            print(f"    (warmup complete)")

            # Real trials
            lats = await benchmark_tts(session, label, text, N_TRIALS)
            s = stats(lats)
            s["word_count"] = word_count
            s["char_count"] = char_count
            s["raw_latencies_s"] = [round(x, 4) for x in lats]
            # Normalize
            s["ms_per_char"] = round(s["mean_ms"] / char_count, 2)
            s["ms_per_word"] = round(s["mean_ms"] / word_count, 2)
            all_results["A2_tts"][label] = s
            print(f"    Stats: mean={s['mean_ms']}ms, median={s['median_ms']}ms, "
                  f"std={s['stddev_ms']}ms, P95={s['p95_ms']}ms")

        # ---------------------------------------------------------------
        # A3: OpenRouter LLM Latency
        # ---------------------------------------------------------------
        print("\n" + "-" * 60)
        print("A3: OpenRouter LLM Response Latency")
        print("-" * 60)

        llm_prompts = {
            "simple_qa": "What is the capital of France?",
            "moderate_qa": "Explain the difference between supervised and unsupervised learning in machine learning.",
            "creative": "Write a short haiku about artificial intelligence.",
        }

        for label, prompt in llm_prompts.items():
            print(f"\n  [{label}] prompt='{prompt[:50]}...'")

            # Warmup
            await benchmark_llm(session, f"{label}-warm", prompt, N_WARMUP_TRIALS)
            print(f"    (warmup complete)")

            # Real trials
            ttfs, totals = await benchmark_llm(session, label, prompt, N_TRIALS)
            if ttfs:
                ttfb_stats = stats(ttfs)
                total_stats = stats(totals)
                ttfb_stats["raw_ttfb_s"] = [round(x, 4) for x in ttfs]
                total_stats["raw_total_s"] = [round(x, 4) for x in totals]
                all_results["A3_llm"][label] = {
                    "prompt_preview": prompt[:80],
                    "ttfb": ttfb_stats,
                    "total": total_stats,
                }
                print(f"    TTFB: mean={ttfb_stats['mean_ms']}ms, P95={ttfb_stats['p95_ms']}ms")
                print(f"    Total: mean={total_stats['mean_ms']}ms, P95={total_stats['p95_ms']}ms")
            else:
                all_results["A3_llm"][label] = {"skipped": True}

        # ---------------------------------------------------------------
        # A4: Network RTT (local loopback)
        # ---------------------------------------------------------------
        print("\n" + "-" * 60)
        print("A4: Network RTT (local loopback)")
        print("-" * 60)

        rtt = benchmark_network_rtt("127.0.0.1", count=50)
        all_results["A4_network"]["loopback"] = rtt
        print(f"  Loopback RTT: {rtt}")

    # Save results
    json_path = OUTPUT_DIR / "m6a_raw_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {json_path}")

    # Generate Markdown report
    md = generate_markdown_report(all_results)
    md_path = OUTPUT_DIR / "M6A_Benchmark_Results.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report saved to: {md_path}")

    return all_results


def generate_markdown_report(results: dict) -> str:
    """Generate a readable Markdown report from benchmark results."""
    meta = results["metadata"]
    lines = []
    lines.append("# M6-A Component Benchmark Results\n")
    lines.append("## Environment\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Date | {meta['date']} |")
    lines.append(f"| Host | {meta['host']} |")
    lines.append(f"| Platform | {meta['platform']} |")
    lines.append(f"| Python | {meta['python']} |")
    lines.append(f"| Server | {meta['server_url']} |")
    lines.append(f"| Health | {json.dumps(meta['health'])} |")
    lines.append(f"| Warmup trials (discarded) | {meta['n_warmup']} |")
    lines.append(f"| Benchmark trials per input | {meta['n_trials']} |")
    lines.append("")

    # A1: ASR
    lines.append("## A1: Whisper ASR Latency\n")
    lines.append("| Input | Duration | Size | Mean (ms) | Median (ms) | StdDev (ms) | P95 (ms) | Min (ms) | Max (ms) |")
    lines.append("|-------|----------|------|-----------|-------------|-------------|----------|----------|----------|")
    for label, s in results["A1_asr"].items():
        safe_label = label.replace("_", " ").title()
        lines.append(
            f"| {safe_label} | {s.get('input_duration_s', '?')}s | {s.get('input_size_bytes', '?')}B | "
            f"{s['mean_ms']} | {s['median_ms']} | {s['stddev_ms']} | {s['p95_ms']} | "
            f"{s['min_ms']} | {s['max_ms']} |"
        )
    lines.append("")

    # A2: TTS
    lines.append("## A2: Kokoro TTS Latency\n")
    lines.append("| Input | Words | Chars | Mean (ms) | Median (ms) | StdDev (ms) | P95 (ms) | ms/char | ms/word |")
    lines.append("|-------|-------|-------|-----------|-------------|-------------|----------|---------|---------|")
    for label, s in results["A2_tts"].items():
        safe_label = label.replace("_", " ").title()
        lines.append(
            f"| {safe_label} | {s['word_count']} | {s['char_count']} | "
            f"{s['mean_ms']} | {s['median_ms']} | {s['stddev_ms']} | {s['p95_ms']} | "
            f"{s['ms_per_char']} | {s['ms_per_word']} |"
        )
    lines.append("")

    # A3: LLM
    lines.append("## A3: OpenRouter LLM Response Latency\n")
    lines.append("| Prompt | TTFB Mean (ms) | TTFB P95 (ms) | Total Mean (ms) | Total P95 (ms) |")
    lines.append("|--------|----------------|---------------|-----------------|----------------|")
    for label, data in results["A3_llm"].items():
        safe_label = label.replace("_", " ").title()
        if data.get("skipped"):
            lines.append(f"| {safe_label} | SKIPPED | SKIPPED | SKIPPED | SKIPPED |")
        else:
            ttfb = data["ttfb"]
            total = data["total"]
            preview = data.get("prompt_preview", "")[:40]
            lines.append(
                f"| {preview}... | {ttfb['mean_ms']} | {ttfb['p95_ms']} | "
                f"{total['mean_ms']} | {total['p95_ms']} |"
            )
    lines.append("")

    # A4: Network
    lines.append("## A4: Network RTT\n")
    for iface, rtt in results["A4_network"].items():
        lines.append(f"### {iface}\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for k, v in rtt.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Limitations
    lines.append("## Limitations\n")
    lines.append("1. **ASR inputs are synthetic or pre-recorded** — real-time VAD-segmented audio may differ.")
    lines.append("2. **TTS latency includes network round-trip** (local loopback) — negligible but present.")
    lines.append("3. **LLM latency includes HTTP overhead** — actual TTFB from OpenRouter may be slightly less.")
    lines.append("4. **No Pi-to-Mac network measurement** — requires Pi availability.")
    lines.append("5. **Cold-start vs warmed-up**: warmup trials were discarded; all reported results are steady-state.")
    lines.append("6. **Concurrent load not tested** — benchmarks run sequentially, not under concurrent user load.")
    lines.append("")

    # Inspection findings
    lines.append("## Instrumentation Inspection Findings\n")
    lines.append("### TaskEvent Timestamps\n")
    lines.append("- `TaskEvent.timestamp` is an ISO-format string (UTC) set at event creation.")
    lines.append("- `Task.created_at` is an ISO-format string set at task construction.")
    lines.append("- **No `Task.completed_at` field exists** — task duration must be computed from `TaskEvent` pairs (STARTED → COMPLETED timestamps).")
    lines.append("- **Actionable**: Background task duration CAN be derived from existing `TaskEvent` timestamps without code changes.")
    lines.append("")
    lines.append("### Pipecat 1.15.0 Metrics API\n")
    lines.append("- `StartFrame(enable_metrics=True)` enables built-in metrics emission.")
    lines.append("- Metrics emitted as `MetricsFrame(list[MetricsData])` downstream.")
    lines.append("- Available types: `TTFBMetricsData` (LLM TTFB), `TTFAMetricsData` (TTS TTFA),")
    lines.append("  `ProcessingMetricsData`, `TurnMetricsData`, `SmartTurnMetricsData`.")
    lines.append("- **Currently disabled** in `pipeline_new.py` (`enable_metrics=False` by default).")
    lines.append("- `OpenRouterLLMService` emits `TTFBMetricsData` when metrics are enabled.")
    lines.append("")
    lines.append("### Generated Audio vs. Physical Playback\n")
    lines.append("- `TTSAudioRawFrame` carries generated audio bytes — NOT confirmation of physical playback.")
    lines.append("- Physical playback occurs when `DuplexAudioTransport` writes to the speaker device.")
    lines.append("- **Interruption-to-playback-stop latency** requires measuring from")
    lines.append("  `UserStartedSpeakingFrame` to the transport's audio buffer flush/stop event.")
    lines.append("- The transport does not emit a 'playback stopped' frame; this must be inferred from")
    lines.append("  `InterruptionFrame` timing or transport callback hooks.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
