"""Quick smoke test for the Genie model server endpoints."""
import io
import json
import wave
import numpy as np
import soundfile as sf
import urllib.request
import sys

BASE = "http://localhost:8300"

def test_health():
    print("=== /health ===")
    req = urllib.request.Request(f"{BASE}/health")
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    print(f"  Status: {data}")
    assert data["status"] == "ok"
    assert data["asr"] is True, "ASR not ready!"
    assert data["tts"] is True, "TTS not ready!"
    print("  PASS ✓")
    return data


def test_transcribe():
    print("\n=== /transcribe ===")
    # Generate a test WAV with some audio content (not silence)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # A simple sine wave — won't transcribe as speech but tests the pipeline
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    wav_data = buf.getvalue()

    req = urllib.request.Request(
        f"{BASE}/transcribe",
        data=wav_data,
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    print(f"  Response: {data}")
    assert "text" in data
    assert "latency_s" in data
    assert data["latency_s"] > 0
    print(f"  Latency: {data['latency_s']}s")
    print("  PASS ✓")
    return data


def test_speak():
    print("\n=== /speak ===")
    payload = json.dumps({"text": "Hello, this is a test of the Kokoro text to speech system."}).encode()
    req = urllib.request.Request(
        f"{BASE}/speak",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    wav_data = resp.read()
    ct = resp.headers.get("Content-Type", "")
    print(f"  Content-Type: {ct}")
    print(f"  WAV size: {len(wav_data)} bytes")

    # Validate WAV structure
    buf = io.BytesIO(wav_data)
    with wave.open(buf, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        frames = w.getnframes()
        duration = frames / sr
        print(f"  Sample rate: {sr} Hz, Channels: {nch}, Duration: {duration:.2f}s")
        assert sr == 24000, f"Expected 24000 Hz, got {sr}"
        assert nch == 1, f"Expected 1 channel, got {nch}"
        assert duration > 0.5, f"Audio too short: {duration}s"

    # Save for manual listening
    with open("/tmp/genie_test_speak.wav", "wb") as f:
        f.write(wav_data)
    print(f"  Saved to /tmp/genie_test_speak.wav")
    print("  PASS ✓")
    return wav_data


def test_lan_access():
    """Test that the server is accessible on 0.0.0.0 (LAN)."""
    print("\n=== LAN accessibility ===")
    # Get the Mac's LAN IP
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"  Hostname: {hostname}, IP: {ip}")
    try:
        req = urllib.request.Request(f"http://{ip}:8300/health", timeout=5)
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        print(f"  LAN health: {data}")
        assert data["status"] == "ok"
        print("  PASS ✓")
    except Exception as e:
        print(f"  WARN: LAN access failed ({e}) — may be firewall restricted")


if __name__ == "__main__":
    try:
        health = test_health()
        transcribe_result = test_transcribe()
        speak_result = test_speak()
        test_lan_access()
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\nTEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
