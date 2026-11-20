#!/usr/bin/env python3
"""Test HTTP endpoints: /speak status code, /transcribe with speech, LAN access."""
import json, wave, io, os, sys, urllib.request, urllib.error

BASE = os.environ.get("GENIE_SERVER_URL", "http://localhost:8300")
# Falls back to Config.load().server_url if GENIE_SERVER_URL not set

def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    return resp.status, dict(resp.headers), resp.read()


def post_bytes(url, data, ct="audio/wav"):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ct}, method="POST")
    resp = urllib.request.urlopen(req)
    return resp.status, dict(resp.headers), resp.read()


def test_health():
    print(f"\n=== GET {BASE}/health ===")
    req = urllib.request.Request(f"{BASE}/health")
    resp = urllib.request.urlopen(req)
    print(f"Status: {resp.status}")
    print(f"Body: {resp.read().decode()}")


def test_speak():
    print(f"\n=== POST {BASE}/speak ===")
    status, headers, body = post_json(f"{BASE}/speak", {
        "text": "Hello, this is a test of the text to speech system."
    })
    print(f"Status: {status}")
    print(f"Content-Type: {headers.get('content-type', 'N/A')}")
    print(f"Body length: {len(body)} bytes")

    # Check WAV validity
    try:
        with wave.open(io.BytesIO(body), "rb") as w:
            print(f"WAV: {w.getnchannels()}ch, {w.getsampwidth()}bytes, {w.getframerate()}Hz, {w.getnframes()} frames ({w.getnframes()/w.getframerate():.2f}s)")
    except Exception as e:
        print(f"WAV parse error: {e}")

    # Save for listening
    with open("/tmp/test_speak.wav", "wb") as f:
        f.write(body)
    print(f"Saved to /tmp/test_speak.wav")
    return status


def test_transcribe_speech():
    """Generate a speech WAV using Kokoro, then transcribe it back."""
    print(f"\n=== POST {BASE}/transcribe (with speech) ===")

    # First get a speech WAV from /speak
    _, _, speech_wav = post_json(f"{BASE}/speak", {
        "text": "The quick brown fox jumps over the lazy dog."
    })
    print(f"Generated speech WAV: {len(speech_wav)} bytes")

    # Transcribe it back
    status, headers, body = post_bytes(f"{BASE}/transcribe", speech_wav)
    print(f"Status: {status}")
    result = json.loads(body.decode())
    print(f"Transcript: {result}")
    return result


def test_transcribe_silence():
    """Transcribe silence — should return empty/low confidence."""
    print(f"\n=== POST {BASE}/transcribe (silence) ===")

    # Generate 3 seconds of silence at 16kHz
    import numpy as np
    silence = np.zeros((48000,), dtype=np.float32)  # 3s at 16kHz
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        samples = (silence * 32767).astype(np.int16)
        w.writeframes(samples.tobytes())
    buf.seek(0)
    silence_wav = buf.getvalue()

    status, headers, body = post_bytes(f"{BASE}/transcribe", silence_wav)
    print(f"Status: {status}")
    result = json.loads(body.decode())
    print(f"Result: {result}")
    return result


def test_lan():
    """Test LAN accessibility."""
    print(f"\n=== LAN test ===")
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
    finally:
        s.close()
    lan_url = f"http://{lan_ip}:8300"
    print(f"LAN IP: {lan_ip}")
    print(f"Testing: {lan_url}/health")
    try:
        req = urllib.request.Request(f"{lan_url}/health")
        resp = urllib.request.urlopen(req, timeout=5)
        print(f"LAN Status: {resp.status}")
        print(f"LAN Body: {resp.read().decode()}")
    except Exception as e:
        print(f"LAN test FAILED: {e}")


if __name__ == "__main__":
    test_health()
    speak_status = test_speak()
    test_transcribe_speech()
    test_transcribe_silence()
    test_lan()
    print(f"\n=== SUMMARY ===")
    print(f"/speak returned HTTP {speak_status} (expected 200)")
