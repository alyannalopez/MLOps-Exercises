"""Genie — Mac-side model server: Whisper ASR + Kokoro TTS, both warm.

Endpoints (HTTP, for the Pi's agent client):
  GET  /health          -> {"status": "ok", "asr": bool, "tts": bool}
  POST /transcribe      -> raw WAV bytes in body (16k mono recommended)
                           returns {"text": "...", "latency_s": ...}
  POST /speak           -> JSON {"text": "..."} (optionally "voice")
                           returns audio/wav bytes (24 kHz mono)

Run:  PYTHONPATH=. .venv-kokoro/bin/python -m genie.server
      (listens on 0.0.0.0:8300)

Uses mlx-whisper on the Mac (Apple Silicon) and Kokoro KPipeline for TTS —
both fully local, no cloud keys.

REQUIRES: .venv-kokoro (Python 3.12) with kokoro==0.9.4, mlx-whisper, fastapi, uvicorn.
"""
from __future__ import annotations

import io
import logging
import sys
import time
import wave

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from genie.config import Config

# ---------------------------------------------------------------------------
# Load config from .env + environment variables
# ---------------------------------------------------------------------------
CFG = Config.load()

SAMPLE_RATE = 24000  # TTS output rate advertised to the client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("genie.server")

app = FastAPI(title="Genie Model Server")

_state: dict = {"asr": False, "tts": False}
_pipeline = None  # Kokoro KPipeline, loaded at startup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wav_bytes(audio_arr: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy array to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        samples = (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Startup — warm both models
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_models() -> None:
    global _pipeline

    t0 = time.perf_counter()

    # --- ASR warm -----------------------------------------------------------
    try:
        from mlx_whisper import transcribe

        silence = np.zeros((8000,), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silence, 16000, format="WAV")
        buf.seek(0)
        transcribe(buf.getvalue(), path_or_hf_repo=CFG.asr_model, language="en")
        _state["asr"] = True
    except Exception as exc:
        log.error("ASR warm failed: %s", exc)
    t1 = time.perf_counter()

    # --- TTS warm -----------------------------------------------------------
    try:
        from kokoro import KPipeline

        _pipeline = KPipeline(lang_code="a")
        # Warm with a short utterance
        list(_pipeline("Hello.", voice=CFG.tts_voice))
        _state["tts"] = True
    except Exception as exc:
        log.error("TTS warm failed: %s", exc)
    t2 = time.perf_counter()

    log.info(
        "ASR warm in %.1fs, TTS warm in %.1fs (asr=%s tts=%s)",
        t1 - t0, t2 - t1, _state["asr"], _state["tts"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "asr": _state["asr"],
        "tts": _state["tts"],
        "asr_model": CFG.asr_model,
        "tts_voice": CFG.tts_voice,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/transcribe")
async def transcribe_ep(request: Request):
    """Accept raw WAV bytes, return JSON transcript."""
    body = await request.body()
    t0 = time.perf_counter()

    audio, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    from mlx_whisper import transcribe

    out = transcribe(audio, path_or_hf_repo=CFG.asr_model, language="en")
    dt = time.perf_counter() - t0

    segs = out.get("segments") or []
    text = (out.get("text") or "").strip().strip(".")

    payload: dict = {
        "text": text,
        "latency_s": round(dt, 3),
    }
    if segs:
        no_speech = sum(s.get("no_speech_prob", 0) for s in segs) / len(segs)
        avg_logprob = sum(s.get("avg_logprob", 0) for s in segs) / len(segs)
        payload.update({
            "no_speech": round(no_speech, 3),
            "avg_logprob": round(avg_logprob, 3),
            "conf": round(1.0 - no_speech, 3),
        })
        # Whisper can hallucinate on silence. Use avg_logprob as a heuristic.
        # Very poor logprob + very short text = likely hallucination.
        if avg_logprob < -0.3 and len(text) < 15:
            payload["speech_detected"] = False
            payload["text"] = ""
        else:
            payload["speech_detected"] = True
    else:
        payload["speech_detected"] = False
        payload["text"] = ""
    return payload


@app.post("/speak")
def speak_ep(payload: dict):
    """Accept JSON {"text": "..."}, return WAV bytes (24 kHz mono)."""
    text = (payload.get("text") or "").strip()
    voice = payload.get("voice", CFG.tts_voice)
    speed = payload.get("speed", CFG.tts_speed)

    if not text:
        return Response(
            '{"error":"empty text"}', status_code=400,
            media_type="application/json",
        )
    if _pipeline is None:
        return Response(
            '{"error":"tts not ready"}', status_code=503,
            media_type="application/json",
        )

    t0 = time.perf_counter()
    results = list(_pipeline(text, voice=voice, speed=speed))

    if results:
        # Concatenate all result segments
        audio_chunks = []
        for r in results:
            audio_chunks.append(r.audio.cpu().numpy().astype(np.float32))
        audio_all = np.concatenate(audio_chunks)
    else:
        audio_all = np.zeros((1,), dtype=np.float32)

    dt = time.perf_counter() - t0
    return Response(
        _wav_bytes(audio_all),
        media_type="audio/wav",
        headers={
            "X-Synthesis-Latency": f"{dt:.3f}",
            "X-Audio-Seconds": f"{len(audio_all) / SAMPLE_RATE:.2f}",
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = CFG.server_port
    log.info("Starting Genie model server on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
