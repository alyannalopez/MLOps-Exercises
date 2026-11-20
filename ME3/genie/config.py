"""Genie configuration — single source of truth for the model server + pipeline.

Loads from .env file (project root) first, then allows environment variables
to override. The .env file is gitignored; .env.example is committed as a
template.

Usage:
    cfg = Config.load()
    print(cfg.openrouter_api_key)   # from .env or OPENROUTER_API_KEY env var
    print(cfg.server_url)           # from .env or GENIE_SERVER_URL env var
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def _get(key: str, default: str = "") -> str:
    """Read a config value: env var wins, then .env file, then default."""
    val = os.environ.get(key)
    if val is not None and val != "":
        return val

    if _HAS_DOTENV and ENV_PATH.exists():
        # dotenv already loaded into os.environ at module import time,
        # but we double-check for clarity.
        pass

    return default


@dataclass
class Config:
    # --- OpenRouter (LLM) ---
    openrouter_api_key: str = ""
    llm_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0.6
    llm_max_tokens: int = 200

    # --- Model server (Mac) ---
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    tts_voice: str = "af_bella"
    tts_speed: float = 1.0
    server_port: int = 8300

    # --- Network ---
    server_url: str = ""

    # --- Audio ---
    mic_device: int | None = None
    speaker_device: int | None = None

    # --- VAD ---
    vad_threshold: float = 0.5
    min_speech_ms: int = 280
    min_silence_ms: int = 400
    endpointing_ms: int = 600

    # --- Wakeword ---
    wakeword: str = "genie"

    # --- Derived ---
    @property
    def asr_url(self) -> str:
        return f"{self.server_url}/transcribe"

    @property
    def speak_url(self) -> str:
        return f"{self.server_url}/speak"

    @property
    def health_url(self) -> str:
        return f"{self.server_url}/health"

    # ------------------------------------------------------------------
    # Class factory
    # ------------------------------------------------------------------
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from .env file + environment variables."""
        # Load .env into os.environ (no-overwrite by default)
        if _HAS_DOTENV and ENV_PATH.exists():
            load_dotenv(str(ENV_PATH), override=False)

        # Auto-detect LAN IP for server_url if not set
        url = _get("GENIE_SERVER_URL", "")
        if not url:
            # Try to detect the Mac's LAN IP
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 1))
                lan_ip = s.getsockname()[0]
                s.close()
                port = int(_get("GENIE_PORT", "8300"))
                url = f"http://{lan_ip}:{port}"
            except Exception:
                port = int(_get("GENIE_PORT", "8300"))
                url = f"http://localhost:{port}"

        def _int(name: str, default: int) -> int:
            v = _get(name, "")
            return int(v) if v else default

        def _float(name: str, default: float) -> float:
            v = _get(name, "")
            return float(v) if v else default

        mic = _get("GENIE_MIC_DEVICE", "")
        spk = _get("GENIE_SPEAKER_DEVICE", "")

        return cls(
            openrouter_api_key=_get("OPENROUTER_API_KEY", ""),
            llm_model=_get("GENIE_LLM_MODEL", "openai/gpt-4o-mini"),
            llm_temperature=_float("GENIE_LLM_TEMPERATURE", 0.6),
            llm_max_tokens=_int("GENIE_LLM_MAX_TOKENS", 200),
            asr_model=_get("GENIE_ASR_MODEL", "mlx-community/whisper-large-v3-turbo"),
            tts_voice=_get("GENIE_TTS_VOICE", "af_bella"),
            tts_speed=_float("GENIE_TTS_SPEED", 1.0),
            server_port=_int("GENIE_PORT", 8300),
            server_url=url,
            mic_device=int(mic) if mic else None,
            speaker_device=int(spk) if spk else None,
            vad_threshold=_float("GENIE_VAD_THRESHOLD", 0.5),
            min_speech_ms=_int("GENIE_MIN_SPEECH_MS", 280),
            min_silence_ms=_int("GENIE_MIN_SILENCE_MS", 400),
            endpointing_ms=_int("GENIE_ENDPOINTING_MS", 600),
            wakeword=_get("GENIE_WAKEWORD", "genie"),
        )
