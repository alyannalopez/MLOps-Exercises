#!/usr/bin/env python3
"""Backup delegate worker (ME3 Step 8) — a SECOND agent harness.

This is a drop-in alternative to ``onit_worker.py``. It speaks the EXACT same
line-oriented stdin/stdout protocol, so ``PersistentOnItBackend`` (or any
backend that talks that protocol) can drive it unchanged. Its job is to give
ME3 a *second* harness to fall back to when the primary (OnIt) is down — e.g.
OnIt's endpoint isn't configured, its venv is broken, or its worker dies.

Protocol (identical to onit_worker.py):

    stdin  : one task per line
    stdout : "===WORKER_READY==="          once, after boot
             "===TASK_BEGIN==="            before each answer
             <answer text>
             "===TASK_END==="              after each answer
             "===TASK_ERROR=== <msg>"      on a task failure (worker stays up)

How a task is answered
----------------------
Two tiers, chosen automatically:

  1. **Pipecat pipeline** — if the ``pipecat`` package is importable, the task
     is run through a minimal Pipecat LLM pipeline (pipecat.services.openai
     OpenAILLMService). This is the "real Pipecat" path you asked for.
  2. **Direct OpenAI-compatible call** — if ``pipecat`` is NOT installed (or
     fails), the task is answered with a single blocking call to the same
     OpenAI-compatible endpoint via urllib (stdlib, no deps). This keeps the
     backup functional with zero extra packages, which is the whole point of a
     backup.

Both tiers read the SAME config as OnIt, via environment variables, so there
are no secrets in code:

    PIPECAT_HOST      OpenAI-compatible /v1 base URL   (required to run)
    PIPECAT_MODEL     model name                       (default: auto/empty)
    OPENAI_API_KEY    key for the endpoint             (optional; many local
                          servers accept an empty/"none" key)

The worker NEVER exits on a task error. It only exits on EOF (stdin closed) or
SIGTERM/SIGINT — which is how ME3 tears it down on Ctrl-C.
"""
import asyncio
import contextlib
import io
import json
import os
import signal
import sys
import urllib.request
import urllib.error
from pathlib import Path

HOST = os.environ.get("PIPECAT_HOST", "").strip()
MODEL = os.environ.get("PIPECAT_MODEL", "").strip()
API_KEY = os.environ.get("OPENAI_API_KEY", "").strip() or "none"
TIMEOUT = float(os.environ.get("PIPECAT_TIMEOUT", "120"))

if not HOST:
    sys.stderr.write(
        "[pipecat-worker] ERROR: PIPECAT_HOST is not set. Point it at your "
        "OpenAI-compatible /v1 endpoint, e.g.\n"
        "  export PIPECAT_HOST=http://<host>:<port>/v1\n"
    )
    sys.exit(2)

# Try to import pipecat lazily so the worker still boots (tier 2) without it.
_PIPECAT_LLM = None
try:
    from pipecat.services.openai.llm import OpenAILLMService  # type: ignore

    _PIPECAT_LLM = OpenAILLMService
    sys.stderr.write("[pipecat-worker] pipecat import OK -> tier 1 (pipeline)\n")
except Exception as e:  # noqa: BLE001
    sys.stderr.write(f"[pipecat-worker] pipecat unavailable ({e!r}) -> tier 2 (direct)\n")

_shutdown = asyncio.Event()


def _sig_handler(*_a):
    _shutdown.set()


# ---------------------------------------------------------------------------
# Tier 2 — direct OpenAI-compatible chat completion (stdlib only)
# ---------------------------------------------------------------------------
def _direct_complete(task: str) -> str:
    url = HOST.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL or "auto",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ME3's delegated research brain. Answer the user's "
                    "task concisely and helpfully. Plain text only."
                ),
            },
            {"role": "user", "content": task},
        ],
        "temperature": 0.4,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected response shape: {e}") from e


# ---------------------------------------------------------------------------
# Tier 1 — minimal Pipecat LLM pipeline (used only if pipecat is installed)
# ---------------------------------------------------------------------------
async def _pipecat_complete(task: str) -> str:
    """Run the task through a single Pipecat OpenAI LLM call.

    We use the LLM service directly (not a full audio pipeline) because ME3
    already owns STT (Whisper) and TTS (Kokoro/Piper); the delegate's job is
    purely "task in -> text out". A full Pipecat audio pipeline here would
    fight ME3 over the mic/speaker.
    """
    llm = _PIPECAT_LLM(api_key=API_KEY, model=MODEL or "gpt-4o-mini", base_url=HOST)
    system = (
        "You are ME3's delegated research brain. Answer concisely. "
        "Plain text only."
    )
    user_msg = {"role": "user", "content": task}

    # Prefer the modern (pipecat >=1.9) API: run_inference(LLMContext) -> str,
    # with the system prompt passed to the service rather than as a message.
    if hasattr(llm, "run_inference"):
        try:
            from pipecat.processors.aggregators.llm_context import LLMContext
        except Exception:  # pragma: no cover - very old pipecat
            LLMContext = None
        if LLMContext is not None:
            # pipecat 1.15 LLMContext takes only `messages`; carry the system
            # prompt as the first message (older pipecat accepted a
            # system_instruction kwarg, but 1.15 does not).
            ctx = LLMContext(messages=[
                {"role": "system", "content": system},
                user_msg,
            ])
            out = await llm.run_inference(ctx)
            answer = (out or "").strip()
            if not answer:
                raise RuntimeError("pipecat pipeline returned empty answer")
            return answer

    # Fallback for older pipecat that exposed get_response(messages).
    if hasattr(llm, "get_response"):
        messages = [{"role": "system", "content": system}, user_msg]
        parts = []
        async for chunk in llm.get_response(messages):
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
            elif isinstance(chunk, dict):
                parts.append(str(chunk.get("text", "")))
        answer = "".join(parts).strip()
        if not answer:
            raise RuntimeError("pipecat pipeline returned empty answer")
        return answer

    raise RuntimeError("pipecat OpenAILLMService has no known inference method")


async def _answer(task: str) -> str:
    if _PIPECAT_LLM is not None:
        try:
            return await _pipecat_complete(task)
        except Exception as e:  # noqa: BLE001 - degrade to tier 2
            sys.stderr.write(f"[pipecat-worker] tier1 failed ({e!r}); trying tier2\n")
            sys.stderr.flush()
    return await asyncio.to_thread(_direct_complete, task)


async def main() -> int:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _sig_handler)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: _shutdown.set())

    print("===WORKER_READY===", flush=True)

    while not _shutdown.is_set():
        line = await asyncio.to_thread(sys.stdin.readline)
        if line == "":  # EOF -> clean shutdown
            break
        task = line.rstrip("\n")
        if not task.strip():
            continue
        try:
            answer = await _answer(task)
            print("===TASK_BEGIN===", flush=True)
            print(answer, flush=True)
            print("===TASK_END===", flush=True)
        except Exception as e:  # noqa: BLE001 - report, keep serving
            print(f"===TASK_ERROR=== {e}", flush=True)

    sys.stderr.write("[pipecat-worker] shutting down\n")
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
