#!/usr/bin/env python3
"""Persistent OnIt worker daemon (ME3 Step 8).

Boots OnIt ONCE, then serves many delegated tasks over stdin/stdout without
re-paying the ~60s cold-boot per task. This is what kills the per-task startup
cost: the agent process stays warm between turns.

Protocol (line-oriented, one task at a time):

    stdin  : one task per line
    stdout : "===WORKER_READY==="          once, after boot
             "===TASK_BEGIN==="            before each answer
             <answer text>
             "===TASK_END==="              after each answer
             "===TASK_ERROR=== <msg>"      on a task failure (worker stays up)

The worker NEVER exits on a task error. It only exits when:
  * stdin is closed (EOF)  -> clean shutdown
  * SIGTERM / SIGINT       -> clean shutdown (this is how ME3 kills it on Ctrl-C)

Configuration (all via environment variables, so no secrets/infra live in code):

    ONIT_HOST      primary OpenAI-compatible / vLLM endpoint  (required to run)
    ONIT_HOST2     secondary endpoint for load-balancing       (optional)
    ONIT_MODEL     model name served by the endpoint           (default: auto)
    ONIT_DATA_PATH scratch dir where OnIt runs tasks           (default: ./onit_work)

The API key is read from the OS keychain by OnIt itself (via `onit setup`);
it is NEVER passed here or written to disk.
"""
import asyncio
import contextlib
import io
import os
import signal
import sys
from pathlib import Path

os.environ["ONIT_APPROVAL_CHANNEL"] = "1"
os.environ["ONIT_AUTO_APPROVE"] = "1"

from src.onit import OnIt  # noqa: E402

HOST = os.environ.get("ONIT_HOST", "").strip()
HOST2 = os.environ.get("ONIT_HOST2", "").strip()
MODEL = os.environ.get("ONIT_MODEL", "").strip()
DATA_PATH = Path(os.environ.get("ONIT_DATA_PATH", str(Path(__file__).resolve().parent / "onit_work")))

if not HOST:
    sys.stderr.write(
        "[worker] ERROR: ONIT_HOST is not set. Point it at your "
        "OpenAI-compatible / vLLM endpoint, e.g.\n"
        "  export ONIT_HOST=http://<host>:<port>/v1\n"
    )
    sys.exit(2)

serving = {
    "host": HOST,
    "model": MODEL or "auto",
    "load_balancer": "sticky",
}
if HOST2:
    serving["host2"] = HOST2
    serving["model2"] = MODEL or "auto"

CONFIG = {
    "serving": serving,
    "data_path": str(DATA_PATH),
    "stream": False,
    "verbose": False,
}

_shutdown = asyncio.Event()


def _sig_handler(*_a):
    # ME3 sent SIGTERM/SIGINT (e.g. Ctrl-C). Request a clean shutdown.
    _shutdown.set()


def boot_agent():
    """Boot OnIt in a thread (its ctor calls asyncio.run internally)."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        agent = OnIt(config=CONFIG)
    return agent


async def main() -> int:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _sig_handler)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: _shutdown.set())

    # Boot once. Redirect the boot chatter to stderr so stdout stays a clean
    # protocol channel.
    sys.stderr.write("[worker] booting OnIt ...\n")
    sys.stderr.flush()
    agent = await asyncio.to_thread(boot_agent)
    sys.stderr.write("[worker] OnIt ready\n")
    sys.stderr.flush()

    print("===WORKER_READY===", flush=True)

    while not _shutdown.is_set():
        # Read one task line. Use a thread so a blocking stdin read doesn't
        # stall the loop, and so a shutdown signal can break the wait.
        line = await asyncio.to_thread(sys.stdin.readline)
        if line == "":            # EOF -> clean shutdown
            break
        task = line.rstrip("\n")
        if not task.strip():
            continue
        try:
            answer = await agent.process_task(
                task, safety_queue=asyncio.Queue(maxsize=10)
            )
            print("===TASK_BEGIN===", flush=True)
            print(answer, flush=True)
            print("===TASK_END===", flush=True)
        except Exception as e:  # noqa: BLE001 - report, keep serving
            print(f"===TASK_ERROR=== {e}", flush=True)

    sys.stderr.write("[worker] shutting down\n")
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
