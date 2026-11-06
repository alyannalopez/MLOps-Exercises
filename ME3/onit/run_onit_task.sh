#!/bin/bash
# ME3 Step 8 — run ONE delegated task through OnIt (one-shot, in-process, no TTY).
#
# Usage:
#   ONIT_HOST=http://<host>:<port>/v1 ./run_onit_task.sh "your task here"
#
# Environment:
#   ONIT_PYTHON    interpreter of the isolated OnIt venv (default: <repo>/onit-venv/bin/python)
#   ONIT_HOST      primary OpenAI-compatible / vLLM endpoint  (required)
#   ONIT_HOST2     secondary endpoint for load-balancing       (optional)
#   ONIT_MODEL     model name served by the endpoint           (default: auto)
#   ONIT_DATA_PATH scratch dir where OnIt runs tasks           (default: ./onit_work)
#
# The API key is read from the OS keychain by OnIt; it is never passed here.
set -euo pipefail
cd "$(dirname "$0")"

TASK="${1:?Usage: run_onit_task.sh \"<task>\"}"
: "${ONIT_HOST:?ONIT_HOST is not set (e.g. http://<host>:<port>/v1)}"
# The isolated OnIt/Pipecat venv lives OUTSIDE the repo, two levels up from
# this script (<sandbox>/onit-venv, where <sandbox> holds ME1/ ME2/ ME3/).
# Override with ONIT_PYTHON if your layout differs.
REPO_ROOT="$(cd ../.. && pwd)"
ONIT_PYTHON="${ONIT_PYTHON:-$REPO_ROOT/onit-venv/bin/python}"

export ONIT_HOST ONIT_HOST2 ONIT_MODEL ONIT_DATA_PATH
exec "$ONIT_PYTHON" - "$TASK" <<'PYEOF'
import asyncio, contextlib, io, os, sys
from pathlib import Path

os.environ["ONIT_APPROVAL_CHANNEL"] = "1"
os.environ["ONIT_AUTO_APPROVE"] = "1"

from src.onit import OnIt

TASK = sys.argv[1]
HOST = os.environ["ONIT_HOST"].strip()
HOST2 = os.environ.get("ONIT_HOST2", "").strip()
MODEL = os.environ.get("ONIT_MODEL", "").strip() or "auto"
DATA_PATH = Path(os.environ.get("ONIT_DATA_PATH", str(Path(__file__).resolve().parent / "onit_work")))

serving = {"host": HOST, "model": MODEL, "load_balancer": "sticky"}
if HOST2:
    serving["host2"] = HOST2
    serving["model2"] = MODEL

CONFIG = {
    "serving": serving,
    "data_path": str(DATA_PATH),
    "stream": False,
    "verbose": False,
}

def boot():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return OnIt(config=CONFIG)

async def main():
    agent = await asyncio.to_thread(boot)
    answer = await agent.process_task(TASK, safety_queue=asyncio.Queue(maxsize=10))
    print("===ANSWER_BEGIN===")
    print(answer)
    print("===ANSWER_END===")

sys.exit(asyncio.run(main()))
PYEOF
