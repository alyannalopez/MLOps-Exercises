"""Step 8 — delegation to a bigger agent harness (black box).

ME3's router (``voiceduplex.loop.brain.route``) handles backchannels and the
ten quick-tool intents offline. Anything else is a *delegated* task: "make a
webpage about X", "write me a report", "summarise this file". Those go to a
real agent harness that can use tools and a big model.

We treat the harness as a BLACK BOX per the Step 8 brief: ME3 measures
turnaround (B1), keep-alive (B4) and result-back (B3) — never the harness
internals. The only contract is:

    backend.dispatch(task) -> str   # final text answer

Backends
--------
OnItBackend   spawn the OnIt agent (own venv) as a subprocess, run one task,
              capture its final answer. Default and only backend for now.

StubBackend   returns a canned "still learning" line. Used when no real
              harness is configured, so the loops stay fully offline/testable.

The loops never import a specific backend — they receive one via
``route(..., delegate=backend)``. Swapping OnIt for Hermes/OpenCode later is a
new Backend class + a config line; the loops and brain don't change.
"""

from __future__ import annotations

import abc
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class DelegateResult:
    """Outcome of one delegated task (black-box view)."""
    ok: bool
    text: str = ""            # final answer (spoken back to the user)
    error: str = ""           # human-readable failure reason
    elapsed: float = 0.0      # wall-clock seconds, dispatch start -> result
    cancelled: bool = False   # True if a barge-in aborted the wait


class DelegateBackend(abc.ABC):
    """Interface every harness backend implements."""

    @abc.abstractmethod
    def dispatch(self, task: str) -> str:
        """Send ``task`` to the harness and return its final text answer.

        May block for a long time (this is expected — the caller runs it in a
        background thread). Should raise on hard failure; the caller catches.
        """

    def is_available(self) -> bool:
        """Cheap readiness probe. Default True; backends override to check
        binary/network."""
        return True

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StubBackend(DelegateBackend):
    """Offline stand-in: rotates canned 'still learning' lines.

    Keeps the full pipeline (route -> delegate -> speak) exercisable with no
    network, no keys, no venv. Used when ``delegate.backend`` is unset.
    """

    _LINES = [
        "I handed that to my deeper brain, but it's still warming up. Try a quick task for now.",
        "That one goes to my research brain, which isn't connected yet. Quick tools work great though.",
        "I'd delegate that, but my harness isn't plugged in. Ask me the time or the lights instead.",
    ]
    _i = 0

    def dispatch(self, task: str) -> str:
        line = self._LINES[self._i % len(self._LINES)]
        self._i += 1
        return line


class OnItBackend(DelegateBackend):
    """Delegate to the OnIt agent (Rowel Atienza) via its one-shot driver.

    Spawns ``run_onit_task.sh "<task>"`` in the isolated OnIt venv, waits for
    it to finish (bounded by ``timeout``), and extracts the answer between the
    ``===ANSWER_BEGIN===`` / ``===ANSWER_END===`` markers the driver prints.

    Everything about the harness (model server, keychain, tools) lives inside
    that script + venv — ME3 only ever sees "task in, text out".
    """

    def __init__(
        self,
        runner: str,
        *,
        timeout: float = 120.0,
        workdir: Optional[str] = None,
        env_extra: Optional[dict] = None,
    ) -> None:
        self.runner = Path(runner)
        self.timeout = timeout
        self.workdir = workdir
        self.env_extra = env_extra or {}

    @property
    def name(self) -> str:
        return "OnIt"

    def is_available(self) -> bool:
        return self.runner.exists() and os.access(self.runner, os.X_OK)

    def dispatch(self, task: str) -> str:
        t0 = time.monotonic()
        env = dict(os.environ)
        env.update(self.env_extra)
        proc = subprocess.run(
            [str(self.runner), task],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.workdir,
            env=env,
        )
        elapsed = time.monotonic() - t0
        out = proc.stdout or ""
        err = proc.stderr or ""

        # Prefer the marker-delimited answer; fall back to the whole stdout.
        if "===ANSWER_BEGIN===" in out and "===ANSWER_END===" in out:
            body = out.split("===ANSWER_BEGIN===", 1)[1].split("===ANSWER_END===", 1)[0]
        else:
            body = out
        answer = body.strip()

        if proc.returncode != 0 and not answer:
            tail = (err.strip().splitlines() or ["(no stderr)"])[-1]
            raise RuntimeError(f"OnIt exited {proc.returncode}: {tail}")
        if not answer:
            raise RuntimeError("OnIt returned an empty answer")
        return answer


class PersistentOnItBackend(DelegateBackend):
    """Delegate to a SINGLE long-lived OnIt worker process.

    Unlike :class:`OnItBackend` (one subprocess per task, ~60s cold-boot each
    time), this backend boots ONE worker daemon (``onit_worker.py`` in the OnIt
    venv) and keeps it warm. Every task is a line written to its stdin; the
    answer is the block between ``===TASK_BEGIN===`` / ``===TASK_END===``.

    Lifecycle is owned by the loop:
      * ``ensure_started()``  — spawn + wait for readiness (called on first use)
      * ``dispatch(task)``    — send one task, read one answer (warm, fast)
      * ``shutdown()``        — SIGTERM the worker, reap it (called on Ctrl-C)

    The worker is a child of the ME3 process, so killing it on Ctrl-C is a
    plain ``terminate()`` + ``wait()`` — no orphaned agents left behind.
    """

    _READY = "===WORKER_READY==="
    _BEGIN = "===TASK_BEGIN==="
    _END = "===TASK_END==="
    _ERROR = "===TASK_ERROR==="

    def __init__(
        self,
        python: str,
        worker_script: str,
        *,
        timeout: float = 120.0,
        boot_timeout: float = 120.0,
        workdir: Optional[str] = None,
        env_extra: Optional[dict] = None,
    ) -> None:
        self.python = Path(python)
        self.worker_script = Path(worker_script)
        self.timeout = timeout
        self.boot_timeout = boot_timeout
        self.workdir = workdir
        self.env_extra = env_extra or {}
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "OnIt (persistent)"

    # -- lifecycle ---------------------------------------------------------
    def ensure_started(self) -> None:
        """Spawn the worker (idempotent) and block until it reports ready."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return  # already warm
            if not (self.python.exists() and self.worker_script.exists()):
                raise RuntimeError(
                    f"persistent OnIt missing: python={self.python} "
                    f"worker={self.worker_script}"
                )
            env = dict(os.environ)
            env.update(self.env_extra)
            self._proc = subprocess.Popen(
                [str(self.python), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                cwd=self.workdir,
                env=env,
            )
            self._wait_ready()

    def _wait_ready(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        deadline = time.monotonic() + self.boot_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"OnIt worker exited during boot (rc={self._proc.returncode})"
                )
            line = self._proc.stdout.readline()
            if line == "":  # worker died without a ready line
                raise RuntimeError("OnIt worker closed stdout during boot")
            if line.strip() == self._READY:
                return
            # ignore any stray pre-ready output
        raise RuntimeError("OnIt worker did not become ready in time")

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, task: str) -> str:
        """Send one task to the warm worker and return its answer."""
        self.ensure_started()
        with self._lock:
            assert self._proc is not None
            assert self._proc.stdin is not None and self._proc.stdout is not None
            proc = self._proc
            t0 = time.monotonic()
            try:
                proc.stdin.write(task + "\n")
                proc.stdin.flush()
                return self._read_answer(proc, t0)
            except BrokenPipeError:
                raise RuntimeError("OnIt worker connection lost (pipe broken)")

    def _read_answer(self, proc: subprocess.Popen, t0: float) -> str:
        assert proc.stdout is not None
        deadline = t0 + self.timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"OnIt worker exited mid-task (rc={proc.returncode}); "
                    "will respawn on next task"
                )
            line = proc.stdout.readline()
            if line == "":
                raise RuntimeError("OnIt worker closed stdout mid-task")
            line = line.rstrip("\n")
            if line.startswith(self._BEGIN):
                buf = []
                while True:
                    l2 = proc.stdout.readline()
                    if l2 == "":
                        raise RuntimeError("OnIt worker closed stdout mid-answer")
                    l2 = l2.rstrip("\n")
                    if l2.startswith(self._END):
                        break
                    buf.append(l2)
                return "\n".join(buf).strip()
            if line.startswith(self._ERROR):
                raise RuntimeError(line[len(self._ERROR):].strip() or "task failed")
            # ignore anything else (shouldn't happen)
        raise RuntimeError("OnIt worker timed out answering")

    # -- teardown ----------------------------------------------------------
    def shutdown(self) -> None:
        """SIGTERM the worker and reap it. Safe to call repeatedly / on Ctrl-C."""
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()  # SIGTERM -> worker's handler -> clean exit
            except OSError:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()  # SIGKILL if it ignored TERM
                    proc.wait(timeout=2.0)
                except OSError:
                    pass
        # Reap any leftover fds.
        for attr in ("stdin", "stdout", "stderr"):
            fh = getattr(proc, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass

    def is_available(self) -> bool:
        return self.python.exists() and self.worker_script.exists()


# ---------------------------------------------------------------------------
# Factory — reads the ``delegate:`` block from config/config.yaml
# ---------------------------------------------------------------------------
class FallbackDelegateBackend(DelegateBackend):
    """Try a PRIMARY harness, and if it fails, transparently retry on a BACKUP.

    This is the resilience layer: ME3's OnIt worker can die (endpoint not
    configured, broken venv, worker crash, timeout). Rather than speaking an
    apology, we hand the same task to a second harness (e.g. Pipecat) and
    speak whichever succeeds. The loop sees ONE backend; it never knows which
    harness actually answered.

    Failure modes that trigger the fallback:
      * primary raises (worker died on boot, task error, timeout)
      * primary is_available() is False (cheap pre-flight probe)
    If BOTH fail, we surface the primary's error (it's the more specific one)
    so the spoken apology is meaningful.
    """

    def __init__(self, primary: DelegateBackend, backup: DelegateBackend) -> None:
        self.primary = primary
        self.backup = backup

    @property
    def name(self) -> str:
        return f"{self.primary.name} (fallback: {self.backup.name})"

    def is_available(self) -> bool:
        return self.primary.is_available() or self.backup.is_available()

    def dispatch(self, task: str) -> str:
        # Primary first.
        if self.primary.is_available():
            try:
                return self.primary.dispatch(task)
            except Exception as e:  # noqa: BLE001 - fall through to backup
                primary_err = e
        else:
            primary_err = RuntimeError(f"{self.primary.name} unavailable")

        # Backup.
        try:
            if self.backup.is_available():
                return self.backup.dispatch(task)
            raise RuntimeError(f"{self.backup.name} unavailable")
        except Exception:  # noqa: BLE001 - both failed; keep primary's reason
            raise primary_err from None

    def shutdown(self) -> None:
        for b in (self.primary, self.backup):
            fn = getattr(b, "shutdown", None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass


def _build_onit(d: dict) -> DelegateBackend:
    """Build the OnIt backend from the ``delegate`` config block."""
    root = Path(__file__).resolve().parents[3]  # .../ME3
    sandbox = root.parent  # .../sandbox
    # Default to the PERSISTENT worker (boots OnIt once, stays warm). Set
    # delegate.mode: oneshot to fall back to the per-task subprocess.
    mode = (d.get("mode") or "persistent").lower()
    if mode == "persistent":
        return PersistentOnItBackend(
            python=d.get("python") or str(sandbox / "onit-venv" / "bin" / "python"),
            worker_script=d.get("worker") or str(sandbox / "onit" / "onit_worker.py"),
            timeout=float(d.get("timeout", 120.0)),
            boot_timeout=float(d.get("boot_timeout", 120.0)),
            workdir=d.get("workdir") or None,
        )
    runner = d.get("runner") or str(sandbox / "onit" / "run_onit_task.sh")
    return OnItBackend(
        runner,
        timeout=float(d.get("timeout", 120.0)),
        workdir=d.get("workdir") or None,
    )


def _build_pipecat(d: dict) -> DelegateBackend:
    """Build the Pipecat backup backend from the ``delegate`` config block.

    Uses the SAME persistent-worker machinery as OnIt, pointed at
    ``pipecat_worker.py``. The worker speaks the identical stdin/stdout
    protocol, so no loop/brain changes are needed. Env vars (PIPECAT_HOST,
    PIPECAT_MODEL, OPENAI_API_KEY) are forwarded from the config block so no
    secrets live in code.
    """
    root = Path(__file__).resolve().parents[3]  # .../ME3
    sandbox = root.parent  # .../sandbox
    env_extra = {}
    if d.get("pipecat_host"):
        env_extra["PIPECAT_HOST"] = d["pipecat_host"]
    if d.get("pipecat_model"):
        env_extra["PIPECAT_MODEL"] = d["pipecat_model"]
    if d.get("openai_api_key"):
        env_extra["OPENAI_API_KEY"] = d["openai_api_key"]
    return PersistentOnItBackend(
        python=d.get("pipecat_python") or d.get("python") or str(sandbox / "onit-venv" / "bin" / "python"),
        worker_script=d.get("pipecat_worker") or str(sandbox / "onit" / "pipecat_worker.py"),
        timeout=float(d.get("timeout", 120.0)),
        boot_timeout=float(d.get("boot_timeout", 120.0)),
        workdir=d.get("workdir") or None,
        env_extra=env_extra,
    )


def build_backend(cfg: dict) -> DelegateBackend:
    """Build the configured backend. ``cfg`` is the parsed config dict (may be
    empty). Returns a StubBackend when nothing is configured, so the loops
    always have *something* to delegate to and never crash.

    ``delegate.backend`` values:
      * ``stub``     — canned offline reply (default when unset)
      * ``onit``     — OnIt harness only
      * ``pipecat``  — Pipecat harness only
      * ``fallback`` — primary (``delegate.primary``, default onit) with an
                       automatic backup (``delegate.backup``, default pipecat).
                       If the primary fails, the task is retried on the backup.
    """
    d = (cfg or {}).get("delegate") or {}
    kind = (d.get("backend") or "stub").lower()
    if kind == "stub":
        return StubBackend()
    if kind == "onit":
        return _build_onit(d)
    if kind == "pipecat":
        return _build_pipecat(d)
    if kind == "fallback":
        primary_kind = (d.get("primary") or "onit").lower()
        backup_kind = (d.get("backup") or "pipecat").lower()
        primary = _build_onit(d) if primary_kind == "onit" else _build_pipecat(d)
        backup = _build_pipecat(d) if backup_kind == "pipecat" else _build_onit(d)
        return FallbackDelegateBackend(primary, backup)
    # Unknown backend -> fall back to stub rather than crash the loop.
    return StubBackend()


# ---------------------------------------------------------------------------
# Fire-and-forget dispatcher used by brain.route()
# ---------------------------------------------------------------------------
def _run_and_collect(
    backend: DelegateBackend,
    task: str,
    on_done: Callable[[DelegateResult], None],
) -> None:
    """Background-thread body: dispatch, then report via ``on_done``."""
    t0 = time.monotonic()
    try:
        text = backend.dispatch(task)
        on_done(DelegateResult(ok=True, text=text, elapsed=time.monotonic() - t0))
    except Exception as e:  # noqa: BLE001 — surface as a spoken apology, never crash
        on_done(
            DelegateResult(
                ok=False,
                error=str(e),
                elapsed=time.monotonic() - t0,
            )
        )


class PendingTask:
    """Handle to a running delegated task. The loop polls ``wait()`` while the
    mic stays live, so a barge-in can cancel the wait."""

    def __init__(self, backend: DelegateBackend, task: str) -> None:
        self.backend = backend
        self.task = task
        self._evt = threading.Event()
        self._result: Optional[DelegateResult] = None
        self._cancelled = False
        threading.Thread(
            target=_run_and_collect,
            args=(backend, task, self._on_done),
            daemon=True,
            name="delegate-task",
        ).start()

    def _on_done(self, res: DelegateResult) -> None:
        self._result = res
        self._evt.set()

    def wait(self, timeout: float) -> DelegateResult:
        """Block up to ``timeout`` seconds. Returns a DelegateResult. If the
        timeout expires first, returns a cancelled result (caller decides
        whether to keep waiting or give up)."""
        if self._evt.wait(timeout):
            return self._result or DelegateResult(ok=False, error="no result", cancelled=True)
        # Timed out: mark cancelled; the background thread keeps running but the
        # loop has moved on (it will speak the apology / keep listening).
        self._cancelled = True
        return DelegateResult(ok=False, error="timed out", cancelled=True)

    def cancel(self) -> None:
        """Best-effort: stop waiting. (The subprocess can't be killed from here
        without its PID; it will finish in the background and its result is
        simply dropped.)"""
        self._cancelled = True
