"""Fallback TTS wrapper — try a PRIMARY backend, retry on a BACKUP.

Mirrors ``voiceduplex.delegate.FallbackDelegateBackend``: ME3 speaks through
ONE TTS object; if the primary (e.g. Piper, or Kokoro) raises mid-synthesis
(venv died, model missing, timeout, OOM), the same text is transparently
re-synthesized on the backup and that audio is returned. The loop never knows
which engine actually spoke.

Failure modes that trigger the fallback:
  * primary.synthesize() raises (worker crash, missing model, timeout)
  * primary.warm() raises (we fall back for that warm-up too)

If BOTH fail, the primary's error is surfaced (it's the more specific one).
"""

from __future__ import annotations

import logging

from .base import Synthesis, TTS

log = logging.getLogger("voiceduplex.tts.fallback")


class FallbackTTS(TTS):
    def __init__(self, primary: TTS, backup: TTS) -> None:
        self.primary = primary
        self.backup = backup
        self._last_used = self.primary.backend_name

    @property
    def backend_name(self) -> str:
        return f"{self.primary.backend_name} (fallback: {self.backup.backend_name})"

    def warm(self) -> None:
        """Warm the primary; on failure warm the backup so the FIRST real
        utterance doesn't pay the cold-start cost of the fallback."""
        try:
            self.primary.warm()
        except Exception as e:  # noqa: BLE001 - fall through to backup
            log.warning("TTS primary %s failed to warm (%s); warming backup",
                        self.primary.backend_name, e)
            self.backup.warm()

    def synthesize(self, text: str) -> Synthesis:
        try:
            synth = self.primary.synthesize(text)
            self._last_used = self.primary.backend_name
            return synth
        except Exception as e:  # noqa: BLE001 - fall through to backup
            primary_err = e
            log.warning("TTS primary %s failed (%s); retrying on %s",
                        self.primary.backend_name, e, self.backup.backend_name)
            try:
                synth = self.backup.synthesize(text)
                self._last_used = self.backup.backend_name
                return synth
            except Exception:  # noqa: BLE001 - both failed; keep primary's reason
                raise primary_err from None
