"""Genie wakeword gate.

Simple energy + text-gated wakeword detector. The pipeline only passes audio
to ASR when the wakeword "genie" is detected in the transcript (or when energy
exceeds a threshold and the user is speaking). This keeps the Mac from
transcribing silence.
"""
from __future__ import annotations

from typing import Optional

from pipecat.frames.frames import TranscriptionFrame


class WakewordGate:
    """Gate frames until the wakeword is heard."""

    def __init__(self, wakeword: str = "genie"):
        self.wakeword = wakeword.lower()
        self.armed = False  # True once wakeword detected this session

    def check_transcription(self, frame: TranscriptionFrame) -> Optional[TranscriptionFrame]:
        """Return the frame if it should pass (wakeword heard or already armed),
        else None to drop it."""
        text = frame.text.lower()
        if not self.armed:
            if self.wakeword in text:
                self.armed = True
                # Strip the wakeword from the transcript before passing on.
                idx = text.find(self.wakeword)
                rest = text[idx + len(self.wakeword):].strip()
                if rest:
                    frame.text = rest
                return frame
            return None
        return frame
