"""VAD — "is a human speaking right now?".

Two interchangeable engines behind one interface:
  - SileroVAD   (primary; ONNX, best quality, ~2 MB)
  - WebRtcVAD   (backup; C lib, lighter, older)

Switch via config `vad.engine: silero|webrtc` (or the probe's --engine flag).
Both emit the same event stream: speech_start / speech_end with timestamps.
This same VAD later drives barge-in detection (Step 6).
"""

from .base import VAD, VADState
from .silero_vad import SileroVAD
from .webrtc_vad import WebRtcVAD

__all__ = ["VAD", "VADState", "SileroVAD", "WebRtcVAD", "make_vad"]


def make_vad(engine: str = "silero", **kwargs) -> VAD:
    return {"silero": SileroVAD, "webrtc": WebRtcVAD}[engine](**kwargs)
