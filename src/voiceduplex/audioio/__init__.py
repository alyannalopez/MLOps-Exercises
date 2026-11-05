"""Audio I/O — THE Mac <-> Pi boundary.

Only this package touches hardware. `make_audio_io()` picks the right
implementation for the platform (or you override via config / arg).
"""

from .base import AudioIO, Frame
from .mac import MacAudioIO
from .pi import PiAudioIO

__all__ = ["AudioIO", "Frame", "MacAudioIO", "PiAudioIO", "make_audio_io"]


def make_audio_io(kind: str | None = None, **kwargs) -> AudioIO:
    """Return the appropriate AudioIO backend.

    kind: "mac" | "pi" | None (None -> auto-detect by platform).
    Extra kwargs are forwarded to the backend constructor.
    """
    import platform

    if kind is None:
        # Auto: Pi is aarch64 Linux; Mac is arm64/x86_64 Darwin.
        kind = "pi" if platform.system() == "Linux" else "mac"

    cls = {"mac": MacAudioIO, "pi": PiAudioIO}[kind]
    if kind == "pi":
        # The Pi always uses whatever mic ALSA/PulseAudio exposes as the system
        # default — a Mac-pinned name like "MacBook Air Microphone" is meaningless
        # there. Ignore input_device so the same config works on both platforms.
        kwargs.pop("input_device", None)
    return cls(**kwargs)
