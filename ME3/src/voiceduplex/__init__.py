"""voiceduplex — full-duplex voice agent for ME3.

Layered so the Mac <-> Pi boundary is a single file (audioio/pi.py).
Everything outside audioio/ is hardware-agnostic and moves to the Pi untouched.
"""

__version__ = "0.1.0"
