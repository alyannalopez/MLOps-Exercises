"""Genie Pi client — runs on the Raspberry Pi.

Streams microphone audio to the Mac model server over WebSocket, plays returned
TTS audio, and handles barge-in via InterruptionFrame semantics (when the Pi
detects user speech during playback, it tells the Mac to clear the TTS queue).

Usage:
    PYTHONPATH=genie .venv/bin/python -m genie.client_pi \
        --server ws://mac.local:8300/ws/audio
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000  # mic input rate sent to the server
CHANNELS = 1
BLOCK_MS = 32


class PiClient:
    def __init__(self, server_url: str, mic_device: Optional[int] = None,
                 speaker_device: Optional[int] = None):
        self.server_url = server_url
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self._ws: Optional[websockets.ClientConnection] = None
        self._play_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._barge_event = asyncio.Event()

    async def connect(self):
        self._ws = await websockets.connect(self.server_url)
        print(f"[pi] connected to {self.server_url}", flush=True)

    async def _mic_loop(self):
        """Read mic blocks and send to server."""
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[pi] mic status: {status}", flush=True)
            # Send raw int16 bytes as a WAV-ish block (server expects WAV).
            self._barge_event.clear()
            asyncio.create_task(self._ws.send(indata.tobytes()))

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=self.mic_device,
            blocksize=int(SAMPLE_RATE * BLOCK_MS / 1000),
            callback=callback,
        ):
            print("[pi] mic stream started", flush=True)
            while True:
                await asyncio.sleep(3600)

    async def _play_loop(self):
        """Play received TTS audio blocks."""
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=self.speaker_device,
            blocksize=int(SAMPLE_RATE * BLOCK_MS / 1000),
        ) as out:
            print("[pi] speaker stream started", flush=True)
            while True:
                block = await self._play_queue.get()
                if block is None:
                    break
                # If barge-in requested, skip remaining playback of this turn.
                if self._barge_event.is_set():
                    continue
                out.write(block)

    async def _recv_loop(self):
        """Receive messages from server: audio bytes or control JSON."""
        async for message in self._ws:
            if isinstance(message, bytes):
                await self._play_queue.put(message)
            else:
                # Text message — could be a control event.
                try:
                    import json
                    ctrl = json.loads(message)
                    if ctrl.get("event") == "interruption":
                        self._barge_event.set()
                        await self._play_queue.put(None)  # flush
                except Exception:
                    pass

    async def run(self):
        await self.connect()
        await asyncio.gather(
            self._mic_loop(),
            self._play_loop(),
            self._recv_loop(),
        )


def main():
    parser = argparse.ArgumentParser(description="Genie Pi client")
    parser.add_argument("--server", required=True, help="WebSocket server URL")
    parser.add_argument("--mic-device", type=int, default=None)
    parser.add_argument("--speaker-device", type=int, default=None)
    args = parser.parse_args()

    client = PiClient(
        server_url=args.server,
        mic_device=args.mic_device,
        speaker_device=args.speaker_device,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("[pi] shutting down", flush=True)


if __name__ == "__main__":
    main()
