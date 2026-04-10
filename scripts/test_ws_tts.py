#!/usr/bin/env python3
"""Interactive websocket TTS tester for local XTTS service.

Usage:
  source .venv/bin/activate
  python scripts/test_ws_tts.py --api-key mac-local-dev-key

Type text lines and press Enter to hear generated chunks.
Type /quit to exit.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from io import BytesIO
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive websocket TTS tester")
    parser.add_argument("--host", default="127.0.0.1:8000", help="API host:port")
    parser.add_argument("--api-key", required=True, help="x-api-key for the service")
    parser.add_argument("--voice-id", default=None, help="Voice profile id from /v1/voices")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument(
        "--save-dir",
        default="outputs/ws_test_chunks",
        help="Directory for saving received chunk wav files",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save received chunks to disk",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not play audio, only save wav chunk files",
    )
    parser.add_argument(
        "--play-mode",
        choices=["stream", "chunk"],
        default="stream",
        help="stream: continuous ffplay PCM stream (recommended), chunk: afplay per chunk",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Expected TTS sample rate for stream mode",
    )
    return parser.parse_args()


def play_wav_bytes_on_mac(wav_bytes: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        subprocess.run(["afplay", tmp.name], check=False)


class ContinuousPCMPlayer:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.proc: subprocess.Popen[bytes] | None = None
        self._last_ffplay_error: str | None = None

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        if shutil.which("ffplay") is None:
            raise RuntimeError("ffplay not found. Install ffmpeg or use --play-mode chunk")

        self.proc = self._spawn_ffplay(channel_flag="-ac")
        # Some ffplay builds do not recognize -ac for this command path.
        if self.proc.poll() is not None:
            self._capture_ffplay_error()
            if self._last_ffplay_error and "option 'ac'" in self._last_ffplay_error.lower():
                self.stop()
                self.proc = self._spawn_ffplay(channel_flag="-channels")

    def _spawn_ffplay(self, channel_flag: str) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                "ffplay",
                "-loglevel",
                "error",
                "-nodisp",
                "-fflags",
                "nobuffer",
                "-f",
                "s16le",
                "-ar",
                str(self.sample_rate),
                channel_flag,
                "1",
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def push_pcm16_bytes(self, pcm_bytes: bytes) -> None:
        if self.proc is None or self.proc.poll() is not None or self.proc.stdin is None:
            self.start()
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("stream player is not available")
        try:
            self.proc.stdin.write(pcm_bytes)
            self.proc.stdin.flush()
        except BrokenPipeError:
            # ffplay can die unexpectedly on some setups; restart once and retry.
            self._capture_ffplay_error()
            self.stop()
            self.start()
            if self.proc is None or self.proc.stdin is None:
                raise RuntimeError("stream player restart failed")
            self.proc.stdin.write(pcm_bytes)
            self.proc.stdin.flush()

    def _capture_ffplay_error(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        try:
            # Read only small stderr tail to avoid blocking.
            err = self.proc.stderr.read(512)
            if err:
                self._last_ffplay_error = err.decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

    def last_error(self) -> str | None:
        return self._last_ffplay_error

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


async def receive_until_done(
    ws: websockets.WebSocketClientProtocol,
    save_dir: Path,
    utterance_id: str,
    play_audio: bool,
    stream_player: ContinuousPCMPlayer | None,
    play_mode: str,
    default_sample_rate: int,
    save_chunks: bool,
) -> None:
    chunk_counter = 0
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        msg_type = msg.get("type")

        if msg_type == "audio_chunk":
            audio_b64 = msg.get("audio")
            if not audio_b64:
                continue
            fmt = msg.get("format", "wav_base64")
            audio_bytes = base64.b64decode(audio_b64)

            if save_chunks:
                if fmt == "pcm16_base64":
                    chunk_file = save_dir / f"{utterance_id}_chunk_{chunk_counter:03d}.pcm"
                    chunk_file.write_bytes(audio_bytes)
                else:
                    chunk_file = save_dir / f"{utterance_id}_chunk_{chunk_counter:03d}.wav"
                    chunk_file.write_bytes(audio_bytes)
                print(f"chunk {chunk_counter} saved: {chunk_file}")
            else:
                print(f"chunk {chunk_counter} received")

            if play_audio:
                if play_mode == "stream" and stream_player is not None:
                    try:
                        if fmt == "pcm16_base64":
                            stream_player.push_pcm16_bytes(audio_bytes)
                        else:
                            pcm, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
                            if sr != stream_player.sample_rate:
                                raise RuntimeError(
                                    f"sample rate mismatch: got {sr}, expected {stream_player.sample_rate}"
                                )
                            if pcm.ndim > 1:
                                pcm = pcm.mean(axis=1)
                            pcm_i16 = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
                            stream_player.push_pcm16_bytes(pcm_i16.tobytes())
                    except Exception as exc:
                        extra = ""
                        if stream_player is not None and stream_player.last_error():
                            extra = f" | ffplay: {stream_player.last_error()}"
                        print(f"stream playback failed ({exc}); falling back to chunk playback{extra}")
                        if fmt == "pcm16_base64":
                            arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                            buf = BytesIO()
                            sample_rate = int(msg.get("sample_rate") or default_sample_rate)
                            sf.write(buf, arr, sample_rate, format="WAV")
                            play_wav_bytes_on_mac(buf.getvalue())
                        else:
                            play_wav_bytes_on_mac(audio_bytes)
                else:
                    if fmt == "pcm16_base64":
                        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                        buf = BytesIO()
                        sample_rate = int(msg.get("sample_rate") or 24000)
                        sf.write(buf, arr, sample_rate, format="WAV")
                        play_wav_bytes_on_mac(buf.getvalue())
                    else:
                        play_wav_bytes_on_mac(audio_bytes)
            chunk_counter += 1
            continue

        if msg_type == "done":
            print("done")
            return

        if msg_type == "error":
            detail = msg.get("detail", "unknown error")
            code = msg.get("code", "")
            raise RuntimeError(f"server error {code}: {detail}")

        if msg_type in {"ready", "config_ack"}:
            print(msg)
            continue

        print(f"info: {msg}")


async def run() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stream_player: ContinuousPCMPlayer | None = None
    if not args.no_play and args.play_mode == "stream":
        stream_player = ContinuousPCMPlayer(sample_rate=args.sample_rate)
        stream_player.start()

    ws_url = f"ws://{args.host}/v1/tts/ws?api_key={args.api_key}"
    print(f"connecting: {ws_url}")

    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            # Consume initial ready message.
            init_msg = json.loads(await ws.recv())
            print(init_msg)

            config_msg = {
                "type": "config",
                "voice_id": args.voice_id,
                "language": args.language,
                "speed": args.speed,
            }
            await ws.send(json.dumps(config_msg))
            print("config sent")

            cfg_ack = json.loads(await ws.recv())
            print(cfg_ack)

            print("Enter text (or /quit):")
            while True:
                text = await asyncio.to_thread(input, "> ")
                if text.strip().lower() in {"/quit", "/exit"}:
                    print("bye")
                    return
                if not text.strip():
                    continue

                utterance_id = str(int(time.time() * 1000))
                await ws.send(json.dumps({"type": "text", "text": text}))
                await receive_until_done(
                    ws,
                    save_dir,
                    utterance_id,
                    play_audio=not args.no_play,
                    stream_player=stream_player,
                    play_mode=args.play_mode,
                    default_sample_rate=args.sample_rate,
                    save_chunks=not args.no_save,
                )
    finally:
        if stream_player is not None:
            stream_player.stop()


if __name__ == "__main__":
    asyncio.run(run())
