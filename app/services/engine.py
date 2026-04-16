from pathlib import Path
from io import BytesIO
from threading import BoundedSemaphore, Lock
from uuid import uuid4

import numpy as np
import soundfile as sf
from TTS.api import TTS

from app.core.settings import Settings


class XTTSEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._semaphore = BoundedSemaphore(value=self.settings.max_concurrent_requests)
        self._model: TTS | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                self._model = TTS(self.settings.model_name).to(self.settings.device)

    def synthesize_to_file(
        self,
        text: str,
        language: str,
        speaker_wav_path: str,
        speed: float,
    ) -> tuple[str, Path]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        acquired = self._semaphore.acquire(timeout=self.settings.request_timeout_seconds)
        if not acquired:
            raise TimeoutError("Request timed out waiting for inference slot")

        request_id = uuid4().hex
        out_file = self.settings.output_dir / f"{request_id}.wav"

        try:
            kwargs = self._build_tts_kwargs(
                text=text,
                language=language,
                speed=speed,
                speaker_wav_path=speaker_wav_path,
            )
            kwargs["file_path"] = str(out_file)

            self._model.tts_to_file(**kwargs)
            return request_id, out_file
        finally:
            self._semaphore.release()

    def synthesize_to_wav_bytes(
        self,
        text: str,
        language: str,
        speaker_wav_path: str,
        speed: float,
    ) -> bytes:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        acquired = self._semaphore.acquire(timeout=self.settings.request_timeout_seconds)
        if not acquired:
            raise TimeoutError("Request timed out waiting for inference slot")

        try:
            kwargs = self._build_tts_kwargs(
                text=text,
                language=language,
                speed=speed,
                speaker_wav_path=speaker_wav_path,
            )

            wav = self._model.tts(**kwargs)
            buf = BytesIO()
            sf.write(buf, wav, 24000, format="WAV")
            return buf.getvalue()
        finally:
            self._semaphore.release()

    def synthesize_to_pcm16_bytes(
        self,
        text: str,
        language: str,
        speaker_wav_path: str,
        speed: float,
    ) -> bytes:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        acquired = self._semaphore.acquire(timeout=self.settings.request_timeout_seconds)
        if not acquired:
            raise TimeoutError("Request timed out waiting for inference slot")

        try:
            kwargs = self._build_tts_kwargs(
                text=text,
                language=language,
                speed=speed,
                speaker_wav_path=speaker_wav_path,
            )
            wav = self._model.tts(**kwargs)
            arr = np.asarray(wav, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            pcm_i16 = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
            return pcm_i16.tobytes()
        finally:
            self._semaphore.release()

    def _build_tts_kwargs(
        self,
        text: str,
        language: str,
        speed: float,
        speaker_wav_path: str,
    ) -> dict[str, object]:
        speaker_path = Path(speaker_wav_path)
        if not speaker_path.exists() or not speaker_path.is_file():
            raise FileNotFoundError("speaker_wav_path does not exist or is not a file")
        return {
            "text": text,
            "language": language,
            "speed": speed,
            "speaker_wav": speaker_wav_path,
        }
