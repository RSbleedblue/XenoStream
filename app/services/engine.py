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

    def list_speakers(self) -> list[str]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        speakers = getattr(self._model, "speakers", None)
        if not speakers:
            return []
        return [str(s) for s in speakers]

    def synthesize_to_file(
        self,
        text: str,
        language: str,
        speaker: str | None,
        speaker_wav_path: str | None,
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
                speaker=speaker,
                speaker_wav_path=speaker_wav_path,
                validate_speaker_wav=True,
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
        speaker: str | None,
        speaker_wav_path: str | None,
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
                speaker=speaker,
                speaker_wav_path=speaker_wav_path,
                validate_speaker_wav=True,
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
        speaker: str | None,
        speaker_wav_path: str | None,
        speed: float,
        validate_speaker_wav: bool = True,
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
                speaker=speaker,
                speaker_wav_path=speaker_wav_path,
                validate_speaker_wav=validate_speaker_wav,
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
        speaker: str | None,
        speaker_wav_path: str | None,
        validate_speaker_wav: bool,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "text": text,
            "language": language,
            "speed": speed,
        }
        if speaker:
            kwargs["speaker"] = speaker
        if speaker_wav_path:
            if validate_speaker_wav:
                speaker_path = Path(speaker_wav_path)
                if not speaker_path.exists() or not speaker_path.is_file():
                    raise FileNotFoundError("speaker_wav_path does not exist or is not a file")
            kwargs["speaker_wav"] = speaker_wav_path
        return kwargs
