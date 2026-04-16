from io import BytesIO
from threading import Lock

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from app.core.settings import Settings

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class STTEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._model: WhisperModel | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                self._model = WhisperModel(
                    self.settings.stt_model_size,
                    device=self.settings.stt_device,
                    compute_type=self.settings.stt_compute_type,
                )

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        if self._model is None:
            raise RuntimeError("STT model not loaded")

        audio, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            duration = len(audio) / sample_rate
            target_len = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        kwargs: dict[str, object] = {
            "beam_size": 1,
            "best_of": 1,
            "vad_filter": True,
        }
        if language:
            kwargs["language"] = language

        segments, _ = self._model.transcribe(audio, **kwargs)
        return " ".join(segment.text.strip() for segment in segments)
