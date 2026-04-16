import logging
from pathlib import Path
from io import BytesIO
import re
from uuid import uuid4

import numpy as np
import soundfile as sf

from app.api.schemas.tts import SynthesizeRequest
from app.services.engine import XTTSEngine
from app.services.voice_store import VoiceStore

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(
        self,
        engine: XTTSEngine,
        voice_store: VoiceStore,
        default_speaker_wav_path: str | None,
        max_text_chars: int,
        stream_chunk_chars: int,
    ) -> None:
        self.engine = engine
        self.voice_store = voice_store
        self.default_speaker_wav_path = default_speaker_wav_path
        self.max_text_chars = max_text_chars
        self.stream_chunk_chars = stream_chunk_chars

    def ensure_ready(self) -> None:
        if not self.engine.is_ready:
            self.engine.load()

    def validate_text_length(self, text: str) -> None:
        if len(text) > self.max_text_chars:
            raise ValueError(f"Text exceeds max_text_chars={self.max_text_chars}")

    def resolve_speaker_wav_path(self, payload: SynthesizeRequest) -> str:
        if payload.voice_id:
            path = str(self.voice_store.get_path(payload.voice_id))
            logger.info("Using voice_id=%s, path=%s", payload.voice_id, path)
            return path
        if self.default_speaker_wav_path:
            logger.info("Using default speaker wav path=%s", self.default_speaker_wav_path)
            return self.default_speaker_wav_path
        raise ValueError(
            "Provide 'voice_id' or set DEFAULT_SPEAKER_WAV_PATH in .env"
        )

    def list_voices(self) -> list[tuple[str, Path]]:
        return [(v.voice_id, v.file_path) for v in self.voice_store.list_voices()]

    def upload_voice(self, voice_id: str, extension: str, content: bytes) -> tuple[str, Path]:
        profile = self.voice_store.create(voice_id=voice_id, extension=extension, content=content)
        return profile.voice_id, profile.file_path

    def delete_voice(self, voice_id: str) -> None:
        self.voice_store.delete(voice_id)

    def synthesize(self, payload: SynthesizeRequest) -> tuple[str, Path]:
        self.validate_text_length(payload.text)
        speaker_wav_path = self.resolve_speaker_wav_path(payload)
        self.ensure_ready()

        return self.engine.synthesize_to_file(
            text=payload.text,
            language=payload.language,
            speaker_wav_path=speaker_wav_path,
            speed=payload.speed,
        )

    def synthesize_resilient(self, payload: SynthesizeRequest) -> tuple[str, Path]:
        """Synthesize text and auto-handle XTTS max-token assertion by splitting and merging."""
        try:
            return self.synthesize(payload)
        except AssertionError as exc:
            detail = str(exc)
            if "maximum of 400 tokens" not in detail:
                raise
            if len((payload.text or "").strip()) <= 8:
                raise

        speaker_wav_path = self.resolve_speaker_wav_path(payload)
        self.ensure_ready()

        pieces = self._split_for_token_limit(payload.text)
        if len(pieces) <= 1:
            # Re-raise as-is if we cannot split further.
            raise AssertionError(" ❗ XTTS can only generate text with a maximum of 400 tokens.")

        wav_parts: list[np.ndarray] = []
        sample_rate: int | None = None
        for piece in pieces:
            wav_bytes = self.engine.synthesize_to_wav_bytes(
                text=piece,
                language=payload.language,
                speaker_wav_path=speaker_wav_path,
                speed=payload.speed,
            )
            arr, sr = sf.read(BytesIO(wav_bytes), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise RuntimeError("Inconsistent sample rate while merging split synthesis")
            wav_parts.append(arr)

        if not wav_parts or sample_rate is None:
            raise RuntimeError("No audio produced during split synthesis")

        merged = np.concatenate(wav_parts)
        request_id = uuid4().hex
        out_file = self.engine.settings.output_dir / f"{request_id}.wav"
        sf.write(str(out_file), merged, sample_rate)
        return request_id, out_file

    def _split_for_token_limit(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        # First pass: sentence-aware split for natural prosody.
        segments = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
        if not segments:
            segments = [text]

        out: list[str] = []
        for seg in segments:
            out.extend(self._bisect_by_chars(seg, max(120, self.max_text_chars // 2)))
        return out

    def _bisect_by_chars(self, text: str, limit: int) -> list[str]:
        t = (text or "").strip()
        if not t:
            return []
        if len(t) <= limit:
            return [t]

        parts: list[str] = []
        current = t
        while len(current) > limit:
            split_at = current.rfind(" ", 0, limit)
            if split_at <= 0:
                split_at = limit
            parts.append(current[:split_at].strip())
            current = current[split_at:].strip()
        if current:
            parts.append(current)
        return parts

    def chunk_text_for_stream(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        sentence_parts = re.split(r"(?<=[.!?])\s+", text)
        out: list[str] = []
        for part in sentence_parts:
            p = part.strip()
            if not p:
                continue
            while len(p) > self.stream_chunk_chars:
                split_at = p.rfind(" ", 0, self.stream_chunk_chars)
                if split_at <= 0:
                    split_at = self.stream_chunk_chars
                out.append(p[:split_at].strip())
                p = p[split_at:].strip()
            if p:
                out.append(p)
        return out

    def prepare_stream_voice(self, *, voice_id: str | None) -> str:
        payload = SynthesizeRequest(
            text="x",
            voice_id=voice_id,
            language="en",
            speed=1.0,
        )
        return self.resolve_speaker_wav_path(payload)

    def synthesize_chunk_pcm16_prepared(
        self,
        *,
        text: str,
        language: str,
        speed: float,
        speaker_wav_path: str,
    ) -> bytes:
        self.validate_text_length(text)
        self.ensure_ready()
        return self.engine.synthesize_to_pcm16_bytes(
            text=text,
            language=language,
            speed=speed,
            speaker_wav_path=speaker_wav_path,
        )
