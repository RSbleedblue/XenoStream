from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.tts import router as tts_router
from app.api.routes.voices import router as voices_router
from app.core.settings import get_settings
from app.services.engine import XTTSEngine
from app.services.tts_service import TTSService
from app.services.voice_store import VoiceStore


def create_app() -> FastAPI:
    settings = get_settings()
    engine = XTTSEngine(settings=settings)
    voice_store = VoiceStore(settings.voices_dir)
    tts_service = TTSService(
        engine=engine,
        voice_store=voice_store,
        default_speaker_wav_path=settings.default_speaker_wav_path,
        max_text_chars=settings.max_text_chars,
        stream_chunk_chars=settings.stream_chunk_chars,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.preload_model:
            engine.load()
        yield

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.tts_service = tts_service

    app.include_router(health_router)
    app.include_router(voices_router)
    app.include_router(tts_router)
    return app
