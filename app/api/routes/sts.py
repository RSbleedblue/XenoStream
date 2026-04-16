import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.api.dependencies import get_stt_engine, get_tts_service
from app.api.schemas.sts import STSResponse
from app.api.schemas.tts import SynthesizeRequest
from app.core.security import verify_api_key
from app.services.stt_engine import STTEngine, ALLOWED_AUDIO_EXTENSIONS
from app.services.tts_service import TTSService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["sts"], dependencies=[Depends(verify_api_key)])


def _validate_and_read_audio(audio_file: UploadFile) -> tuple[bytes, str]:
    extension = Path(audio_file.filename).suffix.lower() if audio_file.filename else ""
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported audio format. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )
    return extension


async def _transcribe_and_build_payload(
    audio_file: UploadFile,
    language: str,
    voice_id: str,
    speed: float,
    stt_engine: STTEngine,
) -> tuple[str, SynthesizeRequest]:
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded audio file is empty",
        )

    try:
        transcribed_text = await run_in_threadpool(
            stt_engine.transcribe, audio_bytes, language
        )
    except Exception as exc:
        logger.exception("STT transcription failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech-to-text transcription failed",
        ) from exc

    if not transcribed_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No speech detected in the audio",
        )

    payload = SynthesizeRequest(
        text=transcribed_text,
        language=language,
        voice_id=voice_id,
        speed=speed,
    )
    return transcribed_text, payload


def _handle_tts_error(exc: Exception) -> None:
    if isinstance(exc, TimeoutError):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, (ValueError, AssertionError)):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    logger.exception("Unhandled TTS failure in STS pipeline")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal synthesis failure",
    ) from exc


@router.post("/sts", response_model=STSResponse)
async def speech_to_speech(
    audio_file: UploadFile = File(...),
    language: str = Form(default="en"),
    voice_id: str = Form(...),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
    stt_engine: STTEngine = Depends(get_stt_engine),
    tts_service: TTSService = Depends(get_tts_service),
) -> STSResponse:
    _validate_and_read_audio(audio_file)

    transcribed_text, payload = await _transcribe_and_build_payload(
        audio_file, language, voice_id, speed, stt_engine
    )

    try:
        request_id, out_file = tts_service.synthesize_resilient(payload)
        return STSResponse(
            request_id=request_id,
            transcribed_text=transcribed_text,
            output_file=str(out_file),
            sample_rate=24000,
        )
    except Exception as exc:
        _handle_tts_error(exc)


@router.post("/sts/file")
async def speech_to_speech_file(
    audio_file: UploadFile = File(...),
    language: str = Form(default="en"),
    voice_id: str = Form(...),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
    stt_engine: STTEngine = Depends(get_stt_engine),
    tts_service: TTSService = Depends(get_tts_service),
) -> FileResponse:
    _validate_and_read_audio(audio_file)

    _, payload = await _transcribe_and_build_payload(
        audio_file, language, voice_id, speed, stt_engine
    )

    try:
        _, out_file = tts_service.synthesize_resilient(payload)
        return FileResponse(path=out_file, media_type="audio/wav", filename=out_file.name)
    except Exception as exc:
        _handle_tts_error(exc)
