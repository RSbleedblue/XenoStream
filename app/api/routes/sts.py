import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import get_stt_engine, get_tts_service
from app.api.schemas.sts import STSResponse
from app.api.schemas.tts import SynthesizeRequest
from app.core.security import verify_api_key
from app.services.stt_engine import STTEngine, ALLOWED_AUDIO_EXTENSIONS
from app.services.tts_service import TTSService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["sts"], dependencies=[Depends(verify_api_key)])


@router.post("/sts", response_model=STSResponse)
async def speech_to_speech(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    voice_id: str = Form(...),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
    stt_engine: STTEngine = Depends(get_stt_engine),
    tts_service: TTSService = Depends(get_tts_service),
) -> STSResponse:
    extension = Path(file.filename).suffix.lower() if file.filename else ""
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported audio format. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    audio_bytes = await file.read()
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

    try:
        request_id, out_file = tts_service.synthesize_resilient(payload)
        return STSResponse(
            request_id=request_id,
            transcribed_text=transcribed_text,
            audio_file=str(out_file),
            sample_rate=24000,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except AssertionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled TTS failure in STS pipeline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal synthesis failure",
        ) from exc
