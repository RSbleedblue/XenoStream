from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_tts_service
from app.api.schemas.voices import VoiceProfileResponse, VoiceProfilesResponse
from app.core.security import verify_api_key
from app.services.tts_service import TTSService

router = APIRouter(prefix="/v1", tags=["voices"], dependencies=[Depends(verify_api_key)])


@router.post("/voices", response_model=VoiceProfileResponse)
async def upload_voice(
    voice_id: str = Form(...),
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    tag: str | None = Form(default=None),
    tts_service: TTSService = Depends(get_tts_service),
) -> VoiceProfileResponse:
    content = await file.read()
    extension = Path(file.filename).suffix if file.filename else ".wav"
    try:
        profile = tts_service.upload_voice(
            voice_id=voice_id,
            extension=extension,
            content=content,
            title=title,
            tag=tag,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return VoiceProfileResponse(
        voice_id=profile.voice_id,
        file_path=str(profile.file_path),
        title=profile.title,
        tag=profile.tag,
    )


@router.get("/voices", response_model=VoiceProfilesResponse)
def list_voices(tts_service: TTSService = Depends(get_tts_service)) -> VoiceProfilesResponse:
    voices = [
        VoiceProfileResponse(
            voice_id=v.voice_id,
            file_path=str(v.file_path),
            title=v.title,
            tag=v.tag,
        )
        for v in tts_service.list_voices()
    ]
    return VoiceProfilesResponse(voices=voices)


@router.delete("/voices/{voice_id}")
def delete_voice(voice_id: str, tts_service: TTSService = Depends(get_tts_service)) -> dict[str, str]:
    try:
        tts_service.delete_voice(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "deleted", "voice_id": voice_id}
