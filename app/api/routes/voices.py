from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_tts_service
from app.api.schemas.voices import VoiceProfileResponse, VoiceProfilesResponse
from app.core.security import verify_api_key
from app.services.tts_service import TTSService

router = APIRouter(prefix="/v1", tags=["voices"], dependencies=[Depends(verify_api_key)])


@router.post("/voices", response_model=VoiceProfileResponse)
async def upload_voice(
    file: UploadFile = File(...),
    tts_service: TTSService = Depends(get_tts_service),
) -> VoiceProfileResponse:
    content = await file.read()
    try:
        voice_id, file_path = tts_service.upload_voice(filename=file.filename or "voice.wav", content=content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return VoiceProfileResponse(voice_id=voice_id, file_path=str(file_path))


@router.get("/voices", response_model=VoiceProfilesResponse)
def list_voices(tts_service: TTSService = Depends(get_tts_service)) -> VoiceProfilesResponse:
    voices = [VoiceProfileResponse(voice_id=v[0], file_path=str(v[1])) for v in tts_service.list_voices()]
    return VoiceProfilesResponse(voices=voices)


@router.delete("/voices/{voice_id}")
def delete_voice(voice_id: str, tts_service: TTSService = Depends(get_tts_service)) -> dict[str, str]:
    try:
        tts_service.delete_voice(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "deleted", "voice_id": voice_id}
