import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.api.dependencies import get_tts_service
from app.api.schemas.voices import (
    VoiceProfileResponse,
    VoiceProfilesResponse,
    VoiceUpdateRequest,
    build_voice_patch,
)
from app.core.security import verify_api_key, verify_api_key_or_query
from app.services.tts_service import TTSService
from app.services.voice_store import (
    MAX_DETAILS,
    MAX_DISPLAY_NAME,
    VoiceProfile,
    media_type_for_path,
    normalize_string_metadata,
    normalize_tags,
)

router = APIRouter(prefix="/v1", tags=["voices"])


def _parse_tags_param(raw: str | None) -> list[str] | None:
    if not raw or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    return normalize_tags([p for p in parts if p]) or None


def _parse_metadata_param(raw: str | None) -> dict[str, str] | None:
    if not raw or not str(raw).strip():
        return None
    try:
        obj: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"metadata is not valid JSON: {exc}",
        ) from exc
    if not isinstance(obj, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="metadata JSON must be an object",
        )
    out = normalize_string_metadata(obj) or None
    return out


def _voice_to_response(p: VoiceProfile) -> VoiceProfileResponse:
    return VoiceProfileResponse(
        voice_id=p.voice_id,
        file_path=str(p.file_path),
        display_name=p.display_name,
        details=p.details,
        tags=p.tags,
        metadata=p.metadata,
    )


@router.post(
    "/voices", response_model=VoiceProfileResponse, dependencies=[Depends(verify_api_key)]
)
async def upload_voice(
    file: UploadFile = File(...),
    display_name: str | None = Form(
        default=None, max_length=MAX_DISPLAY_NAME, description="Label for this voice in your app"
    ),
    details: str | None = Form(
        default=None, max_length=MAX_DETAILS, description="Longer description or notes"
    ),
    tags: str | None = Form(
        default=None, description="Comma-separated tags, e.g. 'personal,english'"
    ),
    metadata: str | None = Form(
        default=None,
        description="JSON object with string (or string-coerced) values for app-specific metadata",
    ),
    tts_service: TTSService = Depends(get_tts_service),
) -> VoiceProfileResponse:
    content = await file.read()
    dname = (display_name.strip() if display_name else None) or None
    det = (details.strip() if details else None) or None
    tag_list = _parse_tags_param(tags)
    meta_dict = _parse_metadata_param(metadata)
    try:
        profile = tts_service.upload_voice(
            filename=file.filename or "voice.wav",
            content=content,
            display_name=dname,
            details=det,
            tags=tag_list,
            metadata=meta_dict,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _voice_to_response(profile)


@router.get(
    "/voices", response_model=VoiceProfilesResponse, dependencies=[Depends(verify_api_key)]
)
def list_voices(tts_service: TTSService = Depends(get_tts_service)) -> VoiceProfilesResponse:
    voices = [_voice_to_response(v) for v in tts_service.list_voices()]
    return VoiceProfilesResponse(voices=voices)


def _download_filename(voice_id: str, file_path: Path, display_name: str | None) -> str:
    base = (display_name or voice_id)[:100]
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", str(base).strip()) or voice_id
    return f"{base}{file_path.suffix}"


# Starlette's FileResponse already streams the file in chunks and honors Range (206) + Accept-Ranges: bytes
# (required for seek/progressive play over the network). Nginx: disable response buffering to the client.
_HOSTED_STREAM_HEADERS = {
    "X-Accel-Buffering": "no",  # nginx: stream to the client; don't buffer the whole file first
    "Cache-Control": "private, no-transform",  # avoid middleboxes re-encoding the audio stream
}


@router.get(
    "/voices/{voice_id}/audio",
    response_class=FileResponse,
    dependencies=[Depends(verify_api_key_or_query)],
    summary="Stream voice sample (hosted playback / seeking)",
    response_description=(
        "Binary audio stream, chunked. Suitable for `AVPlayer`, `Media3`, `fetch`+`blob`, or `<audio>` "
        "with `?api_key=…`. The ASGI response supports HTTP Range and `Accept-Ranges: bytes` (partial / seek)."
    ),
    responses={401: {"description": "Invalid or missing API key"}},
)
def get_voice_audio(
    voice_id: str, tts_service: TTSService = Depends(get_tts_service)
) -> FileResponse:
    """
    Plays the **reference recording** (not synthesized TTS). Auth: `x-api-key` or `?api_key=` (see dependency).

    When you host the API, clients do **not** need the file path on disk: this URL is the public stream.
    """
    try:
        prof = tts_service.get_voice(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    p = prof.file_path
    return FileResponse(
        path=p,
        media_type=media_type_for_path(p),
        filename=_download_filename(voice_id, p, prof.display_name),
        content_disposition_type="inline",
        headers=_HOSTED_STREAM_HEADERS,
    )


@router.get(
    "/voices/{voice_id}",
    response_model=VoiceProfileResponse,
    dependencies=[Depends(verify_api_key)],
)
def get_voice(
    voice_id: str, tts_service: TTSService = Depends(get_tts_service)
) -> VoiceProfileResponse:
    try:
        profile = tts_service.get_voice(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _voice_to_response(profile)


@router.patch(
    "/voices/{voice_id}",
    response_model=VoiceProfileResponse,
    dependencies=[Depends(verify_api_key)],
)
def patch_voice(
    voice_id: str,
    body: VoiceUpdateRequest = Body(...),
    tts_service: TTSService = Depends(get_tts_service),
) -> VoiceProfileResponse:
    data = build_voice_patch(body)
    for key in ("display_name", "details"):
        if key not in data:
            continue
        v = data[key]
        if isinstance(v, str):
            data[key] = v.strip()
    if "display_name" in data and data.get("display_name") == "":
        data["display_name"] = None
    if "details" in data and data.get("details") == "":
        data["details"] = None
    if not data:
        try:
            return _voice_to_response(tts_service.get_voice(voice_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    try:
        profile = tts_service.update_voice(voice_id, data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _voice_to_response(profile)


@router.delete(
    "/voices/{voice_id}", dependencies=[Depends(verify_api_key)]
)
def delete_voice(voice_id: str, tts_service: TTSService = Depends(get_tts_service)) -> dict[str, str]:
    try:
        tts_service.delete_voice(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "deleted", "voice_id": voice_id}
