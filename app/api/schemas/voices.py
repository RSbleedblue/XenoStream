from pydantic import BaseModel


class VoiceProfileResponse(BaseModel):
    voice_id: str
    file_path: str
    title: str | None = None
    tag: str | None = None


class VoiceProfilesResponse(BaseModel):
    voices: list[VoiceProfileResponse]
