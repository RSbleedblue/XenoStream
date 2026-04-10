from pydantic import BaseModel


class VoiceProfileResponse(BaseModel):
    voice_id: str
    file_path: str


class VoiceProfilesResponse(BaseModel):
    voices: list[VoiceProfileResponse]
