from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    language: str = Field(default="en", min_length=2, max_length=10)
    voice_id: str | None = Field(default=None)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class SynthesizeResponse(BaseModel):
    request_id: str
    audio_file: str
    sample_rate: int
