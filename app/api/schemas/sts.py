from pydantic import BaseModel, Field


class STSResponse(BaseModel):
    request_id: str
    transcribed_text: str
    audio_file: str
    sample_rate: int
