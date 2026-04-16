from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "xtts-fastapi"
    app_env: str = Field(default="prod", pattern="^(dev|staging|prod)$")
    host: str = "0.0.0.0"
    port: int = 8000

    api_key: str | None = None
    max_text_chars: int = 800
    stream_chunk_chars: int = 160
    max_concurrent_requests: int = 4
    request_timeout_seconds: int = 90

    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "cpu"
    preload_model: bool = True
    default_speaker_wav_path: str | None = None

    stt_model_size: str = "base"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    preload_stt_model: bool = True

    output_dir: Path = Path("outputs")
    voices_dir: Path = Path("voices")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    return settings
