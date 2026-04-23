from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.voice_store import normalize_string_metadata, normalize_tags

_MAX_NAME = 200
_MAX_DETAILS = 2000


class VoiceProfileResponse(BaseModel):
    """`voice_id` is the TTS key; other fields are for display and organization."""

    voice_id: str
    file_path: str
    display_name: str | None = None
    details: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class VoiceProfilesResponse(BaseModel):
    voices: list[VoiceProfileResponse]


class VoiceUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=_MAX_NAME)
    details: str | None = Field(default=None, max_length=_MAX_DETAILS)
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata_values(cls, v: object) -> object:
        if v is None:
            return v
        if not isinstance(v, dict):
            return v
        return dict(v)

    @field_validator("metadata", mode="after")
    @classmethod
    def _stringify_metadata(cls, v: dict[str, Any] | None) -> dict[str, str] | None:
        if v is None:
            return None
        return normalize_string_metadata(v) or {}


def build_voice_patch(body: VoiceUpdateRequest) -> dict[str, Any]:
    """Apply only fields the client actually sent (Pydantic `model_fields_set`)."""
    present = body.model_fields_set
    out: dict[str, Any] = {}
    if "display_name" in present:
        out["display_name"] = body.display_name
    if "details" in present:
        out["details"] = body.details
    if "tags" in present:
        t = body.tags
        out["tags"] = normalize_tags(list(t) if t is not None else [])
    if "metadata" in present:
        m = body.metadata
        if m is None:
            out["metadata"] = None
        else:
            out["metadata"] = m
    return out
