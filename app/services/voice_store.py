import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
METADATA_FILE_KEYS = frozenset({"display_name", "details", "tags", "metadata"})

# `Content-Type` for GET /v1/voices/{id}/audio and similar.
MEDIA_TYPE_BY_EXT: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def media_type_for_path(path: Path) -> str:
    return MEDIA_TYPE_BY_EXT.get(path.suffix.lower(), "application/octet-stream")

MAX_TAGS = 32
MAX_TAG_LEN = 64
MAX_METADATA_KEYS = 32
MAX_META_KEY_LEN = 64
MAX_META_VALUE_LEN = 2000
MAX_DISPLAY_NAME = 200
MAX_DETAILS = 2000


@dataclass
class VoiceProfile:
    voice_id: str
    file_path: Path
    display_name: str | None = None
    details: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def normalize_tags(value: list[str] | None) -> list[str]:
    if not value:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in value[:MAX_TAGS]:
        t = (raw or "").strip()
        if not t or t in seen:
            continue
        t = t[:MAX_TAG_LEN]
        seen.add(t)
        out.append(t)
    return out


def normalize_string_metadata(value: dict[str, Any] | None) -> dict[str, str]:
    if not value or not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in list(value.items())[:MAX_METADATA_KEYS]:
        if not isinstance(k, str):
            continue
        if not isinstance(v, (str, int, float, bool)) and v is not None:
            continue
        if v is None:
            continue
        k2 = k.strip()[:MAX_META_KEY_LEN]
        if not k2:
            continue
        s = str(v).strip()[:MAX_META_VALUE_LEN]
        if s:
            out[k2] = s
    return out


def _is_storable(data: dict[str, Any]) -> bool:
    if (data.get("display_name") or "").strip():
        return True
    if (data.get("details") or "").strip():
        return True
    t = data.get("tags")
    if isinstance(t, list) and len(t) > 0:
        return True
    m = data.get("metadata")
    if isinstance(m, dict) and len(m) > 0:
        return True
    return False


class VoiceStore:
    def __init__(self, voices_dir: Path) -> None:
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, voice_id: str) -> Path:
        return self.voices_dir / f"{voice_id}.json"

    def _read_raw(self, voice_id: str) -> dict[str, Any]:
        p = self._metadata_path(voice_id)
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _to_profile(
        voice_id: str, file_path: Path, raw: dict[str, Any]
    ) -> VoiceProfile:
        dname: str | None = None
        if "display_name" in raw:
            v = raw.get("display_name")
            if isinstance(v, str) and v.strip():
                dname = v.strip()[:MAX_DISPLAY_NAME]
        dets: str | None = None
        if "details" in raw:
            v = raw.get("details")
            if isinstance(v, str) and v.strip():
                dets = v.strip()[:MAX_DETAILS]
        tgs: list[str] = []
        if "tags" in raw and isinstance(raw.get("tags"), list):
            tgs = normalize_tags(
                [str(x) for x in cast(list[Any], raw.get("tags"))]
            )
        meta2: dict[str, str] = {}
        if "metadata" in raw and isinstance(raw.get("metadata"), dict):
            meta2 = normalize_string_metadata(
                cast(dict[str, Any], raw.get("metadata"))
            )
        return VoiceProfile(
            voice_id=voice_id,
            file_path=file_path,
            display_name=dname,
            details=dets,
            tags=tgs,
            metadata=meta2,
        )

    @staticmethod
    def _simplify_stored_data(data: dict[str, Any]) -> None:
        if "display_name" in data and not (str(data.get("display_name", "")).strip()):
            data.pop("display_name", None)
        if "details" in data and not (str(data.get("details", "")).strip()):
            data.pop("details", None)
        if data.get("tags") == []:
            data.pop("tags", None)
        if data.get("metadata") == {}:
            data.pop("metadata", None)

    def _write_raw(self, voice_id: str, data: dict[str, Any]) -> None:
        self._simplify_stored_data(data)
        path = self._metadata_path(voice_id)
        if not _is_storable(data):
            path.unlink(missing_ok=True)
            return
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _voice_path(self, voice_id: str) -> Path | None:
        matches = [
            p
            for p in self.voices_dir.glob(f"{voice_id}.*")
            if p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        return matches[0] if matches else None

    def create(
        self,
        filename: str,
        content: bytes,
        display_name: str | None = None,
        details: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> VoiceProfile:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported voice file format")
        if not content:
            raise ValueError("Uploaded voice file is empty")

        voice_id = uuid4().hex
        out_path = self.voices_dir / f"{voice_id}{suffix}"
        out_path.write_bytes(content)
        tgs = normalize_tags(tags) if tags else []
        meta = normalize_string_metadata(
            cast(dict[str, Any] | None, metadata) if metadata else None
        )
        dn = (display_name or "").strip()[:MAX_DISPLAY_NAME] or None
        dt = (details or "").strip()[:MAX_DETAILS] or None
        if dn or dt or tgs or meta:
            out: dict[str, Any] = {}
            if dn:
                out["display_name"] = dn
            if dt:
                out["details"] = dt
            if tgs:
                out["tags"] = tgs
            if meta:
                out["metadata"] = meta
            self._write_raw(voice_id, out)
        return self.get_profile(voice_id)

    def list_voices(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        for p in sorted(self.voices_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            profiles.append(
                self._to_profile(
                    p.stem, p, self._read_raw(p.stem)
                )
            )
        return profiles

    def get_path(self, voice_id: str) -> Path:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        return p

    def get_profile(self, voice_id: str) -> VoiceProfile:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        return self._to_profile(voice_id, p, self._read_raw(voice_id))

    def patch_metadata(self, voice_id: str, updates: dict[str, Any]) -> VoiceProfile:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        data: dict[str, Any] = {**self._read_raw(voice_id)}

        for key, v in updates.items():
            if key not in METADATA_FILE_KEYS:
                continue
            if v is None:
                data.pop(key, None)
                continue
            if key == "display_name":
                s = str(v).strip()[:MAX_DISPLAY_NAME]
                if s:
                    data["display_name"] = s
                else:
                    data.pop("display_name", None)
            elif key == "details":
                s = str(v).strip()[:MAX_DETAILS]
                if s:
                    data["details"] = s
                else:
                    data.pop("details", None)
            elif key == "tags":
                if not isinstance(v, list):
                    continue
                tgs = normalize_tags([str(x) for x in v])
                if tgs:
                    data["tags"] = tgs
                else:
                    data.pop("tags", None)
            elif key == "metadata":
                if v is None:
                    data.pop("metadata", None)
                elif isinstance(v, dict):
                    m = normalize_string_metadata(v)
                    if m:
                        data["metadata"] = m
                    else:
                        data.pop("metadata", None)
        self._write_raw(voice_id, data)
        return self._to_profile(voice_id, p, self._read_raw(voice_id))

    def delete(self, voice_id: str) -> None:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        p.unlink(missing_ok=True)
        self._metadata_path(voice_id).unlink(missing_ok=True)
