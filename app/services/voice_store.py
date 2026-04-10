from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass
class VoiceProfile:
    voice_id: str
    file_path: Path


class VoiceStore:
    def __init__(self, voices_dir: Path) -> None:
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def _voice_path(self, voice_id: str) -> Path | None:
        matches = sorted(self.voices_dir.glob(f"{voice_id}.*"))
        return matches[0] if matches else None

    def create(self, filename: str, content: bytes) -> VoiceProfile:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported voice file format")
        if not content:
            raise ValueError("Uploaded voice file is empty")

        voice_id = uuid4().hex
        out_path = self.voices_dir / f"{voice_id}{suffix}"
        out_path.write_bytes(content)
        return VoiceProfile(voice_id=voice_id, file_path=out_path)

    def list_voices(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        for p in sorted(self.voices_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            profiles.append(VoiceProfile(voice_id=p.stem, file_path=p))
        return profiles

    def get_path(self, voice_id: str) -> Path:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        return p

    def delete(self, voice_id: str) -> None:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        p.unlink(missing_ok=True)
