import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass
class VoiceProfile:
    voice_id: str
    file_path: Path
    title: str | None = None
    tag: str | None = None


class VoiceStore:
    def __init__(self, voices_dir: Path) -> None:
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, voice_id: str) -> Path:
        return self.voices_dir / f"{voice_id}.meta.json"

    def _read_meta(self, voice_id: str) -> tuple[str | None, str | None]:
        path = self._meta_path(voice_id)
        if not path.is_file():
            return None, None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None, None
        if not isinstance(raw, dict):
            return None, None
        title = raw.get("title")
        tag = raw.get("tag")
        return (
            title if isinstance(title, str) else None,
            tag if isinstance(tag, str) else None,
        )

    def _write_meta(self, voice_id: str, title: str | None, tag: str | None) -> None:
        if title is None and tag is None:
            return
        data: dict[str, str] = {}
        if title is not None:
            data["title"] = title
        if tag is not None:
            data["tag"] = tag
        self._meta_path(voice_id).write_text(json.dumps(data), encoding="utf-8")

    def _bytes_to_wav(self, content: bytes, source_suffix: str) -> bytes:
        """Return WAV bytes; passthrough for .wav, otherwise decode via ffmpeg (libsndfile-safe)."""
        if source_suffix.lower() == ".wav":
            return content
        if shutil.which("ffmpeg") is None:
            raise ValueError(
                "Non-WAV reference audio requires ffmpeg on PATH. Install ffmpeg or upload WAV."
            )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            input=content,
            capture_output=True,
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"Could not decode reference audio ({source_suffix}): {err[:400]}"
            )
        if not proc.stdout:
            raise ValueError("Could not decode reference audio: ffmpeg produced empty output")
        return proc.stdout

    def _materialize_wav_if_needed(self, p: Path) -> Path:
        if p.suffix.lower() == ".wav":
            return p
        wav_bytes = self._bytes_to_wav(p.read_bytes(), p.suffix)
        out = p.with_suffix(".wav")
        out.write_bytes(wav_bytes)
        p.unlink(missing_ok=True)
        return out

    def _voice_path(self, voice_id: str) -> Path | None:
        matches = [
            p
            for p in sorted(self.voices_dir.glob(f"{voice_id}.*"))
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        return matches[0] if matches else None

    def create(
        self,
        voice_id: str,
        extension: str,
        content: bytes,
        *,
        title: str | None = None,
        tag: str | None = None,
    ) -> VoiceProfile:
        if not voice_id or not voice_id.strip():
            raise ValueError("voice_id is required")
        if " " in voice_id:
            raise ValueError("voice_id must not contain spaces")
        if self._voice_path(voice_id) is not None:
            raise ValueError(f"voice_id '{voice_id}' already exists")

        suffix = extension if extension.startswith(".") else f".{extension}"
        suffix = suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported voice file format")
        if not content:
            raise ValueError("Uploaded voice file is empty")

        wav_bytes = self._bytes_to_wav(content, suffix)
        out_path = self.voices_dir / f"{voice_id}.wav"
        out_path.write_bytes(wav_bytes)
        self._write_meta(voice_id, title, tag)
        t, g = self._read_meta(voice_id)
        return VoiceProfile(voice_id=voice_id, file_path=out_path, title=t, tag=g)

    def list_voices(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        for p in sorted(self.voices_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            vid = p.stem
            t, g = self._read_meta(vid)
            profiles.append(VoiceProfile(voice_id=vid, file_path=p, title=t, tag=g))
        return profiles

    def get_path(self, voice_id: str) -> Path:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        return self._materialize_wav_if_needed(p)

    def delete(self, voice_id: str) -> None:
        p = self._voice_path(voice_id)
        if p is None:
            raise FileNotFoundError("voice_id not found")
        p.unlink(missing_ok=True)
        self._meta_path(voice_id).unlink(missing_ok=True)
