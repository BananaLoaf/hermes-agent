"""Profile-local storage for the OpenAI-compatible Files API."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write


FILE_ID_RE = re.compile(r"^file-[0-9a-f]{32}$")
FILES_API_CACHE_SUBDIR = Path("cache") / "api_files"


class APIFileNotFoundError(FileNotFoundError):
    """Raised when a file ID does not exist in the active profile."""


class APIFileCorruptError(RuntimeError):
    """Raised when a stored Files API record is incomplete or inconsistent."""


@dataclass(frozen=True)
class APIFileRecord:
    """Metadata and content path for one profile-local uploaded file."""

    id: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    content_type: str
    content_path: Path

    def as_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "file",
            "bytes": self.bytes,
            "created_at": self.created_at,
            "filename": self.filename,
            "purpose": self.purpose,
            "status": "processed",
        }


@dataclass(frozen=True)
class StagedAPIFile:
    """Private temporary destination used while an HTTP upload is streaming."""

    id: str
    directory: Path
    content_path: Path


def sanitize_api_filename(filename: Any) -> str:
    """Return a safe display filename without accepting path components."""
    raw = filename if isinstance(filename, str) else "attachment.bin"
    value = Path((raw or "attachment.bin").replace("\\", "/")).name
    value = "".join(char for char in value if ord(char) >= 32 and char != "\x7f")
    value = value.replace("\x00", "").strip()
    if value in {"", ".", ".."}:
        return "attachment.bin"
    return value[:255]


def sanitize_api_content_type(content_type: Any) -> str:
    """Return a header-safe MIME type or the opaque binary fallback."""
    raw = content_type if isinstance(content_type, str) else ""
    value = raw.split(";", 1)[0].strip().lower()
    if not value or len(value) > 255 or "\r" in value or "\n" in value:
        return "application/octet-stream"
    return value


def _api_storage_name(filename: str) -> str:
    """Return a bounded internal filename while preserving a useful suffix."""
    suffix = Path(filename).suffix.lstrip(".")
    safe_suffix = re.sub(r"[^A-Za-z0-9]", "", suffix)[:16].lower()
    return f"content.{safe_suffix}" if safe_suffix else "content"


class APIFileStore:
    """Filesystem-backed Files API store rooted in the active Hermes profile."""

    @staticmethod
    def root() -> Path:
        root = get_hermes_home() / FILES_API_CACHE_SUBDIR
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return root

    @staticmethod
    def _validate_file_id(file_id: str) -> str:
        if not isinstance(file_id, str) or not FILE_ID_RE.fullmatch(file_id):
            raise APIFileNotFoundError(file_id)
        return file_id

    def stage(self) -> StagedAPIFile:
        root = self.root()
        file_id = f"file-{uuid.uuid4().hex}"
        directory = Path(tempfile.mkdtemp(prefix=".upload-", dir=root))
        os.chmod(directory, 0o700)
        content_path = directory / "content"
        fd = os.open(content_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        return StagedAPIFile(file_id, directory, content_path)

    @staticmethod
    def discard(staged: StagedAPIFile | None) -> None:
        if staged is not None:
            shutil.rmtree(staged.directory, ignore_errors=True)

    def commit(
        self,
        staged: StagedAPIFile,
        *,
        filename: str,
        purpose: str,
        content_type: str,
        size: int,
    ) -> APIFileRecord:
        if purpose != "user_data":
            raise ValueError("Only purpose='user_data' is supported.")
        if size < 0:
            raise ValueError("File size must not be negative.")
        if staged.content_path.stat().st_size != size:
            raise ValueError("Uploaded file size does not match the streamed content.")

        display_name = sanitize_api_filename(filename)
        storage_name = _api_storage_name(display_name)
        payload_directory = staged.directory / "payload"
        payload_directory.mkdir(mode=0o700)
        content_path = payload_directory / storage_name
        os.replace(staged.content_path, content_path)

        metadata = {
            "id": staged.id,
            "object": "file",
            "bytes": size,
            "created_at": int(time.time()),
            "filename": display_name,
            "storage_name": storage_name,
            "purpose": purpose,
            "content_type": sanitize_api_content_type(content_type),
        }
        atomic_json_write(staged.directory / "metadata.json", metadata, mode=0o600)
        final_directory = self.root() / staged.id
        os.replace(staged.directory, final_directory)
        return APIFileRecord(
            id=staged.id,
            bytes=size,
            created_at=metadata["created_at"],
            filename=metadata["filename"],
            purpose=purpose,
            content_type=metadata["content_type"],
            content_path=final_directory / "payload" / storage_name,
        )

    def _record_directory(self, file_id: str) -> Path:
        return self.root() / self._validate_file_id(file_id)

    def get(self, file_id: str) -> APIFileRecord:
        directory = self._record_directory(file_id)
        metadata_path = directory / "metadata.json"
        if directory.is_symlink():
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            filename = metadata.get("filename")
            storage_name = metadata.get("storage_name")
            if (
                not isinstance(filename, str)
                or filename != sanitize_api_filename(filename)
                or not isinstance(storage_name, str)
                or storage_name != _api_storage_name(filename)
            ):
                raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.")
            payload_directory = directory / "payload"
            content_path = payload_directory / storage_name
            actual_size = content_path.stat().st_size
        except FileNotFoundError as exc:
            raise APIFileNotFoundError(file_id) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.") from exc

        stored_size = metadata.get("bytes")
        if (
            payload_directory.is_symlink()
            or not payload_directory.is_dir()
            or metadata_path.is_symlink()
            or not metadata_path.is_file()
            or not content_path.is_file()
            or content_path.is_symlink()
            or metadata.get("id") != file_id
            or metadata.get("object") != "file"
            or not isinstance(stored_size, int)
            or isinstance(stored_size, bool)
            or stored_size != actual_size
            or metadata.get("purpose") != "user_data"
        ):
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.")

        try:
            return APIFileRecord(
                id=file_id,
                bytes=actual_size,
                created_at=int(metadata["created_at"]),
                filename=sanitize_api_filename(metadata.get("filename")),
                purpose=str(metadata["purpose"]),
                content_type=sanitize_api_content_type(metadata.get("content_type")),
                content_path=content_path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.") from exc

    def read(self, file_id: str, *, max_bytes: int) -> tuple[APIFileRecord, bytes]:
        record = self.get(file_id)
        if record.bytes > max_bytes:
            raise ValueError(f"Stored file exceeds the {max_bytes // (1024 * 1024)} MiB limit.")
        return record, record.content_path.read_bytes()

    def list(self) -> list[APIFileRecord]:
        records: list[APIFileRecord] = []
        for path in self.root().iterdir():
            if not path.is_dir() or not FILE_ID_RE.fullmatch(path.name):
                continue
            try:
                records.append(self.get(path.name))
            except (APIFileNotFoundError, APIFileCorruptError):
                continue
        return sorted(records, key=lambda record: (record.created_at, record.id), reverse=True)

    def delete(self, file_id: str) -> None:
        directory = self._record_directory(file_id)
        if directory.is_symlink() or not directory.is_dir():
            raise APIFileNotFoundError(file_id)
        shutil.rmtree(directory)
