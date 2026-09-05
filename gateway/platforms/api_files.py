"""Profile-local metadata for files uploaded through the OpenAI-compatible API."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.platforms.base import cache_document_from_bytes, get_document_cache_dir
from hermes_constants import get_hermes_home
from utils import atomic_json_write


FILE_ID_RE = re.compile(r"^file-[0-9a-f]{32}$")
_CACHED_DOCUMENT_RE = re.compile(r"^doc_[0-9a-f]{12}_.+$")
_METADATA_SUBDIR = Path("cache") / "api_files"

UPLOAD_PURPOSES = frozenset({"assistants", "batch", "evals", "fine-tune", "user_data", "vision"})


class APIFileNotFoundError(FileNotFoundError):
    """Raised when a file ID does not exist in the active profile."""


class APIFileCorruptError(RuntimeError):
    """Raised when persisted Files API metadata does not match its cached document."""


@dataclass(frozen=True)
class APIFileRecord:
    """Metadata plus the canonical Telegram-compatible document-cache path."""

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
        }


def sanitize_api_filename(filename: Any) -> str:
    """Return a bounded display filename without accepting path components."""
    raw = filename if isinstance(filename, str) else "attachment.bin"
    value = Path((raw or "attachment.bin").replace("\\", "/")).name
    value = "".join(char for char in value if ord(char) >= 32 and ord(char) != 127)
    value = value.replace("\x00", "").strip()
    return "attachment.bin" if value in {"", ".", ".."} else value[:255]


def sanitize_api_content_type(content_type: Any) -> str:
    """Return a response-header-safe MIME type or the opaque binary fallback."""
    raw = content_type if isinstance(content_type, str) else ""
    value = raw.split(";", 1)[0].strip().lower()
    if not value or len(value) > 255 or "\r" in value or "\n" in value:
        return "application/octet-stream"
    return value


class APIFileStore:
    """Files API metadata backed by the active profile's shared document cache.

    The bytes intentionally go through :func:`cache_document_from_bytes`, the same
    primitive used for Telegram document attachments. Only the small OpenAI lookup
    record lives under ``cache/api_files``.
    """

    @staticmethod
    def metadata_root() -> Path:
        root = get_hermes_home() / _METADATA_SUBDIR
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return root

    @staticmethod
    def _validate_file_id(file_id: str) -> str:
        if not isinstance(file_id, str) or not FILE_ID_RE.fullmatch(file_id):
            raise APIFileNotFoundError(file_id)
        return file_id

    def _metadata_path(self, file_id: str) -> Path:
        return self.metadata_root() / f"{self._validate_file_id(file_id)}.json"

    def create(
        self,
        data: bytes,
        *,
        filename: str,
        purpose: str,
        content_type: str,
    ) -> APIFileRecord:
        if purpose not in UPLOAD_PURPOSES:
            raise ValueError(f"Unsupported file purpose: {purpose!r}.")
        if not data:
            raise ValueError("The uploaded file must not be empty.")

        display_name = sanitize_api_filename(filename)
        cached_path = Path(cache_document_from_bytes(data, display_name))
        file_id = f"file-{uuid.uuid4().hex}"
        metadata = {
            "id": file_id,
            "object": "file",
            "bytes": len(data),
            "created_at": int(time.time()),
            "filename": display_name,
            "purpose": purpose,
            "content_type": sanitize_api_content_type(content_type),
            # A basename, never an absolute path: profile switching must re-root the lookup.
            "content_name": cached_path.name,
        }
        try:
            atomic_json_write(self._metadata_path(file_id), metadata, mode=0o600)
        except BaseException:
            cached_path.unlink(missing_ok=True)
            raise
        return self._record_from_metadata(metadata)

    def _record_from_metadata(self, metadata: dict[str, Any]) -> APIFileRecord:
        try:
            file_id = self._validate_file_id(metadata["id"])
            filename = metadata["filename"]
            purpose = metadata["purpose"]
            content_name = metadata["content_name"]
            stored_size = metadata["bytes"]
            created_at = metadata["created_at"]
        except (KeyError, TypeError, APIFileNotFoundError) as exc:
            raise APIFileCorruptError("Stored Files API metadata is corrupt.") from exc

        if (
            metadata.get("object") != "file"
            or not isinstance(filename, str)
            or filename != sanitize_api_filename(filename)
            or purpose not in UPLOAD_PURPOSES
            or not isinstance(content_name, str)
            or Path(content_name).name != content_name
            or not _CACHED_DOCUMENT_RE.fullmatch(content_name)
            or not isinstance(stored_size, int)
            or isinstance(stored_size, bool)
            or stored_size < 0
            or not isinstance(created_at, int)
            or isinstance(created_at, bool)
        ):
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.")

        content_path = get_document_cache_dir() / content_name
        if content_path.is_symlink() or not content_path.is_file():
            raise APIFileNotFoundError(file_id)
        try:
            actual_size = content_path.stat().st_size
        except OSError as exc:
            raise APIFileCorruptError(f"Stored file {file_id!r} is unavailable.") from exc
        if actual_size != stored_size:
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.")

        return APIFileRecord(
            id=file_id,
            bytes=actual_size,
            created_at=created_at,
            filename=filename,
            purpose=purpose,
            content_type=sanitize_api_content_type(metadata.get("content_type")),
            content_path=content_path,
        )

    def get(self, file_id: str) -> APIFileRecord:
        metadata_path = self._metadata_path(file_id)
        if metadata_path.is_symlink():
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise APIFileNotFoundError(file_id) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.") from exc
        if not isinstance(metadata, dict) or metadata.get("id") != file_id:
            raise APIFileCorruptError(f"Stored file {file_id!r} is corrupt.")
        return self._record_from_metadata(metadata)

    def read(self, file_id: str, *, max_bytes: int) -> tuple[APIFileRecord, bytes]:
        record = self.get(file_id)
        if max_bytes and record.bytes > max_bytes:
            raise ValueError(f"Stored file exceeds the {max_bytes // (1024 * 1024)} MiB limit.")
        try:
            return record, record.content_path.read_bytes()
        except OSError as exc:
            raise APIFileCorruptError(f"Stored file {file_id!r} is unavailable.") from exc

    def list_records(self, *, purpose: str | None = None) -> list[APIFileRecord]:
        records: list[APIFileRecord] = []
        for path in self.metadata_root().glob("file-*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                record = self.get(path.stem)
            except (APIFileNotFoundError, APIFileCorruptError):
                continue
            if purpose is None or record.purpose == purpose:
                records.append(record)
        return sorted(records, key=lambda record: (record.created_at, record.id), reverse=True)

    def delete(self, file_id: str) -> None:
        record = self.get(file_id)
        metadata_path = self._metadata_path(file_id)
        try:
            record.content_path.unlink()
            metadata_path.unlink()
        except FileNotFoundError as exc:
            raise APIFileNotFoundError(file_id) from exc
