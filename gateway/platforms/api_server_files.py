"""OpenAI-compatible Files API routes and ``input_file`` materialization."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast
from urllib.parse import quote

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - mirrors api_server's optional import
    web = None  # type: ignore[assignment]

from gateway.platforms.api_files import (
    APIFileCorruptError,
    APIFileNotFoundError,
    APIFileRecord,
    APIFileStore,
    UPLOAD_PURPOSES,
    sanitize_api_content_type,
    sanitize_api_filename,
)


logger = logging.getLogger("gateway.platforms.api_server")

MAX_API_FILE_BYTES = 20 * 1024 * 1024
MAX_API_FILES_PER_REQUEST = 10
_FILE_PART_TYPES = frozenset({"file", "input_file"})


def contains_file_part(content: Any) -> bool:
    """Return whether one bounded multimodal content array contains a file part."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict)
        and str(part.get("type") or "").strip().lower() in _FILE_PART_TYPES
        for part in content
    )


def validate_file_part_limit(contents: list[Any]) -> None:
    """Reject more than ten files across one model request before caching any bytes."""
    count = sum(
        1
        for content in contents
        if isinstance(content, list)
        for part in content
        if isinstance(part, dict)
        and str(part.get("type") or "").strip().lower() in _FILE_PART_TYPES
    )
    if count > MAX_API_FILES_PER_REQUEST:
        raise ValueError(
            f"too_many_files:At most {MAX_API_FILES_PER_REQUEST} files may be sent in one request."
        )


def _decode_file_data(source: dict[str, Any]) -> tuple[bytes, str, str]:
    file_data = source.get("file_data")
    if not isinstance(file_data, str) or not file_data.strip():
        raise ValueError("invalid_content_part:input_file must include non-empty file_data.")
    filename = source.get("filename", "attachment.bin")
    if not isinstance(filename, str):
        raise ValueError("invalid_content_part:input_file filename must be a string.")

    encoded = file_data.strip()
    content_type = "application/octet-stream"
    if encoded.lower().startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("invalid_file_data:input_file data URLs must contain base64 data.")
        content_type = sanitize_api_content_type(header[5:].split(";", 1)[0])

    if len(encoded) > ((MAX_API_FILE_BYTES + 2) // 3) * 4:
        raise ValueError(
            f"file_too_large:input_file exceeds the {MAX_API_FILE_BYTES // (1024 * 1024)} MiB limit."
        )
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid_file_data:input_file file_data is not valid base64.") from exc
    if not data:
        raise ValueError("invalid_file_data:input_file must not be empty.")
    if len(data) > MAX_API_FILE_BYTES:
        raise ValueError(
            f"file_too_large:input_file exceeds the {MAX_API_FILE_BYTES // (1024 * 1024)} MiB limit."
        )
    return data, sanitize_api_filename(filename), content_type


def _resolve_file_part(part: dict[str, Any]) -> tuple[bytes, str, str, Optional[Path]]:
    nested = part.get("file")
    source = nested if isinstance(nested, dict) else part
    file_id = source.get("file_id")
    file_data = source.get("file_data")
    if file_id is not None and file_data is not None:
        raise ValueError(
            "invalid_content_part:input_file must include exactly one of file_id or file_data."
        )
    if file_id is None:
        data, filename, content_type = _decode_file_data(source)
        return data, filename, content_type, None
    if not isinstance(file_id, str) or not file_id.strip():
        raise ValueError("invalid_content_part:input_file file_id must be a non-empty string.")
    try:
        record, data = APIFileStore().read(file_id.strip(), max_bytes=MAX_API_FILE_BYTES)
    except APIFileNotFoundError as exc:
        raise ValueError(f"file_not_found:No file found with id {file_id!r}.") from exc
    except APIFileCorruptError as exc:
        raise ValueError(f"file_corrupt:Stored file {file_id!r} is unavailable.") from exc
    except ValueError as exc:
        raise ValueError(f"file_too_large:{exc}") from exc
    return data, record.filename, record.content_type, record.content_path


def materialize_file_part(part: dict[str, Any]) -> dict[str, Any]:
    """Turn one OpenAI file part into an agent-native image part or cached-path note."""
    data, filename, content_type, content_path = _resolve_file_part(part)
    ext = os.path.splitext(filename)[1].lower()
    guessed_type = mimetypes.guess_type(filename)[0] or ""
    effective_type = content_type
    if effective_type == "application/octet-stream" and guessed_type:
        effective_type = guessed_type

    from gateway.platforms.base import (
        SUPPORTED_IMAGE_DOCUMENT_TYPES,
        SUPPORTED_VIDEO_TYPES,
        _looks_like_image,
        cache_media_bytes,
        inline_text_document_content,
    )
    from tools.credential_files import to_agent_visible_cache_path

    cached = None
    if content_path is None:
        # Inline ``file_data`` follows the exact same classifier/cache primitives as Telegram.
        cached = cache_media_bytes(data, filename=filename, mime_type=effective_type)
        if cached is None:
            raise ValueError("invalid_file_data:input_file does not contain a supported image.")
        filename = cached.display_name
        effective_type = cached.media_type

    is_image = (
        cached.kind == "image" if cached is not None
        else effective_type.startswith("image/") or ext in SUPPORTED_IMAGE_DOCUMENT_TYPES
    )
    if is_image:
        if content_path is not None and not _looks_like_image(data):
            raise ValueError("invalid_file_data:input_file does not contain a supported image.")
        image_type = effective_type if effective_type.startswith("image/") else SUPPORTED_IMAGE_DOCUMENT_TYPES[ext]
        encoded = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{encoded}"}}

    agent_path = cached.path if cached is not None else to_agent_visible_cache_path(str(content_path))
    is_audio = cached.kind == "audio" if cached is not None else effective_type.startswith("audio/")
    is_video = (
        cached.kind == "video" if cached is not None
        else effective_type.startswith("video/") or ext in SUPPORTED_VIDEO_TYPES
    )
    if is_audio:
        note = (
            f"[The user sent an audio attachment: '{filename}'. It is saved at: {agent_path}. "
            "Inspect or transcribe it before answering when the user's request depends on its contents.]"
        )
    elif is_video:
        note = (
            f"[The user sent a video attachment: '{filename}'. It is saved at: {agent_path}. "
            "Inspect or process it before answering when the user's request depends on its contents.]"
        )
    else:
        inline_content = inline_text_document_content(
            data,
            filename=filename,
            mime_type=effective_type,
        )
        from gateway.run import _build_document_context_note

        note_type = "text/plain" if inline_content is not None else effective_type
        note = _build_document_context_note(
            filename,
            agent_path,
            note_type,
            content_inlined=inline_content is not None,
        )
        if inline_content is not None:
            note = f"{note}\n\n{inline_content}"
    return {"type": "text", "text": note}


class OpenAIFilesRoutesMixin:
    """OpenAI ``/v1/files`` handlers mixed into ``APIServerAdapter``."""

    name: str

    if TYPE_CHECKING:  # supplied by APIServerAdapter at runtime
        def _check_auth(self, request: "web.Request") -> Optional["web.Response"]: ...

    @staticmethod
    def _files_api_error(
        message: str,
        *,
        status: int,
        code: str,
        param: Optional[str] = None,
    ) -> "web.Response":
        from gateway.platforms.api_server import _openai_error

        error_kwargs = {"code": code}
        if param is not None:
            error_kwargs["param"] = param
        return web.json_response(_openai_error(message, **error_kwargs), status=status)

    async def _handle_create_file(self, request: "web.Request") -> "web.Response":
        """POST /v1/files — receive one multipart file into the shared document cache."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not request.content_type.startswith("multipart/"):
            return self._files_api_error(
                "Files must be uploaded as multipart/form-data.",
                status=400,
                code="invalid_request_error",
                param="file",
            )

        filename = "attachment.bin"
        content_type = "application/octet-stream"
        purpose: Optional[str] = None
        payload: Optional[bytearray] = None
        try:
            reader = await request.multipart()
            while True:
                field = cast(Any, await reader.next())
                if field is None:
                    break
                if field.name == "purpose":
                    purpose = (await field.text()).strip()
                    continue
                if field.name != "file":
                    await field.release()
                    continue
                if payload is not None:
                    raise ValueError("invalid_request_error:Upload exactly one file per request.")
                filename = field.filename or filename
                content_type = field.headers.get("Content-Type", content_type)
                payload = bytearray()
                while True:
                    chunk = await field.read_chunk(1024 * 1024)
                    if not chunk:
                        break
                    payload.extend(chunk)
                    if len(payload) > MAX_API_FILE_BYTES:
                        raise ValueError(
                            "file_too_large:File exceeds the "
                            f"{MAX_API_FILE_BYTES // (1024 * 1024)} MiB limit."
                        )

            if payload is None:
                raise ValueError("invalid_request_error:A multipart field named 'file' is required.")
            if not payload:
                raise ValueError("invalid_request_error:The uploaded file must not be empty.")
            if purpose not in UPLOAD_PURPOSES:
                allowed = ", ".join(sorted(UPLOAD_PURPOSES))
                raise ValueError(f"invalid_purpose:purpose must be one of: {allowed}.")
            record = await asyncio.to_thread(
                APIFileStore().create,
                bytes(payload),
                filename=filename,
                purpose=purpose,
                content_type=content_type,
            )
            return web.json_response(record.as_api_dict())
        except ValueError as exc:
            raw = str(exc)
            code, separator, message = raw.partition(":")
            if not separator:
                code, message = "invalid_request_error", raw
            return self._files_api_error(
                message,
                status=413 if code == "file_too_large" else 400,
                code=code,
                param="purpose" if code == "invalid_purpose" else "file",
            )
        except web.HTTPRequestEntityTooLarge:
            raise
        except (AssertionError, web.HTTPBadRequest):
            return self._files_api_error(
                "Invalid multipart upload.", status=400, code="invalid_request_error", param="file"
            )
        except Exception:
            logger.exception("[%s] Failed to receive Files API upload", self.name)
            return self._files_api_error(
                "Failed to store the uploaded file.", status=500, code="file_storage_error", param="file"
            )

    async def _handle_list_files(self, request: "web.Request") -> "web.Response":
        """GET /v1/files — list profile-local files with OpenAI cursor parameters."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            limit = int(request.query.get("limit", "10000"))
        except ValueError:
            return self._files_api_error(
                "limit must be an integer.", status=400, code="invalid_request_error", param="limit"
            )
        if not 1 <= limit <= 10000:
            return self._files_api_error(
                "limit must be between 1 and 10000.",
                status=400,
                code="invalid_request_error",
                param="limit",
            )
        order = request.query.get("order", "desc")
        if order not in {"asc", "desc"}:
            return self._files_api_error(
                "order must be 'asc' or 'desc'.", status=400, code="invalid_request_error", param="order"
            )
        purpose = request.query.get("purpose") or None
        if purpose is not None and purpose not in UPLOAD_PURPOSES:
            return self._files_api_error(
                "Unsupported file purpose.", status=400, code="invalid_purpose", param="purpose"
            )

        records = await asyncio.to_thread(APIFileStore().list_records, purpose=purpose)
        if order == "asc":
            records.reverse()
        after = request.query.get("after")
        if after:
            cursor_index = next((i for i, record in enumerate(records) if record.id == after), None)
            if cursor_index is None:
                return self._files_api_error(
                    "Unknown 'after' cursor.", status=400, code="invalid_request_error", param="after"
                )
            records = records[cursor_index + 1 :]
        page = records[:limit]
        return web.json_response(
            {
                "object": "list",
                "data": [record.as_api_dict() for record in page],
                "first_id": page[0].id if page else None,
                "last_id": page[-1].id if page else None,
                "has_more": len(records) > limit,
            }
        )

    async def _get_api_file_record(
        self, request: "web.Request"
    ) -> tuple[Optional[APIFileRecord], Optional["web.Response"]]:
        file_id = request.match_info.get("file_id", "")
        try:
            return await asyncio.to_thread(APIFileStore().get, file_id), None
        except APIFileNotFoundError:
            return None, self._files_api_error(
                f"No file found with id {file_id!r}.",
                status=404,
                code="file_not_found",
                param="file_id",
            )
        except APIFileCorruptError:
            logger.error("[%s] Stored Files API record %r is corrupt", self.name, file_id)
            return None, self._files_api_error(
                "The stored file is unavailable.",
                status=500,
                code="file_storage_error",
                param="file_id",
            )

    async def _handle_get_file(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        record, error = await self._get_api_file_record(request)
        if error:
            return error
        assert record is not None
        return web.json_response(record.as_api_dict())

    async def _handle_get_file_content(self, request: "web.Request") -> "web.StreamResponse":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        record, error = await self._get_api_file_record(request)
        if error:
            return error
        assert record is not None
        fallback_name = re.sub(r"[^A-Za-z0-9._-]", "_", record.filename) or "attachment.bin"
        disposition = (
            f'attachment; filename="{fallback_name}"; '
            f"filename*=UTF-8''{quote(record.filename, safe='')}"
        )
        return web.FileResponse(
            record.content_path,
            headers={"Content-Type": record.content_type, "Content-Disposition": disposition},
        )

    async def _handle_delete_file(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        file_id = request.match_info.get("file_id", "")
        try:
            await asyncio.to_thread(APIFileStore().delete, file_id)
        except APIFileNotFoundError:
            return self._files_api_error(
                f"No file found with id {file_id!r}.",
                status=404,
                code="file_not_found",
                param="file_id",
            )
        except OSError:
            logger.exception("[%s] Failed to delete Files API record %r", self.name, file_id)
            return self._files_api_error(
                "Failed to delete the stored file.",
                status=500,
                code="file_storage_error",
                param="file_id",
            )
        return web.json_response({"id": file_id, "object": "file", "deleted": True})
