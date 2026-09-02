"""Publish a file from the active terminal backend through the API provider."""

from __future__ import annotations

import logging
import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Optional

from gateway.outbound_files import OutboundFileExporter
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

PUBLISH_FILE_TOOL_NAME = "publish_file"
PUBLISH_FILE_TOOLSET = "outbound_files"


def _configured_exporter() -> Optional[OutboundFileExporter]:
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    platform = config.platforms.get(Platform.API_SERVER)
    raw = platform.extra.get("outbound_files") if platform is not None else None
    return OutboundFileExporter.from_dict(raw)


def outbound_file_publishing_available() -> bool:
    try:
        return _configured_exporter() is not None
    except Exception:
        logger.debug("Outbound file publishing is unavailable", exc_info=True)
        return False


def _backend_filename(path_value: str) -> str:
    return str(path_value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _read_backend_file(path_value: str, task_id: str) -> tuple[bytes, str]:
    """Read a regular file through the active terminal backend."""
    from tools.file_tools import _get_file_ops, _resolve_path_for_task

    resolved = _resolve_path_for_task(path_value, task_id)
    result = _get_file_ops(task_id).read_file_bytes(str(resolved))
    if result.error or result.base64_content is None:
        raise ValueError("file is unavailable in the active terminal environment")
    try:
        content = base64.b64decode(result.base64_content, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("terminal backend returned invalid file data") from exc
    filename = _backend_filename(str(resolved))
    if not filename:
        raise ValueError("file name is unavailable")
    return content, filename


async def publish_file(path: str, task_id: str = "default") -> str:
    """Upload one terminal-backend file and return its public Markdown."""
    try:
        exporter = _configured_exporter()
    except Exception as exc:
        logger.warning("Outbound file configuration is invalid: %s", type(exc).__name__)
        return tool_error("File publishing is not configured correctly.")
    if exporter is None:
        return tool_error("File publishing is not configured.")

    try:
        content, filename = await asyncio.to_thread(_read_backend_file, path, task_id)
    except (OSError, ValueError):
        return tool_error(
            "The file does not exist or is unavailable in the active terminal environment.",
            markdown=exporter.invalid_output(Path(path)),
        )

    with tempfile.TemporaryDirectory(prefix="hermes-publish-") as temp_dir:
        source = Path(temp_dir) / "payload"
        source.write_bytes(content)
        try:
            markdown = await exporter.export_file(source, filename=filename)
        except Exception as exc:
            logger.warning(
                "Outbound file upload through %s failed: %s",
                exporter.config.provider,
                type(exc).__name__,
            )
            return tool_error(
                "File publishing failed.",
                markdown=exporter.invalid_output(Path(filename)),
            )

    return tool_result(success=True, filename=filename, markdown=markdown)


PUBLISH_FILE_SCHEMA = {
    "name": PUBLISH_FILE_TOOL_NAME,
    "description": (
        "Publish a file from the active terminal environment for the user. "
        "Call this after creating any file the user needs to open or download. "
        "The result contains a public Markdown link or image; include its "
        "markdown field verbatim in the final response. Never expose the local path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute path in the active terminal environment, or a path "
                    "relative to its current working directory."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}


registry.register(
    name=PUBLISH_FILE_TOOL_NAME,
    toolset=PUBLISH_FILE_TOOLSET,
    schema=PUBLISH_FILE_SCHEMA,
    handler=lambda args, **kw: publish_file(
        str(args.get("path") or ""), kw.get("task_id") or "default"
    ),
    check_fn=outbound_file_publishing_available,
    is_async=True,
    description="Publish terminal-backend files through the API file provider",
)
