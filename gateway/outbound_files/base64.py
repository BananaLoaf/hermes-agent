"""Base64 image provider for outbound API responses."""

import asyncio
import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Optional

from gateway.outbound_files.config import OutboundFilesConfigError
from gateway.outbound_files.provider import OutboundFileProvider


class Base64OutboundFileProvider(OutboundFileProvider):
    """Inline supported images using the API server's legacy data URL format."""

    max_image_size_bytes: ClassVar[int] = 5 * 1024 * 1024
    image_mime_types: ClassVar[Mapping[str, str]] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(self, options: Mapping[str, Any]):
        self.max_image_size_bytes = self._size_option(
            options, "max_image_size_bytes", self.max_image_size_bytes
        )
        unknown = set(options) - {"max_image_size_bytes"}
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise OutboundFilesConfigError(
                f"unsupported outbound_files.base64 options: {names}"
            )

    @staticmethod
    def _size_option(options: Mapping[str, Any], name: str, default: int) -> int:
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OutboundFilesConfigError(
                f"outbound_files.provider_options.{name} must be a positive integer"
            )
        return value

    def _read_image(self, path: Path) -> Optional[bytes]:
        if path.suffix.lower() not in self.image_mime_types:
            return None
        try:
            with path.open("rb") as file_handle:
                data = file_handle.read(self.max_image_size_bytes + 1)
        except OSError:
            return None
        return data if len(data) <= self.max_image_size_bytes else None

    async def render(self, path: Path) -> Optional[str]:
        if path.suffix.lower() not in self.image_mime_types:
            return "[FILE OMITTED]"
        data = await asyncio.to_thread(self._read_image, path)
        if data is None:
            return None
        encoded = base64.b64encode(data).decode("ascii")
        suffix = path.suffix.lower()
        return f"![image](data:{self.image_mime_types[suffix]};base64,{encoded})"

    def requires_valid_path(self, path: Path) -> bool:
        return path.suffix.lower() in self.image_mime_types

    def system_prompt_hint(self) -> str:
        return (
            "File/media delivery: include MEDIA:/absolute/path in your response. "
            f"Images up to {self.max_image_size_bytes} bytes are inlined as base64 image "
            "data URLs. Non-image file directives are replaced with [FILE OMITTED]."
        )
