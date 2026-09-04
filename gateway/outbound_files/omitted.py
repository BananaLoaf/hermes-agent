"""Disabled provider for outbound API files."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gateway.outbound_files.base64 import Base64OutboundFileProvider
from gateway.outbound_files.config import OutboundFilesConfigError
from gateway.outbound_files.provider import OutboundFileProvider


class OmittedOutboundFileProvider(OutboundFileProvider):
    """Hide local paths when outbound file delivery is disabled."""

    def __init__(self, options: Mapping[str, Any]):
        if options:
            names = ", ".join(sorted(str(key) for key in options))
            raise OutboundFilesConfigError(
                f"unsupported outbound_files.none options: {names}"
            )

    def requires_valid_path(self, path: Path) -> bool:
        return False

    async def render(self, path: Path) -> str:
        if path.suffix.lower() in Base64OutboundFileProvider.image_mime_types:
            return "[IMAGE OMITTED]"
        return "[FILE OMITTED]"

    def system_prompt_hint(self) -> str:
        return (
            "File/media delivery is disabled. Do not use MEDIA:/absolute/path directives: "
            "images would be replaced with [IMAGE OMITTED] and other files with "
            "[FILE OMITTED]."
        )
