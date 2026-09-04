"""Provider-backed rendering of outbound files for API responses."""

from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

from gateway.platforms.base import validate_media_delivery_path

logger = logging.getLogger(__name__)


class OutboundFilesConfigError(ValueError):
    """Raised when outbound file delivery is configured incorrectly."""


@dataclass(frozen=True)
class OutboundFilesConfig:
    provider: str = "base64"
    provider_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Any) -> "OutboundFilesConfig":
        # Preserve the API server's existing implicit base64 behavior.
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise OutboundFilesConfigError("outbound_files must be a mapping")
        unknown = set(raw) - {"provider", "provider_options"}
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise OutboundFilesConfigError(f"unsupported outbound_files options: {names}")
        provider = raw.get("provider", "base64")
        if provider is None:
            provider = "none"
        if not isinstance(provider, str) or not provider.strip():
            raise OutboundFilesConfigError(
                "outbound_files.provider must be a non-empty string"
            )
        provider_options = raw.get("provider_options", {})
        if provider_options is None:
            provider_options = {}
        if not isinstance(provider_options, Mapping):
            raise OutboundFilesConfigError(
                "outbound_files.provider_options must be a mapping"
            )
        return cls(provider=provider.strip().lower(), provider_options=provider_options)


class OutboundFileProvider(ABC):
    """Turn a validated local path into client-visible response text."""

    def requires_valid_path(self, path: Path) -> bool:
        return True

    @abstractmethod
    async def render(self, path: Path) -> Optional[str]:
        """Return replacement text, or None to preserve the MEDIA directive."""

    @abstractmethod
    def system_prompt_hint(self) -> str:
        """Describe this provider's delivery contract to the agent."""


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


class OmittedOutboundFileProvider(OutboundFileProvider):
    """Hide local paths when outbound file delivery is disabled."""

    def requires_valid_path(self, path: Path) -> bool:
        return False

    def __init__(self, options: Mapping[str, Any]):
        if options:
            names = ", ".join(sorted(str(key) for key in options))
            raise OutboundFilesConfigError(
                f"unsupported outbound_files.none options: {names}"
            )

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


_PROVIDER_FACTORIES: dict[
    str, Callable[[Mapping[str, Any]], OutboundFileProvider]
] = {
    "base64": Base64OutboundFileProvider,
    "none": OmittedOutboundFileProvider,
}


def create_outbound_file_provider(config: OutboundFilesConfig) -> OutboundFileProvider:
    factory = _PROVIDER_FACTORIES.get(config.provider)
    if factory is None:
        raise OutboundFilesConfigError(
            f"unsupported outbound_files.provider: {config.provider}"
        )
    return factory(config.provider_options)


class OutboundFileExporter:
    """Validate MEDIA paths before delegating rendering to a provider."""

    def __init__(self, provider: OutboundFileProvider):
        self.provider = provider

    @classmethod
    def from_dict(cls, raw: Any) -> "OutboundFileExporter":
        config = OutboundFilesConfig.from_dict(raw)
        return cls(create_outbound_file_provider(config))

    def system_prompt_hint(self) -> str:
        return self.provider.system_prompt_hint()

    async def export_media_path(self, path: str) -> Optional[str]:
        render_path = path
        if self.provider.requires_valid_path(Path(path)):
            safe_path = validate_media_delivery_path(path)
            if not safe_path:
                return None
            render_path = safe_path
        try:
            return await self.provider.render(Path(render_path))
        except Exception as exc:
            logger.warning(
                "Outbound file provider %s failed: %s",
                type(self.provider).__name__,
                type(exc).__name__,
            )
            return None
