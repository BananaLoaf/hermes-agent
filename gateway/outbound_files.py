"""Provider-backed rendering of outbound files for API responses."""

from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from gateway.platforms.base import validate_media_delivery_path

logger = logging.getLogger(__name__)


class OutboundFilesConfigError(ValueError):
    """Raised when outbound file delivery is configured incorrectly."""


_BASE64_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_DEFAULT_BASE64_MAX_SIZE_BYTES = 5 * 1024 * 1024


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
        provider = raw.get("provider", "base64")
        if provider is None:
            provider = "none"
        if not isinstance(provider, str) or not provider.strip():
            raise OutboundFilesConfigError(
                "outbound_files.provider must be a non-empty string"
            )
        return cls(
            provider=provider.strip().lower(),
            provider_options={key: value for key, value in raw.items() if key != "provider"},
        )


class OutboundFileProvider(ABC):
    """Turn a validated local path into client-visible response text."""

    requires_valid_path = True

    @abstractmethod
    async def render(self, path: Path) -> Optional[str]:
        """Return replacement text, or None to preserve the MEDIA directive."""


class Base64OutboundFileProvider(OutboundFileProvider):
    """Inline supported images using the API server's legacy data URL format."""

    def __init__(self, options: Mapping[str, Any]):
        max_size = options.get("max_size_bytes", _DEFAULT_BASE64_MAX_SIZE_BYTES)
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise OutboundFilesConfigError(
                "outbound_files.max_size_bytes must be a positive integer"
            )
        unknown = set(options) - {"max_size_bytes"}
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise OutboundFilesConfigError(
                f"unsupported outbound_files.base64 options: {names}"
            )
        self.max_size_bytes = max_size

    def _read_image(self, path: Path) -> Optional[bytes]:
        if path.suffix.lower() not in _BASE64_IMAGE_MIME:
            return None
        try:
            with path.open("rb") as file_handle:
                data = file_handle.read(self.max_size_bytes + 1)
        except OSError:
            return None
        return data if len(data) <= self.max_size_bytes else None

    async def render(self, path: Path) -> Optional[str]:
        data = await asyncio.to_thread(self._read_image, path)
        if data is None:
            return None
        mime = _BASE64_IMAGE_MIME[path.suffix.lower()]
        encoded = base64.b64encode(data).decode("ascii")
        return f"![image](data:{mime};base64,{encoded})"


class OmittedOutboundFileProvider(OutboundFileProvider):
    """Hide local paths when outbound file delivery is disabled."""

    requires_valid_path = False

    def __init__(self, options: Mapping[str, Any]):
        if options:
            names = ", ".join(sorted(str(key) for key in options))
            raise OutboundFilesConfigError(
                f"unsupported outbound_files.none options: {names}"
            )

    async def render(self, path: Path) -> str:
        if path.suffix.lower() in _BASE64_IMAGE_MIME:
            return "[IMAGE OMITTED]"
        return "[FILE OMITTED]"


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

    async def export_media_path(self, path: str) -> Optional[str]:
        render_path = path
        if self.provider.requires_valid_path:
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
