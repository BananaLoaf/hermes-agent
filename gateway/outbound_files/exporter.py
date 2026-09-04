"""Outbound file provider construction and path validation."""

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from gateway.outbound_files.base64 import Base64OutboundFileProvider
from gateway.outbound_files.config import OutboundFilesConfig, OutboundFilesConfigError
from gateway.outbound_files.omitted import OmittedOutboundFileProvider
from gateway.outbound_files.provider import OutboundFileProvider
from gateway.platforms.base import validate_media_delivery_path

logger = logging.getLogger(__name__)


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
