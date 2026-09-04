"""Base interface for outbound file providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


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
