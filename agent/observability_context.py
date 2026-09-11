"""Request identity shared with observability plugins."""
from __future__ import annotations

import logging
import threading
from typing import Any


logger = logging.getLogger(__name__)
_PROFILE_WARNING_LOCK = threading.Lock()
_profile_warning_emitted = False


def _warn_profile_resolution_failure(exc: Exception) -> None:
    """Log the unexpected profile lookup failure once without blocking turns."""
    global _profile_warning_emitted
    with _PROFILE_WARNING_LOCK:
        if _profile_warning_emitted:
            return
        _profile_warning_emitted = True
    logger.warning(
        "Failed to resolve the active profile for observability; continuing "
        "without Langfuse user attribution: %s",
        exc,
    )


def _active_profile_name() -> str:
    """Return the active named profile, excluding default/custom homes."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        profile = str(get_active_profile_name() or "").strip()
    except Exception as exc:
        _warn_profile_resolution_failure(exc)
        return ""
    return "" if profile in {"", "default", "custom"} else profile


def agent_observability_context(agent: Any) -> dict[str, str]:
    """Return user identity fields for request-scoped observability hooks."""
    platform_user_id = getattr(agent, "_user_id", None)
    return {
        "platform_user_id": (
            "" if platform_user_id is None else str(platform_user_id).strip()
        ),
        "profile_name": _active_profile_name(),
    }
