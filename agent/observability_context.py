"""Stable request context shared by observability plugins."""
from __future__ import annotations

import os
from typing import Any


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        profile = str(get_active_profile_name() or "").strip()
    except Exception:
        return ""
    return "" if profile in {"", "default", "custom"} else profile


def agent_observability_context(agent: Any) -> dict[str, str]:
    """Return bounded-cardinality identity and request metadata for hooks."""
    platform = str(getattr(agent, "platform", "") or "").strip()
    profile_name = _active_profile_name()
    platform_key = platform.casefold()
    return {
        "platform_user_id": str(getattr(agent, "_user_id", "") or "").strip(),
        "profile_name": profile_name,
        "environment_name": os.getenv("HERMES_ENVIRONMENT_NAME", "").strip(),
        "platform_chat_id": str(getattr(agent, "_chat_id", "") or "").strip(),
        "platform_thread_id": str(getattr(agent, "_thread_id", "") or "").strip(),
        "conversation_id": str(
            getattr(agent, "_gateway_session_key", "") or ""
        ).strip(),
        "gateway_mode": (
            "openai_compatible_api"
            if platform_key == "api_server"
            else platform_key or "unknown"
        ),
    }
