from types import SimpleNamespace

from agent.observability_context import agent_observability_context


def test_agent_observability_context_uses_profile_and_platform_user(
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "manager",
    )
    monkeypatch.setenv("HERMES_ENVIRONMENT_NAME", "production")
    agent = SimpleNamespace(
        platform="telegram",
        _user_id="12345",
        _chat_id="67890",
        _thread_id="42",
        _gateway_session_key="agent:manager:telegram:dm:12345",
    )

    assert agent_observability_context(agent) == {
        "platform_user_id": "12345",
        "profile_name": "manager",
        "environment_name": "production",
        "platform_chat_id": "67890",
        "platform_thread_id": "42",
        "conversation_id": "agent:manager:telegram:dm:12345",
        "gateway_mode": "telegram",
    }


def test_default_profile_is_not_emitted(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "default",
    )
    agent = SimpleNamespace(platform="api_server", _user_id="user@example.com")

    context = agent_observability_context(agent)

    assert context["profile_name"] == ""
    assert context["gateway_mode"] == "openai_compatible_api"


def test_api_uses_active_runtime_profile(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "user_example_com",
    )
    context = agent_observability_context(
        SimpleNamespace(platform="api_server")
    )

    assert context["profile_name"] == "user_example_com"
    assert context["platform_user_id"] == ""
    assert context["gateway_mode"] == "openai_compatible_api"


def test_cli_is_not_reported_as_native_gateway(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "default",
    )

    context = agent_observability_context(SimpleNamespace(platform="cli"))

    assert context["gateway_mode"] == "cli"
