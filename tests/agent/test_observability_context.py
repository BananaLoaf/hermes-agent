import logging
from types import SimpleNamespace

import agent.observability_context as observability_context
from agent.observability_context import agent_observability_context


def test_context_uses_active_profile_and_platform_user(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "manager",
    )
    agent = SimpleNamespace(platform="telegram", _user_id="12345")

    assert agent_observability_context(agent) == {
        "platform_user_id": "12345",
        "profile_name": "manager",
    }


def test_default_profile_is_not_reported_as_named_profile(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "default",
    )

    context = agent_observability_context(
        SimpleNamespace(platform="telegram", _user_id="12345")
    )

    assert context["profile_name"] == ""
    assert context["platform_user_id"] == "12345"


def test_api_server_uses_profile_from_runtime_scope(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "user_example_com",
    )

    context = agent_observability_context(SimpleNamespace(platform="api_server"))

    assert context == {
        "platform_user_id": "",
        "profile_name": "user_example_com",
    }


def test_numeric_zero_platform_user_id_is_converted_to_string(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "default",
    )

    context = agent_observability_context(
        SimpleNamespace(platform="test_platform", _user_id=0)
    )

    assert context["platform_user_id"] == "0"


def test_profile_resolution_failure_warns_once_and_fails_open(
    monkeypatch,
    caplog,
):
    def fail_profile_lookup():
        raise RuntimeError("unexpected profile lookup failure")

    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        fail_profile_lookup,
    )
    monkeypatch.setattr(observability_context, "_profile_warning_emitted", False)

    with caplog.at_level(logging.WARNING, logger=observability_context.__name__):
        first = agent_observability_context(SimpleNamespace(_user_id="123"))
        second = agent_observability_context(SimpleNamespace(_user_id="123"))

    assert first["profile_name"] == ""
    assert second["profile_name"] == ""
    warnings = [
        record
        for record in caplog.records
        if "Failed to resolve the active profile" in record.getMessage()
    ]
    assert len(warnings) == 1
