"""Manual shared Responses-store migration preserves live profile data and isolation."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest


def _record(response_id, session_id, text="old", model="alice"):
    return {
        "response": {"id": response_id, "object": "response", "status": "completed", "model": model},
        "session_id": session_id, "client_session_id": session_id, "instructions": "private instructions",
        "conversation_history": [{"role": "user", "content": text}],
    }


def _store(path, records, conversations=None):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE responses (response_id TEXT PRIMARY KEY, data TEXT NOT NULL, accessed_at REAL NOT NULL)")
        conn.execute("CREATE TABLE conversations (name TEXT PRIMARY KEY, response_id TEXT NOT NULL)")
        conn.executemany("INSERT INTO responses VALUES (?,?,?)", [
            (rid, json.dumps(record), index + 1.0) for index, (rid, record) in enumerate(records.items())])
        conn.executemany("INSERT INTO conversations VALUES (?,?)", (conversations or {}).items())


def _read(path, table="responses"):
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def _profile(root, name, session_ids):
    home = root / "profiles" / name
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins: {enabled: [langfuse]}\n")
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO sessions VALUES (?)", [(sid,) for sid in session_ids])
    return home


@pytest.fixture
def root(tmp_path, monkeypatch):
    home = tmp_path / "custom ? hermes # root"
    home.mkdir()
    (home / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_dry_run_and_apply_preserve_new_rows_source_and_canonical_state(root):
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["alice-chat"])
    bob = _profile(root, "bob", ["bob-chat"])
    old = {"resp_alice": _record("resp_alice", "alice-chat"),
           "resp_bob": _record("resp_bob", "bob-chat", model="bob")}
    source = root / "response_store.db"
    _store(source, old, {"main": "resp_alice", "bob-main": "resp_bob"})
    _store(alice / "response_store.db", {"resp_new": _record("resp_new", "alice-chat", "new")},
           {"main": "resp_new"})
    before_source = source.read_bytes()
    before_state = (alice / "state.db").read_bytes()
    before_config = (alice / "config.yaml").read_bytes()
    before_files = set(root.rglob("*"))
    preview = migrate_response_store()
    assert preview.dry_run
    assert preview.profiles["alice"].copied == 1
    assert set(root.rglob("*")) == before_files
    assert len(_read(alice / "response_store.db")) == 1

    result = migrate_response_store(apply=True)
    assert result.profiles["alice"].copied == result.profiles["bob"].copied == 1
    assert {r[0] for r in _read(alice / "response_store.db")} == {"resp_alice", "resp_new"}
    assert {r[0] for r in _read(bob / "response_store.db")} == {"resp_bob"}
    assert _read(alice / "response_store.db", "conversations") == [("main", "resp_new")]
    assert _read(bob / "response_store.db", "conversations") == [("bob-main", "resp_bob")]
    assert _read(result.profiles["alice"].backup)[0][0] == "resp_new"
    assert source.read_bytes() == before_source
    assert (alice / "state.db").read_bytes() == before_state
    assert (alice / "config.yaml").read_bytes() == before_config
    backups = set(root.rglob("*.bak"))
    again = migrate_response_store(apply=True)
    assert all(row.copied == 0 for row in again.profiles.values())
    assert set(root.rglob("*.bak")) == backups


def test_conflicting_response_ids_and_conversation_tips_are_never_replaced(root):
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["chat"])
    _store(root / "response_store.db", {"resp_same": _record("resp_same", "chat", "old")},
           {"old-name": "resp_same"})
    _store(alice / "response_store.db", {"resp_same": _record("resp_same", "chat", "new")})
    before = _read(alice / "response_store.db")
    result = migrate_response_store(apply=True)
    assert result.profiles["alice"].conflicts == 1
    assert result.profiles["alice"].copied == 0
    assert _read(alice / "response_store.db") == before
    assert _read(alice / "response_store.db", "conversations") == []


def test_ambiguous_unknown_and_malformed_records_are_not_routed_by_model(root):
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["shared-chat"])
    bob = _profile(root, "bob", ["shared-chat", "bob-only"])
    records = {
        "resp_ambiguous": _record("resp_ambiguous", "shared-chat"),
        "resp_unknown": _record("resp_unknown", "unknown", model="alice"),
        "resp_bob": _record("resp_bob", "bob-only", model="alice"),
        "resp_bad": {"session_id": "shared-chat"},
    }
    _store(root / "response_store.db", records)
    report = migrate_response_store(apply=True)
    assert {"resp_ambiguous", "resp_unknown", "resp_bad"} <= report.skipped.keys()
    assert not (alice / "response_store.db").exists()
    assert [r[0] for r in _read(bob / "response_store.db")] == ["resp_bob"]


def test_copy_does_not_run_lru_eviction_or_touch_new_identity_tables(root):
    from gateway.platforms.api_server_response_store import MAX_STORED_RESPONSES
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["chat"])
    _store(root / "response_store.db", {"resp_old": _record("resp_old", "chat")})
    _store(alice / "response_store.db", {
        f"resp_new_{i}": _record(f"resp_new_{i}", "chat", "new") for i in range(MAX_STORED_RESPONSES)})
    with sqlite3.connect(alice / "response_store.db") as conn:
        conn.execute("CREATE TABLE response_keys (request_key TEXT PRIMARY KEY, fingerprint TEXT)")
        conn.execute("INSERT INTO response_keys VALUES ('new-key','new-fingerprint')")
    report = migrate_response_store(apply=True)
    assert report.profiles["alice"].total == MAX_STORED_RESPONSES + 1
    assert len(_read(alice / "response_store.db")) == MAX_STORED_RESPONSES + 1
    assert _read(alice / "response_store.db", "response_keys") == [("new-key", "new-fingerprint")]


def test_tombstoned_profile_still_counts_for_ownership_but_is_not_written(root):
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["shared-chat"])
    _profile(root, "bob", ["shared-chat", "bob-only"])
    from hermes_constants import mark_named_profile_deleted
    mark_named_profile_deleted(root / "profiles" / "bob")
    _store(root / "response_store.db", {
        "resp_shared": _record("resp_shared", "shared-chat"),
        "resp_deleted": _record("resp_deleted", "bob-only"),
    })
    report = migrate_response_store(apply=True)
    assert len(report.skipped) == 2
    assert not (alice / "response_store.db").exists()
    assert not (root / "profiles" / "bob" / "response_store.db").exists()


def test_unreadable_state_schema_aborts_before_any_destination_write(root):
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["chat"])
    broken = root / "profiles" / "broken"
    broken.mkdir()
    (broken / "state.db").write_bytes(b"not sqlite")
    _store(root / "response_store.db", {"resp_old": _record("resp_old", "chat")})
    with pytest.raises(sqlite3.DatabaseError):
        migrate_response_store(apply=True)
    assert not (alice / "response_store.db").exists()


@pytest.mark.platforms("posix")
def test_symlink_destination_is_refused_without_writing_shared_source(root):
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["chat"])
    source = root / "response_store.db"
    _store(source, {"resp_old": _record("resp_old", "chat")})
    (alice / "response_store.db").symlink_to(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="symlink|shared"):
        migrate_response_store(apply=True)
    assert source.read_bytes() == before


def test_cli_parser_defaults_to_preview_and_supports_explicit_source(root, capsys):
    from hermes_cli.subcommands.migrate import build_migrate_parser

    _profile(root, "alice", ["chat"])
    source = root / "saved.sqlite"
    _store(source, {"resp_old": _record("resp_old", "chat")})
    parser = argparse.ArgumentParser()
    build_migrate_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["migrate", "responses", "--source", str(source)])
    assert not args.apply
    assert args.func(args) == 0
    assert "Dry-run" in capsys.readouterr().out
    args = parser.parse_args(["migrate", "responses", "--source", str(source), "--apply"])
    assert args.func(args) == 0
    assert "Copied" in capsys.readouterr().out


def test_backup_reads_committed_wal_contents(root):
    from hermes_cli.response_store_migrate import migrate_response_store

    alice = _profile(root, "alice", ["chat"])
    _store(root / "response_store.db", {"resp_old": _record("resp_old", "chat")})
    destination = alice / "response_store.db"
    _store(destination, {"resp_new": _record("resp_new", "chat", "before WAL")})
    with sqlite3.connect(destination) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        changed = json.dumps(_record("resp_new", "chat", "committed in WAL"))
        writer.execute("UPDATE responses SET data=? WHERE response_id='resp_new'", (changed,))
        writer.commit()
        assert Path(str(destination) + "-wal").stat().st_size > 0
        result = migrate_response_store(apply=True)
        assert _read(result.profiles["alice"].backup)[0][1] == changed
        assert next(row[1] for row in _read(destination) if row[0] == "resp_new") == changed


def test_a_writer_collision_after_preview_is_rechecked_and_preserved(root, monkeypatch):
    from hermes_cli import response_store_migrate as migration

    alice = _profile(root, "alice", ["chat"])
    _store(root / "response_store.db", {"resp_same": _record("resp_same", "chat", "old")})
    _store(alice / "response_store.db", {})
    original = migration._apply_profile

    def writer_then_import(path, rows, conversations, backup):
        with sqlite3.connect(path) as conn:
            conn.execute("INSERT INTO responses VALUES (?,?,?)",
                         ("resp_same", json.dumps(_record("resp_same", "chat", "live writer")), 999.0))
        return original(path, rows, conversations, backup)

    monkeypatch.setattr(migration, "_apply_profile", writer_then_import)
    result = migration.migrate_response_store(apply=True)
    assert result.profiles["alice"].copied == 0
    assert result.profiles["alice"].conflicts == 1
    assert json.loads(_read(alice / "response_store.db")[0][1])["conversation_history"][0]["content"] == "live writer"


def test_backup_failure_prevents_destination_writes(root, monkeypatch):
    from hermes_cli import response_store_migrate as migration

    alice = _profile(root, "alice", ["chat"])
    _store(root / "response_store.db", {"resp_old": _record("resp_old", "chat")})
    _store(alice / "response_store.db", {"resp_new": _record("resp_new", "chat", "new")})
    before = _read(alice / "response_store.db")

    def refused(*args):
        raise OSError("backup filesystem is full")

    monkeypatch.setattr(migration, "_backup", refused)
    with pytest.raises(OSError, match="backup filesystem is full"):
        migration.migrate_response_store(apply=True)
    assert _read(alice / "response_store.db") == before


@pytest.mark.asyncio
async def test_copied_legacy_anchor_continues_through_real_profile_route(root, monkeypatch):
    from types import SimpleNamespace
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from agent import secret_scope
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_cli.response_store_migrate import migrate_response_store
    from hermes_state import SessionDB

    home = root / "profiles" / "alice"
    home.mkdir(parents=True)
    key = "a" * 40
    (home / ".env").write_text(f"API_SERVER_KEY={key}\n")
    db = SessionDB(db_path=home / "state.db")
    db.create_session("chat", "api_server")
    db.close()
    record = _record("resp_old", "chat", "Remember the synthetic earlier message")
    _store(root / "response_store.db", {"resp_old": record})
    migrate_response_store(apply=True)

    secret_scope.set_multiplex_active(False)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": key, "responses_client_managed_session_id": True,
    }))
    adapter.gateway_runner = SimpleNamespace(config=GatewayConfig(multiplex_profiles=True))
    secret_scope.set_multiplex_active(True)
    runs = []

    async def run_agent(**kwargs):
        runs.append(kwargs)
        return ({"final_response": "restored", "messages": [], "api_calls": 1,
                 "session_id": kwargs["session_id"]}, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    adapter._run_agent = run_agent
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    app["api_server_adapter"] = adapter
    headers = {"Authorization": f"Bearer {key}", "X-Hermes-Session-Id": "chat"}
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/p/alice/v1/responses", headers=headers,
                                         json={"input": "Continue", "previous_response_id": "resp_old"})
            assert response.status == 200, await response.text()
            await response.read()
            assert runs[0]["session_id"] == "chat"
            assert runs[0]["conversation_history"] == record["conversation_history"]
    finally:
        await adapter.disconnect()
        secret_scope.set_multiplex_active(False)
