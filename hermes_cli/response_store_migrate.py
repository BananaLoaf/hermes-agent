"""Explicit, additive recovery of legacy shared Responses state into its owning profiles.

Ownership is proved by sessions.id in the canonical state stores, never by a client-supplied
model name. Do not use ResponseStore here: opening/writing it runs LRU eviction, which would
delete existing responses while importing. This command only inserts absent rows.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import uuid
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredResponse:
    response_id: str
    data: str
    accessed_at: float
    session_id: str


@dataclass
class ProfileCopy:
    copied: int = 0
    existing: int = 0
    conflicts: int = 0
    conversations: int = 0
    existing_conversations: int = 0
    total: int = 0
    backup: Path | None = None


@dataclass
class MigrationReport:
    source: Path
    dry_run: bool
    profiles: dict[str, ProfileCopy] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def _read_db(path: Path) -> sqlite3.Connection:
    from hermes_state_holders import read_only_db_uri

    return sqlite3.connect(read_only_db_uri(path), uri=True, timeout=30)


def _source_rows(source: Path, report: MigrationReport) -> tuple[list[StoredResponse], list[tuple[str, str]]]:
    rows = []
    with closing(_read_db(source)) as conn:
        conn.execute("BEGIN")
        stored = conn.execute("SELECT response_id, data, accessed_at FROM responses").fetchall()
        conversations = conn.execute("SELECT name, response_id FROM conversations").fetchall()
    for response_id, data, accessed_at in stored:
        try:
            record = json.loads(data)
            session_id = record.get("session_id") if isinstance(record, dict) else None
            envelope = record.get("response") if isinstance(record, dict) else None
            if (not isinstance(response_id, str) or not response_id.startswith("resp_")
                    or not isinstance(session_id, str) or not session_id
                    or not isinstance(envelope, dict) or envelope.get("id") != response_id
                    or not isinstance(record.get("conversation_history"), list)
                    or not math.isfinite(float(accessed_at))):
                raise ValueError("not a legacy response body with a canonical session id")
            rows.append(StoredResponse(response_id, data, float(accessed_at), session_id))
        except (ValueError, TypeError):
            report.skipped[str(response_id)] = "invalid or unsupported response record"
    return rows, conversations


def _session_owners(homes: dict[str, Path], rows: list[StoredResponse]) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = defaultdict(set)
    session_ids = sorted({row.session_id for row in rows})
    if not session_ids:
        return owners
    for name, home in homes.items():
        path = home / "state.db"
        if not path.is_file():
            continue
        # An unreadable canonical store must abort, not turn a second owner into apparent absence.
        with closing(_read_db(path)) as conn:
            conn.execute("BEGIN")
            for offset in range(0, len(session_ids), 500):
                batch = session_ids[offset:offset + 500]
                marks = ",".join("?" for _ in batch)
                for (session_id,) in conn.execute(f"SELECT id FROM sessions WHERE id IN ({marks})", batch):
                    owners[session_id].add(name)
    return owners


def _route_rows(rows, owners, live, report):
    routed: dict[str, list[StoredResponse]] = defaultdict(list)
    for row in rows:
        candidates = owners.get(row.session_id, set())
        if not candidates:
            report.skipped[row.response_id] = "session not found in any profile state.db"
        elif len(candidates) != 1:
            report.skipped[row.response_id] = "ambiguous session owner: " + ", ".join(sorted(candidates))
        else:
            name = next(iter(candidates))
            if name == "default":
                report.skipped[row.response_id] = "belongs to the default home; retained in source"
            elif name not in live:
                report.skipped[row.response_id] = "owning profile is deleted or inactive"
            else:
                routed[name].append(row)
    return routed


def _validate_targets(source: Path, homes: dict[str, Path], targets: dict[str, list[StoredResponse]]) -> None:
    for name in targets:
        path = homes[name] / "response_store.db"
        if path.is_symlink():
            raise ValueError(f"Refusing symlink destination: {path}")
        if not path.exists():
            continue
        peers = [source, *(home / "state.db" for home in homes.values()),
                 *(home / "response_store.db" for other, home in homes.items() if other != name)]
        if any(peer.is_file() and path.samefile(peer) for peer in peers):
            raise ValueError(f"Refusing shared database destination: {path}")


def _ensure_tables(conn: sqlite3.Connection) -> None:
    # The legacy base schema is also the new runtime's base schema. Its optional identity tables
    # are left alone; a first runtime open adds them to a newly created profile store.
    conn.execute("CREATE TABLE IF NOT EXISTS responses ("
                 "response_id TEXT PRIMARY KEY, data TEXT NOT NULL, accessed_at REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS conversations (name TEXT PRIMARY KEY, response_id TEXT NOT NULL)")


def _same_record(left: str, right: str) -> bool:
    try:
        return json.loads(left) == json.loads(right)
    except (ValueError, TypeError):
        return False


def _copy_rows(conn, rows, conversations, *, apply: bool) -> ProfileCopy:
    result = ProfileCopy(total=conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
    available = set()
    for row in rows:
        existing = conn.execute("SELECT data FROM responses WHERE response_id=?", (row.response_id,)).fetchone()
        if existing is None:
            if apply:
                conn.execute("INSERT OR IGNORE INTO responses (response_id, data, accessed_at) VALUES (?,?,?)",
                             (row.response_id, row.data, row.accessed_at))
            result.copied += 1
            result.total += 1
        elif _same_record(existing[0], row.data):
            result.existing += 1
        else:
            result.conflicts += 1
            continue
        available.add(row.response_id)
    for name, response_id in conversations:
        if response_id not in available:
            continue
        if conn.execute("SELECT 1 FROM conversations WHERE name=?", (name,)).fetchone():
            result.existing_conversations += 1
        else:
            if apply:
                conn.execute("INSERT OR IGNORE INTO conversations (name, response_id) VALUES (?,?)", (name, response_id))
            result.conversations += 1
    return result


def _preview(path, rows, conversations) -> ProfileCopy:
    if path.exists():
        with closing(_read_db(path)) as conn:
            conn.execute("BEGIN")
            return _copy_rows(conn, rows, conversations, apply=False)
    with closing(sqlite3.connect(":memory:")) as conn:
        _ensure_tables(conn)
        return _copy_rows(conn, rows, conversations, apply=False)


def _backup(path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    # SQLite backup includes committed WAL contents; copying only the .db file would not.
    with closing(_read_db(path)) as source, closing(sqlite3.connect(destination)) as target:
        source.backup(target)
    return destination


def _apply_profile(path, rows, conversations, backup_path) -> ProfileCopy:
    backup = None
    # Unlike open_db(), this must not mkdir/recreate a profile that disappeared after discovery.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        backup = _backup(path, backup_path)
    else:
        os.close(fd)
    with closing(sqlite3.connect(path, timeout=30)) as conn:
        from hermes_state_wal import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="response_store.db migration")
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_tables(conn)
            # Re-check collisions under the write lock: a running gateway may have added rows
            # since the preview/backup. No replace, upsert update, pruning or idempotency changes.
            result = _copy_rows(conn, rows, conversations, apply=True)
    result.backup = backup
    return result


def migrate_response_store(*, source: Path | None = None, apply: bool = False) -> MigrationReport:
    """Copy provably owned bodies and absent conversation tips. Source/config/state are read-only.

    Apply is transactional per profile and additive, so an interrupted multi-profile run can be
    repeated. The source is never opened through ResponseStore (even its reads update LRU state).
    """
    from hermes_cli.profiles import _get_default_hermes_home, _iter_named_profile_dirs

    root = _get_default_hermes_home()
    source = Path(source) if source is not None else root / "response_store.db"
    report = MigrationReport(source=source, dry_run=not apply)
    homes = {"default": root, **{path.name: path for path in _iter_named_profile_dirs(live_only=False)}}
    live = {path.name for path in _iter_named_profile_dirs()}
    rows, conversations = _source_rows(source, report)
    routed = _route_rows(rows, _session_owners(homes, rows), live, report)
    _validate_targets(source, homes, routed)
    # Validate every destination before the first write. A bad schema is not an empty store.
    for name, owned in routed.items():
        report.profiles[name] = _preview(homes[name] / "response_store.db", owned, conversations)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    backup_dir = root / "backups" / "response-store-migration" / stamp
    if apply:
        for name, owned in routed.items():
            planned = report.profiles[name]
            if planned.copied or planned.conversations:
                report.profiles[name] = _apply_profile(homes[name] / "response_store.db", owned,
                                                      conversations, backup_dir / f"{name}.bak")
    return report


def cmd_migrate_responses(args) -> int:
    from gateway.platforms.api_server_response_store import MAX_STORED_RESPONSES

    try:
        report = migrate_response_store(source=getattr(args, "source", None), apply=args.apply)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Responses migration failed: {exc}", file=sys.stderr)
        print("Earlier profile copies, if any, remain valid; retrying is safe. Backups are under "
              "<default home>/backups/response-store-migration/.", file=sys.stderr)
        return 1
    print(f"{'Dry-run' if report.dry_run else 'Copied'} shared Responses state from {report.source}")
    for name, result in sorted(report.profiles.items()):
        print(f"  {name}: responses={result.copied}, already present={result.existing}, "
              f"conflicts kept={result.conflicts}, conversation names={result.conversations}, "
              f"existing names kept={result.existing_conversations}")
        if result.backup:
            print(f"    Backup: {result.backup}")
        if result.total > MAX_STORED_RESPONSES:
            print(f"    WARNING: {result.total} responses exceed the runtime LRU limit "
                  f"({MAX_STORED_RESPONSES}); a later runtime open/write can evict older responses.")
    for response_id, reason in sorted(report.skipped.items()):
        print(f"  Skipped {response_id!r}: {reason}")
    print("Source database and profile state/config were not modified.")
    if report.dry_run:
        print("No changes written. Re-run with --apply to copy. Existing destination records always win.")
    return 0
