"""Durable media and attachment display in API sessions."""
import base64
import hashlib
import os
from pathlib import Path

_MIME_EXT = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp'}
# Same per-admission count as ``prompt.submit`` attachments (``session_ingress_media._ATTACHMENT_LIMIT``).
_IMAGE_LIMIT = 10


def _data_url_bytes(url):
    """``(mime, bytes)`` for a committable ``data:image/...;base64,`` URL, else ``None``."""
    header, _, encoded = url.partition(',')
    mime = header[len('data:'):].split(';', 1)[0].strip().lower()
    if not url.lower().startswith('data:') or mime not in _MIME_EXT or ';base64' not in header.lower():
        return None
    try:
        return mime, base64.b64decode(encoded, validate=True)
    except ValueError:
        return None


def _image_urls(content):
    for part in content:
        if isinstance(part, dict) and part.get('type') == 'image_url':
            yield (part.get('image_url') or {}).get('url', '')


def commit_api_images(content):
    """Stage every inline ``data:`` image deterministically (same bytes -> same path, so an
    exact retry keeps its admission digest) and commit the bytes as immutable media."""
    from gateway.platforms.base import get_image_cache_dir
    from gateway.platforms.base import get_inbound_media_max_bytes
    from gateway.session_ingress_media import capture_native_media, sniff_image_mime
    from hermes_state_runtime import RuntimeStoreError
    # The bytes decide the committed type: SDKs often label JPEG bytes ``data:image/png`` and
    # main forwarded those to the provider; only bytes that are no supported image are refused.
    images = [(sniff_image_mime(decoded[1]), decoded[1])
              for decoded in map(_data_url_bytes, _image_urls(content)) if decoded is not None]
    # Validate the whole batch before one byte lands on disk: a refused request must not
    # leave staged or committed bytes behind.
    limit = max(0, get_inbound_media_max_bytes())
    if (len(images) > _IMAGE_LIMIT or (limit and sum(len(data) for _, data in images) > limit)
            or any(mime is None for mime, _ in images)):
        raise RuntimeStoreError('invalid_params')
    staged = []
    for mime, data in images:
        path = Path(get_image_cache_dir()).resolve() / ('api_' + hashlib.sha256(data).hexdigest()[:32] + _MIME_EXT[mime])
        if not path.exists():
            temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
            temporary.write_bytes(data)
            os.replace(temporary, path)
        staged.append(path)
    return capture_native_media(staged)


def restore_api_images(content, media):
    """The transient content keeps its pixels; its text gains the validated committed reference
    per inline image (remote URLs keep their address) so the persisted user row records what
    was seen, exactly like native image turns."""
    from gateway.session_ingress_media import restore_native_media
    paths = iter(restore_native_media(media))
    hints = []
    for url in _image_urls(content):
        if _data_url_bytes(url) is not None:
            hints.append(f'[Image attached at: {next(paths)}]')
        else:
            hints.append(f'[Image attached: {url}]')
    if not hints:
        return content
    parts = [dict(part) for part in content]
    text = next((part for part in parts if part.get('type') == 'text'), None)
    if text is None:
        text = {'type': 'text', 'text': 'What do you see in this image?'}
        parts.insert(0, text)
    text['text'] = text['text'] + '\n\n' + '\n'.join(hints)
    return parts


# A settled admission's inline ``data:`` image is replaced by this reference to its committed bytes.
_RETAINED = 'hermes-retained:'


def _compacted(content, media):
    """``content`` with every committed inline image reduced to its digest, or ``None`` when
    nothing changes. Only bytes this admission committed are compacted (same order as capture)."""
    if not isinstance(content, list) or not media:
        return None
    digests, parts, changed = iter(reference['sha256'] for reference in media), [], False
    for part in content:
        url = (part.get('image_url') or {}).get('url', '') if isinstance(part, dict) and part.get('type') == 'image_url' else ''
        decoded = _data_url_bytes(url) if url else None
        digest = next(digests, None) if decoded is not None else None
        if digest is not None and hashlib.sha256(decoded[1]).hexdigest() == digest:
            part = {**part, 'image_url': {**part['image_url'], 'url': _RETAINED + digest}}
            changed = True
        parts.append(part)
    return parts if changed else None


def compact_settled_api_payloads(db, admission_id=None):
    """Drop what a TERMINAL API admission row no longer needs: one row at settlement, or
    (``admission_id=None``) the startup sweep for rows a crash left between settlement and this pass.

    - The redundant base64 copy of committed images in ``text``: the bytes stay in content-addressed
      ``native-inputs`` (held by ``api_turn_v1.media`` as history context).
    - The caller-supplied ``api_turn_v1.history`` (every turn carries the whole conversation, inline
      images included, so a session's ledger grew quadratically). Execution read it; a terminal row
      is read only by exact-retry projections. Responses rows keep it: their terminal replay rebuilds
      the ``previous_response_id`` snapshot from it.

    ``payload_digest`` is untouched, so an exact retry still matches. Queued/started/unknown rows
    are never rewritten: they may still execute."""
    import json
    from hermes_state_runtime import _json
    # LIKE, not instr(): SQLite LIKE is ASCII case-insensitive, as ``_data_url_bytes`` is.
    candidates = ("SELECT admission_id FROM session_admissions WHERE status='terminal' AND principal_id='api' "
                  "AND ((json_type(payload_json, '$.api_turn_v1.media') IS NOT NULL "
                  "AND payload_json LIKE '%data:image/%') OR (json_type(payload_json, '$.api_turn_v1.history')='array' "
                  "AND request_id NOT LIKE 'responses:%'))")
    if admission_id is None:
        # The scan runs on a read snapshot; a write transaction opens only when there is work.
        with db._read_ctx() as conn:
            ids = [row[0] for row in conn.execute(candidates)]
    else:
        ids = [admission_id]

    def write(conn):
        compacted = 0
        for row_id in ids:
            row = conn.execute(candidates.replace('SELECT admission_id', 'SELECT payload_json, request_id')
                               + ' AND admission_id=?', (row_id,)).fetchone()
            if row is None:
                continue
            payload = json.loads(row[0])
            data = payload['api_turn_v1']
            text = _compacted(payload.get('text'), data.get('media'))
            drop = isinstance(data.get('history'), list) and not row[1].startswith('responses:')
            if text is not None or drop:
                payload = {**payload, 'text': payload.get('text') if text is None else text,
                           'api_turn_v1': {**data, 'history': None} if drop else data}
                conn.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                             (_json(payload), row_id))
                compacted += 1
        return compacted
    return db._execute_write(write) if ids else 0


def rehydrate_api_images(content, media):
    """The original request content of a compacted settled row, for terminal replay projections;
    an image whose committed bytes are gone becomes a text notice rather than a dangling URL."""
    if not isinstance(content, list):
        return content
    from gateway.session_ingress_media import restore_native_media
    from hermes_state_runtime import RuntimeStoreError
    by_digest = {reference['sha256']: reference for reference in media or ()}
    mimes = {ext: mime for mime, ext in _MIME_EXT.items()}
    parts = []
    for part in content:
        url = (part.get('image_url') or {}).get('url', '') if isinstance(part, dict) and part.get('type') == 'image_url' else ''
        reference = by_digest.get(url[len(_RETAINED):]) if url.startswith(_RETAINED) else None
        if url.startswith(_RETAINED):
            try:
                path = Path(restore_native_media([reference])[0]) if reference else None
            except RuntimeStoreError:
                path = None
            if path is None:
                part = {'type': 'text', 'text': '[Image no longer retained]'}
            else:
                data = base64.b64encode(path.read_bytes()).decode()
                part = {**part, 'image_url': {**part['image_url'],
                        'url': f'data:{mimes.get(path.suffix, "image/png")};base64,{data}'}}
        parts.append(part)
    return parts


_REPLY_DISPLAY_SQL = "SELECT display_metadata FROM messages WHERE id=? AND session_id=? AND role='assistant'"


def reply_row_id(result):
    """Reuse the turn receipt's row address, or its committed final message."""
    from agent.persisted_turn_receipt import committed_row_id

    receipt = result.get('persisted_turn') or {}
    if row_id := receipt.get('final_assistant_row_id'):
        return row_id
    messages = result.get('messages') or []
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get('role') != 'assistant' or last.get('tool_calls'):
        return None
    start = result.get('current_turn_user_idx')
    row_id = committed_row_id(last)
    if row_id in receipt.get('row_ids', ()) or (type(start) is int and 0 <= start < len(messages) - 1):
        return row_id
    return None


def stored_reply_display_text(db, result):
    row_id = reply_row_id(result)
    if row_id is not None:
        row = db._read_one(_REPLY_DISPLAY_SQL, (row_id, result.get('session_id')))
        if row is not None:
            return (db._decode_display_metadata(row[0]) or {}).get('display_text')


def retain_rendered_reply(db, result, rendered):
    """Keep the delivered text on the existing transcript row, separate from model context."""
    from agent.redact import redact_sensitive_text

    row_id = reply_row_id(result)
    if row_id is None:
        return rendered
    rendered = redact_sensitive_text(rendered)

    def write(conn):
        row = conn.execute(_REPLY_DISPLAY_SQL, (row_id, result.get('session_id'))).fetchone()
        if row is None:
            return rendered
        metadata = db._decode_display_metadata(row[0]) or {}
        if isinstance(metadata.get('display_text'), str):
            return metadata['display_text']
        metadata['display_text'] = rendered
        conn.execute("UPDATE messages SET display_metadata=? WHERE id=?",
                     (db._encode_display_metadata(metadata), row_id))
        return rendered

    return db._execute_write(write)


def load_session_messages(db, session_id, *, limit, offset, latest, include_compacted):
    """Keep previously exported links when compaction copied a reply before export finished."""
    import json

    messages = db.get_messages(session_id, limit=limit, offset=offset, latest=latest,
                               include_compacted=include_compacted, include_ancestors=True)
    pending = [msg for msg in messages if msg['role'] == 'assistant' and msg.get('message_uid')
               and isinstance(msg.get('content'), str) and 'MEDIA:' in msg['content']
               and 'display_text' not in (msg.get('display_metadata') or {})]
    if not pending:
        return messages
    rows = db._read_all(
        "SELECT message_uid,content,display_metadata FROM messages WHERE role='assistant' "
        "AND session_id IN (SELECT value FROM json_each(?)) "
        "AND message_uid IN (SELECT value FROM json_each(?)) AND display_metadata IS NOT NULL ORDER BY id",
        (json.dumps(db._resume_lineage_ids(session_id)), json.dumps(list({msg['message_uid'] for msg in pending}))))
    rendered = {
        (row['message_uid'], row['content']): text for row in rows
        if isinstance(text := (db._decode_display_metadata(row['display_metadata']) or {}).get('display_text'), str)
    }
    for msg in pending:
        text = rendered.get((msg['message_uid'], db._encode_content(msg['content'])))
        if text is not None:
            msg['display_metadata'] = {**(msg.get('display_metadata') or {}), 'display_text': text}
    return messages
