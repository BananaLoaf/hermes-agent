"""Roll back the transcript of API turns lost with their gateway owner."""
import json


def api_turn_checkpoint(db, conn, session_id, mark):
    tip = db._compression_tip_on(conn, session_id)
    active_ids = [row[0] for row in conn.execute(
        "SELECT id FROM messages WHERE session_id=? AND active=1 ORDER BY id", (tip,))]
    return json.dumps({'tip': tip, 'mark': mark, 'active_ids': active_ids}, separators=(',', ':'))


def _restore_transcript(db, conn, checkpoint):
    from hermes_state_compression import _CHAIN_STEP_SQL, _walk_compression_chain

    tip = checkpoint['tip']
    ids = json.dumps(checkpoint['active_ids'])
    # In-place compaction archives the originals. If another writer removed them entirely,
    # keep the unknown turn fenced instead of replacing good history with an incomplete prefix.
    found = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND id IN (SELECT value FROM json_each(?))",
        (tip, ids)).fetchone()[0]
    if found != len(checkpoint['active_ids']):
        return False

    chain = _walk_compression_chain(tip, lambda sid: conn.execute(_CHAIN_STEP_SQL, (sid,)).fetchone())
    # A summary written during the lost turn can contain its input or partial output. Drop
    # that compaction generation along with the turn and restore the previous active rows.
    conn.execute("UPDATE messages SET active=0, compacted=0 WHERE session_id=? AND id>? "
                 "AND (active=1 OR compacted=1)", (tip, checkpoint['mark']))
    conn.execute("UPDATE messages SET active=1, compacted=0 WHERE session_id=? "
                 "AND id IN (SELECT value FROM json_each(?)) AND (active!=1 OR compacted!=0)",
                 (tip, ids))
    if len(chain) > 1:
        children_json = json.dumps(chain[1:])
        conn.execute("UPDATE messages SET active=0, compacted=0 "
                     "WHERE session_id IN (SELECT value FROM json_each(?)) AND (active=1 OR compacted=1)",
                     (children_json,))
        conn.execute("UPDATE sessions SET message_count=0, tool_call_count=0 "
                     "WHERE id IN (SELECT value FROM json_each(?))", (children_json,))
        # Keep the published continuation addressable by existing API bindings. Its model
        # context becomes the clean pre-turn prefix rather than the discarded handoff.
        if checkpoint['active_ids']:
            db._clone_message_rows(conn, checkpoint['active_ids'], session_id=chain[-1])
    for session_id in dict.fromkeys((tip, chain[-1])):
        counts = db._active_transcript_counts(conn, session_id)
        conn.execute("UPDATE sessions SET message_count=?, tool_call_count=? WHERE id=?", (*counts, session_id))
    return True


def recover_interrupted_api_turn(db, *, epoch, session_id, guard):
    """Roll back a lost turn only when a client continues this API conversation."""
    from hermes_state_runtime import _epoch, _retire_admission_workers
    from hermes_state_terminal import RESULT_PREFIX

    query = (
        "SELECT a.* FROM session_admissions a JOIN sessions s ON s.id=a.target_session_id "
        "WHERE a.target_session_id=? AND a.status='unknown' AND a.owner_epoch!=? "
        "AND a.recovery_checkpoint IS NOT NULL AND a.generation=s.runtime_generation")
    with db._read_ctx() as conn:
        if conn.execute(query, (session_id, epoch)).fetchone() is None:
            return False

    def write(conn):
        _epoch(conn, epoch)
        guard(conn)
        row = conn.execute(query, (session_id, epoch)).fetchone()
        if row is None or not _restore_transcript(db, conn, json.loads(row['recovery_checkpoint'])):
            return False
        admission_id = row['admission_id']
        _retire_admission_workers(conn, row, row['owner_epoch'])
        # Exact retries can observe the interruption, but only a new request can run again.
        result = {'result': {'final_response': '', 'messages': [], 'interrupted': True, 'completed': False,
                             '_rolled_back': True},
                  'usage': {}}
        conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)",
                     (RESULT_PREFIX + admission_id, json.dumps(result)))
        conn.execute("UPDATE session_admissions SET status='terminal', outcome='interrupted', "
                     "recovery_checkpoint=NULL WHERE admission_id=?", (admission_id,))
        conn.execute("UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?", (row['target_session_id'],))
        return True

    return db._execute_write(write)
