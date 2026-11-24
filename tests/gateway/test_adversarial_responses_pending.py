"""A Responses retry stays pinned to its first admission's target, including after pointer loss."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
@pytest.mark.parametrize('compaction', ['none', 'in_place', 'rotation'])
async def test_new_responses_request_rolls_back_lost_turn(api, owner, compaction):
    from gateway.session_api_turn import api_execution
    from gateway.session_results import execution_result
    from hermes_state_runtime import (
        admit_session_input, begin_runtime_epoch, claim_session_input, get_session_admission, recover_session_inputs,
    )

    api._responses_client_managed_session_id = True
    api._api_key = 'test-key'
    calls = []

    async def handle(event):
        calls.append((event.text, api_execution.get()['history']))
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': []}
        return 'answer'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        headers = {'X-Hermes-Session-Id': 'continued-chat', 'Authorization': 'Bearer test-key'}
        first = await client.post('/v1/responses', json={'input': 'first'}, headers=headers)
        assert first.status == 200
        sid = first.headers['X-Hermes-Session-Id']
        good = [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'answer'}]
        owner.db.append_messages_batch(sid, good)
        lost = admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id=sid,
                                   request_id='lost', payload={'text': 'lost input'})
        claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
        owner.db.append_message(sid, 'user', 'lost input')
        owner.db.append_message(sid, 'assistant', 'partial answer')
        queued = admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id=sid,
            request_id='queued', payload={'text': 'already queued', 'api_turn_v1': {'history': good, 'settings': {}}})
        handoff = [{'role': 'assistant', 'content': 'summary containing lost input'}]
        if compaction == 'in_place':
            owner.db.archive_and_compact(sid, handoff)
        elif compaction == 'rotation':
            owner.db.publish_compression_child(parent_session_id=sid, child_session_id='rotated',
                source='api_server', messages=handoff, require_compression_lease=False)
        tip = owner.db.get_compression_tip(sid)
        before = owner.db.get_messages(tip)

        owner.epoch = begin_runtime_epoch(owner.db, instance_id='replacement')
        assert recover_session_inputs(owner.db, epoch=owner.epoch) == 1
        assert get_session_admission(owner.db, admission_id=lost['admission_id'])['status'] == 'unknown'
        assert get_session_admission(owner.db, admission_id=queued['admission_id'])['status'] == 'queued'
        assert owner.db.get_messages(tip) == before

        continuation = await client.post('/v1/responses', json={'input': 'next'}, headers=headers)
        assert continuation.status == 200, await continuation.text()
        assert [text for text, _ in calls] == ['first', 'already queued', 'next']
        assert [msg['content'] for msg in calls[-1][1]] == ['first', 'answer']
        assert get_session_admission(owner.db, admission_id=lost['admission_id'])['outcome'] == 'interrupted'
        assert [msg['content'] for msg in owner.db.get_messages(tip)] == ['first', 'answer']


@pytest.mark.asyncio
async def test_pending_retry_pins_named_conversation_target(api, owner):
    entered, duplicate, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        if event.text == 'slow':
            entered.set()
            if calls.count('slow') > 1:
                duplicate.set()
            await release.wait()
        execution_result.get()['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    first_body = {'input': 'slow', 'conversation': 'shared'}
    first_headers = {'Idempotency-Key': 'pending-original'}
    requests = []
    async with TestClient(TestServer(app)) as client:
        try:
            requests.append(asyncio.create_task(client.post('/v1/responses', json=first_body, headers=first_headers)))
            await asyncio.wait_for(entered.wait(), 10)
            advanced = await client.post('/v1/responses', json={'input': 'fast', 'conversation': 'shared'},
                                         headers={'Idempotency-Key': 'concurrent-other'})
            assert advanced.status == 200
            requests.append(asyncio.create_task(client.post('/v1/responses', json=first_body, headers=first_headers)))
            try:
                await asyncio.wait_for(duplicate.wait(), 2)
            except TimeoutError:
                pass
            release.set()
            responses = await asyncio.gather(*requests)
            assert all(response.status == 200 for response in responses)
            assert calls.count('slow') == 1, 'retry executed the same request on the newly mapped session'
            assert await responses[0].json() == await responses[1].json()
            assert responses[0].headers['X-Hermes-Session-Id'] == responses[1].headers['X-Hermes-Session-Id']
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)


@pytest.mark.asyncio
async def test_unknown_retry_recovers_admission_after_pointer_commit_loss(api, owner):
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
    from gateway.platforms.api_server_response_store import ResponseStore
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handle(event):
        calls.append(event.text)
        started.set()
        await release.wait()
        return 'done'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body, headers = {'input': 'uncertain', 'conversation': 'shared'}, {'Idempotency-Key': 'lost-owner'}
    async with TestClient(TestServer(app)) as client:
        first = asyncio.create_task(client.post('/v1/responses', json=body, headers=headers))
        try:
            await asyncio.wait_for(started.wait(), 10)
            row = owner.db._read_one('SELECT * FROM session_admissions')
            task = owner.sessions[row['target_session_id']].task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            owner.epoch = begin_runtime_epoch(owner.db, instance_id='restarted')
            recover_session_inputs(owner.db, epoch=owner.epoch)
            owner._pause(SessionRef(owner.profile_id, row['target_session_id']), 'unknown_execution')
            assert (await first).status == 409
            # Crash after the ledger commit but before its response-store pointer was saved.
            store = api._current_response_store()
            store._conn.execute('DELETE FROM response_admissions')
            store._conn.commit()
            path = store._db_path
            store.close()
            api._response_store = ResponseStore(db_path=path)
            api._response_store.put('different', {'session_id': 'different-target', 'conversation_history': []})
            api._response_store.set_conversation('shared', 'different')
            retry = await client.post('/v1/responses', json=body, headers=headers)
            assert retry.status == 409
            assert (await retry.json())['error']['code'] == 'unknown_execution'
            assert calls == ['uncertain']
            assert owner.db._read_one('SELECT COUNT(*) FROM session_admissions')[0] == 1
            assert api._response_store._conn.execute('SELECT admission_id FROM response_admissions').fetchone()[0] == row['admission_id']
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_validation_refusal_does_not_reserve_retry_key(api, owner):
    calls = []
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': []}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Idempotency-Key': 'corrected-request'}
    async with TestClient(TestServer(app)) as client:
        refused = await client.post('/v1/responses', json={'input': 42}, headers=headers)
        assert refused.status == 400
        assert api._current_response_store()._conn.execute('SELECT COUNT(*) FROM response_keys').fetchone()[0] == 0
        accepted = await client.post('/v1/responses', json={'input': 'valid'}, headers=headers)
        assert accepted.status == 200
    assert calls == ['valid']


@pytest.mark.asyncio
async def test_concurrent_exact_responses_share_terminal_envelope(api, owner):
    started, release = asyncio.Event(), asyncio.Event()
    async def handle(event):
        from gateway.session_results import execution_result
        started.set()
        await release.wait()
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"a"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'read'},
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body, headers = {'input': 'work', 'stream': True}, {'Idempotency-Key': 'concurrent-exact'}
    pending = []
    async with TestClient(TestServer(app)) as client:
        try:
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            await asyncio.wait_for(started.wait(), 10)
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            # Observe both HTTP requests before the turn can settle.
            deadline = asyncio.get_running_loop().time() + 5
            while sum(len(observers) for observers in getattr(owner, 'api_observers', {}).values()) < 2:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(.01)
            release.set()
            responses = await asyncio.gather(*pending)
            frames = [[json.loads(line[6:]) for line in (await response.text()).splitlines()
                       if line.startswith('data: ')] for response in responses]
            values = [events[-1]['response'] for events in frames]
            assert frames[0][0]['response'] == frames[1][0]['response']
            assert values[0] == values[1]
            retry = await client.post('/v1/responses', json={**body, 'stream': False}, headers=headers)
            assert await retry.json() == values[0]
            assert owner.db._read_one('SELECT COUNT(*) FROM session_admissions')[0] == 1
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_pointer_commit_loss_then_retirement_cannot_reexecute_moved_conversation(api, owner, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result

    calls = []

    async def handle(event):
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': []}
        return 'answer'

    owner.runner._handle_message = handle
    store = api._current_response_store()
    bind = store.bind_request_admission
    def lose_pointer(*args):
        raise RuntimeError('simulated process loss after admission commit')
    monkeypatch.setattr(store, 'bind_request_admission', lose_pointer)
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body = {'input': 'original-once', 'conversation': 'moving'}
    headers = {'Idempotency-Key': 'lost-pointer'}
    async with TestClient(TestServer(app)) as client:
        failed = await client.post('/v1/responses', json=body, headers=headers)
        assert failed.status == 500
        monkeypatch.setattr(store, 'bind_request_admission', bind)
        admitted = owner.db._read_one('SELECT * FROM session_admissions')
        assert admitted['status'] == 'queued'
        # Owner recovery executes the committed request even though no HTTP response was saved.
        ref = SessionRef(owner.profile_id, admitted['target_session_id'])
        owner._schedule(ref)
        await asyncio.wait_for(owner.sessions[ref.session_id].task, 10)
        assert calls == ['original-once']
        assert owner.db.delete_session(ref.session_id)
        # A later request advances the name to an independent live target.
        advanced = await client.post('/v1/responses', json={'input': 'new-target', 'conversation': 'moving'},
                                     headers={'Idempotency-Key': 'successor'})
        assert advanced.status == 200
        retry = await client.post('/v1/responses', json=body, headers=headers)
        await retry.read()
    assert calls.count('original-once') == 1, 'retired tombstone was missed and the retry executed again'
