import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.session_results import execution_result
from gateway.session_api_turn import api_execution
from gateway.platforms.api_files import APIFileStore


@pytest.mark.asyncio
async def test_run_accepts_native_image_and_uploaded_document_on_the_same_turn(api, owner):
    session_id = 'cabinet-attachments'
    api._api_key = 'test-key'
    headers = {'Authorization': 'Bearer test-key'}
    record = APIFileStore().create(b'The number is 42.', filename='source.txt', purpose='user_data',
                                   content_type='text/plain')
    image = ('data:image/png;base64,'
             'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aG1sAAAAASUVORK5CYII=')
    observed = []

    async def handle(event):
        prepared = api_execution.get()
        observed.append((prepared['content'], prepared['input_display']))
        owner.db.append_message(session_id, 'user', event.text)
        if len(observed) == 1:
            execution_result.get()['result'] = {'failed': True, 'error': 'temporary failure'}
            return ''
        execution_result.get()['result'] = {'final_response': 'answer'}
        return 'answer'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_get('/v1/runs/{run_id}/events', api._handle_run_events)
    body = {'session_id': session_id, 'input': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Read both.'},
        {'type': 'image_url', 'image_url': {'url': image}},
        {'type': 'input_file', 'file_id': record.id},
    ]}]}
    async with TestClient(TestServer(app)) as client:
        accepted = await client.post('/v1/runs', headers=headers, json=body)
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        first = await client.get(f'/v1/runs/{run_id}/events', headers=headers)
        assert 'run.failed' in await first.text()
        retried = await client.post('/v1/runs', headers=headers, json={'session_id': session_id, 'retry_of': run_id})
        assert retried.status == 202, await retried.text()
        retry_id = (await retried.json())['run_id']
        second = await client.get(f'/v1/runs/{retry_id}/events', headers=headers)
        assert 'run.completed' in await second.text()
    assert len(observed) == 2
    for content, metadata in observed:
        assert content[1]['image_url']['url'] == image
        assert 'The number is 42.' in content[2]['text']
        assert metadata == {'text': 'Read both.', 'inline_images': 1,
                            'files': [{'id': record.id, 'filename': 'source.txt', 'bytes': record.bytes}]}


@pytest.mark.asyncio
async def test_run_retry_replays_the_failed_input_without_duplication(api, owner):
    session_id = 'cabinet-test'
    api._api_key = 'test-key'
    headers = {'Authorization': 'Bearer test-key'}
    observed = []

    async def handle(event):
        observed.append([message['content'] for message in owner.db.get_messages(session_id)])
        owner.db.append_message(session_id, 'user', event.text)
        if len(observed) == 1:
            owner.db.append_message(session_id, 'assistant', 'partial output')
            execution_result.get()['result'] = {'failed': True, 'error': 'temporary failure'}
            return ''
        owner.db.append_message(session_id, 'assistant', 'answer')
        execution_result.get()['result'] = {'final_response': 'answer'}
        return 'answer'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_get('/v1/runs/{run_id}/events', api._handle_run_events)
    app.router.add_get('/api/sessions/{session_id}', api._handle_get_session)
    async with TestClient(TestServer(app)) as client:
        first = await client.post('/v1/runs', json={'input': 'question', 'session_id': session_id}, headers=headers)
        assert first.status == 202, await first.text()
        run_id = (await first.json())['run_id']
        failure = await client.get(f'/v1/runs/{run_id}/events', headers=headers)
        assert 'run.failed' in await failure.text()

        detail = await client.get(f'/api/sessions/{session_id}?include_latest_run=true', headers=headers)
        assert (await detail.json())['latest_run'] == {'run_id': run_id, 'status': 'failed', 'error': 'temporary failure'}

        body = {'session_id': session_id, 'retry_of': run_id}
        retry_headers = {**headers, 'Idempotency-Key': 'same-retry'}
        retried = await client.post('/v1/runs', json=body, headers=retry_headers)
        assert retried.status == 202, await retried.text()
        new_id = (await retried.json())['run_id']
        answer = await client.get(f'/v1/runs/{new_id}/events', headers=headers)
        assert 'run.completed' in await answer.text()
        repeated = await client.post('/v1/runs', json=body, headers=retry_headers)
        assert (await repeated.json())['run_id'] == new_id
        stale = await client.post('/v1/runs', json=body, headers={**headers, 'Idempotency-Key': 'different-retry'})
        assert stale.status == 409

    assert observed == [[], []]
    assert [row['content'] for row in owner.db.get_messages(session_id)] == ['question', 'answer']
