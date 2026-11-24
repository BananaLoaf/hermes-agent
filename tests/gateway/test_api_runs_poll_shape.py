"""GET /v1/runs/{id} keeps the documented poll shape when the canonical admission answers."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_canonical_run_poll_keeps_object_model_and_created_at(api, owner, monkeypatch):
    from gateway.platforms import api_server_runs

    async def no_execution(*args, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', no_execution)
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', api._handle_get_run)
    async with TestClient(TestServer(app)) as client:
        accepted = await client.post('/v1/runs', json={'input': 'hi', 'model': 'hermes-agent'})
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        polled = await (await client.get(f'/v1/runs/{run_id}')).json()
    assert polled['admission_id'] and polled['status'] == 'queued'
    assert polled['object'] == 'hermes.run' and polled['model'] == 'hermes-agent', polled
    assert isinstance(polled['created_at'], float)


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['zipline', 'base64'])
async def test_run_retains_rendered_reply_only_for_zipline(api, owner, monkeypatch, tmp_path, provider):
    from gateway.outbound_files import OutboundFileExporter
    from gateway.outbound_files.zipline import ZiplineOutboundFileProvider
    from gateway.session_results import admission_result, execution_result

    path = tmp_path / 'answer.png'
    path.write_bytes(b'image')
    raw_reply = f'Here is the image.\nMEDIA:{path}'
    url = 'https://files.example.invalid/u/answer.png'
    uploads = []
    upload_started = asyncio.Event()
    finish_upload = asyncio.Event()

    async def upload(self, path, expiry):
        uploads.append(path)
        upload_started.set()
        await finish_upload.wait()
        return url

    options = {'base_url': 'https://files.example.invalid', 'api_key': 'test'}
    api._outbound_files = OutboundFileExporter.from_dict({
        'provider': provider, 'provider_options': options if provider == 'zipline' else {}})
    monkeypatch.setattr(ZiplineOutboundFileProvider, '_upload', upload)

    async def handle(event):
        messages = [{'role': 'user', 'content': event.text}, {'role': 'assistant', 'content': raw_reply}]
        owner.db.append_messages_batch('media-chat', messages)
        execution_result.get()['result'] = {
            'final_response': raw_reply, 'messages': messages, 'session_id': 'media-chat',
            'persisted_turn': {'final_assistant_row_id': messages[-1]['_row_id']}}
        return raw_reply

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', api._handle_get_run)
    app.router.add_get('/v1/runs/{run_id}/events', api._handle_run_events)
    app.router.add_get('/api/sessions/{session_id}/messages', api._handle_session_messages)
    async with TestClient(TestServer(app)) as client:
        accepted = await client.post('/v1/runs', json={'input': 'make an image', 'session_id': 'media-chat'})
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        if provider == 'zipline':
            try:
                await asyncio.wait_for(upload_started.wait(), timeout=5)
                pending = await (await client.get(f'/v1/runs/{run_id}')).json()
                assert pending['status'] == 'running'
                assert pending['output'] == ''
            finally:
                finish_upload.set()
        events = await (await client.get(f'/v1/runs/{run_id}/events')).text()
        assert 'run.completed' in events
        assert (url if provider == 'zipline' else 'data:image/png;base64,') in events

        polled = await (await client.get(f'/v1/runs/{run_id}')).json()
        history = await (await client.get('/api/sessions/media-chat/messages')).json()
        expected = f'Here is the image.\n![answer.png]({url})' if provider == 'zipline' else raw_reply
        assert polled['output'] == history['data'][-1]['content'] == expected
        row = owner.db.get_messages('media-chat')[-1]
        assert (row['display_metadata'] or {}).get('display_text') == (expected if provider == 'zipline' else None)
        assert admission_result(owner.db, polled['admission_id'])['result']['final_response'] == raw_reply
        assert owner.db.get_messages_as_conversation('media-chat')[-1]['content'] == raw_reply

        # A compactor may have read the assistant row before delivery added its rendering.
        carried = owner.db.get_messages_as_conversation('media-chat')
        for message in carried:
            message.pop('display_metadata', None)
        owner.db.archive_and_compact('media-chat', carried)
        history = await (await client.get('/api/sessions/media-chat/messages?include_compacted=true')).json()
        assert history['data'][-1]['content'] == expected
        assert uploads == ([path] if provider == 'zipline' else [])
