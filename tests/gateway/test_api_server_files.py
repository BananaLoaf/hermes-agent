"""End-to-end contracts for the OpenAI-compatible Files API."""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_files import APIFileStore
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import get_document_cache_dir


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/files") or path == "/v1/responses":
            app.router.add_route(method, path, handler)
    return app


async def _upload(
    client: TestClient,
    *,
    data: bytes,
    filename: str,
    purpose: str = "user_data",
    content_type: str = "application/octet-stream",
):
    form = FormData()
    form.add_field("purpose", purpose)
    form.add_field("file", data, filename=filename, content_type=content_type)
    return await client.post("/v1/files", data=form)


@pytest.fixture
def adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


@pytest.mark.asyncio
async def test_files_api_round_trip_uses_telegram_document_cache(
    adapter, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    payload = b"quarter,value\nQ1,42\n"

    async with TestClient(TestServer(_create_app(adapter))) as client:
        uploaded = await _upload(
            client,
            data=payload,
            filename="report.csv",
            purpose="assistants",
            content_type="text/csv",
        )
        assert uploaded.status == 200, await uploaded.text()
        created = await uploaded.json()
        record = APIFileStore().get(created["id"])

        assert created == record.as_api_dict()
        assert record.content_path.parent == get_document_cache_dir()
        assert record.content_path.name.startswith("doc_")
        assert record.content_path.name.endswith("_report.csv")
        assert record.content_path.read_bytes() == payload

        metadata = await client.get(f"/v1/files/{record.id}")
        assert metadata.status == 200
        assert await metadata.json() == created

        listing = await client.get("/v1/files", params={"purpose": "assistants"})
        assert listing.status == 200
        assert (await listing.json())["data"] == [created]

        content = await client.get(f"/v1/files/{record.id}/content")
        assert content.status == 200
        assert content.headers["Content-Type"].startswith("text/csv")
        assert await content.read() == payload

        deleted = await client.delete(f"/v1/files/{record.id}")
        assert deleted.status == 200
        assert await deleted.json() == {"id": record.id, "object": "file", "deleted": True}
        assert not record.content_path.exists()
        assert (await client.get(f"/v1/files/{record.id}")).status == 404


@pytest.mark.asyncio
async def test_responses_file_id_reuses_cached_path_without_copy(
    adapter, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        uploaded = await _upload(
            client,
            data=b"file id attachment",
            filename="source.txt",
            content_type="text/plain",
        )
        assert uploaded.status == 200, await uploaded.text()
        file_id = (await uploaded.json())["id"]
        record = APIFileStore().get(file_id)

        with patch.object(adapter, "_run_agent", new=MagicMock()) as run_agent:
            async def _stub(**kwargs):
                run_agent.captured = kwargs
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )

            run_agent.side_effect = _stub
            response = await client.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": [{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Read it."},
                            {"type": "input_file", "file_id": file_id},
                        ],
                    }],
                },
            )

        assert response.status == 200, await response.text()
        user_message = run_agent.captured["user_message"]
        assert "Read it." in user_message
        assert "[Content of source.txt]:\nfile id attachment" in user_message
        assert str(record.content_path) in user_message
        assert list(get_document_cache_dir().iterdir()) == [record.content_path]
