"""End-to-end tests for inline image inputs on /v1/chat/completions and /v1/responses.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.api_server as api_server_module
from gateway.config import PlatformConfig
from gateway.platforms.api_files import APIFileNotFoundError, APIFileStore
from gateway.platforms.api_server import (
    APIServerAdapter,
    _content_has_visible_payload,
    _normalize_multimodal_content,
    _normalize_request_multimodal_content,
    _session_chat_user_message,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Pure-function tests for _normalize_multimodal_content
# ---------------------------------------------------------------------------


class TestNormalizeMultimodalContent:
    def test_string_passthrough(self):
        assert _normalize_multimodal_content("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _normalize_multimodal_content(None) == ""

    def test_text_only_list_collapses_to_string(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _normalize_multimodal_content(content) == "hi\nthere"

    def test_responses_input_text_canonicalized(self):
        content = [{"type": "input_text", "text": "hello"}]
        assert _normalize_multimodal_content(content) == "hello"


    def test_input_image_converted_to_canonical_shape(self):
        content = [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
        ]
        out = _normalize_multimodal_content(content)
        assert out == [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]

    def test_input_file_is_cached_and_reported_like_gateway_document(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        payload = base64.b64encode(b"quarter,value\nQ1,42\n").decode("ascii")

        out = _normalize_multimodal_content(
            [
                {"type": "input_text", "text": "Analyze this file."},
                {
                    "type": "input_file",
                    "filename": "report.csv",
                    "file_data": f"data:text/csv;base64,{payload}",
                },
            ]
        )

        assert isinstance(out, str)
        assert "Analyze this file." in out
        assert "The user sent a text document: 'report.csv'" in out
        assert "[Content of report.csv]:\nquarter,value" in out
        cached_files = list((tmp_path / "cache" / "documents").iterdir())
        assert len(cached_files) == 1
        assert cached_files[0].read_bytes() == b"quarter,value\nQ1,42\n"

    def test_input_file_resolves_profile_local_file_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = APIFileStore()
        staged = store.stage()
        staged.content_path.write_bytes(b"uploaded through Files API")
        record = store.commit(
            staged,
            filename="notes.txt",
            purpose="user_data",
            content_type="text/plain",
            size=26,
        )

        out = _normalize_multimodal_content(
            [{"type": "input_file", "file_id": record.id}]
        )

        assert "The user sent a text document: 'notes.txt'" in out
        assert "[Content of notes.txt]:\nuploaded through Files API" in out
        assert str(record.content_path) in out
        assert not (tmp_path / "cache" / "documents").exists()

        repeated = _normalize_multimodal_content(
            [{"type": "input_file", "file_id": record.id}]
        )
        assert str(record.content_path) in repeated
        assert not (tmp_path / "cache" / "documents").exists()

        store.delete(record.id)
        assert not record.content_path.exists()

    def test_input_file_rejects_unknown_file_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(ValueError, match="No file found"):
            _normalize_multimodal_content(
                [{"type": "input_file", "file_id": f"file-{'0' * 32}"}]
            )

    def test_chat_completions_nested_file_shape_is_supported(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        encoded = base64.b64encode(b"hello").decode("ascii")
        out = _normalize_multimodal_content(
            [
                {
                    "type": "file",
                    "file": {
                        "filename": "hello.txt",
                        "file_data": f"data:text/plain;base64,{encoded}",
                    },
                }
            ]
        )
        assert "[Content of hello.txt]:\nhello" in out

    def test_input_file_rejects_invalid_base64(self):
        with pytest.raises(ValueError, match="not valid base64"):
            _normalize_multimodal_content(
                [
                    {
                        "type": "input_file",
                        "filename": "broken.bin",
                        "file_data": "data:application/octet-stream;base64,%%%",
                    }
                ]
            )

    def test_rejects_more_than_ten_inline_files_before_caching(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        encoded = base64.b64encode(b"small").decode("ascii")
        parts = [
            {
                "type": "input_file",
                "filename": f"file-{index}.txt",
                "file_data": f"data:text/plain;base64,{encoded}",
            }
            for index in range(11)
        ]

        with pytest.raises(ValueError, match="At most 10 files"):
            _normalize_multimodal_content(parts)

        assert not (tmp_path / "cache" / "documents").exists()

    def test_ten_files_each_at_per_file_limit_are_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(api_server_module, "MAX_API_FILE_BYTES", 4)
        encoded = base64.b64encode(b"four").decode("ascii")
        parts = [
            {
                "type": "input_file",
                "filename": f"file-{index}.txt",
                "file_data": f"data:text/plain;base64,{encoded}",
            }
            for index in range(10)
        ]

        normalized = _normalize_multimodal_content(parts)

        assert normalized.count("[Content of file-") == 10
        assert len(list((tmp_path / "cache" / "documents").iterdir())) == 10

    @pytest.mark.asyncio
    async def test_input_file_decoding_and_caching_are_offloaded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        encoded = base64.b64encode(b"offloaded").decode("ascii")

        async def run_in_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch(
            "gateway.platforms.api_server.asyncio.to_thread",
            new=AsyncMock(side_effect=run_in_thread),
        ) as to_thread:
            result = await _normalize_request_multimodal_content(
                [
                    {
                        "type": "input_file",
                        "filename": "offloaded.bin",
                        "file_data": encoded,
                    }
                ]
            )

        assert "offloaded.bin" in result
        to_thread.assert_awaited_once()


class TestContentHasVisiblePayload:
    def test_list_with_image_only(self):
        assert _content_has_visible_payload([{"type": "image_url", "image_url": {"url": "x"}}])

    def test_list_with_file_only(self):
        assert _content_has_visible_payload([{"type": "input_file", "file_data": "eA=="}])

    @pytest.mark.asyncio
    async def test_session_chat_accepts_file_only_message(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        encoded = base64.b64encode(b"file-only session input").decode("ascii")

        message, error = await _session_chat_user_message(
            {
                "message": [
                    {
                        "type": "input_file",
                        "filename": "only.txt",
                        "file_data": f"data:text/plain;base64,{encoded}",
                    }
                ]
            }
        )

        assert error is None
        assert "The user sent a text document: 'only.txt'" in message
        assert "[Content of only.txt]:\nfile-only session input" in message


# ---------------------------------------------------------------------------
# HTTP integration — real aiohttp client hitting the adapter handlers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/files", adapter._handle_list_files)
    app.router.add_post("/v1/files", adapter._handle_create_file)
    app.router.add_get("/v1/files/{file_id}", adapter._handle_get_file)
    app.router.add_get("/v1/files/{file_id}/content", adapter._handle_get_file_content)
    app.router.add_delete("/v1/files/{file_id}", adapter._handle_delete_file)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestFilesAPIHTTP:
    @staticmethod
    async def _upload(cli, *, data=b"original bytes", filename="report.bin", content_type="application/octet-stream"):
        form = FormData()
        form.add_field("purpose", "user_data")
        form.add_field("file", data, filename=filename, content_type=content_type)
        return await cli.post("/v1/files", data=form)

    def test_store_is_isolated_by_active_profile_home(self, tmp_path, monkeypatch):
        first_home = tmp_path / "first"
        second_home = tmp_path / "second"
        monkeypatch.setenv("HERMES_HOME", str(first_home))
        store = APIFileStore()
        staged = store.stage()
        staged.content_path.write_bytes(b"private")
        record = store.commit(
            staged,
            filename="private.txt",
            purpose="user_data",
            content_type="text/plain",
            size=7,
        )

        monkeypatch.setenv("HERMES_HOME", str(second_home))
        with pytest.raises(APIFileNotFoundError):
            store.get(record.id)

    @pytest.mark.asyncio
    async def test_upload_metadata_content_list_and_delete(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await self._upload(
                cli,
                data=b"quarter,value\nQ1,42\n",
                filename="report.csv",
                content_type="text/csv",
            )
            assert response.status == 200, await response.text()
            created = await response.json()
            file_id = created["id"]
            assert file_id.startswith("file-")
            assert created["bytes"] == 20
            assert created["filename"] == "report.csv"
            assert created["purpose"] == "user_data"

            stored = APIFileStore().get(file_id)
            assert stored.content_path.name == "content.csv"
            assert stored.content_path.read_bytes() == b"quarter,value\nQ1,42\n"

            metadata_response = await cli.get(f"/v1/files/{file_id}")
            assert metadata_response.status == 200
            assert (await metadata_response.json())["id"] == file_id

            list_response = await cli.get("/v1/files")
            assert list_response.status == 200
            listed = await list_response.json()
            assert [item["id"] for item in listed["data"]] == [file_id]
            assert listed["has_more"] is False

            content_response = await cli.get(f"/v1/files/{file_id}/content")
            assert content_response.status == 200
            assert content_response.headers["Content-Type"].startswith("text/csv")
            assert await content_response.read() == b"quarter,value\nQ1,42\n"

            delete_response = await cli.delete(f"/v1/files/{file_id}")
            assert delete_response.status == 200
            assert (await delete_response.json())["deleted"] is True
            assert (await cli.get(f"/v1/files/{file_id}")).status == 404

    @pytest.mark.asyncio
    async def test_upload_enforces_twenty_megabyte_limit(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(api_server_module, "MAX_API_FILE_BYTES", 4)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await self._upload(cli, data=b"12345")
            payload = await response.json()

        assert response.status == 413
        assert payload["error"]["code"] == "file_too_large"
        assert list((tmp_path / "cache" / "api_files").iterdir()) == []


class TestChatCompletionsMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_file_limit_applies_across_all_messages_before_caching(
        self, adapter, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        encoded = base64.b64encode(b"small").decode("ascii")

        def attachments(offset, count):
            return [
                {
                    "type": "input_file",
                    "filename": f"file-{index}.txt",
                    "file_data": f"data:text/plain;base64,{encoded}",
                }
                for index in range(offset, offset + count)
            ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {"role": "user", "content": attachments(0, 6)},
                        {"role": "assistant", "content": attachments(6, 5)},
                    ],
                },
            )
            payload = await response.json()

        assert response.status == 400
        assert payload["error"]["code"] == "too_many_files"
        assert not (tmp_path / "cache" / "documents").exists()

    @pytest.mark.asyncio
    async def test_inline_image_preserved_to_run_agent(self, adapter):
        """Multimodal user content reaches _run_agent as a list of parts."""
        image_payload = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new=MagicMock(),
            ) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": image_payload}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == image_payload


class TestResponsesMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_file_limit_includes_input_and_explicit_history(
        self, adapter, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        encoded = base64.b64encode(b"small").decode("ascii")

        def attachments(offset, count):
            return [
                {
                    "type": "input_file",
                    "filename": f"file-{index}.txt",
                    "file_data": f"data:text/plain;base64,{encoded}",
                }
                for index in range(offset, offset + count)
            ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": [{"role": "user", "content": attachments(0, 6)}],
                    "conversation_history": [
                        {"role": "user", "content": attachments(6, 5)}
                    ],
                },
            )
            payload = await response.json()

        assert response.status == 400
        assert payload["error"]["code"] == "too_many_files"
        assert not (tmp_path / "cache" / "documents").exists()

    @pytest.mark.asyncio
    async def test_input_image_canonicalized_and_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Describe."},
                                    {
                                        "type": "input_image",
                                        "image_url": "https://example.com/cat.png",
                                    },
                                ],
                            }
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            expected = [
                {"type": "text", "text": "Describe."},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]
            assert mock_run.captured["user_message"] == expected

    @pytest.mark.asyncio
    async def test_input_file_reaches_agent_as_cached_document_note(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        encoded = base64.b64encode(b"raw attachment bytes").decode("ascii")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Inspect it."},
                                    {
                                        "type": "input_file",
                                        "filename": "archive.bin",
                                        "file_data": (
                                            "data:application/octet-stream;base64," + encoded
                                        ),
                                    },
                                ],
                            }
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            user_message = mock_run.captured["user_message"]
            assert "Inspect it." in user_message
            assert "The user sent a document: 'archive.bin'" in user_message
            cached = list((tmp_path / "cache" / "documents").iterdir())
            assert len(cached) == 1
            assert cached[0].read_bytes() == b"raw attachment bytes"

    @pytest.mark.asyncio
    async def test_uploaded_file_id_reaches_agent_as_cached_document_note(
        self, adapter, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            upload = FormData()
            upload.add_field("purpose", "user_data")
            upload.add_field(
                "file",
                b"file id attachment",
                filename="source.txt",
                content_type="text/plain",
            )
            upload_response = await cli.post("/v1/files", data=upload)
            assert upload_response.status == 200, await upload_response.text()
            file_id = (await upload_response.json())["id"]

            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )

                mock_run.side_effect = _stub
                response = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Read it."},
                                    {"type": "input_file", "file_id": file_id},
                                ],
                            }
                        ],
                    },
                )

            assert response.status == 200, await response.text()
            user_message = mock_run.captured["user_message"]
            assert "Read it." in user_message
            assert "The user sent a text document: 'source.txt'" in user_message
            assert "[Content of source.txt]:\nfile id attachment" in user_message
            stored = APIFileStore().get(file_id)
            assert str(stored.content_path) in user_message
            assert not (tmp_path / "cache" / "documents").exists()
