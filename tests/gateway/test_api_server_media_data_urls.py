"""Provider-backed MEDIA rendering for the API server."""

import base64
import asyncio
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aiohttp")

from gateway.outbound_files import (  # noqa: E402
    Base64OutboundFileProvider,
    OmittedOutboundFileProvider,
    OutboundFileExporter,
    OutboundFilesConfigError,
    create_outbound_file_provider,
    OutboundFilesConfig,
)
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_media_tag_is_inlined_as_a_data_url(tmp_path):
    path = tmp_path / "shot.png"
    contents = b"png"
    path.write_bytes(contents)
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict(None)

    rendered = await adapter._render_outbound_text(f"Here: MEDIA:{path}")

    encoded = base64.b64encode(contents).decode("ascii")
    assert rendered == f"Here: ![image](data:image/png;base64,{encoded})"


@pytest.mark.asyncio
async def test_non_image_and_missing_media_are_left_untouched(tmp_path):
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    missing = tmp_path / "missing.png"
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict(None)

    document_text = f"MEDIA:{document}"
    missing_text = f"MEDIA:{missing}"
    assert await adapter._render_outbound_text(document_text) == document_text
    assert await adapter._render_outbound_text(missing_text) == missing_text


@pytest.mark.asyncio
async def test_base64_provider_enforces_configured_size_limit(tmp_path):
    path = tmp_path / "large.png"
    path.write_bytes(b"12345")
    provider = Base64OutboundFileProvider({"max_size_bytes": 4})

    assert await provider.render(path) is None


def test_base64_provider_preserves_legacy_default_limit():
    provider = Base64OutboundFileProvider({})

    assert provider.max_size_bytes == 5 * 1024 * 1024


def test_unknown_provider_is_rejected():
    config = OutboundFilesConfig.from_dict({"provider": "unknown"})

    with pytest.raises(OutboundFilesConfigError, match="unsupported.*unknown"):
        create_outbound_file_provider(config)


@pytest.mark.parametrize("provider", [None, "none", "NONE"])
def test_none_provider_selects_omitted_file_renderer(provider):
    config = OutboundFilesConfig.from_dict({"provider": provider})

    assert isinstance(create_outbound_file_provider(config), OmittedOutboundFileProvider)


@pytest.mark.asyncio
async def test_none_provider_omits_images_and_other_files_without_exposing_paths():
    exporter = OutboundFileExporter.from_dict({"provider": None})

    assert await exporter.export_media_path("/missing/private.png") == "[IMAGE OMITTED]"
    assert await exporter.export_media_path("/missing/private.pdf") == "[FILE OMITTED]"


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1024"])
def test_base64_provider_rejects_invalid_size_limit(value):
    with pytest.raises(OutboundFilesConfigError, match="positive integer"):
        Base64OutboundFileProvider({"max_size_bytes": value})


@pytest.mark.asyncio
async def test_streaming_processor_holds_only_media_candidate(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"png")
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict(None)
    processor = adapter._new_media_response_processor()

    assert await processor.feed("Ready\nMED") == "Ready\n"
    assert await processor.feed(f"IA:{str(path)[:-4]}") == ""
    rendered = await processor.feed(".png\nDone")

    assert rendered.startswith("![image](data:image/png;base64,")
    assert rendered.endswith("\nDone")


@pytest.mark.asyncio
async def test_responses_stream_replaces_split_media_before_writing(tmp_path):
    import gateway.platforms.api_server as api_server
    from gateway.config import PlatformConfig

    path = tmp_path / "chart.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}\nDone"
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test"}))
    written = []

    class FakeStreamResponse:
        async def prepare(self, _request):
            return None

        async def write(self, payload):
            written.append(payload)

    stream_q = api_server.ThreadSafeAsyncQueue()
    stream_q.put_nowait("Ready\nMED")
    stream_q.put_nowait(f"IA:{str(path)[:-4]}")
    stream_q.put_nowait(".png\nDone")
    stream_q.put_nowait(None)

    async def agent_result():
        return (
            {"final_response": raw, "messages": []},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    request = MagicMock()
    request.headers = {}
    with patch.object(
        api_server.web,
        "StreamResponse",
        return_value=FakeStreamResponse(),
    ):
        await adapter._write_sse_responses(
            request=request,
            response_id=f"resp_{uuid.uuid4().hex[:28]}",
            model="hermes-agent",
            created_at=int(time.time()),
            stream_q=stream_q,
            agent_task=asyncio.create_task(agent_result()),
            agent_ref=[None],
            conversation_history=[],
            user_message="create a chart",
            instructions=None,
            conversation=None,
            store=False,
            session_id="session-1",
        )

    wire_output = b"".join(written).decode("utf-8")
    assert str(path) not in wire_output
    assert "data:image/png;base64," in wire_output


@pytest.mark.asyncio
async def test_runs_stream_replaces_split_media_before_writing(tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig

    path = tmp_path / "chart.png"
    path.write_bytes(b"png")
    raw = f"Ready\nMEDIA:{path}"
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)

    def create_agent(**kwargs):
        agent = MagicMock()

        def run_conversation(**_run_kwargs):
            kwargs["stream_delta_callback"]("Ready\nMED")
            kwargs["stream_delta_callback"](f"IA:{path}")
            return {"final_response": raw}

        agent.run_conversation.side_effect = run_conversation
        agent.session_prompt_tokens = 1
        agent.session_completion_tokens = 1
        agent.session_total_tokens = 2
        return agent

    with patch.object(adapter, "_create_agent", side_effect=create_agent):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/runs", json={"input": "make chart"})
            run_id = (await response.json())["run_id"]
            for _ in range(40):
                status = await (await client.get(f"/v1/runs/{run_id}")).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.05)
            events = await client.get(f"/v1/runs/{run_id}/events")
            wire_output = await events.text()

    assert str(path) not in wire_output
    assert str(path) not in status["output"]
    assert "data:image/png;base64," in wire_output
    assert "data:image/png;base64," in status["output"]
