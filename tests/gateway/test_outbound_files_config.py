import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from unittest.mock import MagicMock, patch

import pytest

from gateway.media_response_processor import MediaResponseProcessor
from gateway.outbound_files import (
    Base64OutboundFileProvider,
    OutboundFileExporter,
    OutboundFileProvider,
    OutboundFileUploadError,
    OutboundFilesConfig,
    OutboundFilesConfigError,
    UploadedFile,
    ZiplineOutboundFileProvider,
    _content_type_for_path,
    _is_inline_image,
    create_outbound_file_provider,
)


def _root(**overrides):
    section = {
        "provider": "zipline",
        "base_url": "https://files.example.com/",
        "api_key": "secret-token",
        "file_expiry": "7d",
        "image_expiry": None,
    }
    section.update(overrides)
    return section


def test_missing_section_disables_outbound_files():
    assert OutboundFilesConfig.from_dict(None) is None


def test_creates_base64_provider_with_default_limit():
    config = OutboundFilesConfig.from_dict({"provider": "base64"})

    provider = create_outbound_file_provider(config)

    assert isinstance(provider, Base64OutboundFileProvider)
    assert provider.max_size_bytes == 1024 * 1024


@pytest.mark.parametrize("max_size", [True, False, 0, -1, 1.5, "1024"])
def test_base64_provider_rejects_invalid_size_limit(max_size):
    config = OutboundFilesConfig.from_dict(
        {"provider": "base64", "max_size_bytes": max_size}
    )

    with pytest.raises(
        OutboundFilesConfigError,
        match="max_size_bytes must be a positive integer",
    ):
        create_outbound_file_provider(config)


@pytest.mark.asyncio
async def test_base64_provider_returns_data_url(tmp_path):
    path = tmp_path / "report.pdf"
    contents = b"small report"
    path.write_bytes(contents)
    provider = Base64OutboundFileProvider({"max_size_bytes": len(contents)})

    uploaded = await provider.upload(path, expiry="7d")

    assert uploaded.url == (
        "data:application/pdf;base64," + base64.b64encode(contents).decode("ascii")
    )


@pytest.mark.asyncio
async def test_base64_provider_rejects_file_over_limit(tmp_path):
    path = tmp_path / "large.bin"
    path.write_bytes(b"12345")
    provider = Base64OutboundFileProvider({"max_size_bytes": 4})

    with pytest.raises(OutboundFileUploadError, match="exceeds base64 size limit"):
        await provider.upload(path, expiry=None)


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("image.png", "image/png"),
        ("image.apng", "image/apng"),
        ("image.jpg", "image/jpeg"),
        ("image.jpeg", "image/jpeg"),
        ("image.gif", "image/gif"),
        ("image.webp", "image/webp"),
        ("image.avif", "image/avif"),
        ("image.svg", "image/svg+xml"),
        ("image.bmp", "image/bmp"),
    ],
)
def test_reliable_web_images_use_inline_templates(filename, content_type):
    path = Path(filename)

    assert _is_inline_image(path) is True
    assert _content_type_for_path(path) == content_type


@pytest.mark.parametrize("filename", ["image.tif", "image.tiff", "image.heic"])
def test_browser_specific_images_remain_downloadable_files(filename):
    assert _is_inline_image(Path(filename)) is False


def test_parses_provider_independent_config_and_durations():
    config = OutboundFilesConfig.from_dict(
        _root(
            image_expiry="5h",
            templates={
                "image": "IMAGE {filename}: {url}",
                "file": "FILE {filename}: {url}",
                "invalid_image": "IMAGE UNAVAILABLE",
                "invalid_file": "FILE UNAVAILABLE",
            },
        )
    )

    assert config.provider == "zipline"
    assert config.file_expiry == "7d"
    assert config.image_expiry == "5h"
    assert config.image_template == "IMAGE {filename}: {url}"
    assert config.file_template == "FILE {filename}: {url}"
    assert config.invalid_image_template == "IMAGE UNAVAILABLE"
    assert config.invalid_file_template == "FILE UNAVAILABLE"
    assert config.provider_options == {
        "base_url": "https://files.example.com/",
        "api_key": "secret-token",
    }
    assert config.expiry_for(image=False) == "7d"
    assert config.expiry_for(image=True) == "5h"
    assert "secret-token" not in repr(config)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1d", "1d"), ("5h", "5h"), ("30m", "30m"), ("250ms", "250ms")],
)
def test_accepts_zipline_relative_durations(value, expected):
    config = OutboundFilesConfig.from_dict(_root(file_expiry=value))
    assert config.file_expiry == expected


@pytest.mark.parametrize(
    "root",
    [
        "zipline",
        _root(provider=""),
        _root(file_expiry=7),
        _root(file_expiry="0d"),
        _root(file_expiry="7D"),
        _root(file_expiry="tomorrow"),
        _root(image_expiry=""),
    ],
)
def test_rejects_invalid_common_config(root):
    with pytest.raises(OutboundFilesConfigError):
        OutboundFilesConfig.from_dict(root)


@pytest.mark.parametrize(
    "templates",
    [
        [],
        {"image": "no URL here"},
        {"file": "{unknown}"},
        {"image": "{url!r}"},
        {"invalid_image": "UNAVAILABLE {url}"},
        {"invalid_file": "UNAVAILABLE {filename}"},
        {"video": "{url}"},
    ],
)
def test_rejects_invalid_output_templates(templates):
    with pytest.raises(OutboundFilesConfigError):
        OutboundFilesConfig.from_dict(_root(templates=templates))


def test_provider_factory_rejects_unknown_provider():
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "s3",
            "file_expiry": "7d",
            "image_expiry": None,
        }
    )
    with pytest.raises(OutboundFilesConfigError, match="unsupported"):
        create_outbound_file_provider(config)


@pytest.mark.parametrize(
    "root",
    [
        _root(base_url="files.example.com"),
        _root(base_url="https://user:pass@files.example.com"),
        _root(base_url="https://files.example.com?secret=1"),
        _root(api_key=""),
    ],
)
def test_zipline_provider_rejects_invalid_options(root):
    config = OutboundFilesConfig.from_dict(root)
    with pytest.raises(OutboundFilesConfigError):
        create_outbound_file_provider(config)


@pytest.mark.asyncio
async def test_zipline_upload_uses_token_expiry_and_returned_url(tmp_path, monkeypatch):
    uploaded_path = tmp_path / "report.txt"
    uploaded_path.write_bytes(b"report")
    captured = {}

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self, **_kwargs):
            return {
                "files": [
                    {
                        "id": "file-1",
                        "name": "random.txt",
                        "type": "text/plain",
                        "url": "/u/random.txt",
                    }
                ]
            }

    class FakeSession:
        def __init__(self, **kwargs):
            captured["session_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            captured["form"] = kwargs["data"]
            return FakeResponse()

    monkeypatch.setattr("gateway.outbound_files.aiohttp.ClientSession", FakeSession)
    provider = ZiplineOutboundFileProvider(
        {
            "base_url": "https://files.example.com/",
            "api_key": "token",
        }
    )

    result = await provider.upload(uploaded_path, expiry="5h")

    assert result == UploadedFile(
        url="https://files.example.com/u/random.txt",
        provider_file_id="file-1",
    )
    assert captured["url"] == "https://files.example.com/api/upload"
    assert captured["headers"] == {
        "Authorization": "token",
        "x-zipline-format": "random",
        "x-zipline-original-name": "true",
        "x-zipline-deletes-at": "5h",
    }


class RecordingProvider(OutboundFileProvider):
    def __init__(self):
        self.uploads = []

    async def upload(self, path: Path, *, expiry):
        self.uploads.append((path, expiry))
        return UploadedFile(url=f"https://files.example.com/u/{quote(path.name)}")


@pytest.mark.asyncio
async def test_exporter_selects_expiry_and_markdown_by_file_kind(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    image = tmp_path / "chart.png"
    svg = tmp_path / "diagram.svg"
    document = tmp_path / "quarterly report.pdf"
    image.write_bytes(b"png")
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    document.write_bytes(b"pdf")
    provider = RecordingProvider()
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "test",
            "file_expiry": "7d",
            "image_expiry": None,
        }
    )
    exporter = OutboundFileExporter(config, provider)

    rendered = await MediaResponseProcessor(exporter.export_media_path).render(
        f"Chart:\nMEDIA:{image}\nDiagram:\nMEDIA:{svg}\nReport:\nMEDIA:{document}"
    )

    assert f"![chart.png](https://files.example.com/u/chart.png)" in rendered
    assert f"![diagram.svg](https://files.example.com/u/diagram.svg)" in rendered
    assert (
        "[Download quarterly report.pdf]"
        "(https://files.example.com/u/quarterly%20report.pdf)"
    ) in rendered
    assert "MEDIA:" not in rendered
    assert provider.uploads == [(image, None), (svg, None), (document, "7d")]


@pytest.mark.asyncio
async def test_exporter_uses_configured_templates(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    image = tmp_path / "chart.png"
    document = tmp_path / "quarterly report.pdf"
    image.write_bytes(b"png")
    document.write_bytes(b"pdf")
    provider = RecordingProvider()
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "test",
            "templates": {
                "image": "IMAGE[{filename}]={url}",
                "file": "FILE[{filename}]={url}",
            },
        }
    )
    exporter = OutboundFileExporter(config, provider)

    assert await exporter.export_file(image) == (
        "IMAGE[chart.png]=https://files.example.com/u/chart.png"
    )
    assert await exporter.export_file(document) == (
        "FILE[quarterly report.pdf]="
        "https://files.example.com/u/quarterly%20report.pdf"
    )


@pytest.mark.asyncio
async def test_exporter_renders_configured_timezone_variables(tmp_path, monkeypatch):
    configured_timezone = timezone(timedelta(hours=3))
    fixed_now = datetime(2026, 8, 25, 12, 34, 56, tzinfo=configured_timezone)
    monkeypatch.setattr("gateway.outbound_files._configured_now", lambda: fixed_now)
    image = tmp_path / "chart.png"
    document = tmp_path / "report.pdf"
    image.write_bytes(b"png")
    document.write_bytes(b"pdf")
    provider = RecordingProvider()
    timestamp_template = (
        "{date}|{time}|{datetime}|"
        "{expiration_date}|{expiration_time}|{expiration_datetime}|{url}"
    )
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "test",
            "file_expiry": "7d",
            "image_expiry": None,
            "templates": {
                "image": timestamp_template,
                "file": timestamp_template,
            },
        }
    )
    exporter = OutboundFileExporter(config, provider)

    assert await exporter.export_file(image) == (
        "2026-08-25|12:34|2026-08-25 12:34||||" "https://files.example.com/u/chart.png"
    )
    assert await exporter.export_file(document) == (
        "2026-08-25|12:34|2026-08-25 12:34|"
        "2026-09-01|12:34|2026-09-01 12:34|"
        "https://files.example.com/u/report.pdf"
    )


@pytest.mark.asyncio
async def test_exporter_handles_quoted_unknown_extension_and_hides_invalid_path(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    artifact = tmp_path / "custom artifact.unknown"
    artifact.write_bytes(b"artifact")
    missing_file = tmp_path / "missing.unknown"
    missing_image = tmp_path / "missing.png"
    provider = RecordingProvider()
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "test",
            "file_expiry": "7d",
            "image_expiry": None,
            "templates": {
                "invalid_image": "IMAGE UNAVAILABLE",
                "invalid_file": "FILE UNAVAILABLE",
            },
        }
    )
    exporter = OutboundFileExporter(config, provider)

    rendered = await MediaResponseProcessor(exporter.export_media_path).render(
        f'MEDIA:"{artifact}"\nMEDIA:"{missing_file}"\nMEDIA:"{missing_image}"'
    )

    assert "https://files.example.com/u/custom%20artifact.unknown" in rendered
    assert str(artifact) not in rendered
    assert str(missing_file) not in rendered
    assert str(missing_image) not in rendered
    assert "IMAGE UNAVAILABLE" in rendered
    assert "FILE UNAVAILABLE" in rendered
    assert provider.uploads == [(artifact, "7d")]


@pytest.mark.asyncio
async def test_responses_stream_exports_split_media_marker_without_path_leak(
    tmp_path, monkeypatch
):
    import asyncio
    import time
    import uuid

    import gateway.platforms.api_server as api_mod
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    document = tmp_path / "quarterly report.pdf"
    document.write_bytes(b"pdf")
    provider = RecordingProvider()
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "test",
            "file_expiry": "7d",
            "image_expiry": None,
        }
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter._outbound_files = OutboundFileExporter(config, provider)

    written_payloads = []

    class FakeStreamResponse:
        async def prepare(self, _request):
            return None

        async def write(self, payload):
            written_payloads.append(payload)

    stream_q = api_mod.ThreadSafeAsyncQueue()
    path_text = str(document)
    path_split = path_text.rfind(" ")
    assert path_split > 0
    stream_q.put_nowait("Ready\nMED")
    stream_q.put_nowait(f"IA:{path_text[:path_split]}")
    stream_q.put_nowait(path_text[path_split:])
    stream_q.put_nowait(" Uploaded.")
    stream_q.put_nowait(None)

    allow_agent_to_finish = asyncio.Event()

    async def agent_result():
        await allow_agent_to_finish.wait()
        return (
            {
                "final_response": f"Ready\nMEDIA:{document} Uploaded.",
                "messages": [],
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    request = MagicMock()
    request.headers = {}
    response_id = f"resp_{uuid.uuid4().hex[:28]}"
    with patch.object(api_mod.web, "StreamResponse", return_value=FakeStreamResponse()):
        stream_task = asyncio.create_task(
            adapter._write_sse_responses(
                request=request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=asyncio.create_task(agent_result()),
                agent_ref=[None],
                conversation_history=[],
                user_message="create report",
                instructions=None,
                conversation=None,
                store=False,
                session_id="session-1",
            )
        )
        for _ in range(20):
            if provider.uploads:
                break
            await asyncio.sleep(0.01)
        assert provider.uploads == [(document, "7d")]
        assert not stream_task.done()
        allow_agent_to_finish.set()
        await stream_task

    wire_output = b"".join(written_payloads).decode("utf-8")
    assert str(document) not in wire_output
    assert "https://files.example.com/u/quarterly%20report.pdf" in wire_output
    assert provider.uploads == [(document, "7d")]


@pytest.mark.asyncio
async def test_session_transcript_reuses_exported_final_without_uploading_twice(
    tmp_path, monkeypatch
):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    raw = f"Done\nMEDIA:{document}"
    provider = RecordingProvider()
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "test",
            "file_expiry": "7d",
            "image_expiry": None,
        }
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._outbound_files = OutboundFileExporter(config, provider)
    processor = adapter._new_outbound_response_processor()
    exported_final = await processor.render(raw)

    original = [{"role": "assistant", "content": raw}]
    exported = await adapter._render_outbound_transcript(
        processor,
        original,
        raw_final_response=raw,
        exported_final_response=exported_final,
    )

    assert str(document) not in exported[0]["content"]
    assert exported[0]["content"] == exported_final
    assert original[0]["content"] == raw
    assert provider.uploads == [(document, "7d")]


@pytest.mark.asyncio
async def test_runs_stream_exports_media_without_local_path_leak(tmp_path, monkeypatch):
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    raw = f"Ready\nMEDIA:{document}"
    provider = RecordingProvider()
    config = OutboundFilesConfig.from_dict(
        {
            "provider": "test",
            "file_expiry": "7d",
            "image_expiry": None,
        }
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._outbound_files = OutboundFileExporter(config, provider)

    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)

    def create_agent(**kwargs):
        agent = MagicMock()

        def run_conversation(**_run_kwargs):
            kwargs["stream_delta_callback"]("Ready\nMED")
            kwargs["stream_delta_callback"](f"IA:{document}")
            return {"final_response": raw}

        agent.run_conversation.side_effect = run_conversation
        agent.session_prompt_tokens = 1
        agent.session_completion_tokens = 1
        agent.session_total_tokens = 2
        return agent

    with patch.object(adapter, "_create_agent", side_effect=create_agent):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/runs", json={"input": "make report"})
            run_id = (await response.json())["run_id"]
            for _ in range(40):
                status = await (await client.get(f"/v1/runs/{run_id}")).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.05)
            events = await client.get(f"/v1/runs/{run_id}/events")
            wire_output = await events.text()

    assert str(document) not in wire_output
    assert str(document) not in status["output"]
    assert "https://files.example.com/u/report.pdf" in wire_output
    assert "https://files.example.com/u/report.pdf" in status["output"]
    assert provider.uploads == [(document, "7d")]


def test_managed_gateway_config_reaches_api_server_parser(tmp_path, monkeypatch):
    from gateway.config import Platform, load_gateway_config
    from hermes_cli import managed_scope

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "gateway:\n"
        "  api_server:\n"
        "    enabled: true\n"
        "    outbound_files:\n"
        "      provider: zipline\n"
        "      base_url: https://user-files.example.com\n"
        "      api_key: user-key\n",
        encoding="utf-8",
    )

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "gateway:\n"
        "  api_server:\n"
        "    outbound_files:\n"
        "      base_url: https://central-files.example.com/\n"
        "      api_key: managed-key\n"
        "      image_expiry: null\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    for key in (
        "API_SERVER_ENABLED",
        "API_SERVER_KEY",
        "API_SERVER_PORT",
        "API_SERVER_HOST",
        "API_SERVER_CORS_ORIGINS",
        "API_SERVER_MODEL_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    managed_scope.invalidate_managed_cache()

    gateway_config = load_gateway_config()
    raw = gateway_config.platforms[Platform.API_SERVER].extra["outbound_files"]
    config = OutboundFilesConfig.from_dict(raw)

    assert config.provider_options["base_url"] == "https://central-files.example.com/"
    assert config.provider_options["api_key"] == "managed-key"
    assert config.image_expiry is None
