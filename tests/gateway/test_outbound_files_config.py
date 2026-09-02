from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from unittest.mock import MagicMock, patch

import pytest

from gateway.outbound_files import (
    OutboundFileExporter,
    OutboundFileProvider,
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
async def test_exporter_selects_expiry_and_markdown_by_file_kind(tmp_path):
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

    rendered = [
        await exporter.export_file(image),
        await exporter.export_file(svg),
        await exporter.export_file(document),
    ]

    assert "![chart.png](https://files.example.com/u/chart.png)" in rendered
    assert "![diagram.svg](https://files.example.com/u/diagram.svg)" in rendered
    assert (
        "[Download quarterly report.pdf]"
        "(https://files.example.com/u/quarterly%20report.pdf)"
    ) in rendered
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
