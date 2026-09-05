"""Zipline coverage for the modular outbound-file provider contract."""

from datetime import datetime, timezone

import pytest

from gateway.outbound_files import (
    OutboundFileExporter,
    OutboundFilesConfig,
    OutboundFilesConfigError,
    ZiplineOutboundFileProvider,
    create_outbound_file_provider,
)


def _options(**overrides):
    options = {
        "base_url": "https://files.example.com/",
        "api_key": "secret-token",
        "file_expiry": "7d",
        "image_expiry": None,
    }
    options.update(overrides)
    return options


def _config(**overrides):
    return OutboundFilesConfig.from_dict({
        "provider": "zipline",
        "provider_options": _options(**overrides),
    })


def test_zipline_is_created_from_nested_provider_options():
    provider = create_outbound_file_provider(_config(file_expiry="14d"))

    assert isinstance(provider, ZiplineOutboundFileProvider)
    assert provider.base_url == "https://files.example.com"
    assert provider.file_expiry == "14d"
    assert "secret-token" not in repr(provider)


def test_zipline_reads_api_key_from_hermes_environment(monkeypatch):
    monkeypatch.setenv("ZIPLINE_API_KEY", "environment-token")
    options = _options()
    options.pop("api_key")

    provider = create_outbound_file_provider(
        OutboundFilesConfig.from_dict({
            "provider": "zipline",
            "provider_options": options,
        })
    )

    assert provider.api_key == "environment-token"


@pytest.mark.parametrize(
    "options",
    [
        _options(base_url="files.example.com"),
        _options(base_url="https://user:pass@files.example.com"),
        _options(base_url="https://files.example.com?secret=1"),
        _options(api_key=""),
        _options(file_expiry=7),
        _options(file_expiry="0d"),
        _options(image_expiry="tomorrow"),
        _options(unknown=True),
    ],
)
def test_zipline_rejects_invalid_options(options):
    config = OutboundFilesConfig.from_dict({
        "provider": "zipline",
        "provider_options": options,
    })

    with pytest.raises(OutboundFilesConfigError):
        create_outbound_file_provider(config)


@pytest.mark.parametrize(
    "option",
    [
        {"image_template": "image without URL"},
        {"file_template": "{unknown}"},
        {"image_template": "{url!r}"},
        {"invalid_image_template": "Unavailable: {url}"},
        {"invalid_file_template": "Unavailable: {filename}"},
    ],
)
def test_zipline_rejects_invalid_templates(option):
    with pytest.raises(OutboundFilesConfigError):
        create_outbound_file_provider(_config(**option))


@pytest.mark.asyncio
async def test_zipline_upload_renders_template_and_expiry(tmp_path, monkeypatch):
    uploaded_path = tmp_path / "quarterly report.pdf"
    uploaded_path.write_bytes(b"report")
    captured = {}

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self, **_kwargs):
            return {"files": [{"id": "file-1", "url": "/u/random.pdf"}]}

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
            return FakeResponse()

    monkeypatch.setattr(
        "gateway.outbound_files.zipline.aiohttp.ClientSession", FakeSession
    )
    monkeypatch.setattr(
        "gateway.outbound_files.zipline.hermes_now",
        lambda: datetime(2026, 9, 5, 12, 30, tzinfo=timezone.utc),
    )
    provider = create_outbound_file_provider(
        _config(
            file_expiry="5h",
            file_template="[{filename} until {expiration_datetime}]({url})",
        )
    )

    rendered = await provider.render(uploaded_path)

    assert rendered == (
        "[quarterly report.pdf until 2026-09-05 17:30]"
        "(https://files.example.com/u/random.pdf)"
    )
    assert captured["url"] == "https://files.example.com/api/upload"
    assert captured["headers"] == {
        "Authorization": "secret-token",
        "x-zipline-format": "random",
        "x-zipline-original-name": "true",
        "x-zipline-deletes-at": "5h",
    }


@pytest.mark.asyncio
async def test_zipline_uses_image_expiry_and_markdown(tmp_path, monkeypatch):
    image = tmp_path / "chart.png"
    image.write_bytes(b"png")

    async def fake_upload(self, path, expiry):
        assert path == image
        assert expiry is None
        return "https://files.example.com/u/chart.png"

    monkeypatch.setattr(ZiplineOutboundFileProvider, "_upload", fake_upload)
    provider = create_outbound_file_provider(_config())

    assert await provider.render(image) == (
        "![chart.png](https://files.example.com/u/chart.png)"
    )


@pytest.mark.asyncio
async def test_zipline_exporter_uses_safe_placeholders(tmp_path, monkeypatch):
    provider = create_outbound_file_provider(
        _config(
            invalid_image_template="[IMAGE UNAVAILABLE]",
            invalid_file_template="[FILE UNAVAILABLE]",
        )
    )
    exporter = OutboundFileExporter(provider)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"report")

    async def failed_upload(*_args, **_kwargs):
        raise RuntimeError("provider secret must not escape")

    monkeypatch.setattr(ZiplineOutboundFileProvider, "_upload", failed_upload)

    assert await exporter.export_media_path(str(document)) == "[FILE UNAVAILABLE]"
    assert await exporter.export_media_path("/missing/private.png") == (
        "[IMAGE UNAVAILABLE]"
    )
