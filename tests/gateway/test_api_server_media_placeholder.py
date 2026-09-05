"""Fallback MEDIA handling for API responses without a file publisher."""

import pytest

pytest.importorskip("aiohttp")

from gateway.outbound_files import OutboundFileExporter  # noqa: E402
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_media_tag_is_replaced_without_reading_the_file(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"contents must not be encoded")
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict({"provider": "none"})

    rendered = await adapter._render_outbound_text(f"Here you go: MEDIA:{path}")

    assert rendered == "Here you go: [IMAGE OMITTED]"


@pytest.mark.asyncio
async def test_streaming_fallback_withholds_path_and_emits_placeholder():
    adapter = object.__new__(APIServerAdapter)
    adapter._outbound_files = OutboundFileExporter.from_dict({"provider": "none"})
    processor = adapter._new_media_response_processor()

    assert await processor.feed("Ready\nMED") == "Ready\n"
    assert await processor.feed("IA:/tmp/private") == ""
    assert await processor.feed(".pdf\nDone") == "[FILE OMITTED]\nDone"
