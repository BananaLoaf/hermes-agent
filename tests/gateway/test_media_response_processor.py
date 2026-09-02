import pytest

from gateway.media_response_processor import MediaResponseProcessor, MediaStreamBuffer


def test_stream_buffer_never_emits_a_split_media_marker():
    buffer = MediaStreamBuffer()

    assert buffer.feed("Ready\nMED") == "Ready\n"
    assert buffer.feed("IA:/tmp/partial-") == ""
    assert buffer.feed("report") == ""
    assert buffer.feed(".pdf\nDone") == "MEDIA:/tmp/partial-report.pdf\nDone"
    assert buffer.finish("Ready\nMEDIA:/tmp/partial-report.pdf\nDone") == ""


def test_stream_buffer_releases_quoted_path_at_closing_quote():
    buffer = MediaStreamBuffer()

    assert buffer.feed('MEDIA:"/tmp/partial') == ""
    assert buffer.feed(' report.custom"') == 'MEDIA:"/tmp/partial report.custom"'
    assert buffer.finish('MEDIA:"/tmp/partial report.custom"') == ""


@pytest.mark.asyncio
async def test_processor_holds_only_media_and_resumes_streaming():
    transformed = []

    async def replace_media(path: str) -> str:
        transformed.append(path)
        return "[Download](https://files.test/report)"

    processor = MediaResponseProcessor(replace_media)

    assert await processor.feed("Ready\nMED") == "Ready\n"
    assert await processor.feed("IA:/tmp/partial-") == ""
    rendered = await processor.feed("report.pdf\nDone")

    assert rendered == "[Download](https://files.test/report)\nDone"
    assert transformed == ["/tmp/partial-report.pdf"]
    assert await processor.feed("\nContinued") == "\nContinued"
    assert transformed == ["/tmp/partial-report.pdf"]
    assert (
        await processor.finish(
            "Ready\nMEDIA:/tmp/partial-report.pdf\nDone\nContinued"
        )
        == ""
    )


@pytest.mark.asyncio
async def test_processor_can_transform_complete_text_without_stream_buffering():
    async def replace_media(_path: str) -> str:
        return "[Download](https://files.test/report)"

    processor = MediaResponseProcessor(replace_media, intercept_stream=False)

    assert processor.buffers_stream is False
    assert await processor.feed("MED") == "MED"
    assert await processor.feed("IA:/tmp/report.pdf") == "IA:/tmp/report.pdf"
    assert await processor.finish("MEDIA:/tmp/report.pdf") == ""
    assert await processor.render("MEDIA:/tmp/report.pdf") == (
        "[Download](https://files.test/report)"
    )
