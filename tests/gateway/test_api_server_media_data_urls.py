"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import _resolve_media_to_data_urls  # noqa: E402

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


class TestResolveMediaToDataUrls(unittest.TestCase):
    def _write_png(self, tmpdir_name="hermes_media_test"):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix=tmpdir_name))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    def test_media_tag_inlined(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"Here you go: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_backtick_wrapped_tag(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"See `MEDIA:{p}` above")
        self.assertIn("data:image/png;base64,", out)

    def test_missing_file_left_untouched(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_left_untouched(self):
        text = "MEDIA:/tmp/archive.zip"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_svg_inlined_as_image(self):
        import tempfile
        from pathlib import Path

        directory = Path(tempfile.mkdtemp(prefix="hermes_svg_media_test"))
        path = directory / "diagram.svg"
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")

        out = _resolve_media_to_data_urls(f"MEDIA:{path}")

        self.assertIn("data:image/svg+xml;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_apng_and_avif_are_inlined_as_images(self):
        import tempfile
        from pathlib import Path

        directory = Path(tempfile.mkdtemp(prefix="hermes_modern_media_test"))
        image_types = ((".apng", "image/apng"), (".avif", "image/avif"))
        for suffix, mime_type in image_types:
            with self.subTest(suffix=suffix):
                path = directory / f"image{suffix}"
                path.write_bytes(b"image")

                out = _resolve_media_to_data_urls(f"MEDIA:{path}")

                self.assertIn(f"data:{mime_type};base64,", out)
                self.assertNotIn("MEDIA:", out)


if __name__ == "__main__":
    unittest.main()
