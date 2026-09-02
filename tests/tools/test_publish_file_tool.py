import base64
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from gateway.outbound_files import (
    OutboundFileExporter,
    OutboundFileProvider,
    OutboundFilesConfig,
    UploadedFile,
)
from tools import publish_file_tool


class RecordingProvider(OutboundFileProvider):
    def __init__(self):
        self.uploads = []

    async def upload(self, path: Path, *, expiry, filename=None):
        self.uploads.append((path.read_bytes(), expiry, filename, path))
        return UploadedFile(url="https://files.example.com/u/random")


def _exporter(provider):
    config = OutboundFilesConfig.from_dict({
        "provider": "test",
        "file_expiry": "7d",
        "image_expiry": None,
    })
    return OutboundFileExporter(config, provider)


def test_read_backend_file_uses_terminal_file_operations(monkeypatch):
    seen = {}

    class FakeFileOps:
        def read_file_bytes(self, path):
            seen["read_path"] = path
            return SimpleNamespace(
                error=None,
                base64_content=base64.b64encode(b"backend bytes").decode(),
            )

    monkeypatch.setattr(
        "tools.file_tools._resolve_path_for_task",
        lambda path, task_id: (
            seen.update(path=path, task_id=task_id)
            or PurePosixPath("/workspace/report.pdf")
        ),
    )
    monkeypatch.setattr(
        "tools.file_tools._get_file_ops",
        lambda task_id: seen.update(file_ops_task_id=task_id) or FakeFileOps(),
    )

    content, filename = publish_file_tool._read_backend_file(
        "report.pdf", "session-task"
    )

    assert content == b"backend bytes"
    assert filename == "report.pdf"
    assert seen == {
        "path": "report.pdf",
        "task_id": "session-task",
        "file_ops_task_id": "session-task",
        "read_path": "/workspace/report.pdf",
    }


@pytest.mark.asyncio
async def test_publish_file_uploads_from_active_terminal_task(monkeypatch):
    provider = RecordingProvider()
    seen = {}

    def read_backend(path, task_id):
        seen.update(path=path, task_id=task_id)
        return b"report", "report.pdf"

    monkeypatch.setattr(publish_file_tool, "_read_backend_file", read_backend)
    monkeypatch.setattr(
        publish_file_tool, "_configured_exporter", lambda: _exporter(provider)
    )

    result = json.loads(
        await publish_file_tool.publish_file("report.pdf", task_id="session-task")
    )

    assert result == {
        "success": True,
        "filename": "report.pdf",
        "markdown": "[Download report.pdf](https://files.example.com/u/random)",
    }
    assert seen == {"path": "report.pdf", "task_id": "session-task"}
    assert provider.uploads[0][:3] == (b"report", "7d", "report.pdf")
    assert not provider.uploads[0][3].exists()


@pytest.mark.asyncio
async def test_publish_file_reports_backend_read_failure(monkeypatch):
    provider = RecordingProvider()

    def fail_read(_path, _task_id):
        raise ValueError("private backend error")

    monkeypatch.setattr(publish_file_tool, "_read_backend_file", fail_read)
    monkeypatch.setattr(
        publish_file_tool, "_configured_exporter", lambda: _exporter(provider)
    )

    result = json.loads(await publish_file_tool.publish_file("missing.pdf"))

    assert "terminal environment" in result["error"]
    assert result["markdown"] == "[File unavailable]"
    assert "private backend error" not in json.dumps(result)
    assert provider.uploads == []


@pytest.mark.asyncio
async def test_publish_file_uses_configured_failure_output(monkeypatch):
    class FailingProvider(OutboundFileProvider):
        async def upload(self, path: Path, *, expiry, filename=None):
            raise RuntimeError("provider secret must not leak")

    config = OutboundFilesConfig.from_dict({
        "provider": "test",
        "templates": {"invalid_image": "[Configured image failure]"},
    })
    monkeypatch.setattr(
        publish_file_tool,
        "_read_backend_file",
        lambda _path, _task_id: (b"image", "image.png"),
    )
    monkeypatch.setattr(
        publish_file_tool,
        "_configured_exporter",
        lambda: OutboundFileExporter(config, FailingProvider()),
    )

    result = json.loads(await publish_file_tool.publish_file("image.png"))

    assert result["markdown"] == "[Configured image failure]"
    assert "provider secret" not in json.dumps(result)
