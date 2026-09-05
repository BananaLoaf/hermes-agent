"""Zipline provider for outbound API files."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import Formatter
from typing import ClassVar, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

try:
    import aiohttp
except ImportError:  # Keep gateway imports usable without the messaging extra.
    aiohttp = None

from gateway.outbound_files.config import OutboundFilesConfigError
from gateway.outbound_files.provider import OutboundFileProvider
from hermes_time import now as hermes_now


class OutboundFileUploadError(RuntimeError):
    """Raised when Zipline cannot publish a file."""


_DURATION_RE = re.compile(r"^(?P<amount>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h|d|w|y)$")
_DURATION_SECONDS = {
    "ms": 0.001,
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
    "y": 365 * 24 * 60 * 60,
}
_IMAGE_MIME_TYPES: Mapping[str, str] = {
    ".png": "image/png",
    ".apng": "image/apng",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
}
_TEMPLATE_FIELDS = frozenset({
    "url",
    "filename",
    "date",
    "time",
    "datetime",
    "expiration_date",
    "expiration_time",
    "expiration_datetime",
})


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutboundFilesConfigError(
            f"outbound_files.provider_options.{name} must be a non-empty string"
        )
    return value.strip()


def _optional_duration(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not (duration := value.strip()):
        raise OutboundFilesConfigError(
            f"outbound_files.provider_options.{name} must be a duration or null"
        )
    match = _DURATION_RE.fullmatch(duration)
    if match is None or float(match.group("amount")) <= 0:
        raise OutboundFilesConfigError(
            f"outbound_files.provider_options.{name} must be a positive duration"
        )
    return duration


def _template(value: object, name: str, *, require_url: bool) -> str:
    template = _required_string(value, name)
    fields: set[str] = set()
    try:
        parsed = Formatter().parse(template)
        for _literal, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name not in _TEMPLATE_FIELDS or format_spec or conversion:
                raise OutboundFilesConfigError(
                    f"outbound_files.provider_options.{name} has an unsupported placeholder"
                )
            fields.add(field_name)
    except ValueError as exc:
        raise OutboundFilesConfigError(
            f"outbound_files.provider_options.{name} is not a valid template"
        ) from exc
    if require_url and "url" not in fields:
        raise OutboundFilesConfigError(
            f"outbound_files.provider_options.{name} must contain {{url}}"
        )
    if not require_url and fields:
        raise OutboundFilesConfigError(
            f"outbound_files.provider_options.{name} must not contain placeholders"
        )
    return template


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OutboundFilesConfigError(
            "outbound_files.provider_options.base_url must be an absolute HTTP(S) URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise OutboundFilesConfigError(
            "outbound_files.provider_options.base_url must not contain credentials, "
            "a query, or a fragment"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _normalize_public_url(value: object, base_url: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutboundFileUploadError("Zipline returned no file URL")
    url = urljoin(f"{base_url}/", value.strip())
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OutboundFileUploadError("Zipline returned an invalid file URL")
    if parsed.username or parsed.password:
        raise OutboundFileUploadError("Zipline returned an unsafe file URL")
    return url


def _timestamp_values(expiry: Optional[str]) -> dict[str, str]:
    created_at = hermes_now()
    expiration_at = None
    if expiry:
        match = _DURATION_RE.fullmatch(expiry)
        assert match is not None
        seconds = float(match.group("amount")) * _DURATION_SECONDS[match.group("unit")]
        expiration_at = (
            created_at.astimezone(timezone.utc) + timedelta(seconds=seconds)
        ).astimezone(created_at.tzinfo)

    def parts(value: Optional[datetime], prefix: str = "") -> dict[str, str]:
        if value is None:
            return {
                f"{prefix}date": "",
                f"{prefix}time": "",
                f"{prefix}datetime": "",
            }
        return {
            f"{prefix}date": value.strftime("%Y-%m-%d"),
            f"{prefix}time": value.strftime("%H:%M"),
            f"{prefix}datetime": value.strftime("%Y-%m-%d %H:%M"),
        }

    return {**parts(created_at), **parts(expiration_at, "expiration_")}


def _markdown_label(value: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", value).strip() or "file"
    return cleaned.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


@dataclass(frozen=True)
class ZiplineOutboundFileProvider(OutboundFileProvider):
    """Upload files through Zipline and render their public URLs as Markdown."""

    base_url: str
    api_key: Optional[str] = field(default=None, repr=False)
    file_expiry: Optional[str] = "7d"
    image_expiry: Optional[str] = None
    image_template: str = "![{filename}]({url})"
    file_template: str = "[Download {filename}]({url})"
    invalid_image_template: str = "[Image unavailable]"
    invalid_file_template: str = "[File unavailable]"
    image_mime_types: ClassVar[Mapping[str, str]] = _IMAGE_MIME_TYPES

    def __post_init__(self) -> None:
        if aiohttp is None:
            raise OutboundFilesConfigError(
                "outbound_files.provider=zipline requires aiohttp"
            )
        object.__setattr__(
            self,
            "base_url",
            _normalize_base_url(_required_string(self.base_url, "base_url")),
        )
        api_key = (
            self.api_key if self.api_key is not None else os.getenv("ZIPLINE_API_KEY")
        )
        object.__setattr__(self, "api_key", _required_string(api_key, "api_key"))
        object.__setattr__(
            self, "file_expiry", _optional_duration(self.file_expiry, "file_expiry")
        )
        object.__setattr__(
            self, "image_expiry", _optional_duration(self.image_expiry, "image_expiry")
        )
        object.__setattr__(
            self,
            "image_template",
            _template(self.image_template, "image_template", require_url=True),
        )
        object.__setattr__(
            self,
            "file_template",
            _template(self.file_template, "file_template", require_url=True),
        )
        object.__setattr__(
            self,
            "invalid_image_template",
            _template(
                self.invalid_image_template, "invalid_image_template", require_url=False
            ),
        )
        object.__setattr__(
            self,
            "invalid_file_template",
            _template(
                self.invalid_file_template, "invalid_file_template", require_url=False
            ),
        )

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in self.image_mime_types

    def invalid_output(self, path: Path) -> str:
        return (
            self.invalid_image_template
            if self._is_image(path)
            else self.invalid_file_template
        )

    async def _upload(self, path: Path, expiry: Optional[str]) -> str:
        headers = {
            "Authorization": self.api_key,
            "x-zipline-format": "random",
            "x-zipline-original-name": "true",
        }
        if expiry is not None:
            headers["x-zipline-deletes-at"] = expiry
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        timeout = aiohttp.ClientTimeout(total=120)
        try:
            with path.open("rb") as file_handle:
                form = aiohttp.FormData()
                form.add_field(
                    "file", file_handle, filename=path.name, content_type=content_type
                )
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.base_url}/api/upload", headers=headers, data=form
                    ) as response:
                        if not 200 <= response.status < 300:
                            raise OutboundFileUploadError(
                                f"Zipline upload failed with HTTP {response.status}"
                            )
                        payload = await response.json(content_type=None)
        except OutboundFileUploadError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise OutboundFileUploadError("Zipline upload failed") from exc

        files = payload.get("files") if isinstance(payload, Mapping) else None
        if (
            not isinstance(files, list)
            or len(files) != 1
            or not isinstance(files[0], Mapping)
        ):
            raise OutboundFileUploadError("Zipline returned an invalid upload response")
        return _normalize_public_url(files[0].get("url"), self.base_url)

    async def render(self, path: Path) -> str:
        image = self._is_image(path)
        expiry = self.image_expiry if image else self.file_expiry
        url = await self._upload(path, expiry)
        template = self.image_template if image else self.file_template
        return template.format(
            url=url,
            filename=_markdown_label(path.name),
            **_timestamp_values(expiry),
        )

    def system_prompt_hint(self) -> str:
        return (
            "File/media delivery is enabled. Include MEDIA:/absolute/path in your "
            "response to upload an existing local file through the configured Zipline "
            "service and replace the directive with a public link."
        )
