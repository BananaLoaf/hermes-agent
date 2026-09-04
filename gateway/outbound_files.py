"""Export local Hermes files through configurable external providers."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

try:
    import aiohttp
except ImportError:  # Keep gateway imports usable without the messaging extra.
    aiohttp = None

from gateway.platforms.base import validate_media_delivery_path
from hermes_time import now as hermes_now

logger = logging.getLogger(__name__)


class OutboundFilesConfigError(ValueError):
    """Raised when an explicitly configured outbound-files backend is invalid."""


class OutboundFileUploadError(RuntimeError):
    """Raised when an outbound provider cannot publish a file."""


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
_INLINE_IMAGE_MIME_BY_EXTENSION = {
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
_DEFAULT_IMAGE_TEMPLATE = "![{filename}]({url})"
_DEFAULT_FILE_TEMPLATE = "[Download {filename}]({url})"
_DEFAULT_INVALID_IMAGE_TEMPLATE = "[Image unavailable]"
_DEFAULT_INVALID_FILE_TEMPLATE = "[File unavailable]"
_OUTPUT_TEMPLATE_FIELDS = frozenset(
    {
        "url",
        "filename",
        "date",
        "time",
        "datetime",
        "expiration_date",
        "expiration_time",
        "expiration_datetime",
    }
)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OutboundFilesConfigError(
            f"outbound_files.{key} must be a non-empty string"
        )
    return value.strip()


def _optional_duration(
    data: Mapping[str, Any], key: str, default: Optional[str]
) -> Optional[str]:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise OutboundFilesConfigError(
            f"outbound_files.{key} must be a duration such as 7d or 5h, or null"
        )
    duration = value.strip()
    match = _DURATION_RE.fullmatch(duration)
    if match is None:
        raise OutboundFilesConfigError(
            f"outbound_files.{key} must be a duration such as 7d or 5h, or null"
        )
    if float(match.group("amount")) <= 0:
        raise OutboundFilesConfigError(
            f"outbound_files.{key} duration must be greater than zero"
        )
    return duration


def _configured_now() -> datetime:
    return hermes_now()


def _duration_delta(value: str) -> timedelta:
    match = _DURATION_RE.fullmatch(value)
    if match is None:  # Values are validated while loading the config.
        raise OutboundFilesConfigError(f"invalid outbound file duration: {value}")
    seconds = float(match.group("amount")) * _DURATION_SECONDS[match.group("unit")]
    return timedelta(seconds=seconds)


def _timestamp_parts(value: Optional[datetime], *, prefix: str = "") -> dict[str, str]:
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


def _template_timestamp_values(expiry: Optional[str]) -> dict[str, str]:
    created_at = _configured_now()
    expiration_at = None
    if expiry:
        expiration_at = (
            created_at.astimezone(timezone.utc) + _duration_delta(expiry)
        ).astimezone(created_at.tzinfo)
    return {
        **_timestamp_parts(created_at),
        **_timestamp_parts(expiration_at, prefix="expiration_"),
    }


def _output_template(
    value: Any,
    *,
    key: str,
    default: str,
    allowed_fields: frozenset[str] = _OUTPUT_TEMPLATE_FIELDS,
    required_fields: frozenset[str] = frozenset({"url"}),
) -> str:
    if value is None:
        value = default
    if not isinstance(value, str) or not value.strip():
        raise OutboundFilesConfigError(
            f"outbound_files.templates.{key} must be a non-empty string"
        )

    template = value
    fields: set[str] = set()
    try:
        for _literal, field_name, format_spec, conversion in Formatter().parse(
            template
        ):
            if field_name is None:
                continue
            if field_name not in allowed_fields:
                raise OutboundFilesConfigError(
                    f"outbound_files.templates.{key} contains unsupported "
                    f"placeholder: {field_name}"
                )
            if format_spec or conversion:
                raise OutboundFilesConfigError(
                    f"outbound_files.templates.{key} placeholders must not use "
                    "format specifiers or conversions"
                )
            fields.add(field_name)
    except ValueError as exc:
        raise OutboundFilesConfigError(
            f"outbound_files.templates.{key} is not a valid template"
        ) from exc

    missing = required_fields - fields
    if missing:
        placeholders = ", ".join(f"{{{field}}}" for field in sorted(missing))
        raise OutboundFilesConfigError(
            f"outbound_files.templates.{key} must contain {placeholders}"
        )
    return template


def _output_templates(raw: Mapping[str, Any]) -> tuple[str, str, str, str]:
    templates = raw.get("templates")
    if templates is None:
        templates = {}
    if not isinstance(templates, Mapping):
        raise OutboundFilesConfigError("outbound_files.templates must be a mapping")

    unknown = set(templates) - {
        "image",
        "file",
        "invalid_image",
        "invalid_file",
    }
    if unknown:
        raise OutboundFilesConfigError(
            "outbound_files.templates contains unsupported keys: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    return (
        _output_template(
            templates.get("image"),
            key="image",
            default=_DEFAULT_IMAGE_TEMPLATE,
        ),
        _output_template(
            templates.get("file"),
            key="file",
            default=_DEFAULT_FILE_TEMPLATE,
        ),
        _output_template(
            templates.get("invalid_image"),
            key="invalid_image",
            default=_DEFAULT_INVALID_IMAGE_TEMPLATE,
            allowed_fields=frozenset(),
            required_fields=frozenset(),
        ),
        _output_template(
            templates.get("invalid_file"),
            key="invalid_file",
            default=_DEFAULT_INVALID_FILE_TEMPLATE,
            allowed_fields=frozenset(),
            required_fields=frozenset(),
        ),
    )


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OutboundFilesConfigError(
            "outbound_files.base_url must be an absolute HTTP(S) URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise OutboundFilesConfigError(
            "outbound_files.base_url must not contain credentials, a query, or a fragment"
        )
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _normalize_public_url(value: Any, *, base_url: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutboundFileUploadError("outbound provider returned no file URL")
    url = urljoin(f"{base_url}/", value.strip())
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OutboundFileUploadError("outbound provider returned an invalid file URL")
    if parsed.username or parsed.password:
        raise OutboundFileUploadError("outbound provider returned an unsafe file URL")
    return url


@dataclass(frozen=True)
class OutboundFilesConfig:
    """Provider-independent outbound file settings."""

    provider: str
    file_expiry: Optional[str] = "7d"
    image_expiry: Optional[str] = None
    image_template: str = _DEFAULT_IMAGE_TEMPLATE
    file_template: str = _DEFAULT_FILE_TEMPLATE
    invalid_image_template: str = _DEFAULT_INVALID_IMAGE_TEMPLATE
    invalid_file_template: str = _DEFAULT_INVALID_FILE_TEMPLATE
    provider_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["OutboundFilesConfig"]:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise OutboundFilesConfigError("outbound_files must be a mapping")

        provider = _required_string(raw, "provider").lower()
        (
            image_template,
            file_template,
            invalid_image_template,
            invalid_file_template,
        ) = _output_templates(raw)
        common_keys = {"provider", "file_expiry", "image_expiry", "templates"}
        return cls(
            provider=provider,
            file_expiry=_optional_duration(raw, "file_expiry", "7d"),
            image_expiry=_optional_duration(raw, "image_expiry", None),
            image_template=image_template,
            file_template=file_template,
            invalid_image_template=invalid_image_template,
            invalid_file_template=invalid_file_template,
            provider_options={
                key: value for key, value in raw.items() if key not in common_keys
            },
        )

    def expiry_for(self, *, image: bool) -> Optional[str]:
        return self.image_expiry if image else self.file_expiry


@dataclass(frozen=True)
class UploadedFile:
    """Provider-neutral result of publishing one local file."""

    url: str
    provider_file_id: Optional[str] = None


class OutboundFileProvider(ABC):
    """Publish a local file and return its externally reachable URL."""

    @abstractmethod
    async def upload(self, path: Path, *, expiry: Optional[str]) -> UploadedFile:
        """Upload ``path`` with an optional relative expiry duration."""


class Base64OutboundFileProvider(OutboundFileProvider):
    """Embed small local files in data URLs without an external service."""

    _DEFAULT_MAX_SIZE_BYTES = 1024 * 1024

    def __init__(self, options: Mapping[str, Any]):
        max_size = options.get("max_size_bytes", self._DEFAULT_MAX_SIZE_BYTES)
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise OutboundFilesConfigError(
                "outbound_files.max_size_bytes must be a positive integer"
            )
        self._max_size_bytes = max_size

    @property
    def max_size_bytes(self) -> int:
        return self._max_size_bytes

    def _read_bounded(self, path: Path) -> bytes:
        if not path.is_file():
            raise OutboundFileUploadError("outbound upload source is not a file")
        try:
            with path.open("rb") as file_handle:
                data = file_handle.read(self._max_size_bytes + 1)
        except OSError as exc:
            raise OutboundFileUploadError("outbound file read failed") from exc
        if len(data) > self._max_size_bytes:
            raise OutboundFileUploadError("outbound file exceeds base64 size limit")
        return data

    async def upload(self, path: Path, *, expiry: Optional[str]) -> UploadedFile:
        del expiry  # Data URLs have no server-side expiration.
        data = await asyncio.to_thread(self._read_bounded, path)
        encoded = base64.b64encode(data).decode("ascii")
        return UploadedFile(url=f"data:{_content_type_for_path(path)};base64,{encoded}")


class ZiplineOutboundFileProvider(OutboundFileProvider):
    """Upload files through Zipline's multipart API."""

    def __init__(self, options: Mapping[str, Any]):
        if aiohttp is None:
            raise OutboundFilesConfigError(
                "outbound_files.provider=zipline requires aiohttp"
            )
        self._base_url = _normalize_base_url(_required_string(options, "base_url"))
        self._api_key = _required_string(options, "api_key")

    @property
    def base_url(self) -> str:
        return self._base_url

    async def upload(self, path: Path, *, expiry: Optional[str]) -> UploadedFile:
        if not path.is_file():
            raise OutboundFileUploadError("outbound upload source is not a file")
        content_type = _content_type_for_path(path)
        headers = {
            "Authorization": self._api_key,
            "x-zipline-format": "random",
            "x-zipline-original-name": "true",
        }
        if expiry is not None:
            headers["x-zipline-deletes-at"] = expiry

        timeout = aiohttp.ClientTimeout(total=120)
        try:
            with path.open("rb") as file_handle:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    file_handle,
                    filename=path.name,
                    content_type=content_type,
                )
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self._base_url}/api/upload",
                        headers=headers,
                        data=form,
                    ) as response:
                        if response.status < 200 or response.status >= 300:
                            raise OutboundFileUploadError(
                                f"Zipline upload failed with HTTP {response.status}"
                            )
                        try:
                            payload = await response.json(content_type=None)
                        except Exception as exc:
                            raise OutboundFileUploadError(
                                "Zipline returned an invalid upload response"
                            ) from exc
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
        item = files[0]
        file_id = item.get("id")
        return UploadedFile(
            url=_normalize_public_url(item.get("url"), base_url=self._base_url),
            provider_file_id=file_id if isinstance(file_id, str) and file_id else None,
        )


_PROVIDER_FACTORIES: dict[str, Callable[[Mapping[str, Any]], OutboundFileProvider]] = {
    "base64": Base64OutboundFileProvider,
    "zipline": ZiplineOutboundFileProvider,
}


def create_outbound_file_provider(config: OutboundFilesConfig) -> OutboundFileProvider:
    factory = _PROVIDER_FACTORIES.get(config.provider)
    if factory is None:
        raise OutboundFilesConfigError(
            f"unsupported outbound_files.provider: {config.provider}"
        )
    return factory(config.provider_options)


def _is_inline_image(path: Path) -> bool:
    return path.suffix.lower() in _INLINE_IMAGE_MIME_BY_EXTENSION


def _content_type_for_path(path: Path) -> str:
    return (
        _INLINE_IMAGE_MIME_BY_EXTENSION.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def _markdown_label(value: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", value).strip() or "file"
    return cleaned.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


class OutboundFileExporter:
    """Validate MEDIA paths, publish them, and render provider-neutral Markdown."""

    def __init__(self, config: OutboundFilesConfig, provider: OutboundFileProvider):
        self.config = config
        self.provider = provider

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["OutboundFileExporter"]:
        config = OutboundFilesConfig.from_dict(raw)
        if config is None:
            return None
        return cls(config, create_outbound_file_provider(config))

    async def export_file(self, path: Path) -> str:
        image = _is_inline_image(path)
        expiry = self.config.expiry_for(image=image)
        uploaded = await self.provider.upload(
            path,
            expiry=expiry,
        )
        label = _markdown_label(path.name)
        template = self.config.image_template if image else self.config.file_template
        return template.format(
            url=uploaded.url,
            filename=label,
            **_template_timestamp_values(expiry),
        )

    def invalid_output(self, path: Path) -> str:
        template = (
            self.config.invalid_image_template
            if _is_inline_image(path)
            else self.config.invalid_file_template
        )
        return template.format()

    async def export_media_path(self, path: str) -> str:
        """Publish one intercepted media path and return its replacement text."""
        safe_path = validate_media_delivery_path(path)
        if not safe_path:
            return self.invalid_output(Path(path))
        try:
            return await self.export_file(Path(safe_path))
        except Exception as exc:
            logger.warning(
                "Outbound file upload through %s failed: %s",
                self.config.provider,
                type(exc).__name__,
            )
            return self.invalid_output(Path(safe_path))
