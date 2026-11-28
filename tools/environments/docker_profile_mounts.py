"""Structured Docker bind mounts, resolved only inside the owning profile scope."""
from __future__ import annotations

from pathlib import PurePosixPath
import posixpath
import string


def expand(value: str, context: dict) -> str:
    if not isinstance(value, str):
        raise ValueError('Docker mount paths must be strings')
    fields = list(string.Formatter().parse(value))
    if any(field not in context or spec or conversion for _, field, spec, conversion in fields if field is not None):
        raise ValueError('Unknown or formatted Docker mount placeholder')
    return value.format_map(context)


def _path(value: str, context: dict) -> str:
    value = expand(value, context)
    if not value.startswith('/') or any(c in value for c in '\n\r\0,:') or '..' in PurePosixPath(value).parts:
        raise ValueError('Structured Docker mount paths must be absolute, without traversal or Docker separators')
    return posixpath.normpath(value)


def normalize_mounts(volumes: list, scope: dict) -> list[dict]:
    mounts, targets = [], set()
    context = {key: scope[key] for key in ('profile', 'profile_home')}
    hidden_files = scope.get('hidden_files', [])
    if not isinstance(hidden_files, list) or any(not isinstance(path, str) for path in hidden_files):
        raise ValueError('docker_hidden_files must be a list of in-container file paths')
    entries = [(entry, False) for entry in volumes]
    entries += [(dict(source=scope.get('mask_source'), target=path, read_only=True), True) for path in hidden_files]
    for entry, hidden in entries:
        if isinstance(entry, str):
            parts = entry.rsplit(':', 2)
            if len(parts) not in (2, 3) or len(parts) == 3 and parts[2] not in ('ro', 'rw'):
                raise ValueError('Profile Docker mounts must be explicit bind mounts')
            entry = dict(source=parts[0], target=parts[1], read_only=len(parts) == 3 and parts[2] == 'ro')
        if not isinstance(entry, dict) or set(entry) - {'source', 'target', 'read_only'}:
            raise ValueError('Invalid structured Docker mount')
        read_only = entry.get('read_only', False)
        if type(read_only) is not bool:
            raise ValueError('Docker mount read_only must be a boolean')
        source_value = entry.get('source')
        if hidden and not source_value:
            raise ValueError('Hidden files require terminal.docker_mask_source')
        if not hidden and not source_value:
            raise ValueError('Non-hidden Docker mounts require an explicit source')
        source, target = _path(source_value, context), _path(entry.get('target'), context)
        if target == '/' or target in targets:
            raise ValueError('Docker mount targets must be unique and cannot replace /')
        targets.add(target)
        mounts.append(dict(source=source, target=target, read_only=read_only, hidden=hidden))
    # Parent mounts must exist before their file masks, independent of YAML order.
    mounts.sort(key=lambda m: (m['hidden'], len(PurePosixPath(m['target']).parts)))
    _check_masks(mounts)
    return mounts


def _check_masks(mounts: list[dict]) -> None:
    hidden_targets = {m['target'] for m in mounts if m['hidden']}
    for mask in (m for m in mounts if m['hidden']):
        parents = [m for m in mounts if not m['hidden'] and m['target'] != mask['target'] and PurePosixPath(mask['target']).is_relative_to(m['target'])]
        if not parents:
            raise ValueError('A hidden file must mask a file inside an explicit directory mount')
        parent = max(parents, key=lambda m: len(m['target']))
        relative = PurePosixPath(mask['target']).relative_to(parent['target'])
        original = PurePosixPath(parent['source']) / relative
        for other in mounts:
            source = PurePosixPath(other['source'])
            if not other['hidden'] and original.is_relative_to(source):
                alias = str(PurePosixPath(other['target']) / original.relative_to(source))
                if alias not in hidden_targets:
                    raise ValueError('A second mount exposes the hidden file without a mask')
            if not other['read_only'] and PurePosixPath(mask['source']).is_relative_to(source):
                raise ValueError('The hidden-file filler cannot also be exposed through a writable mount')


def bind_spec(mount: dict) -> str:
    return f"type=bind,source={mount['source']},target={mount['target']}" + (',readonly' if mount['read_only'] else '')
