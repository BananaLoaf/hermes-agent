"""Structured Docker bind mounts, resolved only inside the owning profile scope."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import posixpath
import string
import shlex

OMITTED = 'File contents omitted for security\n'


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
        _check_profile_source(source, scope, context, hidden)
        mounts.append(dict(source=source, target=target, read_only=read_only, hidden=hidden))
    # Parent mounts must exist before their file masks, independent of YAML order.
    mounts.sort(key=lambda m: (m['hidden'], len(PurePosixPath(m['target']).parts)))
    _check_masks(mounts)
    return mounts


def _check_profile_source(source, scope, context, hidden):
    roots = [scope.get('profiles_root', '')]
    path = PurePosixPath(source)
    if 'profiles' in path.parts:
        roots.append(str(PurePosixPath(*path.parts[:path.parts.index('profiles') + 1])))
    allowed = any(path.is_relative_to(_path(p, context)) for p in scope.get('allowed_sources', []))
    for root in filter(None, roots):
        root = PurePosixPath(root)
        if not hidden and path == root:
            raise ValueError('Mount individual profiles, not the profiles root')
        if not hidden and root.is_relative_to(path) and not allowed:
            raise ValueError('A source containing private profiles needs administrator approval')
        if not hidden and path.is_relative_to(root) and path != root:
            if path.relative_to(root).parts[0] != scope['profile'] and not allowed:
                raise ValueError('Another profile source needs docker_allowed_profile_sources approval')


def _check_masks(mounts: list[dict]) -> None:
    hidden_targets = {m['target'] for m in mounts if m['hidden']}
    for mask in (m for m in mounts if m['hidden']):
        parents = [m for m in mounts if not m['hidden'] and m['target'] != mask['target'] and PurePosixPath(mask['target']).is_relative_to(m['target'])]
        if not parents:
            raise ValueError('A hidden file must mask a file inside an explicit directory mount')
        parent = max(parents, key=lambda m: len(m['target']))
        original = PurePosixPath(parent['source']) / PurePosixPath(mask['target']).relative_to(parent['target'])
        mask['parent_source'] = parent['source']
        mask['relative_target'] = str(PurePosixPath(mask['target']).relative_to(parent['target']))
        for other in mounts:
            source = PurePosixPath(other['source'])
            if not other['hidden'] and original.is_relative_to(source):
                alias = str(PurePosixPath(other['target']) / original.relative_to(source))
                if alias not in hidden_targets:
                    raise ValueError('A second mount exposes the hidden file without a mask')
            if not other['read_only'] and PurePosixPath(mask['source']).is_relative_to(source):
                raise ValueError('The hidden-file filler cannot also be exposed through a writable mount')


def bind_spec(mount: dict, target: str | None = None) -> str:
    return f"type=bind,source={mount['source']},target={target or mount['target']}" + (',readonly' if mount['read_only'] else '')


# Only filler bytes are read; protected originals are inspected for type/link metadata.
def validation_script(mounts: list[dict], uid: int, gids: list[int], prefix: str = '') -> str:
    q = shlex.quote
    lines = [
        'set -euo pipefail',
        f'uid={uid}', f'gids={q(" ".join(map(str, gids)))}',
        'fail() { printf "%s\\n" "$1" >&2; exit 1; }',
        'metadata() { read -r permissions links owner group rest < <(LC_ALL=C ls -Lldn -- "$1") || fail "Cannot inspect bind metadata"; }',
        'protected() { metadata "$1"; [[ "$uid" != 0 && "$owner" != "$uid" && "$permissions" != *+ && "${permissions:8:1}" != w ]] '
        '&& { [[ "${permissions:5:1}" != w ]] || [[ " $gids " != *" $group "* ]]; }; }',
    ]
    for index, mount in enumerate(mounts):
        path = f'{prefix}/inputs/{index}'
        lines.append(f'[[ -f {q(path)} || -d {q(path)} ]] || fail "Bind source must be a file or directory"')
        if not mount['hidden']:
            continue
        parent = f'{prefix}/parents/{index}'
        lines.extend([
            f'[[ -f {q(path)} ]] || fail "Hidden filler must be a regular file"',
            f'protected {q(path)} || fail "Hidden filler must be administrator-owned and non-writable"',
            '[[ "$permissions" != *w* ]] || fail "Hidden filler has write permission bits"',
            f"printf '%s\\n' 'File contents omitted for security' | cmp -s - {q(path)} || fail 'Invalid hidden filler contents'",
            f'protected {q(parent)} || fail "Hidden filler parent is writable"',
            f'[[ ! -L {q(parent + "/" + Path(mount["source"]).name)} ]] || fail "Hidden filler cannot be a symlink"',
        ])
        for number in range(mount.get('ancestor_count', 0)):
            lines.append(f'protected {q(f"{prefix}/ancestors/{index}/{number}")} || fail "Hidden filler ancestor is writable"')
        base = f'{prefix}/originals/{index}'
        for part in mount['relative_target'].split('/'):
            base += '/' + part
            lines.append(f'[[ ! -L {q(base)} ]] || fail "Hidden target cannot traverse symlinks"')
        lines.extend([
            f'[[ -f {q(base)} ]] || fail "Hidden target must be a regular file"',
            f'metadata {q(base)}',
            '[[ "$links" == 1 ]] || fail "Hidden target has hard-link aliases"',
        ])
    return '\n'.join([*lines, "printf 'validated\\n'"])


def validate_daemon_sources(docker: str, image: str, mounts: list[dict], run) -> None:
    if not mounts:
        return
    argv = [docker, 'run', '--rm', '--network=none', '--read-only', '--cap-drop=ALL']
    validation_mounts = [dict(m) for m in mounts]
    for index, mount in enumerate(validation_mounts):
        argv += ['--mount', bind_spec(dict(mount, read_only=True), f'/inputs/{index}')]
        if mount['hidden']:
            for source, target in ((str(PurePosixPath(mount['source']).parent), f'/parents/{index}'),
                                   (mount['parent_source'], f'/originals/{index}')):
                argv += ['--mount', bind_spec(dict(source=source, target=target, read_only=True))]
            ancestors = list(Path(mount['source']).parents)
            mount['ancestor_count'] = len(ancestors)
            for number, parent in enumerate(ancestors):
                argv += ['--mount', bind_spec(dict(source=str(parent), target=f'/ancestors/{index}/{number}', read_only=True))]
    uid = os.geteuid() if hasattr(os, 'geteuid') else -1
    gids = list(os.getgroups()) if hasattr(os, 'getgroups') else []
    if any(m['hidden'] for m in mounts) and uid < 0:
        raise ValueError('Hidden file mounts require a POSIX runtime identity')
    if uid >= 0 and hasattr(os, 'getegid'):
        argv += ['--user', f'{uid}:{os.getegid()}']  # windows-footgun: ok — guarded by hasattr above
        for gid in gids:
            argv += ['--group-add', str(gid)]
    argv += ['--entrypoint', 'bash', image, '-c', validation_script(validation_mounts, uid, gids)]
    result = run(argv, timeout=120)
    if result.returncode or result.stdout.strip() != 'validated':
        raise ValueError('Docker bind-source validation failed: ' + result.stderr.strip())
