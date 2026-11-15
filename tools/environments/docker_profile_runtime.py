"""Docker adapter seams for profile-scoped ownership and structured mount validation."""
import hashlib
import json
import logging

from .docker_profile_mounts import bind_spec, normalize_mounts, validate_daemon_sources

logger = logging.getLogger(__name__)


def prepare_mounts(environment, volumes, image, extra_args):
    scope = environment._profile_scope
    structured = any(isinstance(m, dict) for m in volumes or [])
    if not scope and not structured:
        return volumes
    if scope is None:
        raise ValueError('Structured mounts require terminal.isolation_scope: profile')
    for arg in extra_args or []:
        if arg in {'--mount', '--volume', '--volumes-from', '--name', '--label', '-l'} or arg.startswith(
                ('--mount=', '--volume=', '--volumes-from=', '--name=', '--label=', '-v')):
            raise ValueError('Use docker_volumes instead of Docker mount/name/label extra arguments in profile mode')
    if environment._persistent is not True or environment._persist_across_processes is not True:
        raise ValueError('Profile Docker scope requires persistent filesystem and cross-process reuse')
    mounts = normalize_mounts(volumes or [], scope)
    from .docker import find_docker
    from .remote_common import run_capture
    validate_daemon_sources(find_docker() or 'docker', image, mounts, run_capture)
    environment._profile_mounts = mounts
    return mounts


def append_mount_arg(args: list[str], mount) -> None:
    if isinstance(mount, dict):
        args.extend(['--mount', bind_spec(mount)])
    elif isinstance(mount, str) and ':' in mount and mount.strip():
        args.extend(['-v', mount.strip()])
    else:
        logger.warning('Ignoring invalid Docker volume entry: %r', mount)


def apply_profile_labels(environment, image, resources, user_args, extra, *, requested):
    scope = environment._profile_scope
    if not scope:
        return
    material = dict(image=image, requested=requested, mounts=environment._profile_mounts, resources=resources,
                    user=user_args, extra=extra, env=environment._env,
                    forward_env=environment._forward_env,
                    automatic_mounts=environment._labels.get('hermes-environment', ''),
                    egress=environment._labels.get('hermes-egress', ''), cwd=environment.cwd)
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    environment._labels.update({
        'hermes-profile': scope['profile'],
        'hermes-terminal-scope': 'profile', 'hermes-terminal-owner': scope['identity'],
        'hermes-terminal-config-hash': digest,
    })


def validate_automatic_aliases(environment, args):
    """A native credential/cache mount must not re-expose an explicitly masked file."""
    if not environment._profile_scope:
        return
    from .docker import _split_volume_spec
    from .docker_profile_mounts import _check_masks
    from pathlib import PurePosixPath
    mounts = environment._profile_mounts
    declared = {m['target'] for m in mounts}
    extras = []
    for index, arg in enumerate(args[:-1]):
        if arg == '-v' and (parts := _split_volume_spec(args[index + 1])):
            if parts[1] in declared or any(m['hidden'] and PurePosixPath(m['target']).is_relative_to(parts[1]) for m in mounts):
                raise ValueError('Automatic Docker mount conflicts with an explicit or hidden target')
            extras.append(dict(source=parts[0], target=parts[1], read_only=args[index + 1].endswith(':ro'), hidden=False))
    _check_masks([*mounts, *extras])


def claims_target(args: list[str], target: str) -> bool:
    from .docker import _split_volume_spec
    for index, arg in enumerate(args[:-1]):
        value = args[index + 1]
        if arg == '-v' and (parts := _split_volume_spec(value)) and parts[1] == target:
            return True
        if arg == '--mount' and f'target={target}' in value.split(','):
            return True
    return False
