"""Ownership and lifecycle for opt-in, deployment/profile Docker terminal containers."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import threading

from hermes_constants import get_hermes_home, profile_name_for_home
from tools.environments.docker_profile_firewall import apply_firewall, firewall_config
from tools.terminal_scope import terminal_env


def profile_scope() -> dict | None:
    mode = terminal_env('TERMINAL_ISOLATION_SCOPE', 'default')
    if mode not in {'default', 'profile'}:
        raise ValueError('terminal.isolation_scope must be default or profile')
    if mode != 'profile' or terminal_env('TERMINAL_ENV', 'local') != 'docker':
        return None
    if terminal_env('TERMINAL_CONTAINER_PERSISTENT', 'true').lower() != 'true':
        raise ValueError('Profile Docker isolation requires container_persistent: true')
    if terminal_env('TERMINAL_DOCKER_SHARED_CONTAINER_KEY') or terminal_env('TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE', 'false').lower() == 'true':
        raise ValueError('Profile Docker isolation cannot share containers or auto-mount session workspaces')
    home = get_hermes_home().resolve()
    profile = profile_name_for_home(home) or 'default'
    name = terminal_env('TERMINAL_DOCKER_CONTAINER_NAME')
    if not name or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
        raise ValueError('Profile Docker scope requires a valid docker_container_name')
    deployment = terminal_env('TERMINAL_DOCKER_DEPLOYMENT_ID', '').strip()
    if not deployment:
        raise ValueError('Profile Docker scope requires docker_deployment_id')
    identity = hashlib.sha256(f'{deployment}\0{name}\0{home}'.encode()).hexdigest()[:16]
    return dict(profile=profile, profile_home=str(home), name=name, deployment=deployment,
                identity=identity,
                **firewall_config(),
                mask_source=terminal_env('TERMINAL_DOCKER_MASK_SOURCE', ''),
                hidden_files=json.loads(terminal_env('TERMINAL_DOCKER_HIDDEN_FILES', '[]')))


def material_key(config: dict) -> str:
    fields = {key: value for key, value in config.items()
              if key.startswith(('docker_', 'container_')) or key in {'cwd', 'host_cwd'}}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def cache_key(config: dict) -> str | None:
    scope = config.get('docker_profile_scope')
    if not scope:
        return None
    key = 'docker-profile:' + scope['identity']
    from tools import terminal_tool as tt
    with tt._env_lock:
        existing = tt._active_environments.get(key)
    if existing is not None:
        existing._hermes_profile_lifetime = config['lifetime_seconds']
    if existing is not None and getattr(existing, '_hermes_profile_config', None) != material_key(config):
        with tt._creation_locks_lock:
            lock = tt._creation_locks.setdefault(key, threading.Lock())
        with lock:
            with tt._env_lock:
                existing = tt._active_environments.get(key)
                changed = existing is not None and getattr(existing, '_hermes_profile_config', None) != material_key(config)
                if changed:
                    tt._active_environments.pop(key)
                    tt._last_activity.pop(key, None)
            if changed:
                from tools.file_tools import clear_file_ops_cache
                clear_file_ops_cache(key)
                existing.cleanup()
                existing.wait_for_cleanup()
    return key


def creation_lock(scope: dict | None):
    if scope is None:
        return nullcontext()
    from hermes_cli.active_sessions import _FileLock
    return _FileLock(Path(scope['profile_home']) / 'sandboxes' / 'docker' / f".{scope['identity']}.lock")


def connect(environment, task_label, profile_name, egress_label, network):
    scope = environment._profile_scope
    if scope is None:
        reused = environment._persist_across_processes and environment._attach_existing_container(task_label, profile_name, egress_label, network)
        if not reused:
            environment._container_id = environment._docker_run(environment.cwd)
        return reused
    from tools.environments.remote_common import run_capture
    with creation_lock(scope):
        probe = run_capture([environment._docker_exe, 'inspect', scope['name']], timeout=15)
        if probe.returncode == 0:
            current = json.loads(probe.stdout)[0]
            labels = current.get('Config', {}).get('Labels') or {}
            owned = all(labels.get(k) == environment._labels[k] for k in
                        ('hermes-profile', 'hermes-terminal-scope', 'hermes-terminal-owner',
                         'hermes-terminal-deployment'))
            if not owned:
                raise RuntimeError('Profile Docker name belongs to another owner; refusing to replace it')
            if labels.get('hermes-terminal-config-hash') == environment._labels['hermes-terminal-config-hash']:
                if not current.get('State', {}).get('Running'):
                    started = run_capture([environment._docker_exe, 'start', current['Id']], timeout=30)
                    if started.returncode:
                        detail = started.stderr.strip() or started.stdout.strip() or f'exit code {started.returncode}'
                        raise RuntimeError(f"Cannot start profile Docker container {scope['name']}: {detail}")
                apply_firewall(environment._docker_exe, current['Id'], scope)
                environment._container_id = current['Id']
                return True
            if not environment._image_available_locally():
                raise RuntimeError('Replacement image is unavailable; existing profile container was preserved')
            run_capture([environment._docker_exe, 'rm', '-f', current['Id']], timeout=30, check=True)
        elif 'No such' not in probe.stderr:
            raise RuntimeError('Cannot inspect profile Docker identity: ' + probe.stderr.strip())
        container_id = environment._docker_run(environment.cwd)
        apply_firewall(environment._docker_exe, container_id, scope)
        environment._container_id = container_id
        return False
