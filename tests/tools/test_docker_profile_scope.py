"""Profile Docker ownership, mount validation and persistent reuse through the native backend."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import threading

import pytest

from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tools.terminal_scope import install_and_reset_profile_terminal_scope
from tools.environments import docker as docker_env
from tools.environments.docker_profile import profile_scope
from tools.environments.docker_profile_mounts import normalize_mounts

@pytest.fixture
def profiles(tmp_path, monkeypatch):
    root = tmp_path / 'deployment'
    root.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    homes = []
    for name in ('alice', 'bob'):
        home = root / 'profiles' / name
        home.mkdir(parents=True)
        (home / 'config.yaml').write_text(json.dumps({'terminal': {
            'backend': 'docker', 'isolation_scope': 'profile', 'docker_container_name': f'client-one-{name}-terminal',
            'docker_deployment_id': 'client-one',
            'cwd': '/workspace', 'container_persistent': True,
        }}))
        homes.append(home)
    return homes


@contextmanager
def selected(home):
    token = set_hermes_home_override(home)
    try:
        with install_and_reset_profile_terminal_scope(home):
            yield profile_scope()
    finally:
        reset_hermes_home_override(token)


class Daemon:
    def __init__(self):
        self.containers = {}
        self.calls = []
        self.lock = threading.RLock()
        self.serial = 0

    def __call__(self, args, **kwargs):
        with self.lock:
            self.calls.append(args)
            verb = args[1]
            out, err, code = '', '', 0
            if verb == 'inspect' and '--format' in args:
                value = next((v for v in self.containers.values() if args[-1] == v['Id']), None)
                template = args[args.index('--format') + 1]
                if value and 'FinishedAt' in template:
                    out = '2000-01-01T00:00:00Z'
                elif value and 'hermes-terminal-scope' in template:
                    out = value['Config']['Labels'].get('hermes-terminal-scope', '')
            elif verb == 'ps':
                out = '\n'.join(v['Id'] for v in self.containers.values() if not v['State']['Running'])
            elif verb == 'inspect' and '--format' not in args:
                value = self.containers.get(args[2])
                if value:
                    out = json.dumps([value])
                else:
                    code, err = 1, 'Error: No such object: ' + args[2]
            elif verb == 'run' and '-d' in args:
                name = args[args.index('--name') + 1]
                assert name not in self.containers, 'duplicate profile container creation'
                self.serial += 1
                labels = {args[i + 1].split('=', 1)[0]: args[i + 1].split('=', 1)[1]
                          for i, value in enumerate(args) if value == '--label'}
                value = dict(Id=f'container-{self.serial}', Config=dict(Labels=labels), State=dict(Running=True))
                self.containers[name] = value
                out = value['Id']
            elif verb in ('start', 'stop', 'rm'):
                identity = args[-1]
                for name, value in list(self.containers.items()):
                    if identity in (name, value['Id']):
                        if verb == 'rm':
                            del self.containers[name]
                        else:
                            value['State']['Running'] = verb == 'start'
            result = subprocess.CompletedProcess(args, code, out, err)
            if code and kwargs.get('check'):
                raise subprocess.CalledProcessError(code, args, out, err)
            return result


@pytest.fixture
def daemon(monkeypatch):
    daemon = Daemon()
    from tools.environments import remote_common
    monkeypatch.setattr(remote_common, 'run_capture', daemon)
    monkeypatch.setattr(docker_env, 'run_capture', daemon)
    monkeypatch.setattr(docker_env, '_ensure_docker_available', lambda: None)
    monkeypatch.setattr(docker_env, '_image_uses_init_entrypoint', lambda *args: False)
    monkeypatch.setattr(docker_env, 'find_docker', lambda: 'docker')
    for name in ('strip_sandbox_privileges', 'reclaim_sandbox_files'):
        monkeypatch.setattr(docker_env, name, lambda *args: None)
    monkeypatch.setattr(docker_env.DockerEnvironment, 'init_session', lambda self: None)
    monkeypatch.setattr(docker_env.DockerEnvironment, '_image_available_locally', lambda self: True)
    monkeypatch.setattr(docker_env.subprocess, 'run', daemon)
    yield daemon
    assert docker_env.DockerEnvironment.wait_for_all_teardowns(timeout=5)


def environment(home, image='image-one', memory=0):
    with selected(home) as scope:
        return docker_env.DockerEnvironment(image=image, cwd='/workspace', cpu=0, memory=memory, disk=0,
            task_id='same-session-id', profile_scope=scope, persistent_filesystem=True)


def test_profile_reuse_restart_configuration_and_foreign_ownership(profiles, daemon):
    alice, bob = profiles
    first, other = environment(alice), environment(bob)
    assert first._container_id != other._container_id
    assert environment(alice)._container_id == first._container_id
    replacement = environment(alice, image='image-two')
    assert replacement._container_id != first._container_id
    assert environment(bob)._container_id == other._container_id
    assert len(daemon.containers) == 2
    labels = replacement._labels
    assert replacement._profile_scope['name'] == 'client-one-alice-terminal'
    assert labels['hermes-profile'] == 'alice'
    assert labels['hermes-terminal-scope'] == 'profile'
    assert labels['hermes-terminal-deployment'] == 'client-one'
    scope_name = replacement._profile_scope['name']
    daemon.containers[scope_name]['State']['Running'] = False
    assert environment(alice, image='image-two')._container_id == replacement._container_id
    daemon.containers[scope_name]['Config']['Labels']['hermes-terminal-owner'] = 'foreign'
    with pytest.raises(RuntimeError, match='another owner'):
        environment(alice)
    assert scope_name in daemon.containers


def test_same_container_name_cannot_cross_deployments(profiles, daemon):
    profile = profiles[0]
    first = environment(profile)
    config_path = profile / 'config.yaml'
    config = json.loads(config_path.read_text())
    config['terminal']['docker_deployment_id'] = 'client-two'
    config_path.write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match='another owner'):
        environment(profile, image='image-two')

    assert daemon.containers[first._profile_scope['name']]['Id'] == first._container_id
    assert not any(call[1] == 'rm' for call in daemon.calls)


def test_profile_restart_reports_docker_error(profiles, daemon, monkeypatch):
    from tools.environments import remote_common

    first = environment(profiles[0])
    name = first._profile_scope['name']
    daemon.containers[name]['State']['Running'] = False

    def failed_start(args, **kwargs):
        if args[1] == 'start':
            return subprocess.CompletedProcess(args, 1, '', 'network namespace is unavailable\n')
        return daemon(args, **kwargs)

    monkeypatch.setattr(remote_common, 'run_capture', failed_start)
    with pytest.raises(RuntimeError, match=f'{name}: network namespace is unavailable'):
        environment(profiles[0])
    assert daemon.containers[name]['Id'] == first._container_id


def test_simultaneous_first_use_creates_only_one_container(profiles, daemon):
    barrier = threading.Barrier(4)
    def create(_):
        barrier.wait(timeout=5)
        return environment(profiles[0])._container_id
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(create, range(4)))
    assert len(set(ids)) == 1
    assert len(daemon.containers) == 1


def test_sessions_subagents_and_profile_mounts_use_separate_keys(profiles):
    from tools import terminal_tool as terminal
    before = dict(terminal._task_env_overrides)
    try:
        keys = []
        for home in profiles:
            with selected(home):
                first = terminal._resolve_container_task_id('session-one')
                terminal.register_task_env_overrides('session-two', {'cwd': '/other', 'cwd_source': 'session'})
                assert terminal._resolve_container_task_id('session-two') == first
                terminal.register_container_alias('child', 'session-one')
                assert terminal._resolve_container_task_id('child') == first
                terminal.register_task_env_overrides('benchmark', {'docker_image': 'another-image'})
                with pytest.raises(ValueError, match='per-task'):
                    terminal._resolve_container_task_id('benchmark')
                keys.append(first)
        assert keys[0] != keys[1]
    finally:
        terminal._task_env_overrides.clear()
        terminal._task_env_overrides.update(before)


def test_mount_placeholders_and_explicit_sources(profiles):
    with selected(profiles[0]) as scope:
        own = dict(source='/host/client-one/profiles/{profile}/workspace', target='/workspace', read_only=False)
        assert normalize_mounts([own], scope)[0]['source'] == '/host/client-one/profiles/alice/workspace'
        for invalid in [own | {'source': '/host/{unknown}'}, own | {'target': '/workspace/../secret'}]:
            with pytest.raises(ValueError):
                normalize_mounts([invalid], scope)
        with pytest.raises(ValueError, match='unique'):
            normalize_mounts([own, own], scope)


def test_hidden_masks_are_ordered_and_cannot_have_unmasked_mount_aliases(profiles):
    with selected(profiles[0]) as scope:
        parent = dict(source='/host/profiles/alice', target='/profile')
        scope |= {'mask_source': '/admin/filler', 'hidden_files': ['/profile/config.yaml']}
        mounts = normalize_mounts([parent], scope)
        assert mounts[-1]['hidden'] and mounts[-1]['read_only']
        with pytest.raises(ValueError, match='explicit directory mount'):
            normalize_mounts([], scope)
        with pytest.raises(ValueError, match='second mount'):
            normalize_mounts([parent, parent | {'target': '/alternate'}], scope)
        with pytest.raises(ValueError, match='Invalid structured'):
            normalize_mounts([parent | {'hidden': True}], scope)


def test_mask_source_setting_is_profile_scoped(profiles):
    for index, home in enumerate(profiles):
        config = json.loads((home / 'config.yaml').read_text())
        config['terminal']['docker_mask_source'] = f'/admin/filler-{index}'
        config['terminal']['docker_hidden_files'] = ['/profile/config.yaml']
        (home / 'config.yaml').write_text(json.dumps(config))
        with selected(home) as scope:
            parent = dict(source=f'/host/profiles/{home.name}', target='/profile', read_only=True)
            mounts = normalize_mounts([parent], scope)
            assert mounts[-1]['source'] == f'/admin/filler-{index}'
            with pytest.raises(ValueError, match='docker_mask_source'):
                normalize_mounts([parent], scope | {'mask_source': ''})
            for entry in ({'target': '/data'},
                          {'target': '/data', 'source': ''}, {'target': '/data', 'source': None}):
                with pytest.raises(ValueError, match='explicit source'):
                    normalize_mounts([entry], scope | {'hidden_files': []})


def test_live_mount_change_evicts_only_own_native_environment(profiles, monkeypatch):
    from tools import terminal_tool as terminal
    from tools.registry import registry
    from tools import file_tools
    from tools.terminal_tool_backends import _ENV_BUILDERS
    from tools.environments.local import LocalEnvironment
    from tools.terminal_tool_lifecycle import cleanup_all_environments
    monkeypatch.setattr(terminal, '_terminal_config_bridge_attempted', True)
    class TestEnvironment(LocalEnvironment):
        def wait_for_cleanup(self):
            return True
    monkeypatch.setitem(_ENV_BUILDERS, 'docker', lambda **kw: TestEnvironment(cwd=kw['cwd'], timeout=kw['timeout']))
    monkeypatch.setattr(terminal, '_maybe_reap_docker_orphans', lambda config: None)
    monkeypatch.setattr(terminal, '_start_cleanup_thread', lambda: None)
    for home in profiles:
        workspace = home / 'workspace'
        workspace.mkdir()
        cfg = json.loads((home / 'config.yaml').read_text())
        cfg['terminal'].update(cwd=str(workspace), docker_volumes=[])
        (home / 'config.yaml').write_text(json.dumps(cfg))
    try:
        environments = []
        for home in profiles:
            with selected(home):
                result = json.loads(registry.dispatch('write_file', {'path': 'note.txt', 'content': home.name}, task_id='same-id'))
                assert 'error' not in result
                environments.append(terminal._active_environments[terminal._resolve_container_task_id('same-id')])
        cfg = json.loads((profiles[0] / 'config.yaml').read_text())
        cfg['terminal']['docker_volumes'] = [{'source': '/host/profiles/alice/workspace', 'target': '/new-workspace'}]
        (profiles[0] / 'config.yaml').write_text(json.dumps(cfg))
        with selected(profiles[0]):
            result = json.loads(registry.dispatch('read_file', {'path': 'note.txt'}, task_id='same-id'))
            assert 'alice' in str(result)
            current = terminal._active_environments[terminal._resolve_container_task_id('same-id')]
            assert current is not environments[0]
        with selected(profiles[1]):
            current = terminal._active_environments[terminal._resolve_container_task_id('same-id')]
            assert current is environments[1]
            assert 'bob' in json.loads(registry.dispatch('read_file', {'path': 'note.txt'}, task_id='same-id'))['content']
    finally:
        cleanup_all_environments()
        file_tools.clear_file_ops_cache()


def test_structured_mounts_keep_command_guard_host_access_checks():
    from tools.terminal_tool import _docker_has_host_access
    assert _docker_has_host_access({'env_type': 'docker', 'docker_volumes': [
        {'source': '/host/workspace', 'target': '/workspace', 'read_only': False}]})


def test_prompt_probe_and_idle_reaping_keep_the_profile_container(profiles, daemon):
    from tools import terminal_tool
    from tools.terminal_tool_backends import _create_environment
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    first = environment(profiles[0])
    identity = first._container_id
    with selected(profiles[0]) as scope:
        probe = _create_environment(env_type='docker', image='image-one', cwd='/workspace', timeout=30,
            container_config={'docker_profile_scope': scope, 'container_cpu': 0, 'container_memory': 0,
                              'container_disk': 0, 'container_persistent': True},
            task_id='prompt-backend-probe', probe_only=True)
        assert probe._container_id == identity
        probe.cleanup(force_remove=True)
        assert probe.wait_for_cleanup(timeout=5)
    assert len(daemon.containers) == 1
    terminal_tool._active_environments[first._task_id] = first
    terminal_tool._last_activity[first._task_id] = 0
    _cleanup_inactive_envs()
    assert first._task_id not in terminal_tool._active_environments
    assert next(iter(daemon.containers.values()))['State']['Running']
    assert not any(call[1] == 'stop' for call in daemon.calls)
    daemon(['docker', 'stop', identity])
    assert docker_env.reap_orphan_containers(max_age_seconds=0, docker_exe='docker') == 0
    assert environment(profiles[0])._container_id == identity


@pytest.mark.parametrize('explicit_home', [False, True])
def test_profile_uses_configured_mounts_only(profiles, daemon, explicit_home):
    home = '/opt/data/profiles/alice'
    workspace = f'{home}/workspace'
    with selected(profiles[0]) as scope:
        env = docker_env.DockerEnvironment(
            image='image-one', cwd=workspace, profile_scope=scope, persistent_filesystem=True,
            env={'HOME': home} if explicit_home else None,
            volumes=[dict(source='/host/profiles/alice/workspace', target=workspace)],
            extra_args=['--network=container:client-one-vpn'],
        )
    actual = next(args for args in daemon.calls if args[1] == 'run' and '-d' in args)
    if explicit_home:
        assert 'HOME' in actual
    assert env._home_dir is None
    assert not any('target=/root' in arg for arg in actual)
    assert env._workspace_dir is None
    assert f'type=bind,source=/host/profiles/alice/workspace,target={workspace}' in actual
    assert not any('target=/workspace' in arg for arg in actual)
    assert '--read-only' not in actual
    assert '--network=container:client-one-vpn' in actual
    assert '--network=none' not in actual


def test_profile_uses_only_configured_binds_and_masks(profiles, daemon, monkeypatch):
    from tools import credential_files

    profile = profiles[0]
    target = str(profile)
    mounts = [dict(source='/host/alice', target=target, read_only=True)]

    def unexpected_mounts():
        pytest.fail('Profile Docker uses explicit mounts')

    monkeypatch.setattr(docker_env, '_readonly_skill_mount_args', unexpected_mounts)
    with selected(profile) as scope:
        scope.update(mask_source='/admin/filler', hidden_files=[f'{target}/auth.json'])
        docker_env.DockerEnvironment(image='image-one', cwd=target, profile_scope=scope,
                                     persistent_filesystem=True, volumes=mounts)
        attachment = str(profile / 'attachments' / 'one.pdf')
        assert credential_files.to_agent_visible_cache_path(attachment) == attachment
        assert credential_files.from_agent_visible_cache_path(attachment) == attachment

    creates = [args for args in daemon.calls if args[1] == 'run' and '--mount' in args]
    assert len(creates) == 1 and '-d' in creates[0]
    assert f'type=bind,source=/host/alice,target={target},readonly' in creates[0]
    assert f'type=bind,source=/admin/filler,target={target}/auth.json,readonly' in creates[0]


def test_resource_setting_change_replaces_only_its_profile(profiles, daemon, monkeypatch):
    monkeypatch.setattr(docker_env, '_cgroup_limits_available', lambda image: False)
    alice, bob = environment(profiles[0]), environment(profiles[1])
    changed = environment(profiles[0], memory=1024)
    assert changed._container_id != alice._container_id
    assert environment(profiles[1])._container_id == bob._container_id
