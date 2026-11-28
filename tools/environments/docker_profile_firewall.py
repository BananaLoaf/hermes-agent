"""Install opt-in egress rules in a profile sandbox's own network namespace."""

from ipaddress import IPv4Network, collapse_addresses
import json
import subprocess

from tools.terminal_scope import terminal_env
from tools.environments.remote_common import run_capture


def firewall_config() -> dict:
    image = terminal_env('TERMINAL_DOCKER_FIREWALL_IMAGE', '').strip()
    denied = json.loads(terminal_env('TERMINAL_DOCKER_DENIED_CIDRS', '[]'))
    if not isinstance(denied, list) or any(not isinstance(item, str) for item in denied):
        raise ValueError('terminal.docker_denied_cidrs must be a list of IPv4 addresses or CIDRs')
    if not image:
        if denied:
            raise ValueError('terminal.docker_denied_cidrs requires docker_firewall_image')
        return {}
    networks = collapse_addresses(IPv4Network(item, strict=False) for item in denied)
    return {'firewall': {'image': image, 'denied': [str(network) for network in networks]}}


def apply_firewall(docker: str, container_id: str, scope: dict) -> None:
    firewall = scope.get('firewall')
    if firewall is None:
        return
    result = run_capture(
        [docker, 'inspect', '--format', '{{json .HostConfig}}', container_id],
        timeout=15, check=True,
    )
    config = json.loads(result.stdout)
    mode = config['NetworkMode']
    extra_caps = {cap.upper().removeprefix('CAP_') for cap in config.get('CapAdd') or []}
    if mode == 'host' or mode.startswith('container:'):
        raise RuntimeError('Profile Docker firewall requires a private network namespace')
    allowed_caps = {'DAC_OVERRIDE', 'CHOWN', 'FOWNER', 'SETUID', 'SETGID'}
    unexpected_caps = extra_caps - allowed_caps
    if config['Privileged'] or unexpected_caps:
        raise RuntimeError(
            'Profile Docker firewall requires a confined terminal container '
            f'(privileged={config["Privileged"]}, unexpected capabilities={sorted(unexpected_caps)})'
        )
    deny_rule = ''
    if firewall['denied']:
        deny_rule = 'ip daddr { ' + ', '.join(firewall['denied']) + ' } drop'
    rules = f'''
add table inet hermes_terminal
delete table inet hermes_terminal
table inet hermes_terminal {{
    chain input {{
        type filter hook input priority filter; policy drop;
        iifname "lo" accept
        ct state established,related accept
    }}
    chain output {{
        type filter hook output priority filter; policy drop;
        oifname "lo" accept
        {deny_rule}
        meta nfproto ipv4 accept
    }}
}}
'''
    # Only the short-lived installer gets NET_ADMIN. The agent sandbox keeps
    # the Docker backend's normal capability and no-new-privileges settings.
    result = subprocess.run(
        [docker, 'run', '--rm', '-i', '--pull=missing', '--user=0:0',
         '--cap-drop=ALL', '--cap-add=NET_ADMIN', '--security-opt=no-new-privileges',
         '--read-only', '--network', f'container:{container_id}', '--entrypoint=/usr/sbin/nft',
         firewall['image'], '-f', '-'],
        input=rules, capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError('Cannot apply profile Docker firewall: ' + result.stderr.strip())
