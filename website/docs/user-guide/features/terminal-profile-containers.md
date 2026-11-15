---
title: Persistent terminal containers per profile
---

The native Docker terminal can keep one container per deployment and profile. All native
terminal, file, process and media tools continue to use the existing runtime.

Enable it in each profile's `config.yaml`:

```yaml
terminal:
  backend: docker
  isolation_scope: profile
  docker_container_name: prompower-terminal-alice
  container_persistent: true
  docker_persist_across_processes: true
  docker_image: nousresearch/hermes-sandbox:desktop
  cwd: /workspace
  docker_profiles_root: /opt/docker-volumes/prompower/hermes/profiles
  docker_volumes:
    - source: /opt/docker-volumes/prompower/hermes/profiles/{profile}/workspace
      target: /workspace
      read_only: false
```

`docker_container_name` is the exact Docker container name. Set a unique name for each
profile and deployment in your configuration template. Hermes validates Docker name syntax
but does not normalize or append anything to the name. Ownership labels based on the name
and canonical profile home, plus a material-configuration digest, are checked before reuse.
A foreign container occupying the name is never removed automatically.

Chats, subagents and prompt probes reuse the profile container, including after a gateway
restart. A filesystem lock serializes first use across processes sharing the profile home.
Changing image, mounts, network, resource limits or other material Docker settings replaces
only that profile's container. Images are checked before an existing container is removed.
Idle cleanup stops the container without deleting its filesystem; the orphan reaper preserves
profile containers so the next use can restart them. An explicit force teardown may remove it.

The default `isolation_scope: default` retains existing session/workspace behavior. Profile
scope requires persistence and cannot be combined with `docker_shared_container_key` or
automatic session-workspace mounts. Per-task image/backend overrides are refused in this mode;
change the profile configuration instead. It does not mount the entire profile home by default.

## Structured bind mounts

Mount fields are `source`, `target`, and `read_only` (default false).
Supported placeholders are `{profile}` and `{profile_home}`. Profile home means the path visible to Hermes; it is not automatically
translated to a host-daemon path. Sources must be absolute paths on the **Docker daemon host**.
Targets must be absolute Linux paths; traversal, unknown placeholders, duplicate targets and
Docker-option separators in structured paths are refused.

Sources are checked through the daemon with strict `--mount`, which refuses a missing source
instead of creating an empty directory. The short-lived checker uses the configured image's
Bash and ordinary Unix utilities; it does not require Python in the sandbox image. The complete
mount list is profile-scoped and is never copied into another profile's process environment.

A path under another `profiles/<name>` directory requires explicit administrator approval:

```yaml
terminal:
  docker_allowed_profile_sources:
    - /opt/docker-volumes/prompower/hermes/profiles/bob/workspace
  docker_volumes:
    - source: /opt/docker-volumes/prompower/hermes/profiles/alice/workspace
      target: /workspace
    - source: /opt/docker-volumes/prompower/hermes/profiles/bob/workspace
      target: /colleagues/bob
      read_only: true
```

Allowlists are exact source roots and their descendants, not profile names or wildcard patterns.
Mounting the profiles root itself is refused. Docker mount/name/label overrides in
`docker_extra_args` are refused in profile mode; declare mounts through `docker_volumes`.

## Hidden files

For an explicitly mounted profile directory, a hidden mount can cover an existing sensitive
file with an administrator-managed filler:

```yaml
terminal:
  docker_mask_source: /opt/docker-volumes/prompower/hermes/hidden-file
  docker_volumes:
    - source: /opt/docker-volumes/prompower/hermes/profiles/{profile}
      target: /profile
  docker_hidden_files:
    - /profile/config.yaml
```

`docker_mask_source` is resolved inside the active profile and used as the host source
for every path in `docker_hidden_files`. These masks are always read-only. Ordinary
`docker_volumes` entries require `source` and do not accept a `hidden` field.

The filler must contain exactly `File contents omitted for security` followed by a newline.
It and its parent hierarchy must be administrator-owned and non-writable by the Hermes runtime;
use a non-root runtime and a protected directory such as `/etc/hermes`. The filler itself must
have no write permission bits. The terminal receives the file read-only. Checks run through
the host daemon, so the source does not have to exist at the same path in the parent container.

The mask is applied after its directory mount regardless of YAML ordering. Hidden targets must
be regular files with no hard-link aliases, and their path must not traverse a symlink. Extra
mounts, including native credential/cache mounts, cannot expose the original under an unmasked
alias. The underlying sensitive file is inspected for metadata only, never read or copied into
the terminal container before the mask is applied.

This is opt-in. Deployment templates, enabled toolsets and standalone plugins are not changed
by enabling support in the Hermes codebase.
