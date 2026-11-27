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
  docker_deployment_id: prompower
  container_persistent: true
  docker_persist_across_processes: true
  docker_image: nousresearch/hermes-sandbox:desktop
  cwd: /workspace
  docker_volumes:
    - source: /opt/docker-volumes/prompower/hermes/profiles/{profile}/workspace
      target: /workspace
      read_only: false
```

`docker_container_name` is the exact Docker container name. Set a unique name for each
profile and deployment in your configuration template. `docker_deployment_id` identifies the
installation on the Docker host. Use the same stable ID across that installation's profiles
and give each installation its own ID. Hermes checks the deployment label, profile ownership
and material configuration before reuse.

Chats, subagents and prompt probes reuse the profile container, including after a gateway
restart. A filesystem lock serializes first use across processes sharing the profile home.
Changing image, mounts, network, resource limits or other material Docker settings replaces
only that profile's container. Images are checked before an existing container is removed.
Idle cleanup releases the cached gateway handle. The container stays available to other
sessions and processes. The orphan reaper preserves it, and an explicit force teardown may
remove it.

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

In profile mode, Docker uses the bind mounts declared in `docker_volumes`. Mount paths for
skills, caches, and attachments should match the paths used by the Hermes process if the
terminal needs to access them. Set `docker_env.HOME` if the shell should use the mounted
profile path as its home directory. Provide bind mounts through `docker_volumes`, not
`docker_extra_args`.

## Hidden files

Hidden mounts use an administrator-managed filler. Set hidden paths inside an explicit
directory mount:

```yaml
terminal:
  docker_mask_source: /opt/docker-volumes/prompower/hermes/hidden-file
  docker_volumes:
    - source: /opt/docker-volumes/prompower/hermes/profiles/{profile}
      target: /profile
      read_only: true
  docker_hidden_files:
    - /profile/config.yaml
```

`docker_mask_source` is resolved inside the active profile and used as the host source
for every path in `docker_hidden_files`. These masks are always read-only. Ordinary
`docker_volumes` entries require `source` and do not accept a `hidden` field.

The filler should be a read-only file prepared by the deployment on the Docker host.
Hermes passes the configured masks to Docker after the directory mounts.

This is opt-in. Deployment templates, enabled toolsets and standalone plugins are not changed
by enabling support in the Hermes codebase.
