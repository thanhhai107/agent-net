"""Sandbox execution configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from nika.config import _REPO_ROOT

ENV_AGENT_SANDBOX = "NIKA_AGENT_SANDBOX"
ENV_SANDBOX_IMAGE = "NIKA_SANDBOX_IMAGE"
ENV_SANDBOX_ENV_FILE = "NIKA_SANDBOX_ENV_FILE"
ENV_SANDBOX_NETWORK = "NIKA_SANDBOX_NETWORK"
ENV_SANDBOX_KEEP = "NIKA_SANDBOX_KEEP"
ENV_SANDBOX_CPUS = "NIKA_SANDBOX_CPUS"
ENV_SANDBOX_MEMORY = "NIKA_SANDBOX_MEMORY"
ENV_SANDBOX_CODEX_AUTH_FILE = "NIKA_SANDBOX_CODEX_AUTH_FILE"
ENV_SANDBOX_HTTP_PROXY = "NIKA_SANDBOX_HTTP_PROXY"
ENV_SANDBOX_HTTPS_PROXY = "NIKA_SANDBOX_HTTPS_PROXY"
ENV_SANDBOX_NO_PROXY = "NIKA_SANDBOX_NO_PROXY"
ENV_SANDBOX_AUTO_PROXY = "NIKA_SANDBOX_AUTO_PROXY"

ENV_SANDBOX_EXECUTION = "NIKA_SANDBOX_EXECUTION"
ENV_SESSION_DIR = "NIKA_SESSION_DIR"

DEFAULT_SANDBOX_IMAGE = "nika/agent-sandbox:latest"
DEFAULT_SANDBOX_NETWORK = "bridge"
DEFAULT_SANDBOX_ENV_FILE = _REPO_ROOT / ".env"
DEFAULT_SANDBOX_LOCAL_ENV_FILE = _REPO_ROOT / ".env.sandbox.local"
DEFAULT_CLASH_HTTP_PROXY = "http://127.0.0.1:7890"
SANDBOX_NETWORK_HOST = "host"
SANDBOX_GATEWAY_HOST_BRIDGE = "host.docker.internal"
SANDBOX_GATEWAY_HOST_HOSTNET = "127.0.0.1"


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool
    image: str
    env_file: Path
    network: str
    keep_container: bool
    cpus: str | None
    memory: str | None
    codex_auth_file: Path | None
    http_proxy: str | None
    https_proxy: str | None
    no_proxy: str | None


def sandbox_uses_host_network(network: str) -> bool:
    return network.strip().lower() == SANDBOX_NETWORK_HOST


def sandbox_gateway_agent_host(network: str) -> str:
    """Return the MCP gateway hostname reachable from the sandbox container."""
    if sandbox_uses_host_network(network):
        return SANDBOX_GATEWAY_HOST_HOSTNET
    return SANDBOX_GATEWAY_HOST_BRIDGE


def load_sandbox_env_values(*paths: Path) -> dict[str, str]:
    """Merge key/value pairs from optional sandbox env files (later paths win)."""
    from dotenv import dotenv_values

    merged: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        merged.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    return merged


def sandbox_local_env_file() -> Path:
    return DEFAULT_SANDBOX_LOCAL_ENV_FILE


def _env_bool_value(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _sandbox_auto_proxy_enabled(values: dict[str, str]) -> bool:
    for key in (ENV_SANDBOX_AUTO_PROXY,):
        if _env_bool_value(os.environ.get(key)) or _env_bool_value(values.get(key)):
            return True
    return False


def _clash_proxy_reachable(proxy_url: str) -> bool:
    """Best-effort check that a local Clash mixed-port accepts connections."""
    from urllib.parse import urlparse

    parsed = urlparse(proxy_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 7890
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def resolve_sandbox_proxy(
    *,
    network: str,
    env_file: Path,
    local_env_file: Path | None = None,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
    no_proxy: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve optional outbound proxy env for sandbox containers.

    Proxy is **off by default**. Enable it only when sandbox containers cannot
    reach LLM API endpoints directly (e.g. OpenAI, Anthropic). Set
    ``NIKA_SANDBOX_HTTP_PROXY`` / ``NIKA_SANDBOX_HTTPS_PROXY`` in the
    gitignored ``.env.sandbox.local``, or ``NIKA_SANDBOX_AUTO_PROXY=true`` to
    auto-detect a local Clash mixed port on host network.
    """
    local_path = local_env_file or sandbox_local_env_file()
    file_values = load_sandbox_env_values(env_file, local_path)

    def _from_sources(*keys: str) -> str | None:
        for key in keys:
            for source in (os.environ, file_values):
                value = str(source.get(key, "")).strip()
                if value:
                    return value
        return None

    resolved_http = (http_proxy or "").strip() or _from_sources(ENV_SANDBOX_HTTP_PROXY)
    resolved_https = (https_proxy or "").strip() or _from_sources(
        ENV_SANDBOX_HTTPS_PROXY
    )
    resolved_no = (no_proxy or "").strip() or _from_sources(ENV_SANDBOX_NO_PROXY)

    if (
        _sandbox_auto_proxy_enabled(file_values)
        and sandbox_uses_host_network(network)
        and not resolved_http
        and not resolved_https
        and _clash_proxy_reachable(DEFAULT_CLASH_HTTP_PROXY)
    ):
        resolved_http = DEFAULT_CLASH_HTTP_PROXY
        resolved_https = DEFAULT_CLASH_HTTP_PROXY

    if not resolved_no:
        resolved_no = "localhost,127.0.0.1,host.docker.internal"

    return resolved_http, resolved_https, resolved_no


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def resolve_sandbox_config(
    *,
    enabled: bool | None = None,
    image: str | None = None,
    env_file: str | Path | None = None,
    network: str | None = None,
    keep_container: bool | None = None,
    cpus: str | None = None,
    memory: str | None = None,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
    no_proxy: str | None = None,
) -> SandboxConfig:
    """Resolve sandbox settings from CLI flags and environment."""
    resolved_enabled = enabled if enabled is not None else _env_bool(ENV_AGENT_SANDBOX)
    resolved_image = (
        image or os.environ.get(ENV_SANDBOX_IMAGE, "").strip() or DEFAULT_SANDBOX_IMAGE
    )
    env_path_raw = env_file or os.environ.get(ENV_SANDBOX_ENV_FILE, "").strip()
    resolved_env_file = Path(env_path_raw) if env_path_raw else DEFAULT_SANDBOX_ENV_FILE
    if not resolved_env_file.is_absolute():
        resolved_env_file = (_REPO_ROOT / resolved_env_file).resolve()

    local_env_values = load_sandbox_env_values(sandbox_local_env_file())
    resolved_network = (
        network
        or os.environ.get(ENV_SANDBOX_NETWORK, "").strip()
        or str(local_env_values.get(ENV_SANDBOX_NETWORK, "")).strip()
        or DEFAULT_SANDBOX_NETWORK
    )
    resolved_keep = (
        keep_container if keep_container is not None else _env_bool(ENV_SANDBOX_KEEP)
    )
    resolved_cpus = cpus or os.environ.get(ENV_SANDBOX_CPUS, "").strip() or None
    resolved_memory = memory or os.environ.get(ENV_SANDBOX_MEMORY, "").strip() or None

    codex_auth_raw = os.environ.get(ENV_SANDBOX_CODEX_AUTH_FILE, "").strip()
    codex_auth_file: Path | None = None
    if codex_auth_raw:
        codex_auth_file = Path(codex_auth_raw).expanduser().resolve()
    else:
        default_auth = Path.home() / ".codex" / "auth.json"
        if default_auth.is_file():
            codex_auth_file = default_auth

    proxy_http, proxy_https, proxy_no = resolve_sandbox_proxy(
        network=resolved_network,
        env_file=resolved_env_file,
        local_env_file=sandbox_local_env_file(),
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        no_proxy=no_proxy,
    )

    return SandboxConfig(
        enabled=resolved_enabled,
        image=resolved_image,
        env_file=resolved_env_file,
        network=resolved_network,
        keep_container=resolved_keep,
        cpus=resolved_cpus,
        memory=resolved_memory,
        codex_auth_file=codex_auth_file,
        http_proxy=proxy_http,
        https_proxy=proxy_https,
        no_proxy=proxy_no,
    )
