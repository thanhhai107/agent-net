"""Claude Code CLI configuration: model defaults and authentication.

NIKA drives ``claude -p`` as a subprocess.  Authentication supports:

1. **Environment API key** — ``ANTHROPIC_API_KEY`` (native Anthropic or any
   provider that accepts the standard header).
2. **Environment token + base URL** — ``ANTHROPIC_AUTH_TOKEN`` with optional
   ``ANTHROPIC_BASE_URL`` (e.g. DeepSeek's Anthropic-compatible endpoint).
3. **Claude Code login** — ``claude auth login`` OAuth session (no env vars;
   subprocess runs without ``--bare`` so the CLI can read stored credentials).

Model selection reads from env when ``-m`` / ``--model`` is not passed:

``ANTHROPIC_MODEL`` → ``CLAUDE_CODE_SUBAGENT_MODEL`` →
``ANTHROPIC_DEFAULT_SONNET_MODEL``. If none are set, pass ``-m/--model`` or configure ``.env``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

_CLAUDE_MODEL_ENV_KEYS = (
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
)
def default_claude_model() -> str:
    """Return the Claude model id from environment variables."""
    for key in _CLAUDE_MODEL_ENV_KEYS:
        if value := os.environ.get(key, "").strip():
            return value
    raise ValueError(
        "Missing Claude model: set ANTHROPIC_MODEL (or CLAUDE_CODE_SUBAGENT_MODEL / "
        "ANTHROPIC_DEFAULT_SONNET_MODEL) in .env or pass -m/--model."
    )


def resolve_claude_model(model: str | None) -> str:
    """Use *model* when set; otherwise fall back to :func:`default_claude_model`."""
    if model and model.strip():
        return model.strip()
    return default_claude_model()


def has_env_claude_credentials() -> bool:
    """True when API credentials are supplied via environment variables."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    )


def claude_cli_logged_in(*, timeout_s: float = 10.0) -> bool:
    """True when ``claude auth status`` reports an active login session."""
    if shutil.which("claude") is None:
        return False
    try:
        proc = subprocess.run(
            ["claude", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            return False
        data = json.loads(proc.stdout)
        return bool(data.get("loggedIn"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return False


def claude_credentials_available(*, check_cli_login: bool = True) -> bool:
    """True when the Claude CLI is installed and credentials are configured."""
    if shutil.which("claude") is None:
        return False
    if has_env_claude_credentials():
        return True
    if not check_cli_login:
        return False
    return claude_cli_logged_in()


def use_bare_claude_mode() -> bool:
    """Whether to pass ``--bare`` to the Claude subprocess.

    Bare mode isolates the run to environment-based API auth and skips
    keychain / OAuth reads.  Use it when credentials come from ``.env``.
    """
    return has_env_claude_credentials()


def prepare_claude_subprocess_env(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the subprocess environment for ``claude -p``.

  * Copies *base* or ``os.environ``.
  * Maps ``ANTHROPIC_AUTH_TOKEN`` → ``ANTHROPIC_API_KEY`` when the latter is unset
    (required by ``--bare`` and some third-party Anthropic-compatible APIs).
  * Forwards ``ANTHROPIC_BASE_URL`` unchanged when present.
    """
    env = dict(base if base is not None else os.environ)
    if env.get("ANTHROPIC_AUTH_TOKEN", "").strip() and not env.get("ANTHROPIC_API_KEY", "").strip():
        env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_AUTH_TOKEN"]
    return env


def describe_claude_auth() -> dict[str, Any]:
    """Summarize detected auth mode (for logging and documentation)."""
    if has_env_claude_credentials():
        mode = "env_token" if os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip() else "env_api_key"
        return {
            "mode": mode,
            "bare": True,
            "base_url": os.environ.get("ANTHROPIC_BASE_URL", "").strip() or None,
            "model_default": default_claude_model(),
        }
    if claude_cli_logged_in():
        return {
            "mode": "claude_login",
            "bare": False,
            "base_url": None,
            "model_default": default_claude_model(),
        }
    return {
        "mode": "none",
        "bare": False,
        "base_url": None,
        "model_default": default_claude_model(),
    }
