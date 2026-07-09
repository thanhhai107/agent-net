"""Docker sandbox execution for NIKA troubleshooting agents."""

from agent.sandbox.config import SandboxConfig, resolve_sandbox_config
from agent.sandbox.image import ensure_sandbox_image

SANDBOX_SUPPORTED_AGENTS = (
    "local_cli.codex_cli",
    "local_cli.claude_cli",
    "sdk.codex_sdk",
    "sdk.claude_sdk",
)

__all__ = [
    "SANDBOX_SUPPORTED_AGENTS",
    "SandboxConfig",
    "ensure_sandbox_image",
    "resolve_sandbox_config",
]
