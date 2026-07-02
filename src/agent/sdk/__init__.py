"""Claude / Codex SDK agents (planned).

Direct integration with vendor SDKs, bypassing LangChain chat models:
- Anthropic SDK for Claude (`agent.claude_sdk`, planned)
- Cursor SDK (``cursor-sdk`` / ``@cursor/sdk``) for Codex (`agent.codex_sdk`, planned)

Expected layout::

    sdk/
      claude_sdk/       # Claude SDK agent (planned)
      codex_sdk/        # Codex SDK agent (planned)

CLI-based agents live under ``agent.local_cli``:
- ``agent.local_cli.claude_cli`` — Claude Code CLI subprocess workers
- ``agent.local_cli.codex_cli`` — Codex CLI subprocess workers

Both phases (diagnosis → submission) should still write to
``{session_dir}/messages.jsonl`` via ``AgentCallbackLogger`` or an SDK-specific
adapter with the same event schema.
"""

__all__: list[str] = []
