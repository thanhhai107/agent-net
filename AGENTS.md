# Agent Instructions

## Project Snapshot

- NIKA is a Python 3.12 network troubleshooting benchmark and orchestration platform.
- It uses Kathara and Docker to deploy network labs, inject failures, run troubleshooting agents, and evaluate results.
- The package uses a `src/` layout. The console entry point is `nika = nika.cli.main:main`.
- Dependencies are managed with `uv`; `.env` is loaded from the repository root by `src/nika/config.py`.

## Repository Map

- `src/nika/cli/`: Typer CLI groups for `session`, `env`, `failure`, `exec`, `agent`, `eval`, `benchmark`, and `traffic`.
- `src/nika/workflows/`: command workflows that coordinate CLI actions.
- `src/nika/orchestrator/`: task definitions and injectable problem classes.
- `src/nika/net_env/`: Network lab definitions split by backend — `kathara/` (Kathara labs) and `containerlab/` (Containerlab labs).
- `src/nika/service/`: Kathara APIs and MCP servers exposed to troubleshooting agents.
- `src/nika/generator/`: fault and traffic generators.
- `src/nika/evaluator/`: metrics, trace parsing, and LLM judge support.
- `src/agent/`: troubleshooting agent implementations and shared agent utilities.
- `benchmark/`: benchmark YAML cases and regeneration script.
- `tests/`: unit tests plus Docker/Kathara integration tests (`agents/`, `benchmark/`, `integration/`, `failure_inject_verify/`, `net_env_verify/`, `runtime/`).

## Common Commands

- Install dependencies: `uv sync`
- Run the CLI during development: `uv run nika --help`
- Run all pytest tests: `uv run --with pytest pytest`
- Run unittest agent tests: `uv run python -m unittest discover -s tests/agents -p 'test_*.py' -v`
- Run a focused unittest module: `uv run python -m unittest tests.agents.test_mock -v`
- Regenerate benchmark YAML: `uv run python benchmark/generate_benchmark.py`

## Architecture Rules

- Keep CLI parsing in `src/nika/cli/commands/`; put reusable behavior in `src/nika/workflows/` or lower-level modules.
- Session-scoped workflows should accept or resolve `session_id` consistently with the existing CLI behavior.
- Runtime state belongs under `runtime/`; experiment artifacts belong under `results/{session_id}/`.
- Relative result paths must resolve from the repository root, matching `resolve_results_root()`.
- New Kathara network scenarios belong under `src/nika/net_env/kathara/`; Containerlab scenarios under `src/nika/net_env/containerlab/`. Both must be registered in the environment pool.
- New injectable problems belong under `src/nika/orchestrator/problems/` and should expose explicit typed injection parameters.
- New agents must implement the shared troubleshooting contract, register in `agent.registry.create_agent()`, and write standard artifacts.

## Agent System Rules

- Agent implementations live under `src/agent/` and share a two-phase pipeline: diagnosis, then submission.
- Supported production agent ids include `byo.langgraph`, `byo.mcp_agent`, `byo.autogen`, `local_cli.codex_cli`, `local_cli.claude_cli`, `community.sade`, `sdk.claude_sdk`, and `sdk.codex_sdk`.
- The `mock` agent is test-only and deterministic; prefer it for no-credential pipeline tests.
- SDK agents require `uv sync --extra sdk --prerelease=allow` (`claude-agent-sdk` + `openai-codex`).
- Agent runs should write `messages.jsonl` and `submission.json` in the session result directory.
- MCP server behavior lives under `src/nika/service/mcp_server/`; avoid duplicating tool behavior in agent code.
- Shared agent skills live under `src/agent/skills/`; helpers in `agent.utils.skills`. See `docs/agent-skills.md`. SADE keeps its own skill library under `src/agent/community/sade/.claude/`.

## Testing Guidance

- Prefer focused tests near the changed behavior. Many integration tests require Docker, Kathara, CLIs, or API credentials.
- For pure Python changes, use targeted `pytest` or `unittest` commands before broader suites.
- For CLI behavior, test through the CLI path when practical because config resolution and session selection are part of the behavior.
- For benchmark resume or artifact logic, verify against a temporary or isolated `--result_dir`.
- Do not assume Docker/Kathara integration tests can run in every environment; report skipped or unavailable checks clearly.

## Environment and Secrets

- Use `.env.example` as the template; do not commit real API keys or local credentials.
- CLI flags override `.env` values.
- Common agent variables include `NIKA_AGENT_TYPE`, `NIKA_MAX_STEPS`, `NIKA_MODEL`, and agent-specific model/provider variables.
- Claude and Codex local CLI agents require their respective CLIs on `PATH` plus configured authentication.

## Operational Cautions

- Do not delete `runtime/` or `results/` broadly unless the user explicitly asks; these may contain active sessions or experiment outputs.
- Use `nika session close` or `nika session wipe` for lab cleanup instead of manually removing Docker/Kathara state.
- Avoid changing generated benchmark YAML by hand unless the task is specifically about benchmark cases.
- Preserve existing lab config files, startup files, P4 programs, and Kubernetes manifests unless the change directly targets them.
- Network tests can be slow and environment-sensitive; keep verification commands specific and explain any external prerequisites.

