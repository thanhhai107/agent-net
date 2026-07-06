# NIKA tests

## Layout

| Directory | Purpose |
|-----------|---------|
| `tests/agents/` | Per-agent unit tests and integration pipelines |
| `tests/benchmark/` | Benchmark batch run and resume logic |
| `tests/integration/` | End-to-end session pipeline (env → inject → agent → eval) |
| `tests/failure_inject_verify/` | Failure injection smoke tests (Kathara + Containerlab) |
| `tests/net_env_verify/` | Network environment deploy and topology checks |
| `tests/runtime/` | Runtime/backend unit tests and session index |

Shared helpers: [`integration_base.py`](integration_base.py), [`integration_pipeline.py`](integration_pipeline.py)

## Agent tests (`tests/agents/`)

Each module contains **unit tests** (no Docker) and, for LLM-backed agents, an **integration
pipeline** on `simple_bgp` / `link_down`:

| Module | Agent | Unit tests | Pipeline requires |
|--------|-------|------------|-------------------|
| `test_mock.py` | `mock` | CLI config, judge env | — (no pipeline; see `tests/integration/`) |
| `test_codex_cli.py` | `local_cli.codex_cli` | CLI config | Docker + Codex + OpenAI |
| `test_claude_cli.py` | `local_cli.claude_cli` | CLI config | Docker + Claude CLI |
| `test_langgraph.py` | `byo.langgraph` | CLI config | Docker + `DEEPSEEK_API_KEY` |
| `test_mcp_agent.py` | `byo.mcp_agent` | CLI config | Docker + `OPENAI_API_KEY` |
| `test_autogen.py` | `byo.autogen` | CLI config | Docker + `DEEPSEEK_API_KEY` |
| `test_sade.py` | `community.sade` | config + MCP adapter | Docker + `claude-agent-sdk` + Anthropic creds |
| `test_claude_sdk.py` | `sdk.claude_sdk` | config + MCP adapter | Docker + `claude-agent-sdk` + Anthropic creds |
| `test_codex_sdk.py` | `sdk.codex_sdk` | config + MCP TOML | Docker + `openai-codex` + `~/.codex/auth.json` |

```shell
# All agent tests (unit + pipeline; missing credentials skip pipeline only)
uv run python -m unittest discover -s tests/agents -p 'test_*.py' -v

# Unit tests only (no Docker)
uv run python -m unittest tests.agents.test_mock -v
```

## Benchmark tests (`tests/benchmark/`)

| Module | Purpose |
|--------|---------|
| `test_batch.py` | Parallel `nika benchmark run --batch-size N` with mock agent (Docker) |
| `test_resume.py` | Resume/fingerprint unit tests (no Docker) |

```shell
uv run python -m unittest tests.benchmark.test_resume -v
uv run python -m unittest tests.benchmark.test_batch -v   # requires Docker
```

## Integration pipeline (`tests/integration/`)

| Module | Purpose |
|--------|---------|
| `test_pipeline.py` | Core session pipeline: env → inject → MCP → mock agent → close → eval |

```shell
uv run python -m unittest tests.integration.test_pipeline -v   # requires Docker
```

## Mock agent (test-only)

The **mock agent** (`src/agent/mock/mock_agent.py`) is a deterministic stand-in for
LLM-backed agents. It runs a fixed two-phase MCP tool sequence without API keys.

```shell
nika agent run -a mock -m mock-v1 -n 5 --session_id <id>
```

Mock runs expect perfect detection/RCA scores (`detection_score == 1.0`,
`rca_accuracy == 1.0`) because the agent reads ground truth from the session.

## Failure injection verify (`tests/failure_inject_verify/`)

| Module | Backend |
|--------|---------|
| `test_kathara_failure_inject.py` | Kathara (Docker) |
| `test_clab_failure_inject.py` | Containerlab (skipped without `clab` on PATH) |

## Runtime unit tests (`tests/runtime/`)

Pure Python tests (no Docker): backend resolution, session index, system logger, etc.

```shell
uv run --with pytest pytest tests/runtime/ -v
uv run python -m unittest tests.runtime.test_session_index -v
```
