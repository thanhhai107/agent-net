# NIKA tests

## Layout

| Directory | Purpose |
|-----------|---------|
| `tests/agents/` | Per-agent unit tests and integration pipelines |
| `tests/benchmark/` | Benchmark batch run and resume logic |
| `tests/integration/` | End-to-end session pipeline (env → inject → agent → eval) |
| `tests/service/` | Service-layer unit tests (PingMesh, MCP servers, Containerlab APIs) |
| `tests/failure_inject_verify/` | Failure injection smoke tests (Kathara + Containerlab) |
| `tests/net_env_verify/` | Network environment deploy and topology checks |
| `tests/problems/` | Problem registry unit tests |
| `tests/evaluator/` | Rule-based scoring unit tests |
| `tests/runtime/` | Runtime/backend unit tests and session index |

Shared helpers: [`integration_base.py`](integration_base.py), [`integration_pipeline.py`](integration_pipeline.py)

## Agent tests (`tests/agents/`)

Each module contains **unit tests** (no Docker) and, for LLM-backed agents, an **integration
pipeline** on `simple_bgp` / `link_down`:

| Module | Agent | Unit tests | Pipeline requires |
|--------|-------|------------|-------------------|
| `test_agent_config.py` | shared config | agent model/env resolution, judge env | — |
| `test_codex_cli.py` | `local_cli.codex_cli` | Codex TOML/display/worker config | Docker + Codex + OpenAI |
| `test_claude_cli.py` | `local_cli.claude_cli` | Claude JSON/display/auth helpers | Docker + Claude CLI |
| `test_langgraph.py` | `byo.langgraph` | — | Docker + `DEEPSEEK_API_KEY` |
| `test_mcp_agent.py` | `byo.mcp_agent` | — | Docker + `OPENAI_API_KEY` |
| `test_autogen.py` | `byo.autogen` | — | Docker + `DEEPSEEK_API_KEY` |
| `test_sade.py` | `community.sade` | SDK env + MCP adapter | Docker + `claude-agent-sdk` + Anthropic creds |
| `test_claude_sdk.py` | `sdk.claude_sdk` | SDK env + MCP adapter | Docker + `claude-agent-sdk` + Anthropic creds |
| `test_codex_sdk.py` | `sdk.codex_sdk` | auth/reasoning + MCP TOML | Docker + `openai-codex` + `~/.codex/auth.json` |
| `test_mcp_server_selection.py` | shared MCP | diagnosis server selection (`pingmesh_mcp_server`, routing/switch keywords) | — |

```shell
# All agent tests (unit + pipeline; missing credentials skip pipeline only)
uv run python -m unittest discover -s tests/agents -p 'test_*.py' -v

# Unit tests only (no Docker)
uv run python -m unittest tests.agents.test_agent_config -v
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
| `test_pipeline_kathara.py` | Kathara pipeline: env → inject → MCP → mock agent → close → eval |
| `test_pipeline_clab.py` | Containerlab min3clos pipeline (same steps) |
| `test_pingmesh.py` | PingMesh MCP snapshot: healthy mesh → inject `link_down` → faulty mesh |

```shell
uv run python -m unittest tests.integration.test_pipeline_kathara -v   # requires Docker
uv run python -m unittest tests.integration.test_pipeline_clab -v      # requires containerlab + gnmic
uv run python -m unittest tests.integration.test_pingmesh -v           # requires Docker (+ containerlab for CLab case)
```

### PingMesh integration (`test_pingmesh.py`)

Exercises the `pingmesh_mcp_server` tool (`run_pingmesh_snapshot`) on a live session:

| Test class | Scenario | Notes |
|------------|----------|-------|
| `KatharaPingMeshIntegrationTest` | `rip_small_internet_vpn` (`-s m`) | ≥ 6 endpoints (PCs + VPN/web servers); inject `link_down` on `pc1:eth0` |
| `ContainerlabPingMeshIntegrationTest` | `min3clos` | `client1` / `client2`; inject `link_down` on `leaf1:e1-1` |

Each case asserts a clean cross-endpoint mesh before fault injection, then anomalies
from the affected endpoint after injection.

```shell
# Kathara only (~1 min with lab deploy)
uv run python -m unittest tests.integration.test_pingmesh.KatharaPingMeshIntegrationTest -v

# Containerlab only
uv run python -m unittest tests.integration.test_pingmesh.ContainerlabPingMeshIntegrationTest -v
```

## Service unit tests (`tests/service/`)

| Directory / module | Purpose |
|--------------------|---------|
| `pingmesh/test_parser.py` | Ping output parsing (`loss`, RTT, unreachable) |
| `pingmesh/test_endpoints.py` | Endpoint discovery and Containerlab data-plane IP selection |
| `pingmesh/test_engine.py` | Snapshot orchestration, anomaly summary, parameter bounds |
| `mcp_server/test_pingmesh_mcp.py` | PingMesh MCP tool wiring |
| `mcp_server/test_containerlab_srl_mcp.py` | Containerlab SR Linux MCP tools |
| `containerlab/test_srl_api.py` | Containerlab SRL API helpers |

```shell
uv run python -m unittest discover -s tests/service/pingmesh -p 'test_*.py' -v
uv run python -m unittest tests.service.mcp_server.test_pingmesh_mcp -v
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

## Problem unit tests (`tests/problems/`)

Pure Python tests for problem registration (`prob_pool` auto-discovers `ProblemBase` subclasses by `root_cause_name`).

```shell
uv run --with pytest pytest tests/problems/ -v
```

## Evaluator unit tests (`tests/evaluator/`)

Pure Python tests for rule-based scoring helpers.

```shell
uv run --with pytest pytest tests/evaluator/ -v
```
