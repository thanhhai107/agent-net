# NIKA tests

Test layout mirrors `src/`: `tests/agent/` for `src/agent/`, `tests/nika/` for `src/nika/`.
Shared integration helpers live under `tests/support/`.

## Layout

| Directory | Maps to `src/` | Purpose |
|-----------|----------------|---------|
| `tests/agent/` | `src/agent/` | Per-agent unit tests and integration pipelines |
| `tests/nika/cli/` | `src/nika/cli/` | CLI smoke and import wiring |
| `tests/nika/workflows/` | `src/nika/workflows/` | Benchmark, eval, and end-to-end pipeline tests |
| `tests/nika/problems/` | `src/nika/problems/` | Failure injection smoke tests (Kathara + Containerlab) |
| `tests/nika/net_env/` | `src/nika/net_env/` | Network environment deploy and topology checks |
| `tests/nika/service/` | `src/nika/service/` | Service-layer unit and live API smoke tests |
| `tests/nika/runtime/` | `src/nika/runtime/` | Runtime/backend unit tests and session index |
| `tests/nika/evaluator/` | `src/nika/evaluator/` | Rule-based scoring unit tests |
| `tests/support/` | — | Shared bases, prerequisites, and pipeline helpers |

## Shared support (`tests/support/`)

| Module | Purpose |
|--------|---------|
| `integration_base.py` | Session/env/failure workflow test bases |
| `integration_pipeline.py` | Ordered agent pipeline steps and credential probes |
| `prerequisites.py` | Docker, Containerlab, and image availability checks |
| `api_smoke.py` | Live API smoke mixin and JSON assertions |
| `net_env.py` | `verify_lab` assertion helpers |
| `kathara_api_base.py` | Shared Kathara API smoke test base class |

## Agent tests (`tests/agent/`)

Each module contains **unit tests** (no Docker) and, for LLM-backed agents, an **integration
pipeline** on `simple_bgp` / `link_down`:

| Module | Agent | Unit tests | Pipeline requires |
|--------|-------|------------|-------------------|
| `test_agent_config.py` | shared config | workflow/provider/model resolution | — |
| `test_langgraph.py` | `react` | upstream ReAct integration | Docker + configured LLM API |
| `test_mcp_server_selection.py` | shared MCP | diagnosis server selection | — |

```shell
# All agent tests (unit + pipeline; missing credentials skip pipeline only)
uv run python -m unittest discover -s tests/agent -p 'test_*.py' -v

# Unit tests only (no Docker)
uv run python -m unittest tests.agent.test_agent_config -v
```

## Workflow tests (`tests/nika/workflows/`)

### Benchmark (`benchmark/`)

| Module | Purpose |
|--------|---------|
| `test_batch.py` | Parallel `nika benchmark run --batch-size N` with mock agent (Docker) |
| `test_resume.py` | Resume/fingerprint unit tests (no Docker) |

```shell
uv run python -m unittest tests.nika.workflows.benchmark.test_resume -v
uv run python -m unittest tests.nika.workflows.benchmark.test_batch -v   # requires Docker
```

### Eval (`eval/`)

| Module | Purpose |
|--------|---------|
| `test_clean.py` | `remove_session_results` artifact cleanup |
| `test_session_batch.py` | Batch eval session iteration and LLM judge wiring |

### Integration pipeline (`integration/`)

| Module | Purpose |
|--------|---------|
| `test_pipeline_kathara.py` | Kathara pipeline: env → inject → MCP → mock agent → close → eval |
| `test_pipeline_clab.py` | Containerlab min3clos pipeline (same steps) |

```shell
uv run python -m unittest tests.nika.workflows.integration.test_pipeline_kathara -v   # requires Docker
uv run python -m unittest tests.nika.workflows.integration.test_pipeline_clab -v      # requires containerlab + gnmic
```

## Service tests (`tests/nika/service/`)

MCP gateway wiring and MCP server tool delegation are **not** unit-tested in isolation;
they are covered by the workflow integration pipelines (`test_pipeline_kathara`,
`test_pipeline_clab`, agent pipelines) and live API smokes where applicable.

| Directory / module | Purpose |
|--------------------|---------|
| `pingmesh/test_parser.py` | Ping output parsing (`loss`, RTT, unreachable) |
| `pingmesh/test_endpoints.py` | Endpoint discovery and Containerlab data-plane IP selection |
| `pingmesh/test_engine.py` | Snapshot orchestration, anomaly summary, parameter bounds |
| `pingmesh/test_integration.py` | Live PingMesh MCP: healthy mesh → inject `link_down` → faulty mesh |
| `kathara/test_kathara_*.py` | Live Kathara API smoke tests (Docker) |
| `containerlab/test_containerlab_api.py` | Live Containerlab API smoke tests |
| `containerlab/test_srl_api.py` | SRL API parsing/script logic (mocked runtime) |

```shell
uv run python -m unittest discover -s tests/nika/service/pingmesh -p 'test_*.py' -v
uv run python -m unittest tests.nika.service.pingmesh.test_integration.KatharaPingMeshIntegrationTest -v
```

## Problem injection (`tests/nika/problems/`)

| Module | Backend |
|--------|---------|
| `test_kathara_failure_inject.py` | Kathara (Docker) |
| `test_clab_failure_inject.py` | Containerlab (skipped without `clab` on PATH) |

## Network environment (`tests/nika/net_env/`)

Deploy and `verify_lab` checks for Kathara and Containerlab scenarios.

## Runtime unit tests (`tests/nika/runtime/`)

Pure Python tests (no Docker): backend resolution, session index, system logger, etc.

```shell
uv run --with pytest pytest tests/nika/runtime/ -v
uv run python -m unittest tests.nika.runtime.test_session_index -v
```

## Evaluator unit tests (`tests/nika/evaluator/`)

```shell
uv run --with pytest pytest tests/nika/evaluator/ -v
```

## Mock agent (test-only)

The **mock agent** (`src/agent/mock/mock_agent.py`) is a deterministic stand-in for
LLM-backed agents. It runs a fixed two-phase MCP tool sequence without API keys.

```shell
nika agent run -a mock -m mock-v1 -n 5 --session_id <id>
```

Mock runs expect perfect detection/RCA scores (`detection_score == 1.0`,
`rca_accuracy == 1.0`) because the agent reads ground truth from the session.
