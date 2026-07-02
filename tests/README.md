# NIKA tests

Integration and pipeline tests exercise the full NIKA workflow (env deploy, failure
injection, agent run, session close, eval) against real Docker/Kathara labs.

## Agent tests (`tests/agents/`)

Each module under `tests/agents/` contains **unit tests** (no Docker) and an **integration
pipeline** on the same real scenario:

- **Scenario**: `simple_bgp`
- **Fault**: `link_down` on `pc1` / `eth0`

| Module | Agent | Unit tests | Pipeline requires |
|--------|-------|------------|-------------------|
| `test_mock.py` | `mock` | CLI config, judge env | Docker only |
| `test_codex_cli.py` | `local_cli.codex_cli` | local_cli.codex_cli CLI config | Docker + Codex + OpenAI |
| `test_claude_cli.py` | `local_cli.claude_cli` | local_cli.claude_cli CLI config | Docker + Claude CLI |
| `test_langgraph.py` | `byo.langgraph` | byo.langgraph CLI config | Docker + `DEEPSEEK_API_KEY` |
| `test_mcp_agent.py` | `byo.mcp_agent` | byo.mcp_agent CLI config | Docker + `OPENAI_API_KEY` |

Shared pipeline helpers: [`tests/integration_pipeline.py`](integration_pipeline.py)

```shell
# All agent tests (unit + pipeline; missing credentials skip pipeline only)
uv run python -m unittest discover -s tests/agents -p 'test_*.py' -v

# Unit tests only (no Docker) — examples
uv run python -m unittest tests.agents.test_mock.MockAgentConfigTest -v
uv run python -m unittest tests.agents.test_codex_cli.CodexMcpTomlTest -v
uv run python -m unittest tests.agents.test_claude_cli.ClaudeDisplayTest -v

# Full pipeline for one agent
uv run python -m unittest tests.agents.test_mcp_agent.McpAgentPipelineTest -v
```

## Mock agent (test-only)

The **mock agent** (`src/agent/mock/mock_agent.py`) is a deterministic stand-in for
LLM-backed agents. It runs a fixed two-phase MCP tool sequence (diagnosis, then
submission) without calling any LLM or requiring API keys.

Tests invoke it via the CLI registry path:

```shell
nika agent run -a mock -m mock-v1 -n 5 --session_id <id>
```

Or programmatically:

```python
from agent.registry import create_agent

agent = create_agent(
    "mock",
    session_id=session_id,
    model="mock-v1",
    max_steps=5,
)
```

### Test modules that use mock

| Module | Purpose |
|--------|---------|
| `test_pipeline.py` | Core session pipeline (env → inject → agent → close → eval) |
| `tests/agents/test_mock.py` | Mock agent — full integration with MCP validation |
| `test_benchmark_batch.py` | Batch benchmark rows without LLM cost |
| `test_benchmark_agents.py` | `MockAgentBenchmarkTest` — agent completion smoke test |

Run mock pipeline tests:

```shell
uv run python -m unittest tests.agents.test_mock -v
uv run python -m unittest tests.test_pipeline -v
uv run python -m unittest tests.test_benchmark_agents.MockAgentBenchmarkTest -v
```

Mock runs expect perfect detection/RCA scores (`detection_score == 1.0`,
`rca_accuracy == 1.0`) because the agent reads ground truth from the session.
