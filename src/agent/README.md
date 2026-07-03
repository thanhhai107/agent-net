# Agent Architecture

`src/agent` hosts multiple troubleshooting agent implementations for NIKA. All share the same entry contract (`protocols.TroubleshootingAgent`) and produce the same session artifacts (`messages.jsonl`, `submission.json`, etc.).

## Directory Layout

```
src/agent/
├── protocols.py          # Shared Protocol interface
├── registry.py           # Type registry and factory for `nika agent run`
├── langgraph/            # LangGraph + LangChain ReAct
│   ├── react_agent.py
│   └── phases/
├── mock/                 # Deterministic mock without an LLM
│   └── mock_agent.py
├── sdk/                  # [planned] SDK-backed agents
│   └── agent.py
├── llm/                  # LangChain model factory (react path)
└── utils/                # MCP config, phases, loggers
```

## Agent Types

| CLI name | Orchestration | LLM access | Status |
|----------|---------------|------------|--------|
| `react` | LangGraph `StateGraph` | LangChain ReAct + `load_model()` | Implemented |
| `plan-execute` | LangGraph `StateGraph` | LangChain structured planner/executor | Implemented |
| `reflexion` | LangGraph `StateGraph` | LangChain iterative reflection | Implemented |
| `mock` | Hand-written two-phase flow | No LLM; fixed tool sequence | Implemented |
| `sdk` | TBD | SDK adapter | Planned |

## Shared Pipeline

Every agent runs **diagnosis** (Kathara MCP, `if_submit=False`) then **submission** (task MCP, `if_submit=True` → `list_avail_problems` + `submit`).

## CLI & Environment

`nika agent run` resolves options from CLI flags first, then `.env`. See [`.env.example`](../../.env.example) for a full template.

### Shared (all agents)

| Flag | Env | Required | Notes |
|------|-----|----------|-------|
| `-a` / `--agent` | `NIKA_AGENT_TYPE` | Yes | `react`, `plan-execute`, `reflexion`, `mock` |
| `-b` / `--backend` | `NIKA_LLM_PROVIDER` | react only | `openai`, `ollama`, `deepseek`, `custom` |
| `-n` / `--max-steps` | `NIKA_MAX_STEPS` | Yes | Limits LangGraph worker steps |
| `-m` / `--model` | `NIKA_MODEL` | No | Overrides agent-specific model env when set |
| `--session-id` | — | No | Target session (default: current running session) |

Model resolution order: `-m` → `NIKA_MODEL` → agent-specific env (below).

### Observability

LangSmith: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (default `NIKA`).

Langfuse: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`.

---

## react

LangGraph orchestration + LangChain ReAct workers per phase.

**Entry**: `agent.langgraph.react_agent.BasicReActAgent`

**Requires**: API key for the chosen provider.

| Provider | API key / URL |
|----------|---------------|
| `openai` | `OPENAI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `ollama` | `OLLAMA_API_URL` (default `http://localhost:11434`) |
| `custom` | `CUSTOM_API_URL`, `CUSTOM_API_KEY`; Netmind is auto-detected when the URL is the Netmind gateway |

| Env | Default in `.env.example` |
|-----|-------------------------|
| `NIKA_REACT_MODEL` | `openai/gpt-oss-20b` |

```bash
# .env
NIKA_AGENT_TYPE=react
NIKA_LLM_PROVIDER=custom
NIKA_MAX_STEPS=20
NIKA_REACT_MODEL=openai/gpt-oss-20b
CUSTOM_API_URL=https://stream-netmind.viettel.vn/gateway/v1
CUSTOM_API_KEY=...

nika agent run                              # all from .env
nika agent run -a react -b deepseek -m deepseek-chat -n 20
```

### Local deployment (Ollama)

ReAct requires a tool-calling model — see [Ollama tool calling](https://github.com/ollama/ollama/blob/main/docs/capabilities/tool-calling.mdx). Install, pull, and server setup: [Ollama FAQ](https://docs.ollama.com/faq).

Common small models: `qwen2.5:7b`, `llama3.2:3b`, `llama3.1:8b`.

```bash
# .env
NIKA_AGENT_TYPE=react
NIKA_LLM_PROVIDER=ollama
NIKA_MAX_STEPS=20
NIKA_REACT_MODEL=qwen2.5:7b
OLLAMA_API_URL=http://localhost:11434

nika agent run -a react -b ollama -m qwen2.5:7b -n 20
```

No API key. `load_model()` validates the model at init — run `ollama pull` first. For a remote host, set `OLLAMA_API_URL` to the server base URL.

---

## mock

Fixed MCP tool sequence; no LLM. For CI and integration tests.

**Entry**: `agent.mock.mock_agent.MockAgent`

`-n` is accepted but does not change behaviour.

| Env | Default in `.env.example` |
|-----|-------------------------|
| `NIKA_MOCK_MODEL` | `mock-v1` |

```bash
# .env
NIKA_AGENT_TYPE=mock
NIKA_MAX_STEPS=5
NIKA_MOCK_MODEL=mock-v1

nika agent run -a mock -n 5
```

---

## sdk (planned)

**Entry**: `agent.sdk.agent.SdkAgent` — not implemented.

```bash
# nika agent run -a sdk   # raises ValueError
```

---

## Example Workflow

```bash
nika env run simple_bgp
nika failure inject link_down --set host_name=pc1 --set intf_name=eth0
nika agent run -a react -b custom -m openai/gpt-oss-20b
nika session close -y
nika eval metrics
```

See the root [README.md](../../README.md#troubleshooting-agents) for a longer walkthrough.

## Adding a New Agent

1. Implement `async def run(task_description) -> dict` in a subpackage.
2. Register in `agent.registry.create_agent()`.
3. Write events to `{session_dir}/messages.jsonl` via `MessageLogger` or `AgentCallbackLogger`.

## CLI Reference

```bash
nika agent list          # agent types, LLM providers, reasoning-effort levels
nika agent run [options] # dispatch via nika/workflows/agent/run.py → registry.create_agent()
```
