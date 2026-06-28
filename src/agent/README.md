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
├── sdk/                  # [planned] Claude / Codex SDK
│   └── agent.py
├── codex_cli/            # LangGraph + Codex CLI
├── claude_cli/           # LangGraph + Claude Code CLI
├── llm/                  # LangChain model factory (react path)
└── utils/                # MCP config, phases, loggers
```

## Agent Types

| CLI name | Orchestration | LLM access | Status |
|----------|---------------|------------|--------|
| `react` | LangGraph `StateGraph` | LangChain ReAct + `load_model()` | Implemented |
| `mock` | Hand-written two-phase flow | No LLM; fixed tool sequence | Implemented |
| `codex_cli` | LangGraph `StateGraph` | `codex exec` subprocess | Implemented |
| `claude_cli` | LangGraph `StateGraph` | `claude -p` subprocess | Implemented |
| `sdk` | TBD | Claude / Codex SDK | Planned |

## Shared Pipeline

Every agent runs **diagnosis** (Kathara MCP, `if_submit=False`) then **submission** (task MCP, `if_submit=True` → `list_avail_problems` + `submit`).

## CLI & Environment

`nika agent run` resolves options from CLI flags first, then `.env`. See [`.env.example`](../../.env.example) for a full template.

### Shared (all agents)

| Flag | Env | Required | Notes |
|------|-----|----------|-------|
| `-a` / `--agent` | `NIKA_AGENT_TYPE` | Yes | `react`, `mock`, `codex_cli`, `claude_cli` |
| `-p` / `--provider` | `NIKA_LLM_PROVIDER` | react only | `openai`, `ollama`, `deepseek` |
| `-n` / `--max-steps` | `NIKA_MAX_STEPS` | Yes | Limits ReAct steps in `react` / `mock` only |
| `-m` / `--model` | `NIKA_MODEL` | No | Overrides agent-specific model env when set |
| `--session-id` | — | No | Target session (default: current running session) |

Model resolution order: `-m` → `NIKA_MODEL` → agent-specific env (below).

### Observability (react, codex_cli, claude_cli)

LangSmith: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (default `NIKA`).

Langfuse (react only): `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`.

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

| Env | Default in `.env.example` |
|-----|-------------------------|
| `NIKA_REACT_MODEL` | `gpt-5-mini` |

```bash
# .env
NIKA_AGENT_TYPE=react
NIKA_LLM_PROVIDER=openai
NIKA_MAX_STEPS=20
NIKA_REACT_MODEL=gpt-5-mini
OPENAI_API_KEY=sk-...

nika agent run                              # all from .env
nika agent run -a react -p deepseek -m deepseek-chat -n 20
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

nika agent run -a react -p ollama -m qwen2.5:7b -n 20
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

## codex_cli

LangGraph orchestration + `codex exec` subprocess per phase. Workspace: `results/{session_id}/codex_workspace/`. MCP config written to an isolated `CODEX_HOME` (does not touch `~/.codex/`).

**Entry**: `agent.codex_cli.agent.CodexCliAgent`

**Requires**: [Codex CLI](https://github.com/openai/codex) on `PATH`. Auth via `codex login` or `OPENAI_API_KEY`.

| Flag | Env | Notes |
|------|-----|-------|
| `-m` / `--model` | `NIKA_CODEX_MODEL` | Default `gpt-5.4-mini` |
| `-e` / `--reasoning-effort` | `NIKA_CODEX_REASONING_EFFORT` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; optional |

```bash
codex login   # once

# .env
NIKA_AGENT_TYPE=codex_cli
NIKA_MAX_STEPS=20
NIKA_CODEX_MODEL=gpt-5.4-mini
# NIKA_CODEX_REASONING_EFFORT=medium

nika agent run -a codex_cli -m gpt-5.4-mini -e medium
```

---

## claude_cli

LangGraph orchestration + `claude -p` subprocess per phase. Workspace: `results/{session_id}/claude_workspace/`. MCP config: `{phase}_mcp_config.json`.

**Entry**: `agent.claude_cli.agent.ClaudeAgent`

**Requires**: [Claude Code](https://docs.anthropic.com/en/docs/claude-code) on `PATH`.

**Auth** (pick one):

| Mode | Setup |
|------|-------|
| Anthropic API | `ANTHROPIC_API_KEY` |
| Compatible proxy | `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` |
| OAuth | `claude auth login` |

When credentials come from env vars, NIKA runs `claude` with `--bare`. With OAuth only, keychain credentials are used.

**Model** (when `-m` omitted, first non-empty wins):

1. `ANTHROPIC_MODEL`
2. `CLAUDE_CODE_SUBAGENT_MODEL`
3. `ANTHROPIC_DEFAULT_SONNET_MODEL`

If none are set, pass `-m` or configure `.env`.

```bash
# .env — DeepSeek via Anthropic-compatible API
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_AUTH_TOKEN=sk-...
ANTHROPIC_MODEL=deepseek-v4-pro[1m]

NIKA_AGENT_TYPE=claude_cli
NIKA_MAX_STEPS=20

nika agent run -a claude_cli
nika agent run -a claude_cli -m deepseek-v4-flash
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
nika agent run -a codex_cli -m gpt-5.4-mini
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
