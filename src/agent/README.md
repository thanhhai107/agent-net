# Agent Architecture

`src/agent` hosts multiple troubleshooting agent implementations for NIKA. All share the same entry contract (`protocols.TroubleshootingAgent`) and produce the same session artifacts (`messages.jsonl`, `submission.json`, etc.).

## Directory Layout

```
src/agent/
├── protocols.py          # Shared Protocol interface
├── registry.py           # Type registry and factory for `nika agent run`
├── byo/                  # Bring-your-own LLM / agent framework backends
│   ├── langgraph/        # -a byo.langgraph (LangChain ReAct workers)
│   │   ├── react_agent.py
│   │   └── phases/
│   ├── mcp_agent/        # -a byo.mcp_agent
│   └── autogen/          # -a byo.autogen
├── local_cli/            # Local CLI subprocess workers
│   ├── codex_cli/        # -a local_cli.codex_cli
│   └── claude_cli/       # -a local_cli.claude_cli
├── community/            # Community-contributed agents
│   └── sade/             # -a community.sade
├── mock/                 # Test-only deterministic agent (see tests/README.md)
│   └── mock_agent.py
├── sdk/                  # SDK agents (claude-agent-sdk, openai-codex)
│   ├── claude_sdk/       # -a sdk.claude_sdk
│   └── codex_sdk/        # -a sdk.codex_sdk
├── llm/                  # LangChain model factory (langgraph path)
└── utils/                # MCP config, phases, loggers
```

## Agent Types

| CLI name | Orchestration | LLM access | Status |
|----------|---------------|------------|--------|
| `byo.langgraph` | LangGraph `StateGraph` | LangChain ReAct + `load_model()` | Implemented |
| `local_cli.codex_cli` | LangGraph `StateGraph` | `codex exec` subprocess | Implemented |
| `local_cli.claude_cli` | LangGraph `StateGraph` | `claude -p` subprocess | Implemented |
| `byo.mcp_agent` | mcp-agent `Workflow` | mcp-agent + OpenAI | Implemented |
| `byo.autogen` | AutoGen `GraphFlow` | AutoGen AgentChat + OpenAI | Implemented |
| `community.sade` | Single Claude Code session + 15-skill library | `claude-agent-sdk` (optional extra `sade`) | Implemented |
| `sdk.claude_sdk` | Native two-phase `ClaudeSDKClient` sessions | `claude-agent-sdk` (optional extra `sdk`) | Implemented |
| `sdk.codex_sdk` | Native two-phase `AsyncCodex` threads | `openai-codex` (optional extra `sdk`) | Implemented |

## Community Agents

Community-contributed agents live under `src/agent/community/<name>/` and implement the
same `protocols.TroubleshootingAgent` contract.

See [`community/sade/README.md`](community/sade/README.md) for SADE setup, DeepSeek
credentials, and the paper citation (arXiv:2605.04530).

## Shared Pipeline

Every agent runs **diagnosis** (Kathara MCP, `if_submit=False`) then **submission** (task MCP, `if_submit=True` → `list_avail_problems` + `submit`).

## CLI & Environment

`nika agent run` resolves options from CLI flags first, then `.env`. See [`.env.example`](../../.env.example) for a full template.

### Shared (all agents)

| Flag | Env | Required | Notes |
|------|-----|----------|-------|
| `-a` / `--agent` | `NIKA_AGENT_TYPE` | Yes | `byo.langgraph`, `byo.mcp_agent`, `byo.autogen`, `local_cli.codex_cli`, `local_cli.claude_cli`, `community.sade` |
| `-p` / `--provider` | `NIKA_LLM_PROVIDER` | byo.langgraph only | `openai`, `ollama`, `deepseek` |
| `-n` / `--max-steps` | `NIKA_MAX_STEPS` | Yes | Limits steps per phase in `byo.langgraph`, `byo.mcp_agent`, `byo.autogen`, and `community.sade` |
| `-m` / `--model` | `NIKA_MODEL` | No | Overrides agent-specific model env when set |
| `--session_id` | — | No | Target session (default: current running session) |

Model resolution order: `-m` → `NIKA_MODEL` → agent-specific env (below).

### Observability (byo.langgraph, local_cli.codex_cli, local_cli.claude_cli)

LangSmith: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (default `NIKA`).

Langfuse (byo.langgraph only): `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`.

---

## byo.langgraph

LangGraph orchestration + LangChain ReAct workers per phase.

**Entry**: `agent.byo.langgraph.react_agent.BasicReActAgent`

**Requires**: API key for the chosen provider.

| Provider | API key / URL |
|----------|---------------|
| `openai` | `OPENAI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `ollama` | `OLLAMA_API_URL` (default `http://localhost:11434`) |

| Env | Default in `.env.example` |
|-----|-------------------------|
| `NIKA_LANGGRAPH_MODEL` | `gpt-5-mini` |

```bash
# .env
NIKA_AGENT_TYPE=byo.langgraph
NIKA_LLM_PROVIDER=openai
NIKA_MAX_STEPS=20
NIKA_LANGGRAPH_MODEL=gpt-5-mini
OPENAI_API_KEY=sk-...

nika agent run                              # all from .env
nika agent run -a byo.langgraph -p deepseek -m deepseek-chat -n 20
```

### Local deployment (Ollama)

Requires a tool-calling model — see [Ollama tool calling](https://github.com/ollama/ollama/blob/main/docs/capabilities/tool-calling.mdx). Install, pull, and server setup: [Ollama FAQ](https://docs.ollama.com/faq).

Common small models: `qwen2.5:7b`, `llama3.2:3b`, `llama3.1:8b`.

```bash
# .env
NIKA_AGENT_TYPE=byo.langgraph
NIKA_LLM_PROVIDER=ollama
NIKA_MAX_STEPS=20
NIKA_LANGGRAPH_MODEL=qwen2.5:7b
OLLAMA_API_URL=http://localhost:11434

nika agent run -a byo.langgraph -p ollama -m qwen2.5:7b -n 20
```

No API key. `load_model()` validates the model at init — run `ollama pull` first. For a remote host, set `OLLAMA_API_URL` to the server base URL.

---

## local_cli.codex_cli

LangGraph orchestration + `codex exec` subprocess per phase. Workspace: `results/{session_id}/codex_workspace/`. MCP config written to an isolated `CODEX_HOME` (does not touch `~/.codex/`).

**Entry**: `agent.local_cli.codex_cli.agent.CodexCliAgent`

**Requires**: [Codex CLI](https://github.com/openai/codex) on `PATH`. Auth via `codex login` or `OPENAI_API_KEY`.

| Flag | Env | Notes |
|------|-----|-------|
| `-m` / `--model` | `NIKA_CODEX_MODEL` | Default `gpt-5.4-mini` |
| `-e` / `--reasoning-effort` | `NIKA_CODEX_REASONING_EFFORT` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; optional |

```bash
codex login   # once

# .env
NIKA_AGENT_TYPE=local_cli.codex_cli
NIKA_MAX_STEPS=20
NIKA_CODEX_MODEL=gpt-5.4-mini
# NIKA_CODEX_REASONING_EFFORT=medium

nika agent run -a local_cli.codex_cli -m gpt-5.4-mini -e medium
```

---

## local_cli.claude_cli

LangGraph orchestration + `claude -p` subprocess per phase. Workspace: `results/{session_id}/claude_workspace/`. MCP config: `{phase}_mcp_config.json`.

**Entry**: `agent.local_cli.claude_cli.agent.ClaudeAgent`

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

NIKA_AGENT_TYPE=local_cli.claude_cli
NIKA_MAX_STEPS=20

nika agent run -a local_cli.claude_cli
nika agent run -a local_cli.claude_cli -m deepseek-v4-flash
```

---

## byo.mcp_agent

mcp-agent ``Workflow`` orchestration + [mcp-agent SDK](https://docs.mcp-agent.com/mcp-agent-sdk/overview) workers per phase.

**Entry**: `agent.byo.mcp_agent.agent.McpAgent`

**Requires**: `OPENAI_API_KEY`.

| Env | Default in `.env.example` |
|-----|-------------------------|
| `NIKA_MCP_AGENT_MODEL` | `gpt-4.1-mini` (use `gpt-4o-mini` if unavailable) |

```bash
# .env
NIKA_AGENT_TYPE=byo.mcp_agent
NIKA_MAX_STEPS=20
NIKA_MCP_AGENT_MODEL=gpt-4.1-mini
OPENAI_API_KEY=sk-...

nika agent run -a byo.mcp_agent -m gpt-4.1-mini -n 20
```

No LangSmith / Langfuse integration in this path (observability deferred).

---

## byo.autogen

AutoGen ``GraphFlow`` orchestration + [AutoGen AgentChat](https://microsoft.github.io/autogen/stable/) workers per phase.

**Entry**: `agent.byo.autogen.agent.AutogenAgent`

**Requires**: `OPENAI_API_KEY` for the default model. When `-m` / `NIKA_AUTOGEN_MODEL` starts with `deepseek`, uses `DEEPSEEK_API_KEY` instead.

| Env | Default in `.env.example` |
|-----|-------------------------|
| `NIKA_AUTOGEN_MODEL` | `gpt-4.1-mini` (use `gpt-4o-mini` if unavailable) |

```bash
# .env
NIKA_AGENT_TYPE=byo.autogen
NIKA_MAX_STEPS=20
NIKA_AUTOGEN_MODEL=gpt-4.1-mini
OPENAI_API_KEY=sk-...

nika agent run -a byo.autogen -m gpt-4.1-mini -n 20
```

No LangSmith / Langfuse integration in this path (observability deferred).

---

## sdk.claude_sdk

Native two-phase pipeline via ``claude-agent-sdk`` ``ClaudeSDKClient`` (no LangGraph). Each phase starts a separate SDK session with phase-specific MCP servers.

**Entry**: `agent.sdk.claude_sdk.agent.ClaudeSdkAgent`

**Requires**: `uv sync --extra sdk --prerelease=allow`

**Auth**: DeepSeek or Anthropic via env (same as `local_cli.claude_cli` option B):

```bash
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_AUTH_TOKEN=sk-...
ANTHROPIC_MODEL=deepseek-v4-pro[1m]
```

| Flag | Env | Notes |
|------|-----|-------|
| `-n` / `--max-steps` | `NIKA_MAX_STEPS` | Passed to SDK `max_turns` per phase |
| `-m` / `--model` | `NIKA_CLAUDE_SDK_MODEL` or `ANTHROPIC_MODEL` chain | |

```bash
nika agent run -a sdk.claude_sdk -n 20
nika agent run -a sdk.claude_sdk -m deepseek-v4-flash
```

---

## sdk.codex_sdk

Native two-phase pipeline via ``openai-codex`` ``AsyncCodex`` threads (no LangGraph). MCP config is written to an isolated `CODEX_HOME` per phase.

**Entry**: `agent.sdk.codex_sdk.agent.CodexSdkAgent`

**Requires**: `uv sync --extra sdk --prerelease=allow`

**Auth**: Local only — `codex login` → `~/.codex/auth.json` (does not use `OPENAI_API_KEY`).

| Flag | Env | Notes |
|------|-----|-------|
| `-m` / `--model` | `NIKA_CODEX_SDK_MODEL` or `NIKA_CODEX_MODEL` | Default `gpt-5.4-mini` |
| `-e` / `--reasoning-effort` | `NIKA_CODEX_REASONING_EFFORT` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |

```bash
codex login   # once

nika agent run -a sdk.codex_sdk -m gpt-5.4-mini -e medium
```

---

## Example Workflow

```bash
nika env run simple_bgp
nika failure inject link_down --set host_name=pc1 --set intf_name=eth0
nika agent run -a local_cli.codex_cli -m gpt-5.4-mini
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
