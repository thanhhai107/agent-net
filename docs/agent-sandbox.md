# Agent Sandbox Execution

NIKA can run supported troubleshooting agents inside an isolated Docker container while the host keeps control of the network lab, MCP tools, and evaluation.

## Supported agents

Sandbox mode works with:

- `local_cli.codex_cli`
- `local_cli.claude_cli`
- `sdk.codex_sdk`
- `sdk.claude_sdk`

Other agent types continue to run on the host when `--sandbox` is not set.

## Build the sandbox image

The image is built **automatically** the first time you run `nika agent run --sandbox` (or when `NIKA_AGENT_SANDBOX=true`) if `docker image inspect` shows it is missing locally. No manual build step is required for normal use.

To rebuild manually (e.g. after changing agent code under `src/agent` or `src/nika`):

```bash
docker build --network=host -t nika/agent-sandbox:latest -f src/agent/sandbox/Dockerfile .
```

Use `--network=host` during build only when the build container cannot reach package mirrors directly (e.g. behind a restrictive network or TUN routing). If you use an outbound proxy for sandbox runs, the same proxy is passed as Docker build args during auto-build.

The image includes Python 3.12, Codex CLI, Claude CLI, SDK packages, and a minimal copy of `src/agent` + `src/nika` for the in-container runner.

## Quick start

1. Copy `.env.example` to `.env` and configure credentials for your agent.
2. Start a session and inject a fault (standard NIKA workflow).
3. Run the agent in sandbox mode:

```bash
uv run nika agent run --sandbox -a local_cli.codex_cli -m gpt-5.4-mini -n 20
```

Equivalent env-based toggle:

```bash
export NIKA_AGENT_SANDBOX=true
uv run nika agent run -a sdk.claude_sdk -m claude-sonnet-4-20250514 -n 20
```

## Architecture

| Component | Runs on |
|-----------|---------|
| Kathara / Containerlab lab | Host |
| MCP HTTP gateway + tool servers | Host |
| Agent (CLI or SDK subprocess) | Sandbox container |
| `messages.jsonl`, `submission.json` | Host `results/{session_id}/` (bind-mounted) |

The container reaches the MCP gateway at `http://host.docker.internal:{port}` (bridge, default) or `http://127.0.0.1:{port}` (host network). Submission phase transitions use `POST /gateway/sessions/{session_id}/phase`.

### Outbound proxy (optional)

**Default: no proxy.** Sandbox agents call LLM APIs directly from the container. You do not need proxy settings unless the container cannot reach those endpoints (network restrictions, regional blocks, etc.).

When proxy is required, put settings in **gitignored** `.env.sandbox.local` (not committed `.env`):

```bash
cp .env.sandbox.local.example .env.sandbox.local
# Edit the file and uncomment the proxy options you need.
```

Two opt-in modes:

1. **Explicit URL** — set `NIKA_SANDBOX_HTTP_PROXY` and `NIKA_SANDBOX_HTTPS_PROXY`.
2. **Auto-detect Clash** — set `NIKA_SANDBOX_NETWORK=host` and `NIKA_SANDBOX_AUTO_PROXY=true` so NIKA uses `http://127.0.0.1:7890` when a local Clash mixed port is reachable.

Example for restricted networks (Clash TUN / `clashui`):

```bash
NIKA_SANDBOX_NETWORK=host
NIKA_SANDBOX_AUTO_PROXY=true
```

`NIKA_SANDBOX_NO_PROXY` defaults to `localhost,127.0.0.1,host.docker.internal` so MCP traffic to the host gateway stays direct.

Verify connectivity when debugging:

```bash
curl -I https://api.openai.com          # direct
curl -x http://127.0.0.1:7890 -I https://api.openai.com   # via proxy
```

## Configuration

| Flag / env | Default | Description |
|------------|---------|-------------|
| `--sandbox` / `NIKA_AGENT_SANDBOX` | off | Enable Docker sandbox execution |
| `--sandbox-image` / `NIKA_SANDBOX_IMAGE` | `nika/agent-sandbox:latest` | Sandbox image |
| `--sandbox-env-file` / `NIKA_SANDBOX_ENV_FILE` | repo `.env` | Whitelisted credential source |
| `NIKA_SANDBOX_NETWORK` | `bridge` | Docker network (`host` only when needed for local proxy auto-detect) |
| `--sandbox-keep-container` / `NIKA_SANDBOX_KEEP` | off | Keep container after exit |
| `--sandbox-cpus` / `NIKA_SANDBOX_CPUS` | none | CPU limit |
| `--sandbox-memory` / `NIKA_SANDBOX_MEMORY` | none | Memory limit |
| `NIKA_SANDBOX_HTTP_PROXY` | off | Optional outbound HTTP proxy (`.env.sandbox.local`) |
| `NIKA_SANDBOX_HTTPS_PROXY` | off | Optional outbound HTTPS proxy |
| `NIKA_SANDBOX_AUTO_PROXY` | off | Optional: auto-use Clash `7890` on host network |
| `NIKA_SANDBOX_NO_PROXY` | localhost defaults | Bypass proxy for MCP gateway (when proxy enabled) |
| `NIKA_SANDBOX_CODEX_AUTH_FILE` | auto | Optional read-only Codex auth mount |

### Credentials

- **Codex CLI/SDK**: `OPENAI_API_KEY` from the env file is preferred. If absent, `~/.codex/auth.json` is mounted read-only when present.
- **Claude CLI/SDK**: use `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` (+ optional `ANTHROPIC_BASE_URL`) in `.env`. OAuth/keychain login is not available inside the container.

Secrets are injected via a whitelist and redacted from logs and `sandbox_manifest.json`.

## Security model

The sandbox container:

- Runs as non-root user `agent`
- Has no Docker socket mount
- Has no host home directory mount
- Only bind-mounts the session results directory and the shared skills library (read-only)
- Uses `--security-opt no-new-privileges`

MCP tools execute on the host and retain lab access; the agent cannot call Kathara, Containerlab, or Docker directly.

## Artifacts

After a sandbox run, check:

```
results/{session_id}/
├── run.json
├── sandbox_manifest.json   # agent type, model, gateway URL (no secrets)
├── messages.jsonl          # diagnosis + submission events
├── submission.json         # structured answer from task MCP
└── codex_workspace/        # or claude_workspace/, codex_sdk_workspace/
```

## Testing

```bash
# Unit tests (no Docker)
uv run python -m unittest tests.agent.test_sandbox_unit -v

# Security probes (requires built image)
uv run python -m unittest tests.agent.test_sandbox_security -v

# Full sandbox agent pipelines (Docker + Kathara + credentials)
uv run python -m unittest tests.agent.test_sandbox_agents -v
```

## Troubleshooting

**Sandbox container exits immediately**

- Confirm Docker is running. The sandbox image is built automatically on first `--sandbox` use.
- To rebuild after code changes: `docker build --network=host -t nika/agent-sandbox:latest -f src/agent/sandbox/Dockerfile .`

**MCP connection errors inside container**

- Gateway must bind `0.0.0.0` in sandbox mode (automatic).
- Default bridge networking uses `host.docker.internal` for the MCP gateway.

**API ConnectionRefused / api_retry in messages.jsonl**

- Usually means the container cannot reach the LLM API. If direct access works on the host, no proxy is needed.
- If the host requires a proxy to reach LLM APIs, create `.env.sandbox.local` from `.env.sandbox.local.example` and enable the proxy options there.

**Codex auth failures**

- Set `OPENAI_API_KEY` in `.env`, or run `codex login` on the host so `~/.codex/auth.json` can be mounted.

**Claude auth failures**

- Configure env API credentials; CLI OAuth is not supported in sandbox mode.

Manual debug shell (legacy helper):

```bash
NIKA_SANDBOX_NETWORK=bridge src/agent/sandbox/run_sandbox.sh bash
```

For production runs, prefer `nika agent run --sandbox`.
