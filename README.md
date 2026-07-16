# NIKA

NIKA is a live-network benchmark for evaluating autonomous troubleshooting agents.
It deploys reproducible Kathara or Containerlab scenarios, injects network faults,
exposes telemetry through MCP tools, runs an agent, and evaluates its submission.

The repository supports three workflows: `react`, `plan-execute`, and `reflexion`.
It also provides Procedural Memory, adapted from Skill-Pro, and Tool Refinement,
adapted from DRAFT.

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker Engine
- [Kathara](https://www.kathara.org/) for Kathara scenarios
- [Containerlab](https://containerlab.dev/) for Containerlab scenarios

## Installation

```bash
git clone https://github.com/sands-lab/nika
cd nika
uv sync --extra studio
cp .env.example .env
```

Configure the OpenAI-compatible endpoint in `.env`:

```dotenv
CUSTOM_API_URL="https://example.com/v1"
CUSTOM_API_BASE="https://example.com/v1"
CUSTOM_API_KEY=""
CUSTOM_TIMEOUT_SECONDS=90
CUSTOM_MAX_RETRIES=5
```

Shared workflow and learning defaults live in [`config/modules.yaml`](config/modules.yaml).

## Experiment Studio

```bash
uv run nika studio
```

The Studio configures baseline and learning experiments, tracks progress, resumes
existing experiments, and compares result metrics.

## CLI Workflow

Start a network session and inject a fault:

```bash
uv run nika env list
uv run nika env run simple_bgp
uv run nika failure describe link_down
uv run nika failure inject link_down --set host_name=pc1 --set intf_name=eth0
```

Run an agent:

```bash
uv run nika agent list
uv run nika agent run \
  --agent react \
  --provider custom \
  --model openai/gpt-oss-120b \
  --max-steps 50
```

Close and evaluate the session:

```bash
uv run nika session close -y
uv run nika eval metrics
uv run nika eval summary
```

## Benchmarks

Benchmark definitions live in [`benchmark/`](benchmark/):

- `benchmark_selected.yaml`: small selected set
- `benchmark_evaluate.yaml`: evaluation set
- `benchmark_evolve.yaml`: learning curriculum with evolve/read-only phases
- `benchmark_full.yaml`: full benchmark set

Run a benchmark directly:

```bash
uv run nika benchmark run \
  --config benchmark/benchmark_evaluate.yaml \
  --agent react \
  --provider custom \
  --model openai/gpt-oss-120b \
  --max-steps 50
```

Learning experiments are composed by `nika.extensions.benchmark`; the Studio stores
the complete command and configuration with each run.

## Configuration

`config/modules.yaml` is the single source of project defaults:

- `baseline`: workflow, provider, model, step and evaluation defaults
- `procedural_memory`: retrieval, evolution, verification and policy parameters
- `tool_refinement`: exploration, analysis, rewriting and publication parameters

Procedural Memory defaults to `behavioral_replay`, a provider-compatible
verification fallback. `policy_logprob` more closely follows Skill-Pro's PPO Gate
when a completion endpoint with echoed prompt log-probabilities is configured, but
it replays the Skill system prompt and serialized action rather than provider-side
chat history and tool schemas.

Candidate verification is an offline admissibility prescreen: the gate measures
the clipped-surrogate improvement over the parent policy, then publishes passing
candidates as `probationary`. Only candidates with positive conservative gain from
later NIKA episodes become `validated`; unresolved probationary skills are retired
when the bank is frozen at the evolve/read boundary. Benchmarks that declare
`evolve_first_cases` (including `benchmark_evolve.yaml`) apply that cutoff in both
Studio and the extension CLI, and Studio reports read-only cases as the primary
endpoint while retaining evolve-phase learning diagnostics separately.

CLI and Studio values override these defaults per experiment. `.env` is reserved for
API connection details and credentials.

## Results

Important case artifacts include:

- `run.json`: session and configuration metadata
- `events.jsonl`: lifecycle and progress events
- `messages.jsonl`: agent trace
- `submission.json`: final diagnosis submission
- `eval_metrics.json`: evaluation metrics

The experiment summary reports detection, localization, root-cause, tool-use and
runtime metrics across completed cases.

## Development

```bash
uv run ruff check src tests benchmark
uv run pytest -q
```

Core source directories:

```text
src/agent/extensions/         ReAct, Plan & Execute, and Reflexion workflows
src/agent/procedural_memory/  Procedural Memory runtime and persistence
src/agent/tool_refinement/    Tool Refinement runtime and persistence
src/nika/                     Environments, orchestration, evaluation, CLI, Studio
benchmark/                    Benchmark definitions and generation utilities
config/                       Shared experiment defaults
```

## Citation

NIKA is described in [arXiv:2512.16381](https://arxiv.org/abs/2512.16381).
