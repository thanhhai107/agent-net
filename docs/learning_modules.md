# Learning Modules

This document is the compact source of truth for agent-side learning modules in
the current NIKA checkout.

## Boundary

NIKA is the benchmark and orchestration platform. It owns scenarios, sessions,
environment startup, fault injection, evaluation, and result artifacts. The
learning logic belongs to the evaluated agent side.

| Layer | Owns | Must not own |
|---|---|---|
| NIKA core | scenarios, sessions, env startup, fault injection, evaluation | agent memory contents, evolved tools, prompt-policy learning |
| Agent runtime | diagnosis workflows, prompt construction, MCP tool loading, submission | benchmark scoring or hidden ground truth |
| Learning modules | memory notes, tool libraries, policy overlays | hidden-answer mutation or benchmark selection |
| Experiment runners | repeatable command workflows | low-level learning implementation details |

`src/agent/composition.py` is the typed boundary for optional extensions:

- `MemoryConfig` controls procedural-memory retrieval and update mode.
- `ToolEvolutionConfig` controls evolved-tool library exposure and update mode.
- `PolicyOverlayConfig` controls prompt-policy injection from Agent Evolution.
- `AgentRunConfig` groups the base agent run and extension config.

## Online Timeline

Benchmark CSV files are one online timeline:

```csv
problem,scenario,topo_size
```

The runner writes an internal `benchmark_index` into `run.json` based on row
order. It no longer reads `stream_id`, `split`, or `sequence_index` from CSV.
Every row can influence later rows through enabled learning modules.

Benchmark CSV runs are intentionally sequential:

- evolving memory updates the memory bank after each evaluated episode;
- Tool Evolution updates the selected tool library after each evaluated episode;
- Agent Evolution learns from all rows in each generation before writing the
  next `policy_overlay.md`.

## Procedural Memory

Procedural memory wraps an existing agent. It retrieves concise guidance before
diagnosis and writes validated notes after evaluation.

```bash
docker compose up -d postgres qdrant

nika benchmark run --file benchmark/benchmark_test.csv \
  -a react \
  -b netmind \
  -m openai/gpt-oss-120b \
  -n 100 \
  --memory memory-gptoss120
```

Use `--memory-read <bank>` for read-only retrieval. See `memory/README.md` for
the detailed memory design and safety notes.

## Tool Evolution

Tool Evolution is not a separate agent type. It is an optional agent-side module
that improves model-facing primitive-tool guidance and can synthesize reusable
candidate tools.

```bash
nika benchmark run --file benchmark/benchmark_test.csv \
  -a react \
  -b netmind \
  -m openai/gpt-oss-120b \
  -n 100 \
  --tools tools-gptoss120 \
  --tool-mode dual
```

Modes:

| Mode | Behavior |
|---|---|
| `mastery` | update guidance for existing primitive tools |
| `distill` | synthesize and validate reusable composite/generated capabilities |
| `dual` | enable both mastery and synthesis |

Persistent libraries live under `runtime/tool_evolution/<library_id>/`. Use a
fresh library id per experimental condition.

## Agent Evolution

Agent Evolution is the outer loop for policy overlays. Each generation runs a
full benchmark batch, writes scored context, and creates a policy overlay for
the next generation.

```bash
nika evolve run --file benchmark/benchmark_test.csv \
  --max-gen 3 \
  -a react \
  -b netmind \
  -m openai/gpt-oss-120b \
  -n 100
```

Artifacts:

```text
runtime/agent_evolution/<run_id>/
results/agent-evolution-<run_id>/
```

Every CSV row contributes to the next policy. `--feedback-mode deterministic`
uses the deterministic planner; `--feedback-mode llm` requires the structured
feedback agent; `auto` tries the LLM and falls back.

## Clean Ablations

Run one learning module at a time before combined experiments:

```bash
# baseline
nika benchmark run --file benchmark/benchmark_test.csv \
  -a react -b netmind -m openai/gpt-oss-120b -n 100

# memory only
nika benchmark run --file benchmark/benchmark_test.csv \
  -a react -b netmind -m openai/gpt-oss-120b -n 100 \
  --memory memory-gptoss120

# tool evolution only
nika benchmark run --file benchmark/benchmark_test.csv \
  -a react -b netmind -m openai/gpt-oss-120b -n 100 \
  --tools tools-gptoss120 --tool-mode dual

# agent evolution only
nika evolve run --file benchmark/benchmark_test.csv \
  -a react -b netmind -m openai/gpt-oss-120b -n 100 \
  --max-gen 3
```

The same modules can be toggled together and launched from the Streamlit UI:

```bash
uv run nika studio
```
