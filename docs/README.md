# Learning Modules And Agent Baselines

This document is the compact source of truth for agent-side learning modules and
agent baselines in the current NIKA checkout.

## Boundary

NIKA is the benchmark and orchestration platform. It owns scenarios, sessions,
environment startup, fault injection, evaluation, and result artifacts. The
learning logic belongs to the evaluated agent side.

| Layer | Owns | Must not own |
|---|---|---|
| NIKA core | scenarios, sessions, env startup, fault injection, evaluation | agent skill contents or refined tool docs |
| Agent runtime | diagnosis workflows, prompt construction, MCP tool loading, submission | benchmark scoring or hidden ground truth |
| Learning modules | Skill-Pro procedural skills, DRAFT tool documentation libraries | hidden-answer mutation or benchmark selection |
| Agent baselines | static agents | benchmark scoring or hidden ground truth |
| Experiment runners | repeatable command workflows | low-level learning implementation details |

`src/agent/composition.py` is the typed boundary for optional extensions:

- `MemoryConfig` controls procedural-memory retrieval and update mode.
- `ToolEvolutionConfig` controls evolved-tool library exposure and update mode.
- `AgentRunConfig` groups the base agent run and extension config.

## Online Timeline

Benchmark YAML files are one online timeline:

```yaml
cases:
  - scenario: dc_clos_bgp
    topo_size: s
    problem: link_down
    inject:
      host_name: pc_0_0
      intf_name: eth0
```

The runner writes an internal `benchmark_index` into `run.json` based on case
order. Each case carries deterministic `inject` parameters, so benchmark target
selection does not rely on runtime randomness. Every case can influence later
cases through enabled learning modules.

Benchmark YAML runs are intentionally sequential:

- evolving memory updates the Skill-Pro skill bank after each evaluated episode;
- Tool Evolution updates DRAFT documentation for fixed primitive tools after each evaluated episode.

## Skill-Pro Memory

Memory wraps an existing agent. It uses a Skill-MDP selector to activate a
reusable procedure before diagnosis and writes candidate skills after
evaluation.

```bash
nika benchmark run --file benchmark/benchmark_test.yaml \
  -a react \
  -b custom \
  -m openai/gpt-oss-20b \
  -n 100 \
  --memory memory-gptoss120
```

Use `--memory-read <bank>` for read-only retrieval. Persistent state is local
JSON under `runtime/memory/<bank>/skills.json`.
Studio-created memory banks use the shared experiment id format
`<benchmark>-<NNNN>` unless a bank id is explicitly provided.

Each skill has:

- activation condition;
- execution steps;
- termination condition.

New banks are bootstrapped with the six Skill-Pro seed procedures
(StructuredCoT, ReActDecision, HypothesisElimination, SelfConsistencyCheck,
ExploreExploitArbitration, and StrategicPlanning); clearing a bank removes
learned state and rebuilds that seed pool.
The local skill pool tracks frequency, average gain, maturity, parent/version
lineage, and reuse count. Offline learning persists ExperiencePool and
GoldenExperiencePool records, stores structured LLM semantic-gradient critiques,
generates best-of-N new/refined candidates, and uses a non-parametric PPO gate
to accept a candidate only when clipped surrogate reward advantage beats the
active/best baseline.
Ground-truth labels can be used as offline evaluation evidence, but retrieved
Skill-Pro context is procedural only: hidden root-cause names and faulty-device
labels are redacted from the critic prompt and not written into skill guidance.
Score-based maintenance retires duplicate, low-value, or over-capacity skills.
`memory_update.json` records whether each accepted/rejected candidate used an
LLM semantic gradient, whether the critic failed, and the bounded learning-call
error when deterministic recovery is used. Memory-bank stats count total,
LLM-produced, and failed LLM gradients.

Skill-Pro and DRAFT learning calls can be decoupled from the diagnosis model
with `NIKA_LEARNING_LLM_BACKEND` and `NIKA_LEARNING_LLM_MODEL`. Leave them
blank to inherit the benchmark agent backend/model; set them to a faster
structured-output-capable model when curator latency dominates.
Runtime Skill-Pro selection defaults to LCB ranking. Use
`--memory-selector llm_topk_lcb` for the Skill-Pro top-k LLM nomination plus
LCB selector, and `--memory-meta-controller llm` for the Skill-Pro
DONE/CONTINUE option-termination controller. These modes are stored in run
metadata as `memory_skill_selector_mode` and `memory_meta_controller_mode`.
Additional runtime knobs are `--memory-max-skill-age`,
`--memory-selector-min-lcb`, and `--memory-selector-nominee-k`. Offline
evolution knobs are `--memory-pool-size`, `--memory-evolution-threshold`,
`--memory-best-of-n`, and `--memory-ppo-epsilon`; these are persisted in
session metadata and reused by the evaluation-time Skill-Pro update.

## Tool Evolution

Tool Evolution is not a separate agent type. It is an optional agent-side module
that improves model-facing documentation for fixed primitive MCP tools using
DRAFT. It does not create new executable tools or MCP servers.

```bash
nika benchmark run --file benchmark/benchmark_test.yaml \
  -a react \
  -b custom \
  -m openai/gpt-oss-20b \
  -n 100 \
  --tools tools-gptoss120
```

Persistent libraries live under `runtime/tool_evolution/<library_id>/state.json`.
Studio-created tool libraries use the shared experiment id format
`<benchmark>-<NNNN>` unless a library id is explicitly provided.
They store tool trials, comprehension gaps, structured LLM documentation
rewrites, Explorer observations, Analyzer suggestions, rewrite history,
tool-level usage summaries, path-rate metrics, mastery/convergence stats,
revisions, LLM rewrite failures, and frozen documents. Use a fresh library id
per experimental condition.

The DRAFT runtime is configurable through CLI, benchmark forwarding, and Studio:
`--tool-doc-chars`, `--tool-prompt-doc-limit`,
`--tool-scoped-prompt-doc-limit`, `--tool-planned-checks`,
`--tool-next-checks`, and `--tool-convergence-threshold`. Planned/next checks
accept `0` when prompt injection for those queues should be disabled.

## Clean Ablations

Run one module or baseline at a time before combined experiments:

```bash
# baseline
nika benchmark run --file benchmark/benchmark_test.yaml \
  -a react -b custom -m openai/gpt-oss-20b -n 100

# memory only
nika benchmark run --file benchmark/benchmark_test.yaml \
  -a react -b custom -m openai/gpt-oss-20b -n 100 \
  --memory memory-gptoss120

# tool evolution only
nika benchmark run --file benchmark/benchmark_test.yaml \
  -a react -b custom -m openai/gpt-oss-20b -n 100 \
  --tools tools-gptoss120

```

Tool and memory modules can be composed with static baselines from the Streamlit UI:

```bash
uv run nika studio
```
