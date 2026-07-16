# Agent Workflows

NIKA exposes three public workflows through `nika agent run` and the Experiment
Studio: `react`, `plan-execute`, and `reflexion`.

The upstream ReAct baseline lives in `agent.byo.langgraph`. The extension layer in
`agent.extensions` preserves that baseline path when training modules are disabled
and composes the same MCP diagnosis/submission contract with advanced workflows and
training modules when requested.

## Layout

```text
byo/langgraph/        Upstream ReAct baseline
extensions/           ReAct composition, Plan & Execute, Reflexion
procedural_memory/    Skill retrieval, evolution, verification, persistence
tool_refinement/      Tool exploration, analysis, rewriting, publication
llm/                  Shared provider factory
utils/                MCP, logging and prompt helpers
mock/                 Internal pipeline test double
```

All workflows use the same two-phase MCP contract:

1. Diagnose the live network with scenario-specific tools.
2. Submit anomaly, localization, and root-cause fields through the task MCP server.

Shared defaults are loaded from `config/modules.yaml`. API endpoint and credential
values are loaded from `.env`.
