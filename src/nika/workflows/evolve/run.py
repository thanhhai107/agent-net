"""SIA-style outer loop for evolving agent prompt policy over benchmark generations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.composition import MemoryConfig, PolicyOverlayConfig, ToolEvolutionConfig
from agent.defaults import DEFAULT_MAX_STEPS
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from nika.config import RESULTS_DIR, RUNTIME_DIR
from nika.workflows.benchmark.run import default_benchmark_csv_path, run_benchmark_from_csv

AGENT_EVOLUTION_DIR = RUNTIME_DIR / "agent_evolution"
FEEDBACK_MODES = frozenset({"auto", "deterministic", "llm"})
_RESULT_FIELDS = (
    "detection_score",
    "localization_accuracy",
    "rca_accuracy",
    "steps",
    "tool_calls",
    "tool_errors",
    "in_tokens",
    "out_tokens",
)
_IP_ADDRESS_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SESSION_ID_PATTERN = re.compile(r"\b\d{8}-\d{6}-[0-9a-f]{6}\b")


class AgentEvolutionFeedback(BaseModel):
    """Structured output produced by the outer-loop feedback agent."""

    observations: list[str] = Field(min_length=1)
    improvement_plan: list[str] = Field(min_length=1)
    policy_rules: list[str] = Field(min_length=3)


@dataclass
class EvolutionCaseResult:
    session_id: str
    scenario: str
    problem: str
    benchmark_index: int | None
    submitted: bool
    metrics: dict[str, Any]
    submission: dict[str, Any] | None

    @property
    def detection_hit(self) -> bool:
        return _metric_hit(self.metrics.get("detection_score"))

    @property
    def localization_hit(self) -> bool:
        return _metric_hit(self.metrics.get("localization_accuracy"))

    @property
    def rca_hit(self) -> bool:
        return _metric_hit(self.metrics.get("rca_accuracy"))


@dataclass
class EvolutionGenerationSummary:
    generation: int
    benchmark_root: str
    context_path: str
    cases: int
    submitted: int
    detection_hits: int
    localization_hits: int
    rca_hits: int
    next_policy_path: str | None = None
    feedback_source: str | None = None


def _metric_hit(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _metric_average(rows: list[EvolutionCaseResult], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.metrics[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_generation_results(benchmark_root: str | Path) -> list[EvolutionCaseResult]:
    root = Path(benchmark_root)
    rows: list[EvolutionCaseResult] = []
    if not root.exists():
        return rows

    for run_path in sorted(root.rglob("run.json")):
        if "0_summary" in run_path.relative_to(root).parts:
            continue
        session_dir = run_path.parent
        run_meta = _load_json(run_path) or {}
        metrics = _load_json(session_dir / "eval_metrics.json") or {}
        submission = _load_json(session_dir / "submission.json")
        problem_names = run_meta.get("problem_names") or []
        benchmark_index = run_meta.get("benchmark_index")
        try:
            benchmark_index = int(benchmark_index)
        except (TypeError, ValueError):
            benchmark_index = None
        rows.append(
            EvolutionCaseResult(
                session_id=str(run_meta.get("session_id") or session_dir.name),
                scenario=str(run_meta.get("scenario_name") or ""),
                problem=str(problem_names[0] if problem_names else ""),
                benchmark_index=benchmark_index,
                submitted=submission is not None,
                metrics=metrics,
                submission=submission,
            )
        )
    rows.sort(
        key=lambda row: (
            row.benchmark_index if row.benchmark_index is not None else 10**9,
            row.session_id,
        )
    )
    return rows


def _score_line(rows: list[EvolutionCaseResult]) -> str:
    total = len(rows)
    submitted = sum(row.submitted for row in rows)
    detected = sum(row.detection_hit for row in rows)
    localized = sum(row.localization_hit for row in rows)
    rca = sum(row.rca_hit for row in rows)
    return (
        f"submitted={submitted}/{total} detection={detected}/{total} "
        f"localization={localized}/{total} rca={rca}/{total}"
    )


def _metric_table(rows: list[EvolutionCaseResult]) -> str:
    lines = [
        "| Index | Session | Scenario | Problem | Submitted | Detection | Localization | RCA | Steps | Tool calls |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.benchmark_index if row.benchmark_index is not None else "-"),
                    row.session_id,
                    row.scenario or "-",
                    row.problem or "-",
                    "1" if row.submitted else "0",
                    str(metrics.get("detection_score", "-")),
                    str(metrics.get("localization_accuracy", "-")),
                    str(metrics.get("rca_accuracy", "-")),
                    str(metrics.get("steps", "-")),
                    str(metrics.get("tool_calls", "-")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_generation_context(
    *,
    run_id: str,
    generation: int,
    benchmark_file: str | Path,
    benchmark_root: str | Path,
    policy_overlay_path: str | Path | None,
    rows: list[EvolutionCaseResult],
) -> str:
    lines = [
        f"# Agent Evolution Context: {run_id} / Generation {generation}",
        "",
        f"**Started**: {datetime.now().isoformat(timespec='seconds')}",
        f"**Benchmark CSV**: {benchmark_file}",
        f"**Benchmark Root**: {benchmark_root}",
        f"**Input Policy Overlay**: {policy_overlay_path or 'None'}",
        "**Feedback Scope**: all benchmark rows",
        "",
        "## Summary",
        "",
        f"- All cases: {_score_line(rows)}",
        f"- Feedback cases: {_score_line(rows)}",
        "",
        "## Averages",
        "",
    ]
    for key in _RESULT_FIELDS:
        avg = _metric_average(rows, key)
        if avg is not None:
            lines.append(f"- {key}: {avg:.2f}")
    lines.extend(["", "## Cases", "", _metric_table(rows), ""])
    return "\n".join(lines)


def build_feedback_context(
    *,
    generation: int,
    rows: list[EvolutionCaseResult],
) -> str:
    """Build a sanitized context for the feedback agent.

    The human-facing context can include case labels. The feedback-agent context
    intentionally omits hidden problem ids, concrete session ids, device names,
    and addresses so generated policy remains general.
    """
    lines = [
        f"# Sanitized Agent Evolution Feedback Context: Generation {generation}",
        "",
        "Do not infer or output incident labels, device names, IP addresses, or session ids.",
        "",
        "## Score Summary",
        "",
        f"- All cases: {_score_line(rows)}",
        f"- Feedback cases: {_score_line(rows)}",
        "",
        "## Feedback Cases",
        "",
        "| Case | Benchmark index | Submitted | Detection | Localization | RCA | Steps | Tool calls | Tool errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        metrics = row.metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(row.benchmark_index if row.benchmark_index is not None else "-"),
                    "1" if row.submitted else "0",
                    str(metrics.get("detection_score", "-")),
                    str(metrics.get("localization_accuracy", "-")),
                    str(metrics.get("rca_accuracy", "-")),
                    str(metrics.get("steps", "-")),
                    str(metrics.get("tool_calls", "-")),
                    str(metrics.get("tool_errors", "-")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Allowed Feedback Scope",
            "",
            "- Improve investigation procedure, evidence discipline, budget management, and tool-use strategy.",
            "- Do not mention benchmark problem ids, host/router/interface names, IP addresses, or session ids.",
            "- Do not tell the agent an answer. The policy must remain useful for unseen transfer cases.",
        ]
    )
    return "\n".join(lines) + "\n"


def _submission_failures(rows: list[EvolutionCaseResult]) -> int:
    return sum(not row.submitted for row in rows)


def _normal_submissions(rows: list[EvolutionCaseResult]) -> int:
    total = 0
    for row in rows:
        submission = row.submission or {}
        if submission.get("is_anomaly") is False:
            total += 1
    return total


def _case_specific_terms(rows: list[EvolutionCaseResult]) -> set[str]:
    terms: set[str] = set()
    for row in rows:
        for value in (row.session_id, row.problem):
            if value:
                terms.add(value.lower())
    return terms


def _policy_has_forbidden_terms(policy: str, rows: list[EvolutionCaseResult]) -> bool:
    lowered = policy.lower()
    if _IP_ADDRESS_PATTERN.search(policy) or _SESSION_ID_PATTERN.search(policy):
        return True
    return any(term and term in lowered for term in _case_specific_terms(rows))


def build_policy_update(
    *,
    generation: int,
    max_generations: int,
    rows: list[EvolutionCaseResult],
) -> tuple[str, str]:
    total = len(rows)
    submitted = sum(row.submitted for row in rows)
    detected = sum(row.detection_hit for row in rows)
    localized = sum(row.localization_hit for row in rows)
    rca = sum(row.rca_hit for row in rows)
    missing = _submission_failures(rows)
    normal_submissions = _normal_submissions(rows)

    observations = [
        f"- Feedback timeline score: submitted={submitted}/{total}, detection={detected}/{total}, localization={localized}/{total}, RCA={rca}/{total}.",
    ]
    if missing:
        observations.append(
            f"- {missing} feedback case(s) did not produce a submission; preserve enough budget for the submission phase."
        )
    if normal_submissions:
        observations.append(
            f"- {normal_submissions} feedback submission(s) marked the injected incident as normal; require positive evidence before declaring normal operation."
        )
    if rca == 0 and total:
        observations.append(
            "- RCA did not score on the feedback timeline; force an explicit final mapping from evidence to a known root-cause class."
        )

    policy_lines = [
        "# Agent Evolution Policy Overlay",
        "",
        f"Generated after generation {generation} of {max_generations}.",
        "",
        "Apply this policy to the next diagnosis run. Keep it general; do not memorize case ids, device ids, or hidden labels.",
        "",
        "## Investigation Discipline",
        "",
        "- Spend the first calls establishing the symptom from reachability or service checks, then localize from the affected edge inward.",
        "- Treat timeouts, missing reachability, failed service checks, and control-plane inconsistencies as anomaly evidence until contradicted by stronger evidence.",
        "- Do not submit an empty or normal diagnosis while any observed end-to-end symptom remains unexplained.",
        "- Before finalizing, write a short evidence ledger: symptom, affected endpoint or service, nearest faulty device or interface, and root-cause hypothesis.",
        "- In the submission phase, map the evidence ledger to one of the available problem names instead of inventing a free-form cause.",
        "",
        "## Tool Use Discipline",
        "",
        "- Prefer targeted protocol or service tools after the initial symptom check; avoid repeating the same ping or config query without a new hypothesis.",
        "- If a multi-step evidence pattern is repeated or missing, use the Tool Evolution manager tools to identify the capability gap, propose a parameterized composite or pure Python helper, and execute it once for verification.",
        "- Use generated or composite candidates as evidence collectors only; verify their observations against primitive tool output before concluding.",
    ]
    if missing:
        policy_lines.extend(
            [
                "",
                "## Budget Guardrail",
                "",
                "- Reserve enough remaining steps for `list_avail_problems` and `submit`; stop exploration once the evidence ledger can support detection, localization, and RCA.",
            ]
        )
    if rca == 0 and total:
        policy_lines.extend(
            [
                "",
                "## RCA Guardrail",
                "",
                "- Never leave `root_cause_name` empty for an anomalous incident. If multiple causes seem plausible, choose the one best supported by direct failed evidence.",
            ]
        )

    improvement_lines = [
        f"# Improvement Plan: Generation {generation + 1}",
        "",
        "## Observations",
        "",
        *observations,
        "",
        "## Policy Changes",
        "",
        "- Add an explicit evidence ledger before submission.",
        "- Add a step-budget guardrail when submissions are missing.",
        "- Add a RCA mapping guardrail when root-cause accuracy is weak.",
        "- Make tool synthesis an explicit fallback when repeated primitive sequences appear.",
        "",
        "## Produced Artifact",
        "",
        "- `policy_overlay.md` is injected into the diagnosis system prompt for the next generation.",
    ]
    return "\n".join(improvement_lines) + "\n", "\n".join(policy_lines) + "\n"


def _format_feedback_markdown(feedback: AgentEvolutionFeedback) -> tuple[str, str]:
    improvement_lines = [
        "# Improvement Plan",
        "",
        "## Observations",
        "",
        *[f"- {item}" for item in feedback.observations],
        "",
        "## Policy Changes",
        "",
        *[f"- {item}" for item in feedback.improvement_plan],
        "",
        "## Produced Artifact",
        "",
        "- `policy_overlay.md` is injected into the diagnosis system prompt for the next generation.",
    ]
    policy_lines = [
        "# Agent Evolution Policy Overlay",
        "",
        "Generated by the Agent Evolution feedback agent.",
        "",
        "Apply this policy to the next diagnosis run. Keep it general; do not memorize case ids, device ids, IP addresses, or hidden labels.",
        "",
        "## Feedback-Agent Rules",
        "",
        *[f"- {item}" for item in feedback.policy_rules],
    ]
    return "\n".join(improvement_lines) + "\n", "\n".join(policy_lines) + "\n"


def _build_feedback_prompt(
    *,
    generation: int,
    max_generations: int,
    sanitized_context: str,
    previous_policy: str,
) -> str:
    previous_policy_block = previous_policy.strip() or "No previous policy overlay."
    return f"""You are the Agent Evolution feedback agent for NIKA.

NIKA is the benchmark/orchestrator. You improve only the evaluated agent's
general diagnosis policy for the next benchmark generation.

Current generation: {generation}
Maximum generations: {max_generations}

Rules:
- Use only the sanitized metrics/context below.
- Do not output benchmark problem ids, hidden labels, concrete device names,
  interface names, IP addresses, or session ids.
- Do not give the agent case-specific answers.
- Produce general troubleshooting policy that can transfer to unseen incidents.
- Keep rules concise and directly actionable inside a system prompt.

Previous policy overlay:
```markdown
{previous_policy_block}
```

Sanitized generation context:
```markdown
{sanitized_context}
```

Return structured fields:
- observations: short evidence-grounded findings from the generation.
- improvement_plan: concrete changes to apply next generation.
- policy_rules: prompt-ready rules for the diagnosis agent.
"""


def _run_llm_feedback_agent(
    *,
    generation: int,
    max_generations: int,
    rows: list[EvolutionCaseResult],
    previous_policy_path: Path | None,
    feedback_llm_backend: str,
    feedback_model: str,
) -> tuple[str, str]:
    sanitized_context = build_feedback_context(
        generation=generation,
        rows=rows,
    )
    previous_policy = (
        previous_policy_path.read_text(encoding="utf-8")
        if previous_policy_path is not None and previous_policy_path.exists()
        else ""
    )
    prompt = _build_feedback_prompt(
        generation=generation,
        max_generations=max_generations,
        sanitized_context=sanitized_context,
        previous_policy=previous_policy,
    )
    model = load_model(feedback_llm_backend, feedback_model).with_structured_output(
        AgentEvolutionFeedback
    )
    feedback = model.invoke(prompt)
    if not isinstance(feedback, AgentEvolutionFeedback):
        feedback = AgentEvolutionFeedback.model_validate(feedback)
    improvement, policy = _format_feedback_markdown(feedback)
    if _policy_has_forbidden_terms(policy, rows):
        raise ValueError("feedback policy contained case-specific terms")
    return improvement, policy


def build_next_policy_update(
    *,
    generation: int,
    max_generations: int,
    rows: list[EvolutionCaseResult],
    feedback_mode: str,
    previous_policy_path: Path | None,
    feedback_llm_backend: str,
    feedback_model: str,
) -> tuple[str, str, str]:
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(
            "feedback_mode must be one of: " + ", ".join(sorted(FEEDBACK_MODES))
        )
    if feedback_mode in {"auto", "llm"}:
        try:
            improvement, policy = _run_llm_feedback_agent(
                generation=generation,
                max_generations=max_generations,
                rows=rows,
                previous_policy_path=previous_policy_path,
                feedback_llm_backend=feedback_llm_backend,
                feedback_model=feedback_model,
            )
            return improvement, policy, "llm"
        except Exception as exc:
            if feedback_mode == "llm":
                raise
            fallback_note = (
                "\n## Feedback-Agent Fallback\n\n"
                f"- LLM feedback failed with `{type(exc).__name__}`; used deterministic fallback.\n"
            )
            improvement, policy = build_policy_update(
                generation=generation,
                max_generations=max_generations,
                rows=rows,
            )
            return improvement + fallback_note, policy, "deterministic-fallback"

    improvement, policy = build_policy_update(
        generation=generation,
        max_generations=max_generations,
        rows=rows,
    )
    return improvement, policy, "deterministic"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_agent_evolution(
    *,
    benchmark_file: str | Path = default_benchmark_csv_path(),
    max_generations: int = 3,
    run_id: str | None = None,
    agent_type: str = "react",
    llm_backend: str = "netmind",
    model: str = "openai/gpt-oss-120b",
    max_steps: int = DEFAULT_MAX_STEPS,
    max_attempts: int = 3,
    parallel: int = 1,
    run_judge: bool = False,
    judge_llm_backend: str | None = None,
    judge_model: str | None = None,
    oracle_routing: bool = False,
    tool_evolution_enabled: bool = False,
    tool_library_id: str | None = None,
    tool_evolution_mode: str = "dual",
    memory_mode: str = "off",
    memory_bank: str = "default",
    memory_top_k: int = 5,
    memory_token_budget: int = 1500,
    initial_policy_overlay: str | Path | None = None,
    feedback_mode: str = "auto",
    feedback_llm_backend: str = DEFAULT_LLM_BACKEND,
    feedback_model: str = DEFAULT_MODEL,
    runtime_root: str | Path = AGENT_EVOLUTION_DIR,
    results_root: str | Path = RESULTS_DIR,
) -> list[EvolutionGenerationSummary]:
    if max_generations < 1:
        raise ValueError("max_generations must be >= 1")
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(
            "feedback_mode must be one of: " + ", ".join(sorted(FEEDBACK_MODES))
        )
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}"

    runtime_dir = Path(runtime_root) / run_id
    benchmark_results_dir = Path(results_root) / f"agent-evolution-{run_id}"
    runtime_dir.mkdir(parents=True, exist_ok=False)
    benchmark_results_dir.mkdir(parents=True, exist_ok=False)

    summaries: list[EvolutionGenerationSummary] = []
    policy_overlay_path = Path(initial_policy_overlay) if initial_policy_overlay else None
    resolved_tool_library = tool_library_id or f"{run_id}-tools"

    _write_json(
        runtime_dir / "run.json",
        {
            "run_id": run_id,
            "benchmark_file": str(benchmark_file),
            "max_generations": max_generations,
            "agent_type": agent_type,
            "llm_backend": llm_backend,
            "model": model,
            "max_steps": max_steps,
            "max_attempts": max_attempts,
            "feedback_mode": feedback_mode,
            "feedback_llm_backend": feedback_llm_backend,
            "feedback_model": feedback_model,
            "tool_evolution_enabled": tool_evolution_enabled,
            "tool_library_id": resolved_tool_library if tool_evolution_enabled else None,
            "tool_evolution_mode": tool_evolution_mode if tool_evolution_enabled else None,
            "memory_mode": memory_mode,
            "memory_bank": memory_bank if memory_mode != "off" else None,
            "memory_top_k": memory_top_k if memory_mode != "off" else None,
            "memory_token_budget": memory_token_budget if memory_mode != "off" else None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    for generation in range(1, max_generations + 1):
        gen_dir = runtime_dir / f"gen_{generation}"
        gen_results_dir = benchmark_results_dir / f"gen_{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        gen_results_dir.mkdir(parents=True, exist_ok=False)

        print(
            "evolve_generation_start "
            f"run_id={run_id} generation={generation}/{max_generations} "
            f"policy_overlay={policy_overlay_path or '-'} "
            f"result_root={gen_results_dir}",
            flush=True,
        )
        run_benchmark_from_csv(
            benchmark_file=str(benchmark_file),
            agent_type=agent_type,
            llm_backend=llm_backend,
            model=model,
            max_steps=max_steps,
            max_attempts=max_attempts,
            parallel=parallel,
            run_judge=run_judge,
            judge_llm_backend=judge_llm_backend,
            judge_model=judge_model,
            oracle_routing=oracle_routing,
            tool_evolution=ToolEvolutionConfig(
                enabled=tool_evolution_enabled,
                library_id=resolved_tool_library,
                mode=tool_evolution_mode,
            ),
            memory=MemoryConfig(
                mode=memory_mode,
                bank=memory_bank,
                top_k=memory_top_k,
                token_budget=memory_token_budget,
            ),
            policy_overlay=PolicyOverlayConfig(
                path=str(policy_overlay_path) if policy_overlay_path else None
            ),
            result_root=gen_results_dir,
        )

        rows = load_generation_results(gen_results_dir)
        context = build_generation_context(
            run_id=run_id,
            generation=generation,
            benchmark_file=benchmark_file,
            benchmark_root=gen_results_dir,
            policy_overlay_path=policy_overlay_path,
            rows=rows,
        )
        context_path = gen_dir / "context.md"
        context_path.write_text(context, encoding="utf-8")
        feedback_context_path = gen_dir / "feedback_context.md"
        feedback_context_path.write_text(
            build_feedback_context(
                generation=generation,
                rows=rows,
            ),
            encoding="utf-8",
        )

        summary = EvolutionGenerationSummary(
            generation=generation,
            benchmark_root=str(gen_results_dir),
            context_path=str(context_path),
            cases=len(rows),
            submitted=sum(row.submitted for row in rows),
            detection_hits=sum(row.detection_hit for row in rows),
            localization_hits=sum(row.localization_hit for row in rows),
            rca_hits=sum(row.rca_hit for row in rows),
        )

        if generation < max_generations:
            improvement, policy, feedback_source = build_next_policy_update(
                generation=generation,
                max_generations=max_generations,
                rows=rows,
                feedback_mode=feedback_mode,
                previous_policy_path=policy_overlay_path,
                feedback_llm_backend=feedback_llm_backend,
                feedback_model=feedback_model,
            )
            next_dir = runtime_dir / f"gen_{generation + 1}"
            next_dir.mkdir(parents=True, exist_ok=True)
            (next_dir / "improvement.md").write_text(
                improvement,
                encoding="utf-8",
            )
            next_policy = next_dir / "policy_overlay.md"
            next_policy.write_text(policy, encoding="utf-8")
            policy_overlay_path = next_policy
            summary.next_policy_path = str(next_policy)
            summary.feedback_source = feedback_source

        _write_json(
            gen_dir / "metrics_summary.json",
            {
                **asdict(summary),
                "cases": [asdict(row) for row in rows],
            },
        )

        summaries.append(summary)
        _write_json(
            runtime_dir / "summary.json",
            {"run_id": run_id, "generations": [asdict(item) for item in summaries]},
        )
        print(
            "evolve_generation_done "
            f"run_id={run_id} generation={generation}/{max_generations} "
            f"cases={summary.cases} submitted={summary.submitted} "
            f"detection={summary.detection_hits} localization={summary.localization_hits} "
            f"rca={summary.rca_hits} context={context_path}",
            flush=True,
        )

    print(
        f"evolve_summary run_id={run_id} runtime_dir={runtime_dir} "
        f"results_dir={benchmark_results_dir}",
        flush=True,
    )
    return summaries
