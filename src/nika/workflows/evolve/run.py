"""SIA-H style harness loop for evolving executable target agents."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.composition import HarnessConfig, MemoryConfig, ToolEvolutionConfig
from agent.defaults import DEFAULT_MAX_STEPS
from agent.harness.runner import validate_target_agent_source
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from nika.config import RESULTS_DIR, RUNTIME_DIR
from nika.utils.kathara_cleanup import ensure_kathara_clean
from nika.workflows.benchmark.run import default_benchmark_csv_path, run_benchmark_from_csv

HARNESS_EVOLUTION_DIR = RUNTIME_DIR / "harness_evolution"
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
_REPO_ROOT = Path(__file__).resolve().parents[4]
_REFERENCE_TARGET_AGENT = _REPO_ROOT / "src" / "agent" / "harness" / "reference_target_agent.py"
_IP_ADDRESS_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SESSION_ID_PATTERN = re.compile(r"\b\d{8}-\d{6}-[0-9a-f]{6}\b")


class TargetAgentArtifact(BaseModel):
    """Structured source artifact produced by the meta-agent."""

    improvement_md: str = Field(min_length=1)
    target_agent_py: str = Field(min_length=200)


@dataclass
class EvolutionCaseResult:
    session_id: str
    session_dir: str
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
    target_agent_path: str
    cases: int
    submitted: int
    detection_hits: int
    localization_hits: int
    rca_hits: int
    next_target_agent_path: str | None = None
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
                session_dir=str(session_dir),
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
        "| Index | Session | Scenario | Problem | Submitted | Detection | "
        "Localization | RCA | Steps | Tool calls |",
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
    target_agent_path: str | Path,
    rows: list[EvolutionCaseResult],
) -> str:
    lines = [
        f"# Harness Evolution Context: {run_id} / Generation {generation}",
        "",
        f"**Started**: {datetime.now().isoformat(timespec='seconds')}",
        f"**Benchmark CSV**: {benchmark_file}",
        f"**Benchmark Root**: {benchmark_root}",
        f"**Target Agent**: {target_agent_path}",
        "**Feedback Scope**: all benchmark rows",
        "",
        "## Summary",
        "",
        f"- All cases: {_score_line(rows)}",
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


def _safe_text(
    value: Any,
    *,
    limit: int = 2400,
    forbidden_terms: tuple[str, ...] = (),
) -> str:
    text = str(value or "")
    text = _IP_ADDRESS_PATTERN.sub("<ip>", text)
    text = _SESSION_ID_PATTERN.sub("<session>", text)
    for term in forbidden_terms:
        term = str(term or "").strip()
        if not term:
            continue
        variants = {
            term,
            term.replace("_", " "),
            term.replace("-", " "),
        }
        for variant in variants:
            if variant:
                text = re.sub(
                    re.escape(variant),
                    "<case-term>",
                    text,
                    flags=re.IGNORECASE,
                )
    return text[:limit]


def _submission_terms(submission: dict[str, Any] | None) -> tuple[str, ...]:
    if not submission:
        return ()
    terms: list[str] = []
    for key in ("root_cause_name", "faulty_devices"):
        value = submission.get(key)
        if isinstance(value, str):
            terms.append(value)
        elif isinstance(value, list):
            terms.extend(str(item) for item in value if item)
    return tuple(terms)


def _execution_digest(
    path: Path,
    *,
    forbidden_terms: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not payload:
        return None
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    compact_messages: list[dict[str, Any]] = []
    for item in messages[-16:]:
        if not isinstance(item, dict):
            continue
        compact_messages.append(
            {
                "event": item.get("event") or item.get("type"),
                "agent": item.get("agent"),
                "name": item.get("name"),
                "content": _safe_text(
                    item.get("content") or item.get("message") or item,
                    limit=800,
                    forbidden_terms=forbidden_terms,
                ),
            }
        )
    return {
        "case": payload.get("case"),
        "diagnosis_report": _safe_text(
            payload.get("diagnosis_report"),
            forbidden_terms=forbidden_terms,
        ),
        "submission_result": _safe_text(
            payload.get("submission_result"),
            forbidden_terms=forbidden_terms,
        ),
        "error": _safe_text(
            payload.get("error"),
            limit=1200,
            forbidden_terms=forbidden_terms,
        ),
        "messages_tail": compact_messages,
    }


def build_feedback_context(
    *,
    generation: int,
    rows: list[EvolutionCaseResult],
    target_agent_path: str | Path | None = None,
) -> str:
    """Build a bounded, non-ground-truth context for the target-agent meta loop."""
    lines = [
        f"# Harness Evolution Feedback Context: Generation {generation}",
        "",
        "NIKA is the harness and evaluator. Improve only the executable target agent.",
        "Do not memorize benchmark ids, session ids, concrete addresses, or hidden labels.",
        "",
        "## Score Summary",
        "",
        f"- All cases: {_score_line(rows)}",
        f"- Target agent: {target_agent_path or '-'}",
        "",
        "## Case Metrics",
        "",
        "| Case | Benchmark index | Submitted | Detection | Localization | RCA | "
        "Steps | Tool calls | Tool errors |",
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

    lines.extend(["", "## Execution Samples", ""])
    for row in rows[:6]:
        forbidden_terms = tuple(
            term
            for term in (
                row.session_id,
                row.problem,
                *_submission_terms(row.submission),
            )
            if term
        )
        digest = _execution_digest(
            Path(row.session_dir) / "agent_execution.json",
            forbidden_terms=forbidden_terms,
        )
        if digest is None:
            error = _load_json(Path(row.session_dir) / "harness_error.json")
            digest = (
                {"error": _safe_text(error, limit=1200, forbidden_terms=forbidden_terms)}
                if error
                else {}
            )
        lines.append(
            "```json\n"
            + json.dumps(digest, indent=2, ensure_ascii=False, default=str)[:7000]
            + "\n```"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _reference_target_agent_source() -> str:
    return _REFERENCE_TARGET_AGENT.read_text(encoding="utf-8")


def _strip_code_fence(source: str) -> str:
    text = source.strip()
    if not text.startswith("```"):
        return text + "\n"
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip() + "\n"
    return text + "\n"


def _write_target_agent(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_strip_code_fence(source), encoding="utf-8")
    validate_target_agent_source(path)


def _copy_initial_target(path: Path, initial_target_agent: str | Path | None) -> None:
    if initial_target_agent is None:
        _write_target_agent(path, _reference_target_agent_source())
        return
    source_path = Path(initial_target_agent)
    _write_target_agent(path, source_path.read_text(encoding="utf-8"))


def _build_initial_target_prompt(*, benchmark_file: str | Path, reference_source: str) -> str:
    return f"""You are the SIA-H meta-agent for NIKA.

Write a complete, standalone Python file named target_agent.py. It will be run by
the NIKA harness for each benchmark case.

Runtime contract:
- Parse --session-id, --dataset-dir, --working-dir, --backend, --model, --max-steps.
- Read only public files from --dataset-dir, especially case_context.json.
- Use MCP tools through the current session id to diagnose and submit.
- Write useful execution artifacts under --working-dir.
- Do not import benchmark CSV readers, failure injection, ground truth, or private labels.
- Do not train or modify model weights.

Benchmark CSV for this evolution run: {benchmark_file}

Reference implementation to improve from:
```python
{reference_source}
```

Return structured fields:
- improvement_md: concise explanation of the initial executable strategy.
- target_agent_py: the full Python source for target_agent.py.
"""


def _build_feedback_prompt(
    *,
    generation: int,
    max_generations: int,
    feedback_context: str,
    current_source: str,
) -> str:
    return f"""You are the SIA-H feedback/meta agent for NIKA.

The target agent is an executable Python program, not a prompt overlay. Improve
the source code for the next generation based on public execution artifacts and
scores from the current generation.

Current generation: {generation}
Maximum generations: {max_generations}

Rules:
- Preserve the CLI contract.
- Read only --dataset-dir public files and use MCP tools for live diagnosis.
- Keep submission through list_avail_problems() and submit().
- Do not reference benchmark CSV files, ground_truth.json, failure injection,
  private problem labels, or hard-coded case/session ids.
- Do not train or modify model weights.
- Prefer general improvements: planning, evidence ledger, retry behavior, tool
  selection, budget management, robust parsing, and crash recovery.

Feedback context:
```markdown
{feedback_context}
```

Current target_agent.py:
```python
{current_source}
```

Return structured fields:
- improvement_md: what changed and why.
- target_agent_py: the full next-generation Python source.
"""


def _invoke_target_meta_agent(
    *,
    prompt: str,
    feedback_llm_backend: str,
    feedback_model: str,
) -> TargetAgentArtifact:
    model = load_model(feedback_llm_backend, feedback_model).with_structured_output(
        TargetAgentArtifact
    )
    artifact = model.invoke(prompt)
    if not isinstance(artifact, TargetAgentArtifact):
        artifact = TargetAgentArtifact.model_validate(artifact)
    return artifact


def build_initial_target_agent(
    *,
    output_path: str | Path,
    benchmark_file: str | Path,
    feedback_mode: str,
    feedback_llm_backend: str,
    feedback_model: str,
) -> tuple[str, str]:
    """Create generation-1 target_agent.py."""
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(
            "feedback_mode must be one of: " + ", ".join(sorted(FEEDBACK_MODES))
        )

    target_path = Path(output_path)
    reference_source = _reference_target_agent_source()
    if feedback_mode in {"auto", "llm"}:
        prompt = _build_initial_target_prompt(
            benchmark_file=benchmark_file,
            reference_source=reference_source,
        )
        (target_path.parent / "initial_meta_prompt.md").write_text(
            prompt,
            encoding="utf-8",
        )
        try:
            artifact = _invoke_target_meta_agent(
                prompt=prompt,
                feedback_llm_backend=feedback_llm_backend,
                feedback_model=feedback_model,
            )
            _write_target_agent(target_path, artifact.target_agent_py)
            (target_path.parent / "improvement.md").write_text(
                artifact.improvement_md,
                encoding="utf-8",
            )
            return artifact.improvement_md, "llm"
        except Exception as exc:
            if feedback_mode == "llm":
                raise
            note = (
                "# Initial Target Agent\n\n"
                f"- Meta-agent failed with `{type(exc).__name__}`; "
                "using the reference executable target.\n"
            )
            _write_target_agent(target_path, reference_source)
            (target_path.parent / "improvement.md").write_text(note, encoding="utf-8")
            return note, "deterministic-fallback"

    note = (
        "# Initial Target Agent\n\n"
        "- Deterministic mode uses the checked-in reference executable target.\n"
    )
    _write_target_agent(target_path, reference_source)
    (target_path.parent / "improvement.md").write_text(note, encoding="utf-8")
    return note, "deterministic"


def build_next_target_update(
    *,
    generation: int,
    max_generations: int,
    rows: list[EvolutionCaseResult],
    feedback_mode: str,
    current_target_path: str | Path,
    next_target_path: str | Path,
    feedback_llm_backend: str,
    feedback_model: str,
) -> tuple[str, str]:
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(
            "feedback_mode must be one of: " + ", ".join(sorted(FEEDBACK_MODES))
        )

    current_path = Path(current_target_path)
    next_path = Path(next_target_path)
    current_source = current_path.read_text(encoding="utf-8")
    feedback_context = build_feedback_context(
        generation=generation,
        rows=rows,
        target_agent_path=current_path,
    )
    next_path.parent.mkdir(parents=True, exist_ok=True)
    (next_path.parent / "feedback_context.md").write_text(
        feedback_context,
        encoding="utf-8",
    )
    prompt = _build_feedback_prompt(
        generation=generation,
        max_generations=max_generations,
        feedback_context=feedback_context,
        current_source=current_source,
    )
    (next_path.parent / "feedback_prompt.md").write_text(prompt, encoding="utf-8")

    if feedback_mode in {"auto", "llm"}:
        try:
            artifact = _invoke_target_meta_agent(
                prompt=prompt,
                feedback_llm_backend=feedback_llm_backend,
                feedback_model=feedback_model,
            )
            _write_target_agent(next_path, artifact.target_agent_py)
            (next_path.parent / "improvement.md").write_text(
                artifact.improvement_md,
                encoding="utf-8",
            )
            return artifact.improvement_md, "llm"
        except Exception as exc:
            if feedback_mode == "llm":
                raise
            improvement = (
                f"# Improvement Plan: Generation {generation + 1}\n\n"
                f"- Meta-agent failed with `{type(exc).__name__}`; "
                "carrying forward the current target agent.\n"
                "- Continue scoring this executable while preserving all artifacts "
                "for later manual or LLM feedback.\n"
            )
            _write_target_agent(next_path, current_source)
            (next_path.parent / "improvement.md").write_text(
                improvement,
                encoding="utf-8",
            )
            return improvement, "deterministic-fallback"

    improvement = (
        f"# Improvement Plan: Generation {generation + 1}\n\n"
        "- Deterministic feedback mode carries forward the current executable "
        "target agent without LLM source rewriting.\n"
    )
    _write_target_agent(next_path, current_source)
    (next_path.parent / "improvement.md").write_text(improvement, encoding="utf-8")
    return improvement, "deterministic"


def collect_generation_artifacts(
    *,
    rows: list[EvolutionCaseResult],
    gen_dir: str | Path,
) -> None:
    gen_path = Path(gen_dir)
    execution_dir = gen_path / "agent_execution"
    execution_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    for row in rows:
        session_dir = Path(row.session_dir)
        suffix = (
            str(row.benchmark_index)
            if row.benchmark_index is not None
            else row.session_id
        )
        execution_path = session_dir / "agent_execution.json"
        error_path = session_dir / "harness_error.json"
        target_execution = execution_dir / f"execution_{suffix}.json"
        if execution_path.exists():
            shutil.copy2(execution_path, target_execution)
        if error_path.exists():
            shutil.copy2(error_path, execution_dir / f"harness_error_{suffix}.json")
        copied.append(
            {
                "session_id": row.session_id,
                "benchmark_index": row.benchmark_index,
                "session_dir": row.session_dir,
                "execution_artifact": (
                    str(target_execution) if target_execution.exists() else None
                ),
                "harness_error": (
                    str(execution_dir / f"harness_error_{suffix}.json")
                    if (execution_dir / f"harness_error_{suffix}.json").exists()
                    else None
                ),
                "metrics": row.metrics,
                "submitted": row.submitted,
            }
        )

    _write_json(
        gen_path / "results.json",
        {
            "cases": copied,
            "score": {
                "submitted": sum(row.submitted for row in rows),
                "detection_hits": sum(row.detection_hit for row in rows),
                "localization_hits": sum(row.localization_hit for row in rows),
                "rca_hits": sum(row.rca_hit for row in rows),
                "total": len(rows),
            },
        },
    )


def run_harness_evolution(
    *,
    benchmark_file: str | Path = default_benchmark_csv_path(),
    max_generations: int = 3,
    run_id: str | None = None,
    llm_backend: str = "netmind",
    model: str = "openai/gpt-oss-120b",
    max_steps: int = DEFAULT_MAX_STEPS,
    run_judge: bool = False,
    judge_llm_backend: str | None = None,
    judge_model: str | None = None,
    tool_evolution_enabled: bool = False,
    tool_library_id: str | None = None,
    tool_evolution_mode: str = "dual",
    memory_mode: str = "off",
    memory_bank: str = "default",
    memory_top_k: int = 5,
    memory_token_budget: int = 1500,
    initial_target_agent: str | Path | None = None,
    feedback_mode: str = "auto",
    feedback_llm_backend: str = DEFAULT_LLM_BACKEND,
    feedback_model: str = DEFAULT_MODEL,
    runtime_root: str | Path = HARNESS_EVOLUTION_DIR,
    results_root: str | Path = RESULTS_DIR,
) -> list[EvolutionGenerationSummary]:
    if max_generations < 1:
        raise ValueError("max_generations must be >= 1")
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(
            "feedback_mode must be one of: " + ", ".join(sorted(FEEDBACK_MODES))
        )

    ensure_kathara_clean(context="harness evolution run")

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}"
    runtime_dir = Path(runtime_root) / run_id
    raw_name = Path(benchmark_file).stem
    benchmark_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_name).strip(".-")
    benchmark_name = benchmark_name or "benchmark"
    benchmark_results_dir = Path(results_root) / f"{benchmark_name}-{run_id}"
    runtime_dir.mkdir(parents=True, exist_ok=False)
    benchmark_results_dir.mkdir(parents=True, exist_ok=False)

    resolved_tool_library = tool_library_id or f"{run_id}-tools"
    summaries: list[EvolutionGenerationSummary] = []

    gen1_dir = runtime_dir / "gen_1"
    gen1_dir.mkdir(parents=True, exist_ok=True)
    target_agent_path = gen1_dir / "target_agent.py"
    if initial_target_agent is None:
        _, initial_source = build_initial_target_agent(
            output_path=target_agent_path,
            benchmark_file=benchmark_file,
            feedback_mode=feedback_mode,
            feedback_llm_backend=feedback_llm_backend,
            feedback_model=feedback_model,
        )
    else:
        _copy_initial_target(target_agent_path, initial_target_agent)
        initial_source = "user-provided"
        (gen1_dir / "improvement.md").write_text(
            "# Initial Target Agent\n\n- Copied from user-provided source.\n",
            encoding="utf-8",
        )

    _write_json(
        runtime_dir / "run.json",
        {
            "run_id": run_id,
            "mode": "sia_harness",
            "benchmark_file": str(benchmark_file),
            "max_generations": max_generations,
            "agent_type": "harness",
            "llm_backend": llm_backend,
            "model": model,
            "max_steps": max_steps,
            "feedback_mode": feedback_mode,
            "feedback_llm_backend": feedback_llm_backend,
            "feedback_model": feedback_model,
            "initial_target_agent_source": initial_source,
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
        target_agent_path = gen_dir / "target_agent.py"
        validate_target_agent_source(target_agent_path)

        print(
            "evolve_generation_start "
            f"run_id={run_id} generation={generation}/{max_generations} "
            f"target_agent={target_agent_path} result_root={gen_results_dir}",
            flush=True,
        )
        run_benchmark_from_csv(
            benchmark_file=str(benchmark_file),
            agent_type="harness",
            llm_backend=llm_backend,
            model=model,
            max_steps=max_steps,
            max_attempts=1,
            run_judge=run_judge,
            judge_llm_backend=judge_llm_backend,
            judge_model=judge_model,
            oracle_routing=False,
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
            harness=HarnessConfig(target_agent_path=str(target_agent_path)),
            harness_allow_failure=True,
            result_root=gen_results_dir,
        )

        rows = load_generation_results(gen_results_dir)
        collect_generation_artifacts(rows=rows, gen_dir=gen_dir)
        context = build_generation_context(
            run_id=run_id,
            generation=generation,
            benchmark_file=benchmark_file,
            benchmark_root=gen_results_dir,
            target_agent_path=target_agent_path,
            rows=rows,
        )
        context_path = gen_dir / "context.md"
        context_path.write_text(context, encoding="utf-8")
        (gen_dir / "feedback_context.md").write_text(
            build_feedback_context(
                generation=generation,
                rows=rows,
                target_agent_path=target_agent_path,
            ),
            encoding="utf-8",
        )

        summary = EvolutionGenerationSummary(
            generation=generation,
            benchmark_root=str(gen_results_dir),
            context_path=str(context_path),
            target_agent_path=str(target_agent_path),
            cases=len(rows),
            submitted=sum(row.submitted for row in rows),
            detection_hits=sum(row.detection_hit for row in rows),
            localization_hits=sum(row.localization_hit for row in rows),
            rca_hits=sum(row.rca_hit for row in rows),
        )

        if generation < max_generations:
            next_dir = runtime_dir / f"gen_{generation + 1}"
            next_dir.mkdir(parents=True, exist_ok=True)
            next_target = next_dir / "target_agent.py"
            _, feedback_source = build_next_target_update(
                generation=generation,
                max_generations=max_generations,
                rows=rows,
                feedback_mode=feedback_mode,
                current_target_path=target_agent_path,
                next_target_path=next_target,
                feedback_llm_backend=feedback_llm_backend,
                feedback_model=feedback_model,
            )
            summary.next_target_agent_path = str(next_target)
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
