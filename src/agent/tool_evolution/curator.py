"""Offline DRAFT learning hook.

DRAFT learns by comparing tool trials against their documentation, identifying
where the agent misunderstood arguments or preconditions, then rewriting the
documentation.  This implementation is gradient-free and never creates new
primitive tools.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any

from agent.learning_llm import (
    format_learning_error,
    learning_backend,
    learning_max_retries,
    learning_model,
    learning_timeout_seconds,
)
from agent.llm.model_factory import load_model
from agent.tool_evolution.models import (
    ComprehensionGap,
    DraftAnalyzerSuggestion,
    DraftExploration,
    DraftRewriteDraft,
    DocumentationRevision,
    DraftRewriteProposal,
    DraftToolStats,
    ToolDocumentation,
    ToolParameterDoc,
    ToolTrial,
    utc_now,
)
from agent.tool_evolution.store import ToolEvolutionStore
from agent.utils.phases import DIAGNOSIS
from nika.evaluator.result_log import MESSAGES_FILENAME
from nika.utils.session import Session


ERROR_MARKERS = (
    "error",
    "exception",
    "validation",
    "not found",
    "unknown",
    "missing",
    "invalid",
    "failed",
)
DRAFT_CONVERGENCE_THRESHOLD = 0.75
DIAGNOSIS_AGENT_NAMES = frozenset({DIAGNOSIS, "diagnosis_agent"})
MEMORY_AGENT_NAME = "memory_agent"
DRAFT_PROMPT_TEXT_LIMIT = 360
INTEGRATED_GUIDANCE_MARKER = "[Integrated learning guidance - not evidence]"


def _short_text(value: Any, *, limit: int = 700) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if INTEGRATED_GUIDANCE_MARKER in text:
        text = text.split(INTEGRATED_GUIDANCE_MARKER, 1)[0]
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text.strip()


def _trim_text(value: Any, *, limit: int = DRAFT_PROMPT_TEXT_LIMIT) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _stable_id(*parts: Any, prefix: str) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _text_similarity(left: str, right: str) -> float:
    left = left.strip().lower()
    right = right.strip().lower()
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    seq = difflib.SequenceMatcher(None, left, right).ratio()
    lhs = set(left.split())
    rhs = set(right.split())
    jaccard = len(lhs & rhs) / max(len(lhs | rhs), 1)
    return round((seq + jaccard) / 2, 6)


def _tool_usage_description(doc: ToolDocumentation) -> str:
    description = doc.description.strip() or f"diagnose with `{doc.name}`"
    summary = f"{doc.name} is a primitive diagnostic tool that can {description.rstrip('.')}"
    if doc.parameters:
        summary += " using parameters: " + ", ".join(sorted(doc.parameters)[:8])
    if doc.failure_modes:
        summary += ". Avoid known failure modes: " + "; ".join(doc.failure_modes[:2])
    return summary[:800]


def _library_doc_summary(doc: ToolDocumentation) -> str:
    return (
        doc.tool_usage_description
        or doc.description
        or doc.refined_description(max_chars=200)
    )


def _library_usage_description(docs: dict[str, ToolDocumentation]) -> str:
    if not docs:
        return ""
    summaries = [
        f"- {name}: {_library_doc_summary(doc)}" for name, doc in sorted(docs.items())
    ]
    lines = [
        "DRAFT-refined primitive diagnostic tools:",
        *summaries[:12],
    ]
    return "\n".join(lines)[:4000]


def _path_rates(
    *,
    trials: list[ToolTrial],
    documented_tools_at_start: set[str],
) -> tuple[float, float]:
    unique_tools = {trial.tool_name for trial in trials}
    if not unique_tools:
        return 0.0, 0.0
    documented = unique_tools & documented_tools_at_start
    successful = {trial.tool_name for trial in trials if trial.success}
    return (
        round(len(documented) / len(unique_tools), 6),
        round(len(successful) / len(unique_tools), 6),
    )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {"_raw": text}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _argument_signature(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, default=str)


def _runtime_draft_hints(entries: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    hints: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        if (
            entry.get("agent") != MEMORY_AGENT_NAME
            or entry.get("event") != "skill_transition"
        ):
            continue
        exploration_id = str(entry.get("draft_exploration_id") or "").strip()
        if not exploration_id:
            continue
        tool_name = str(entry.get("tool") or "")
        if not tool_name:
            continue
        arguments = _parse_arguments(entry.get("tool_input"))
        hints[(tool_name, _argument_signature(arguments))].append(
            {
                "planned_exploration_id": exploration_id,
                "planned_next_exploration": str(
                    entry.get("draft_next_exploration") or ""
                ),
            }
        )
    return hints


def _pop_runtime_draft_hint(
    hints: dict[tuple[str, str], list[dict[str, str]]],
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, str]:
    queue = hints.get((tool_name, _argument_signature(arguments)))
    if not queue:
        return {}
    return queue.pop(0)


def extract_tool_trials(
    trace_path: str | Path,
    *,
    session_id: str,
    task_description: str = "",
    agent_filter: str | Collection[str] | None = DIAGNOSIS_AGENT_NAMES,
) -> tuple[list[ToolTrial], dict[str, str]]:
    path = Path(trace_path)
    if not path.exists():
        return [], {}

    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runtime_draft_hints = _runtime_draft_hints(entries)
    starts: dict[str, dict[str, Any]] = {}
    anonymous_starts: list[tuple[str, dict[str, Any]]] = []
    docs: dict[str, str] = {}
    trials: list[ToolTrial] = []
    unnamed_index = 0
    for entry in entries:
        entry_agent = str(entry.get("agent") or "")
        if isinstance(agent_filter, str):
            if entry_agent != agent_filter:
                continue
        elif agent_filter is not None and entry_agent not in agent_filter:
            continue
        event = entry.get("event")
        raw_run_id = str(entry.get("run_id") or "")
        run_id = raw_run_id or f"anon-{unnamed_index}"
        if event == "tool_start":
            unnamed_index += 1
            tool = entry.get("tool") or {}
            name = str(tool.get("name") or "unknown_tool")
            docs.setdefault(name, str(tool.get("description") or ""))
            start = {
                "tool_name": name,
                "arguments": _parse_arguments(entry.get("input")),
                "timestamp": str(entry.get("timestamp") or utc_now()),
            }
            starts[run_id] = start
            if not raw_run_id:
                anonymous_starts.append((run_id, start))
        elif event in {"tool_end", "tool_error"}:
            start = starts.get(run_id)
            if start is None and not raw_run_id and anonymous_starts:
                run_id, start = anonymous_starts.pop(0)
            if start is None:
                continue
            status = "success" if event == "tool_end" else "error"
            output = entry.get("output") or entry.get("error") or ""
            draft_hint = _pop_runtime_draft_hint(
                runtime_draft_hints,
                tool_name=start["tool_name"],
                arguments=start["arguments"],
            )
            trial = ToolTrial(
                trial_id=_stable_id(session_id, run_id, start, output, prefix="trial"),
                session_id=session_id,
                tool_name=start["tool_name"],
                task_description=task_description,
                arguments=start["arguments"],
                status=status,
                output_summary=_short_text(output) if status == "success" else "",
                error_summary=_short_text(output) if status == "error" else "",
                planned_exploration_id=draft_hint.get(
                    "planned_exploration_id",
                    "",
                ),
                planned_next_exploration=draft_hint.get(
                    "planned_next_exploration",
                    "",
                ),
                timestamp=start["timestamp"],
            )
            trials.append(trial)
            starts.pop(run_id, None)
    return trials, docs


def _infer_parameter_doc(name: str, value: Any) -> ToolParameterDoc:
    constraints: list[str] = []
    lowered = name.lower()
    if any(token in lowered for token in ("host", "router", "device", "node")):
        constraints.append("Use exact NIKA/Kathara device names from the scenario.")
    if "interface" in lowered or lowered in {"iface", "ifname"}:
        constraints.append("Use exact interface names observed on the target device.")
    if isinstance(value, bool):
        type_hint = "bool"
    elif isinstance(value, int):
        type_hint = "int"
    elif isinstance(value, float):
        type_hint = "float"
    elif isinstance(value, list):
        type_hint = "list"
    elif isinstance(value, dict):
        type_hint = "object"
    else:
        type_hint = "str"
    return ToolParameterDoc(
        name=name,
        type_hint=type_hint,
        description=f"Observed argument `{name}`.",
        constraints=constraints,
        examples=[value] if value not in (None, "", [], {}) else [],
    )


def identify_comprehension_gaps(trials: list[ToolTrial]) -> list[ComprehensionGap]:
    gaps: list[ComprehensionGap] = []
    for trial in trials:
        if trial.status != "error":
            continue
        text = trial.error_summary.lower()
        if any(marker in text for marker in ("validation", "missing", "invalid")):
            gap_type = "argument_schema"
            recommendation = "Clarify required parameters, expected types, and allowed values."
        elif any(marker in text for marker in ("not found", "unknown", "no such")):
            gap_type = "environment_reference"
            recommendation = (
                "Warn that device/interface/router names must come from observed "
                "topology."
            )
        else:
            gap_type = "precondition"
            recommendation = (
                "Document when the tool can fail and what evidence should be "
                "collected first."
            )
        gaps.append(
            ComprehensionGap(
                gap_id=_stable_id(
                    trial.tool_name,
                    gap_type,
                    trial.error_summary,
                    prefix="gap",
                ),
                tool_name=trial.tool_name,
                gap_type=gap_type,
                evidence=trial.error_summary,
                recommendation=recommendation,
                session_id=trial.session_id,
            )
        )
    return gaps


def identify_diagnostic_semantic_gaps(
    trials: list[ToolTrial],
    *,
    metrics: dict[str, Any],
) -> list[ComprehensionGap]:
    """Find DRAFT gaps where tool calls succeeded but diagnosis stayed weak."""
    if not trials:
        return []
    loc_score = max(
        float(metrics.get("localization_accuracy") or 0.0),
        float(metrics.get("localization_f1") or 0.0),
    )
    rca_score = max(
        float(metrics.get("rca_accuracy") or 0.0),
        float(metrics.get("rca_f1") or 0.0),
    )
    if max(loc_score, rca_score) >= 0.6:
        return []
    latest_success_by_tool: dict[str, ToolTrial] = {}
    for trial in trials:
        if trial.success:
            latest_success_by_tool[trial.tool_name] = trial
    if not latest_success_by_tool:
        return []
    gaps: list[ComprehensionGap] = []
    for sample in latest_success_by_tool.values():
        gaps.append(
            ComprehensionGap(
                gap_id=_stable_id(
                    sample.tool_name,
                    "diagnostic_semantic_gap",
                    sample.output_summary,
                    round(loc_score, 3),
                    round(rca_score, 3),
                    prefix="gap",
                ),
                tool_name=sample.tool_name,
                gap_type="diagnostic_semantic_gap",
                evidence=_trim_text(sample.output_summary, limit=500),
                recommendation=(
                    "Document how this successful output should be interpreted for "
                    "localization/RCA, which contradictory signals matter, and what "
                    "follow-up probe distinguishes competing root causes."
                ),
                session_id=sample.session_id,
            )
        )
    return gaps


def _exploration_from_trial(
    *,
    trial: ToolTrial,
    doc: ToolDocumentation,
    analyzer_suggestion: str = "",
    next_exploration: str = "",
) -> DraftExploration:
    observation = trial.output_summary if trial.success else trial.error_summary
    user_query = trial.task_description or (
        f"Explore `{trial.tool_name}` with arguments "
        f"{json.dumps(trial.arguments, ensure_ascii=False, default=str)}."
    )
    return DraftExploration(
        exploration_id=_stable_id(
            trial.trial_id,
            doc.content_hash(),
            analyzer_suggestion,
            prefix="explore",
        ),
        session_id=trial.session_id,
        tool_name=trial.tool_name,
        user_query=user_query,
        parameters=trial.arguments,
        observation=observation,
        status=trial.status,
        document_hash=doc.content_hash(),
        analyzer_suggestion=analyzer_suggestion,
        next_exploration=next_exploration,
    )


def _diagnosis_scores_are_weak(metrics: dict[str, Any]) -> bool:
    if not metrics:
        return False
    loc_values = [
        float(metrics[key] or 0.0)
        for key in ("localization_accuracy", "localization_f1")
        if key in metrics
    ]
    rca_values = [
        float(metrics[key] or 0.0)
        for key in ("rca_accuracy", "rca_f1")
        if key in metrics
    ]
    loc_score = max(loc_values or [1.0])
    rca_score = max(rca_values or [1.0])
    return min(loc_score, rca_score) < 0.6


_TOOL_VALIDATION_EXPLORATION_MARKERS = (
    "boundary",
    "invalid",
    "schema",
    "validation behavior",
    "parameter semantics",
    "automatic sampling",
    "error case",
)


def _looks_like_tool_validation_exploration(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _TOOL_VALIDATION_EXPLORATION_MARKERS)


def _default_next_exploration(
    doc: ToolDocumentation,
    *,
    diagnosis_context: bool = False,
) -> str:
    if doc.exploration_suggestions:
        latest = doc.exploration_suggestions[-1]
        if diagnosis_context and _looks_like_tool_validation_exploration(latest):
            return (
                "Run one topology-grounded diagnostic check with this tool and "
                "record how the output affects localization/RCA."
            )
        return latest
    if diagnosis_context:
        return (
            "Run one topology-grounded diagnostic check with this tool and "
            "record how the output affects localization/RCA."
        )
    if doc.parameters:
        return (
            "Explore one valid topology-grounded call and one boundary case "
            "to verify parameter semantics."
        )
    return (
        "Explore one minimal valid call using observed topology identifiers "
        "and record how the output should affect localization/RCA."
    )


def _planned_exploration_from_doc(
    *,
    doc: ToolDocumentation,
    session_id: str = "",
    task_description: str = "",
    analyzer_suggestion: str = "",
) -> DraftExploration:
    diagnosis_context = bool(str(task_description or "").strip())
    next_exploration = _default_next_exploration(
        doc,
        diagnosis_context=diagnosis_context,
    )
    parameters = {
        name: param.examples[-1]
        for name, param in sorted(doc.parameters.items())
        if param.examples
    }
    user_query = task_description or (
        f"Actively explore `{doc.name}` for NIKA diagnosis: {next_exploration}"
    )
    return DraftExploration(
        exploration_id=_stable_id(
            "planned",
            session_id,
            doc.name,
            doc.content_hash(),
            next_exploration,
            prefix="explore",
        ),
        session_id=session_id,
        tool_name=doc.name,
        intent="diagnosis_check" if diagnosis_context else "tool_validation",
        user_query=user_query,
        parameters=parameters,
        observation="",
        status="planned",
        document_hash=doc.content_hash(),
        analyzer_suggestion=analyzer_suggestion,
        next_exploration=next_exploration,
    )


def _planned_parameters_match(
    planned_parameters: dict[str, Any],
    trial_arguments: dict[str, Any],
) -> bool:
    if not planned_parameters:
        return True
    for key, value in planned_parameters.items():
        if key not in trial_arguments or trial_arguments[key] != value:
            return False
    return True


def _consume_planned_explorations(
    *,
    state: Any,
    trials: list[ToolTrial],
) -> int:
    consumed = 0
    for trial in trials:
        if trial.planned_exploration_id:
            matched = next(
                (
                    exploration
                    for exploration in state.explorations
                    if exploration.exploration_id == trial.planned_exploration_id
                    and exploration.status == "planned"
                ),
                None,
            )
            if matched is not None:
                summary = trial.output_summary if trial.success else trial.error_summary
                matched.status = "consumed"
                matched.consumed_by_trial_id = trial.trial_id
                matched.consumed_at = utc_now()
                matched.observation = (
                    f"Consumed by {trial.status} trial {trial.trial_id}: "
                    f"{_trim_text(summary, limit=500)}"
                ).strip()
                consumed += 1
                continue
        for exploration in state.explorations:
            if exploration.status != "planned":
                continue
            if exploration.tool_name != trial.tool_name:
                continue
            if not _planned_parameters_match(exploration.parameters, trial.arguments):
                continue
            summary = trial.output_summary if trial.success else trial.error_summary
            exploration.status = "consumed"
            exploration.consumed_by_trial_id = trial.trial_id
            exploration.consumed_at = utc_now()
            exploration.observation = (
                f"Consumed by {trial.status} trial {trial.trial_id}: "
                f"{_trim_text(summary, limit=500)}"
            ).strip()
            consumed += 1
            break
    return consumed


def _planned_exploration_already_covered(
    *,
    state: Any,
    planned: DraftExploration,
    similarity_threshold: float = 0.9,
) -> bool:
    planned_text = (planned.next_exploration or planned.user_query).strip()
    if not planned_text:
        return False
    for exploration in state.explorations:
        if exploration.status not in {"planned", "consumed"}:
            continue
        if exploration.tool_name != planned.tool_name:
            continue
        consumed_text = (
            exploration.next_exploration or exploration.user_query
        ).strip()
        if not consumed_text:
            continue
        if not _planned_parameters_match(exploration.parameters, planned.parameters):
            continue
        if consumed_text == planned_text:
            return True
        if _text_similarity(consumed_text, planned_text) < similarity_threshold:
            continue
        return True
    return False


def _upsert_planned_explorations(
    *,
    state: Any,
    by_tool: dict[str, list[ToolTrial]],
    metrics: dict[str, Any],
    session_id: str = "",
    task_description: str = "",
    documented_path_rate: float = 0.0,
    success_path_rate: float = 0.0,
) -> int:
    weak_diagnosis = _diagnosis_scores_are_weak(metrics)
    seen = {item.exploration_id for item in state.explorations}
    added = 0
    for tool_name, doc in sorted(state.documents.items()):
        trials = by_tool.get(tool_name, [])
        if trials:
            latest_session_id = trials[-1].session_id
            latest_task = trials[-1].task_description
        else:
            latest_session_id = session_id
            latest_task = task_description
        should_plan = not trials or weak_diagnosis or bool(doc.exploration_suggestions)
        if not should_plan:
            continue
        analyzer_suggestion = doc.analyzer_suggestions[-1] if doc.analyzer_suggestions else ""
        planned = _planned_exploration_from_doc(
            doc=doc,
            session_id=latest_session_id,
            task_description=latest_task,
            analyzer_suggestion=analyzer_suggestion,
        )
        already_covered = _planned_exploration_already_covered(
            state=state,
            planned=planned,
        )
        if planned.exploration_id not in seen and not already_covered:
            state.explorations.append(planned)
            seen.add(planned.exploration_id)
            added += 1
            _append_unique(
                doc.exploration_suggestions,
                [planned.next_exploration],
                limit=12,
            )
            _append_unique(doc.explored_queries, [planned.user_query], limit=20)
        state.documents[tool_name] = doc
        _refresh_tool_stats(
            state=state,
            doc=doc,
            convergence_score=doc.last_convergence_score,
            documented_path_rate=documented_path_rate,
            success_path_rate=success_path_rate,
        )
    return added


def _analyzer_suggestion_for_tool(
    *,
    tool_name: str,
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
    doc: ToolDocumentation,
) -> DraftAnalyzerSuggestion:
    error_count = sum(not trial.success for trial in trials)
    success_count = sum(trial.success for trial in trials)
    gap_text = "; ".join(gap.recommendation for gap in gaps[:4])
    semantic_gaps = [
        gap for gap in gaps if gap.gap_type == "diagnostic_semantic_gap"
    ]
    if semantic_gaps:
        suggestion = (
            f"Clarify how successful `{tool_name}` outputs should affect "
            "localization/RCA decisions: "
            + "; ".join(gap.recommendation for gap in semantic_gaps[:3])
        )
        next_exploration = (
            "Run a follow-up diagnostic probe that distinguishes the current "
            "localization/RCA alternatives for the same symptom."
        )
    elif gaps:
        suggestion = (
            f"Clarify `{tool_name}` documentation using observed failures: {gap_text}"
        )
        next_exploration = (
            "Try a minimal call with topology-derived identifiers and one "
            "boundary/error case."
        )
    elif success_count:
        suggestion = (
            f"Preserve successful `{tool_name}` usage patterns and add concise positive examples."
        )
        next_exploration = (
            "Explore a nearby valid input variation to confirm the documented "
            "parameter semantics."
        )
    else:
        suggestion = (
            f"No usable `{tool_name}` execution feedback yet; keep documentation "
            "concise and request a concrete trial."
        )
        next_exploration = "Collect one concrete tool call with explicit arguments and output."
    if error_count and not doc.failure_modes:
        suggestion += " Add known failure modes before adding new examples."
    return DraftAnalyzerSuggestion(
        suggestion_id=_stable_id(
            tool_name,
            [trial.trial_id for trial in trials],
            suggestion,
            prefix="suggest",
        ),
        tool_name=tool_name,
        session_id=trials[-1].session_id if trials else "",
        trial_ids=[trial.trial_id for trial in trials],
        suggestion=suggestion,
        next_exploration=next_exploration,
    )


def _recent_revision_hashes(
    revisions: list[DocumentationRevision],
    tool_name: str,
    *,
    limit: int = 3,
) -> list[str]:
    return [
        revision.after_hash
        for revision in revisions
        if revision.tool_name == tool_name
    ][-limit:]


def _append_unique(target: list[Any], items: list[Any], *, limit: int = 12) -> None:
    for item in items:
        if item in (None, "", [], {}) or item in target:
            continue
        target.append(item)
        if len(target) >= limit:
            break


def _refresh_tool_stats(
    *,
    state: Any,
    doc: ToolDocumentation,
    convergence_score: float | None = None,
    documented_path_rate: float = 0.0,
    success_path_rate: float = 0.0,
) -> None:
    tool_name = doc.name
    all_tool_trials = [
        trial for trial in state.trials if trial.tool_name == tool_name
    ]
    doc.trial_count = len(all_tool_trials)
    doc.success_count = sum(trial.success for trial in all_tool_trials)
    doc.error_count = sum(trial.status == "error" for trial in all_tool_trials)
    if convergence_score is not None:
        doc.last_convergence_score = convergence_score
    state.tool_stats[tool_name] = DraftToolStats(
        tool_name=tool_name,
        trials=doc.trial_count,
        successes=doc.success_count,
        errors=doc.error_count,
        gaps=sum(gap.tool_name == tool_name for gap in state.gaps),
        revisions=sum(rev.tool_name == tool_name for rev in state.revisions),
        llm_rewrites=sum(
            rev.tool_name == tool_name and rev.metrics.get("llm_rewrite") == 1.0
            for rev in state.revisions
        ),
        explorations=sum(item.tool_name == tool_name for item in state.explorations),
        planned_explorations=sum(
            item.tool_name == tool_name and item.status == "planned"
            for item in state.explorations
        ),
        consumed_explorations=sum(
            item.tool_name == tool_name and item.status == "consumed"
            for item in state.explorations
        ),
        mastery_score=doc.mastery_score,
        convergence_score=doc.last_convergence_score,
        documented_path_rate=documented_path_rate,
        success_path_rate=success_path_rate,
        mastered=doc.frozen and "converged" in doc.frozen_reason,
    )


def _draft_rewrite_prompt(
    *,
    tool_name: str,
    doc: ToolDocumentation,
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
    suggestions: list[DraftAnalyzerSuggestion],
    explorations: list[DraftExploration],
    metrics: dict[str, Any],
) -> str:
    doc_payload = doc.model_dump(
        exclude={
            "rewrite_history",
            "explored_queries",
            "updated_at",
            "version",
            "frozen",
            "frozen_reason",
            "trial_count",
            "success_count",
            "error_count",
            "mastery_score",
            "last_convergence_score",
        },
        mode="json",
    )
    doc_payload["description"] = _trim_text(doc_payload.get("description"), limit=500)
    doc_payload["tool_usage_description"] = _trim_text(
        doc_payload.get("tool_usage_description"),
        limit=500,
    )
    doc_payload["usage_notes"] = [
        _trim_text(item) for item in doc_payload.get("usage_notes", [])[-5:]
    ]
    doc_payload["failure_modes"] = [
        _trim_text(item) for item in doc_payload.get("failure_modes", [])[-4:]
    ]
    doc_payload["analyzer_suggestions"] = [
        _trim_text(item) for item in doc_payload.get("analyzer_suggestions", [])[-4:]
    ]
    trial_payload = [
        {
            "session_id": trial.session_id,
            "tool_name": trial.tool_name,
            "arguments": trial.arguments,
            "status": trial.status,
            "output_summary": _trim_text(trial.output_summary),
            "error_summary": _trim_text(trial.error_summary),
            "planned_exploration_id": trial.planned_exploration_id,
            "planned_next_exploration": _trim_text(
                trial.planned_next_exploration
            ),
        }
        for trial in trials[-6:]
    ]
    gap_payload = [gap.model_dump(mode="json") for gap in gaps[-8:]]
    suggestion_payload = [
        {
            "suggestion": _trim_text(item.suggestion, limit=500),
            "next_exploration": _trim_text(item.next_exploration),
        }
        for item in suggestions[-4:]
    ]
    exploration_payload = [
        {
            "tool_name": item.tool_name,
            "parameters": item.parameters,
            "status": item.status,
            "observation": _trim_text(item.observation),
            "analyzer_suggestion": _trim_text(item.analyzer_suggestion, limit=500),
            "next_exploration": _trim_text(item.next_exploration),
        }
        for item in explorations[-4:]
    ]
    return (
        "You are the DRAFT documentation rewriting agent for NIKA primitive "
        "network-diagnosis tools. Refine documentation only; do not invent new "
        "tools, APIs, commands, hidden labels, or benchmark answers.\n\n"
        "DRAFT loop mapping: Explorer = observed diagnosis tool calls; "
        "Analyzer = suggestions from output/error feedback; Rewriter = update "
        "the documentation and propose the next useful exploration direction.\n\n"
        f"Tool: {tool_name}\n"
        f"Current documentation:\n{json.dumps(doc_payload, indent=2, ensure_ascii=False)}\n\n"
        "Explorer observations:\n"
        f"{json.dumps(exploration_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Recent tool trials:\n{json.dumps(trial_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Analyzer suggestions:\n{json.dumps(suggestion_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Comprehension gaps:\n{json.dumps(gap_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Evaluation metrics:\n{json.dumps(metrics, indent=2, ensure_ascii=False, default=str)}\n\n"
        "Return a documentation rewrite that helps future agents understand "
        "preconditions, argument types, allowed values, constraints, common "
        "failure modes, and safe usage examples. Also return "
        "`tool_usage_description` as a concise selection-time tool summary and "
        "`suggestions_for_exploring` as one concise direction for the next "
        "useful tool trial. For `parameters`, return a mapping from parameter "
        "name to a short description."
    )


def _proposal_from_draft(
    draft: DraftRewriteDraft,
    *,
    tool_name: str,
) -> DraftRewriteProposal:
    return DraftRewriteProposal(
        tool_name=draft.tool_name or tool_name,
        description=draft.description,
        tool_usage_description=draft.tool_usage_description,
        preconditions=draft.preconditions,
        parameters={
            name: ToolParameterDoc(
                name=name,
                type_hint="unknown",
                description=description,
            )
            for name, description in draft.parameters.items()
        },
        constraints=draft.constraints,
        failure_modes=draft.failure_modes,
        usage_notes=draft.usage_notes,
        suggestions_for_exploring=draft.suggestions_for_exploring,
        confidence=draft.confidence,
        rationale=draft.rationale,
    )


def _invoke_draft_rewriter(
    *,
    tool_name: str,
    doc: ToolDocumentation,
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
    suggestions: list[DraftAnalyzerSuggestion],
    explorations: list[DraftExploration],
    metrics: dict[str, Any],
    llm_backend: str | None,
    model: str | None,
) -> tuple[DraftRewriteProposal | None, str]:
    selected_backend = learning_backend(llm_backend)
    selected_model = learning_model(model)
    if not selected_backend or not selected_model:
        return None, ""
    try:
        llm = load_model(
            selected_backend,
            selected_model,
            timeout=learning_timeout_seconds(),
            max_retries=learning_max_retries(),
        )
        rewriter = llm.with_structured_output(DraftRewriteDraft)
        raw_proposal = rewriter.invoke(
            _draft_rewrite_prompt(
                tool_name=tool_name,
                doc=doc,
                trials=trials,
                gaps=gaps,
                suggestions=suggestions,
                explorations=explorations,
                metrics=metrics,
            )
        )
        if isinstance(raw_proposal, DraftRewriteProposal):
            proposal = raw_proposal
        else:
            draft = (
                raw_proposal
                if isinstance(raw_proposal, DraftRewriteDraft)
                else DraftRewriteDraft.model_validate(raw_proposal)
            )
            proposal = _proposal_from_draft(draft, tool_name=tool_name)
        if proposal.tool_name != tool_name:
            proposal.tool_name = tool_name
        return proposal, ""
    except Exception as exc:
        return None, format_learning_error(exc)


def _apply_draft_proposal(
    doc: ToolDocumentation,
    proposal: DraftRewriteProposal,
) -> None:
    if proposal.tool_usage_description.strip():
        doc.tool_usage_description = proposal.tool_usage_description.strip()
    if proposal.description.strip():
        doc.description = proposal.description.strip()
    _append_unique(doc.preconditions, proposal.preconditions)
    _append_unique(doc.constraints, proposal.constraints)
    _append_unique(doc.failure_modes, proposal.failure_modes)
    rationale = [proposal.rationale] if proposal.rationale else []
    _append_unique(doc.usage_notes, proposal.usage_notes + rationale)
    if proposal.suggestions_for_exploring:
        _append_unique(
            doc.exploration_suggestions,
            [proposal.suggestions_for_exploring],
            limit=12,
        )
    _append_unique(doc.positive_examples, proposal.positive_examples, limit=10)
    _append_unique(doc.negative_examples, proposal.negative_examples, limit=10)
    for name, param in proposal.parameters.items():
        if not name:
            continue
        current = doc.parameters.get(name)
        if current is None:
            doc.parameters[name] = param
            continue
        if param.type_hint != "unknown":
            current.type_hint = param.type_hint
        if param.description:
            current.description = param.description
        _append_unique(current.constraints, param.constraints, limit=8)
        _append_unique(current.examples, param.examples, limit=5)


def rewrite_documentation(
    store: ToolEvolutionStore,
    *,
    trials: list[ToolTrial],
    tool_descriptions: dict[str, str],
    metrics: dict[str, Any],
    llm_backend: str | None = None,
    model: str | None = None,
    documented_tools_at_start: set[str] | None = None,
    session_id: str = "",
    task_description: str = "",
    convergence_threshold: float = DRAFT_CONVERGENCE_THRESHOLD,
) -> list[DocumentationRevision]:
    state = store.load()
    start_docs = (
        set(state.documents)
        if documented_tools_at_start is None
        else documented_tools_at_start
    )
    documented_path_rate, success_path_rate = _path_rates(
        trials=trials,
        documented_tools_at_start=start_docs,
    )
    for name, description in tool_descriptions.items():
        state.documents.setdefault(
            name,
            ToolDocumentation(name=name, description=description.strip()),
        )

    seen_trials = {trial.trial_id for trial in state.trials}
    for trial in trials:
        if trial.trial_id not in seen_trials:
            state.trials.append(trial)
            seen_trials.add(trial.trial_id)

    by_tool: dict[str, list[ToolTrial]] = defaultdict(list)
    for trial in trials:
        by_tool[trial.tool_name].append(trial)
    planned_consumed = _consume_planned_explorations(state=state, trials=trials)

    revisions: list[DocumentationRevision] = []
    accuracy = float(metrics.get("rca_accuracy") or 0.0)
    for tool_name, tool_trials in by_tool.items():
        doc = state.documents.setdefault(
            tool_name,
            ToolDocumentation(name=tool_name),
        )
        before_hash = doc.content_hash()
        before_description = doc.refined_description(max_chars=4000)
        if doc.frozen:
            _refresh_tool_stats(
                state=state,
                doc=doc,
                documented_path_rate=documented_path_rate,
                success_path_rate=success_path_rate,
            )
            state.documents[tool_name] = doc
            continue

        for trial in tool_trials:
            for key, value in trial.arguments.items():
                if key.startswith("_"):
                    continue
                current = doc.parameters.get(key)
                inferred = _infer_parameter_doc(key, value)
                if current is None:
                    doc.parameters[key] = inferred
                else:
                    for constraint in inferred.constraints:
                        if constraint not in current.constraints:
                            current.constraints.append(constraint)
                    if value not in current.examples and len(current.examples) < 5:
                        current.examples.append(value)

            if trial.success and trial.arguments:
                example = {"arguments": trial.arguments}
                if example not in doc.positive_examples:
                    doc.positive_examples.append(example)
            elif trial.status == "error":
                example = {
                    "arguments": trial.arguments,
                    "error": trial.error_summary[:300],
                }
                if example not in doc.negative_examples:
                    doc.negative_examples.append(example)

        gaps = identify_comprehension_gaps(tool_trials)
        gaps.extend(
            identify_diagnostic_semantic_gaps(tool_trials, metrics=metrics)
        )
        for gap in gaps:
            if gap.recommendation not in doc.usage_notes:
                doc.usage_notes.append(gap.recommendation)
            if gap.evidence and gap.evidence not in doc.failure_modes:
                doc.failure_modes.append(gap.evidence[:300])
            if gap.gap_type == "precondition":
                note = (
                    "Run topology or service discovery before using this tool "
                    "when names are uncertain."
                )
                if note not in doc.preconditions:
                    doc.preconditions.append(note)
            if gap.gap_type == "diagnostic_semantic_gap":
                note = (
                    "After a successful call, interpret the output against the "
                    "current hypothesis and choose a follow-up probe that can "
                    "distinguish localization/RCA alternatives."
                )
                if note not in doc.usage_notes:
                    doc.usage_notes.append(note)
            if not any(item.gap_id == gap.gap_id for item in state.gaps):
                state.gaps.append(gap)

        grounded_constraint = (
            "Tool arguments must be grounded in currently observed topology evidence."
        )
        if grounded_constraint not in doc.constraints:
            doc.constraints.append(grounded_constraint)

        suggestion = _analyzer_suggestion_for_tool(
            tool_name=tool_name,
            trials=tool_trials,
            gaps=gaps,
            doc=doc,
        )
        tool_suggestions = [suggestion]
        if not any(
            item.suggestion_id == suggestion.suggestion_id
            for item in state.analyzer_suggestions
        ):
            state.analyzer_suggestions.append(suggestion)
        _append_unique(doc.analyzer_suggestions, [suggestion.suggestion], limit=20)
        _append_unique(doc.exploration_suggestions, [suggestion.next_exploration], limit=12)

        explorations = [
            _exploration_from_trial(
                trial=trial,
                doc=doc,
                analyzer_suggestion=suggestion.suggestion,
                next_exploration=suggestion.next_exploration,
            )
            for trial in tool_trials
        ]
        seen_explorations = {item.exploration_id for item in state.explorations}
        for exploration in explorations:
            if exploration.exploration_id not in seen_explorations:
                state.explorations.append(exploration)
                seen_explorations.add(exploration.exploration_id)
            _append_unique(doc.explored_queries, [exploration.user_query], limit=20)

        proposal, llm_error = _invoke_draft_rewriter(
            tool_name=tool_name,
            doc=doc,
            trials=tool_trials,
            gaps=gaps,
            suggestions=tool_suggestions,
            explorations=explorations,
            metrics=metrics,
            llm_backend=llm_backend,
            model=model,
        )
        if proposal is not None:
            _apply_draft_proposal(doc, proposal)
        if not doc.tool_usage_description:
            doc.tool_usage_description = _tool_usage_description(doc)

        after_hash = doc.content_hash()
        after_description = doc.refined_description(max_chars=4000)
        convergence_score = _text_similarity(before_description, after_description)
        doc.last_convergence_score = convergence_score
        if after_description not in doc.rewrite_history:
            doc.rewrite_history.append(after_description)
            doc.rewrite_history = doc.rewrite_history[-12:]

        all_tool_trials = [
            trial for trial in state.trials if trial.tool_name == tool_name
        ]
        doc.trial_count = len(all_tool_trials)
        doc.success_count = sum(trial.success for trial in all_tool_trials)
        doc.error_count = sum(trial.status == "error" for trial in all_tool_trials)
        doc.mastery_score = round(
            (doc.success_count / max(doc.trial_count, 1)) * (0.5 + 0.5 * accuracy),
            6,
        )

        recent_hashes = _recent_revision_hashes(state.revisions, tool_name)
        unchanged_streak = (
            len(recent_hashes) >= 2
            and len(set(recent_hashes + [after_hash])) == 1
        )
        converged = (
            convergence_score >= convergence_threshold
            and len(doc.rewrite_history) >= 2
            and doc.mastery_score >= 0.5
        )
        no_signal = not gaps and accuracy <= 0 and len(tool_trials) >= 3
        if unchanged_streak or converged or no_signal:
            doc.frozen = True
            doc.frozen_reason = (
                "DRAFT adaptive termination: documentation converged."
                if converged
                else "DRAFT adaptive termination: no useful documentation change."
            )
        if after_hash != before_hash:
            doc.version += 1
            doc.updated_at = utc_now()
        revision = DocumentationRevision(
            revision_id=_stable_id(
                tool_name,
                before_hash,
                after_hash,
                len(state.revisions),
                prefix="rev",
            ),
            tool_name=tool_name,
            before_hash=before_hash,
            after_hash=after_hash,
            changed=after_hash != before_hash,
            reason=(
                "DRAFT LLM documentation rewrite"
                if proposal is not None
                else "DRAFT trial feedback rewrite"
            ),
            metrics={
                "rca_accuracy": accuracy,
                "tool_error_rate": sum(not item.success for item in tool_trials)
                / max(len(tool_trials), 1),
                "llm_attempted": 1.0 if llm_backend and model else 0.0,
                "llm_rewrite": 1.0 if proposal is not None else 0.0,
                "llm_failed": 1.0 if llm_error else 0.0,
                "convergence_score": convergence_score,
                "mastery_score": doc.mastery_score,
                "documented_path_rate": documented_path_rate,
                "success_path_rate": success_path_rate,
            },
            analyzer_suggestion_ids=[suggestion.suggestion_id],
            llm_error=llm_error,
        )
        state.documents[tool_name] = doc
        state.revisions.append(revision)
        _refresh_tool_stats(
            state=state,
            doc=doc,
            convergence_score=convergence_score,
            documented_path_rate=documented_path_rate,
            success_path_rate=success_path_rate,
        )
        revisions.append(revision)

    planned_added = _upsert_planned_explorations(
        state=state,
        by_tool=by_tool,
        metrics=metrics,
        session_id=session_id,
        task_description=task_description,
        documented_path_rate=documented_path_rate,
        success_path_rate=success_path_rate,
    )
    if planned_added:
        for revision in revisions:
            revision.metrics["planned_explorations_added"] = float(planned_added)
    if planned_consumed:
        for revision in revisions:
            revision.metrics["planned_explorations_consumed"] = float(planned_consumed)

    state.library_usage_description = _library_usage_description(state.documents)
    store.save(state)
    return revisions


def finalize_tool_evolution_session(
    *,
    session_id: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    session = Session()
    session.load_closed_session(session_id=session_id)
    library_id = getattr(session, "tool_library_id", "default")
    raw_convergence_threshold = getattr(
        session,
        "tool_convergence_threshold",
        DRAFT_CONVERGENCE_THRESHOLD,
    )
    if raw_convergence_threshold is None:
        raw_convergence_threshold = DRAFT_CONVERGENCE_THRESHOLD
    convergence_threshold = float(raw_convergence_threshold)
    session_dir = Path(session.session_dir)
    trace_path = session_dir / MESSAGES_FILENAME
    store = ToolEvolutionStore(library_id)
    documented_tools_at_start = set(store.load().documents)
    trials, tool_descriptions = extract_tool_trials(
        trace_path,
        session_id=session_id,
        task_description=str(getattr(session, "task_description", "") or ""),
    )
    added_trials = store.record_trials(trials)
    documented_path_rate, success_path_rate = _path_rates(
        trials=trials,
        documented_tools_at_start=documented_tools_at_start,
    )
    revisions = rewrite_documentation(
        store,
        trials=trials,
        tool_descriptions=tool_descriptions,
        metrics=metrics,
        llm_backend=getattr(session, "llm_backend", None),
        model=getattr(session, "model", None),
        documented_tools_at_start=documented_tools_at_start,
        session_id=session_id,
        task_description=str(getattr(session, "task_description", "") or ""),
        convergence_threshold=convergence_threshold,
    )
    state = store.load()
    llm_attempts = sum(
        revision.metrics.get("llm_attempted") == 1.0 for revision in revisions
    )
    llm_failures = sum(
        revision.metrics.get("llm_failed") == 1.0 for revision in revisions
    )
    llm_errors = [
        revision.llm_error
        for revision in revisions
        if revision.llm_error
    ]
    report = {
        "status": "updated",
        "method": "DRAFT",
        "library_id": store.library_id,
        "draft_trials": len(trials),
        "draft_trials_added": added_trials,
        "draft_document_revisions": sum(revision.changed for revision in revisions),
        "draft_comprehension_gaps": len(state.gaps),
        "draft_frozen_documents": sum(doc.frozen for doc in state.documents.values()),
        "draft_documented_tools": len(state.documents),
        "draft_unique_trial_tools": len({trial.tool_name for trial in trials}),
        "draft_explorations": len(state.explorations),
        "draft_planned_explorations": sum(
            exploration.status == "planned" for exploration in state.explorations
        ),
        "draft_consumed_explorations": sum(
            exploration.status == "consumed" for exploration in state.explorations
        ),
        "draft_analyzer_suggestions": len(state.analyzer_suggestions),
        "draft_mastered_tools": sum(stat.mastered for stat in state.tool_stats.values()),
        "draft_documented_path_rate": documented_path_rate,
        "draft_success_path_rate": success_path_rate,
        "draft_converged_documents": sum(
            "converged" in doc.frozen_reason for doc in state.documents.values()
        ),
        "draft_llm_attempts": llm_attempts,
        "draft_llm_failures": llm_failures,
        "draft_llm_revisions": sum(
            revision.metrics.get("llm_rewrite") == 1.0 for revision in revisions
        ),
        "draft_llm_errors": llm_errors[:5],
        "draft_config": {
            "convergence_threshold": convergence_threshold,
            "tool_doc_chars": getattr(session, "tool_doc_chars", None),
            "prompt_doc_limit": getattr(session, "tool_prompt_doc_limit", None),
            "scoped_prompt_doc_limit": getattr(
                session,
                "tool_scoped_prompt_doc_limit",
                None,
            ),
            "planned_checks": getattr(session, "tool_planned_checks", None),
            "next_checks": getattr(session, "tool_next_checks", None),
        },
    }
    (session_dir / "tool_evolution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
