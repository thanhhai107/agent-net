"""Offline DRAFT learning hook.

DRAFT learns by comparing tool trials against their documentation, identifying
where the agent misunderstood arguments or preconditions, then rewriting the
documentation.  This implementation is gradient-free and never creates new
primitive tools.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.tool_evolution.models import (
    ComprehensionGap,
    DocumentationRevision,
    DraftRewriteProposal,
    ToolDocumentation,
    ToolParameterDoc,
    ToolTrial,
    utc_now,
)
from agent.tool_evolution.store import ToolEvolutionStore
from agent.llm.model_factory import load_model
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


def _short_text(value: Any, *, limit: int = 700) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _stable_id(*parts: Any, prefix: str) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


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


def extract_tool_trials(
    trace_path: str | Path,
    *,
    session_id: str,
    agent_filter: str | None = "diagnosis_agent",
) -> tuple[list[ToolTrial], dict[str, str]]:
    path = Path(trace_path)
    if not path.exists():
        return [], {}

    starts: dict[str, dict[str, Any]] = {}
    docs: dict[str, str] = {}
    trials: list[ToolTrial] = []
    unnamed_index = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if agent_filter and entry.get("agent") != agent_filter:
            continue
        event = entry.get("event")
        run_id = str(entry.get("run_id") or f"anon-{unnamed_index}")
        if event == "tool_start":
            unnamed_index += 1
            tool = entry.get("tool") or {}
            name = str(tool.get("name") or "unknown_tool")
            docs.setdefault(name, str(tool.get("description") or ""))
            starts[run_id] = {
                "tool_name": name,
                "arguments": _parse_arguments(entry.get("input")),
                "timestamp": str(entry.get("timestamp") or utc_now()),
            }
        elif event in {"tool_end", "tool_error"}:
            start = starts.get(run_id)
            if start is None:
                continue
            status = "success" if event == "tool_end" else "error"
            output = entry.get("output") or entry.get("error") or ""
            trial = ToolTrial(
                trial_id=_stable_id(session_id, run_id, start, output, prefix="trial"),
                session_id=session_id,
                tool_name=start["tool_name"],
                arguments=start["arguments"],
                status=status,
                output_summary=_short_text(output) if status == "success" else "",
                error_summary=_short_text(output) if status == "error" else "",
                timestamp=start["timestamp"],
            )
            trials.append(trial)
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
            recommendation = "Warn that device/interface/router names must come from observed topology."
        else:
            gap_type = "precondition"
            recommendation = "Document when the tool can fail and what evidence should be collected first."
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


def _draft_rewrite_prompt(
    *,
    tool_name: str,
    doc: ToolDocumentation,
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
    metrics: dict[str, Any],
) -> str:
    trial_payload = [
        trial.model_dump(exclude={"trial_id"}, mode="json") for trial in trials[-12:]
    ]
    gap_payload = [gap.model_dump(mode="json") for gap in gaps[-8:]]
    return (
        "You are the DRAFT documentation rewriting agent for NIKA primitive "
        "network-diagnosis tools. Refine documentation only; do not invent new "
        "tools, APIs, commands, hidden labels, or benchmark answers.\n\n"
        f"Tool: {tool_name}\n"
        f"Current documentation:\n{doc.model_dump_json(indent=2)}\n\n"
        f"Recent tool trials:\n{json.dumps(trial_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Comprehension gaps:\n{json.dumps(gap_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Evaluation metrics:\n{json.dumps(metrics, indent=2, ensure_ascii=False, default=str)}\n\n"
        "Return a documentation rewrite that helps future agents understand "
        "preconditions, argument types, allowed values, constraints, common "
        "failure modes, and safe usage examples."
    )


def _invoke_draft_rewriter(
    *,
    tool_name: str,
    doc: ToolDocumentation,
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
    metrics: dict[str, Any],
    llm_backend: str | None,
    model: str | None,
) -> DraftRewriteProposal | None:
    if not llm_backend or not model:
        return None
    try:
        rewriter = load_model(llm_backend, model).with_structured_output(
            DraftRewriteProposal
        )
        proposal = rewriter.invoke(
            _draft_rewrite_prompt(
                tool_name=tool_name,
                doc=doc,
                trials=trials,
                gaps=gaps,
                metrics=metrics,
            )
        )
        if not isinstance(proposal, DraftRewriteProposal):
            proposal = DraftRewriteProposal.model_validate(proposal)
        if proposal.tool_name != tool_name:
            proposal.tool_name = tool_name
        return proposal
    except Exception:
        return None


def _apply_draft_proposal(
    doc: ToolDocumentation,
    proposal: DraftRewriteProposal,
) -> None:
    if proposal.description.strip():
        doc.description = proposal.description.strip()
    _append_unique(doc.preconditions, proposal.preconditions)
    _append_unique(doc.constraints, proposal.constraints)
    _append_unique(doc.failure_modes, proposal.failure_modes)
    _append_unique(doc.usage_notes, proposal.usage_notes + ([proposal.rationale] if proposal.rationale else []))
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
) -> list[DocumentationRevision]:
    state = store.load()
    for name, description in tool_descriptions.items():
        state.documents.setdefault(
            name,
            ToolDocumentation(name=name, description=description.strip()),
        )

    by_tool: dict[str, list[ToolTrial]] = defaultdict(list)
    for trial in trials:
        by_tool[trial.tool_name].append(trial)

    revisions: list[DocumentationRevision] = []
    accuracy = float(metrics.get("rca_accuracy") or 0.0)
    for tool_name, tool_trials in by_tool.items():
        doc = state.documents.setdefault(tool_name, ToolDocumentation(name=tool_name))
        before_hash = doc.content_hash()
        if doc.frozen:
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

        gaps = [gap for gap in identify_comprehension_gaps(tool_trials)]
        for gap in gaps:
            if gap.recommendation not in doc.usage_notes:
                doc.usage_notes.append(gap.recommendation)
            if gap.evidence and gap.evidence not in doc.failure_modes:
                doc.failure_modes.append(gap.evidence[:300])
            if gap.gap_type == "precondition":
                note = "Run topology or service discovery before using this tool when names are uncertain."
                if note not in doc.preconditions:
                    doc.preconditions.append(note)
            if not any(item.gap_id == gap.gap_id for item in state.gaps):
                state.gaps.append(gap)

        if "Tool arguments must be grounded in currently observed topology evidence." not in doc.constraints:
            doc.constraints.append(
                "Tool arguments must be grounded in currently observed topology evidence."
            )

        proposal = _invoke_draft_rewriter(
            tool_name=tool_name,
            doc=doc,
            trials=tool_trials,
            gaps=gaps,
            metrics=metrics,
            llm_backend=llm_backend,
            model=model,
        )
        if proposal is not None:
            _apply_draft_proposal(doc, proposal)

        after_hash = doc.content_hash()
        recent_hashes = _recent_revision_hashes(state.revisions, tool_name)
        unchanged_streak = len(recent_hashes) >= 2 and len(set(recent_hashes + [after_hash])) == 1
        if unchanged_streak or (not gaps and accuracy <= 0 and len(tool_trials) >= 3):
            doc.frozen = True
            doc.frozen_reason = "DRAFT adaptive termination: no useful documentation change."
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
                "llm_rewrite": 1.0 if proposal is not None else 0.0,
            },
        )
        state.documents[tool_name] = doc
        state.revisions.append(revision)
        revisions.append(revision)

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
    session_dir = Path(session.session_dir)
    trace_path = session_dir / MESSAGES_FILENAME
    trials, tool_descriptions = extract_tool_trials(trace_path, session_id=session_id)
    store = ToolEvolutionStore(library_id)
    added_trials = store.record_trials(trials)
    revisions = rewrite_documentation(
        store,
        trials=trials,
        tool_descriptions=tool_descriptions,
        metrics=metrics,
        llm_backend=getattr(session, "llm_backend", None),
        model=getattr(session, "model", None),
    )
    state = store.load()
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
        "draft_llm_revisions": sum(
            revision.metrics.get("llm_rewrite") == 1.0 for revision in revisions
        ),
    }
    (session_dir / "tool_evolution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
