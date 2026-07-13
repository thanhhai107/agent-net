"""Self-driven DRAFT exploration while the diagnosis tool phase is active."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from agent.tool_refinement.curator import (
    DRAFT_EXPLORATION_SIMILARITY_THRESHOLD,
    _exploration_is_read_only,
    _exploration_signature,
    _stable_id,
    _text_similarity,
    extract_tool_trials,
    rewrite_documentation,
)
from agent.tool_refinement.generalization import is_runtime_identifier_parameter
from agent.tool_refinement.models import (
    DraftExploration,
    DraftExplorerDraft,
    ToolDocumentation,
    ToolTrial,
    utc_now,
)
from agent.tool_refinement.store import ToolRefinementStore
from agent.utils.tool_output import classify_tool_outcome, compact_tool_output
from nika.evaluator.result_log import MESSAGES_FILENAME

DRAFT_EXPLORER_REFLECTION_LIMIT = 3


def _flatten_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {item for value_item in value for item in _flatten_strings(value_item)}
    if isinstance(value, dict):
        return {
            item
            for value_item in value.values()
            for item in _flatten_strings(value_item)
        }
    return set()


def _grounded_identifier_values(
    trials: list[ToolTrial],
    *,
    tool_name: str | None = None,
    session_id: str | None = None,
) -> dict[str, set[str]]:
    grounded: dict[str, set[str]] = {}
    for trial in trials:
        if tool_name is not None and trial.tool_name != tool_name:
            continue
        if session_id is not None and trial.session_id != session_id:
            continue
        for name, value in trial.arguments.items():
            if is_runtime_identifier_parameter(name):
                grounded.setdefault(name, set()).update(_flatten_strings(value))
    return grounded


def _validate_parameters(
    tool: BaseTool,
    doc: ToolDocumentation,
    parameters: dict[str, Any],
    *,
    grounded_identifiers: dict[str, set[str]],
    observed_trials: list[ToolTrial],
) -> tuple[dict[str, Any] | None, str]:
    properties = doc.source_schema.get("properties")
    allowed = set(properties) if isinstance(properties, dict) else set(doc.parameters)
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        return None, "unknown parameters: " + ", ".join(unknown)

    required = set(doc.source_schema.get("required") or [])
    missing = sorted(required - set(parameters))
    if missing:
        return None, "missing required parameters: " + ", ".join(missing)

    normalized = dict(parameters)
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_validate"):
        try:
            parsed = args_schema.model_validate(parameters)
            normalized = parsed.model_dump(exclude_none=True)
        except Exception as exc:
            return None, f"source schema rejected parameters: {exc}"

    for name, value in normalized.items():
        if not is_runtime_identifier_parameter(name):
            continue
        proposed = _flatten_strings(value)
        observed = grounded_identifiers.get(name, set())
        if proposed and (not observed or not proposed <= observed):
            return (
                None,
                f"runtime identifier `{name}` must reuse values observed in this environment",
            )
    property_schemas = properties if isinstance(properties, dict) else {}
    for name, value in normalized.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parameter_schema = property_schemas.get(name)
        parameter_schema = (
            parameter_schema if isinstance(parameter_schema, dict) else {}
        )
        allowed_values = {
            item
            for item in parameter_schema.get("enum", [])
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        }
        default = parameter_schema.get("default")
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            allowed_values.add(default)
        allowed_values.update(
            trial.arguments[name]
            for trial in observed_trials
            if name in trial.arguments
            and isinstance(trial.arguments[name], (int, float))
            and not isinstance(trial.arguments[name], bool)
        )
        minimum = parameter_schema.get("minimum")
        maximum = parameter_schema.get("maximum")
        bounded = (
            isinstance(minimum, (int, float))
            and isinstance(maximum, (int, float))
            and minimum <= value <= maximum
        )
        if value not in allowed_values and not bounded:
            return (
                None,
                f"numeric parameter `{name}` must use a source default, enum, "
                "bounded range, or value observed in this session",
            )
    return normalized, ""


def _explorer_prompt(
    *,
    tool_name: str,
    doc: ToolDocumentation,
    observed_trials: list[ToolTrial],
    prior_explorations: list[DraftExploration],
    grounded_identifiers: dict[str, set[str]],
    feedback: list[str],
) -> str:
    trial_payload = [
        {
            "parameters": trial.arguments,
            "status": trial.status,
            "observation": trial.output_summary or trial.error_summary,
        }
        for trial in observed_trials[-6:]
    ]
    explored_payload = [
        {
            "query": _generalized_exploration_query(item),
            "parameters": _generalized_parameters(item.parameters),
            "status": item.status,
        }
        for item in prior_explorations[-8:]
    ]
    grounded_payload = {
        name: sorted(values) for name, values in sorted(grounded_identifiers.items())
    }
    return (
        "You are the self-driven Explorer in DRAFT. Generate exactly one natural "
        "single-tool exploration that improves understanding of the primitive "
        f"tool `{tool_name}`. The exploration must be read-only and must not infer "
        "a diagnosis, faulty device, root-cause label, or benchmark answer.\n\n"
        "Use only parameter names from the immutable source schema. Runtime host, "
        "router, node, source, target, and interface values must be copied exactly "
        "from `grounded_identifier_values`; never invent topology identifiers. "
        "Other values must satisfy the source schema. Do not generate mutating "
        "commands or actions. Make the query materially different from prior "
        "explorations and follow the Rewriter's next direction when present.\n\n"
        f"Source documentation:\n{doc.description}\n\n"
        f"Source schema:\n{json.dumps(doc.source_schema, ensure_ascii=False)}\n\n"
        f"Current refined documentation:\n{doc.refined_description(max_chars=1800)}\n\n"
        f"Next exploration direction:\n{doc.next_exploration_direction}\n\n"
        f"Grounded identifier values:\n{json.dumps(grounded_payload, ensure_ascii=False)}\n\n"
        f"Observed trials:\n{json.dumps(trial_payload, ensure_ascii=False, default=str)}\n\n"
        f"Prior explorations:\n{json.dumps(explored_payload, ensure_ascii=False, default=str)}\n\n"
        f"Reflection feedback:\n{json.dumps(feedback, ensure_ascii=False)}"
    )


def _generalized_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        name: f"<{name}>" if is_runtime_identifier_parameter(name) else value
        for name, value in parameters.items()
    }


def _generalized_exploration_query(exploration: DraftExploration) -> str:
    query = exploration.user_query
    for name, value in exploration.parameters.items():
        if not is_runtime_identifier_parameter(name):
            continue
        for identifier in sorted(_flatten_strings(value), key=len, reverse=True):
            query = query.replace(identifier, f"<{name}>")
    return query


async def _generate_candidate(
    *,
    llm: Any,
    tool: BaseTool,
    doc: ToolDocumentation,
    trials: list[ToolTrial],
    prior_explorations: list[DraftExploration],
    session_id: str,
    similarity_threshold: float = DRAFT_EXPLORATION_SIMILARITY_THRESHOLD,
    reflection_limit: int = DRAFT_EXPLORER_REFLECTION_LIMIT,
) -> tuple[DraftExplorerDraft | None, dict[str, Any], int, str]:
    current_trials = [trial for trial in trials if trial.session_id == session_id]
    grounded = _grounded_identifier_values(current_trials)
    explorer = llm.with_structured_output(DraftExplorerDraft)
    feedback: list[str] = []
    last_error = ""
    for reflection_count in range(reflection_limit + 1):
        try:
            raw = await explorer.ainvoke(
                _explorer_prompt(
                    tool_name=tool.name,
                    doc=doc,
                    observed_trials=[
                        item for item in current_trials if item.tool_name == tool.name
                    ],
                    prior_explorations=prior_explorations,
                    grounded_identifiers=grounded,
                    feedback=feedback,
                )
            )
            candidate = (
                raw
                if isinstance(raw, DraftExplorerDraft)
                else DraftExplorerDraft.model_validate(raw)
            )
        except Exception as exc:
            return None, {}, reflection_count, f"Explorer generation failed: {exc}"

        parameters, validation_error = _validate_parameters(
            tool,
            doc,
            candidate.parameters,
            grounded_identifiers=grounded,
            observed_trials=current_trials,
        )
        if validation_error:
            last_error = validation_error
            feedback.append(validation_error)
            continue
        assert parameters is not None
        if not _exploration_is_read_only(
            tool_name=tool.name,
            parameters=parameters,
            text=candidate.user_query,
        ):
            last_error = "candidate is not read-only"
            feedback.append(last_error)
            continue

        signature = _exploration_signature(
            tool_name=tool.name,
            user_query=candidate.user_query,
            parameters=parameters,
        )
        max_similarity = max(
            (
                _text_similarity(
                    signature,
                    _exploration_signature(
                        tool_name=item.tool_name,
                        user_query=item.user_query,
                        parameters=item.parameters,
                    ),
                )
                for item in prior_explorations
                if item.tool_name == tool.name
            ),
            default=0.0,
        )
        if max_similarity >= similarity_threshold:
            last_error = (
                f"candidate similarity {max_similarity:.3f} exceeds "
                f"{similarity_threshold:.3f}"
            )
            feedback.append(last_error)
            continue
        return candidate, parameters, reflection_count, ""
    return None, {}, reflection_limit, last_error


async def run_active_exploration(
    *,
    session_id: str,
    session_dir: str | Path,
    task_description: str,
    tools: list[BaseTool],
    store: ToolRefinementStore,
    llm: Any,
    llm_backend: str,
    model: str,
    convergence_threshold: float,
    exploration_similarity_threshold: float = DRAFT_EXPLORATION_SIMILARITY_THRESHOLD,
    explorer_reflection_limit: int = DRAFT_EXPLORER_REFLECTION_LIMIT,
    analyzer_model: str | None = None,
    rewriter_model: str | None = None,
) -> dict[str, Any]:
    """Run one DRAFT exploration episode for each used, unfrozen tool."""

    trace_path = Path(session_dir) / MESSAGES_FILENAME
    passive_trials, _ = extract_tool_trials(
        trace_path,
        session_id=session_id,
        task_description=task_description,
    )
    if not passive_trials:
        return {"status": "skipped", "reason": "no diagnosis tool trials"}
    store.record_trials(passive_trials)
    tools_by_name = {tool.name: tool for tool in tools}
    used_tools = list(dict.fromkeys(trial.tool_name for trial in passive_trials))
    initial_state = store.load()
    grounded = _grounded_identifier_values(
        initial_state.trials,
        session_id=session_id,
    )
    exploration_counts = {
        name: sum(item.tool_name == name for item in initial_state.explorations)
        for name in tools_by_name
    }
    discovery_candidates: list[str] = []
    for name, tool in tools_by_name.items():
        if name in used_tools:
            continue
        doc = initial_state.documents.get(name)
        if doc is None or doc.frozen:
            continue
        required = set(doc.source_schema.get("required") or [])
        required_identifiers = {
            parameter
            for parameter in required
            if is_runtime_identifier_parameter(parameter)
        }
        if all(grounded.get(parameter) for parameter in required_identifiers):
            discovery_candidates.append(name)
    discovery_candidates.sort(key=lambda name: (exploration_counts.get(name, 0), name))
    scheduled_tools = [*used_tools, *discovery_candidates[:1]]
    executed = 0
    skipped: dict[str, str] = {}

    for tool_name in scheduled_tools:
        state = store.load()
        doc = state.documents.get(tool_name)
        tool = tools_by_name.get(tool_name)
        if doc is None or tool is None:
            skipped[tool_name] = "primitive tool or source document unavailable"
            continue
        if doc.frozen:
            skipped[tool_name] = "documentation already converged"
            continue
        if any(
            item.session_id == session_id
            and item.tool_name == tool_name
            and item.trial_id.startswith("active_trial_")
            for item in state.explorations
        ):
            skipped[tool_name] = "active exploration already completed for session"
            continue

        all_trials = list(state.trials)
        prior = [item for item in state.explorations if item.tool_name == tool_name]
        candidate, parameters, reflections, error = await _generate_candidate(
            llm=(
                llm
                if (not analyzer_model or analyzer_model == model)
                and (not rewriter_model or rewriter_model == model)
                else None
            ),
            tool=tool,
            doc=doc,
            trials=all_trials,
            prior_explorations=prior,
            session_id=session_id,
            similarity_threshold=exploration_similarity_threshold,
            reflection_limit=explorer_reflection_limit,
        )
        if candidate is None:
            skipped[tool_name] = error or "no diverse valid exploration"
            continue

        try:
            output = await tool.ainvoke(parameters)
            status = classify_tool_outcome(output, event="tool_end")
            summary = compact_tool_output(output)
        except Exception as exc:
            output = f"{type(exc).__name__}: {exc}"
            status = "error"
            summary = compact_tool_output(output)

        trial_id = _stable_id(
            session_id,
            tool_name,
            candidate.user_query,
            parameters,
            output,
            prefix="active_trial",
        )
        trial = ToolTrial(
            trial_id=trial_id,
            session_id=session_id,
            tool_name=tool_name,
            task_description=candidate.user_query,
            arguments=parameters,
            status=status,
            output_summary=summary if status != "error" else "",
            error_summary=summary if status == "error" else "",
            timestamp=utc_now(),
        )
        signature = _exploration_signature(
            tool_name=tool_name,
            user_query=candidate.user_query,
            parameters=parameters,
        )
        max_similarity = max(
            (
                _text_similarity(
                    signature,
                    _exploration_signature(
                        tool_name=item.tool_name,
                        user_query=item.user_query,
                        parameters=item.parameters,
                    ),
                )
                for item in prior
            ),
            default=0.0,
        )
        exploration = DraftExploration(
            exploration_id=_stable_id(trial_id, prefix="active_explore"),
            session_id=session_id,
            trial_id=trial_id,
            tool_name=tool_name,
            intent=candidate.intent,
            user_query=candidate.user_query,
            parameters=parameters,
            observation=summary,
            status=status,
            document_hash=doc.content_hash(),
            analyzer_suggestion=doc.next_exploration_direction,
            diversity_score=round(1.0 - max_similarity, 6),
            reflection_count=reflections,
            read_only=True,
        )
        with store.exclusive():
            latest_state = store.load()
            if not any(item.trial_id == trial_id for item in latest_state.trials):
                latest_state.trials.append(trial)
            if not any(
                item.exploration_id == exploration.exploration_id
                for item in latest_state.explorations
            ):
                latest_state.explorations.append(exploration)
            store.save(latest_state)
        await asyncio.to_thread(
            rewrite_documentation,
            store,
            trials=[trial],
            tool_descriptions={tool_name: doc.description},
            metrics={},
            llm_backend=llm_backend,
            model=model,
            analyzer_model=analyzer_model,
            rewriter_model=rewriter_model,
            convergence_threshold=convergence_threshold,
            llm=llm,
        )
        executed += 1

    return {
        "status": "completed",
        "used_tools": used_tools,
        "scheduled_tools": scheduled_tools,
        "active_explorations": executed,
        "skipped": skipped,
    }
