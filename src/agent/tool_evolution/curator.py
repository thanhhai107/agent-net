"""Post-incident curation, trace distillation, validation, and metrics."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agent.tool_evolution.models import (
    CompositeStep,
    CompositeTool,
    ToolEvolutionMode,
    ToolParameter,
    ToolUsageExample,
    ToolVerificationReport,
    ValidationEvidence,
)
from agent.tool_evolution.runtime import (
    COMPOSABLE_PRIMITIVE_TOOLS,
    SAFE_PRIMITIVE_TOOLS,
    _validate_argument_safety,
    _validate_step_argument_policy,
)
from agent.tool_evolution.store import ToolEvolutionStore
from agent.utils.mcp_servers import select_diagnosis_servers
from nika.utils.session import Session


_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_CURATION_VERSION = 3


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _output_shape(value: Any) -> str:
    if isinstance(value, dict):
        keys = sorted(str(key) for key in value)[:12]
        return "object fields: " + ", ".join(keys) if keys else "empty object"
    if isinstance(value, list):
        return "list response"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return "text response"
        return _output_shape(parsed)
    if value is None:
        return "missing response"
    return f"{type(value).__name__} response"


def _sanitize_value(
    value: Any,
    devices: set[str],
    forbidden: set[str] | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item, devices, forbidden)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item, devices, forbidden) for item in value]
    if not isinstance(value, str):
        return value
    sanitized = value
    for device in sorted(devices, key=len, reverse=True):
        sanitized = re.sub(
            rf"\b{re.escape(device)}\b",
            "<device>",
            sanitized,
            flags=re.IGNORECASE,
        )
    for token in sorted(forbidden or set(), key=len, reverse=True):
        if token:
            sanitized = re.sub(
                re.escape(token),
                "<redacted>",
                sanitized,
                flags=re.IGNORECASE,
            )
    return _IPV4.sub("<ip>", sanitized)[:300]


def _context_fingerprint(session: Session) -> str:
    scenario = str(getattr(session, "scenario_name", "unknown"))
    tier = str(getattr(session, "scenario_topo_size", "") or "fixed")
    topology = sorted(
        sorted((str(left), str(right)))
        for edge in getattr(session, "topology", []) or []
        if isinstance(edge, (list, tuple)) and len(edge) == 2
        for left, right in [edge]
    )
    topology_hash = hashlib.sha256(
        json.dumps(topology, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"{scenario}:{tier}:{topology_hash}"


def _full_incident_success(metrics: dict[str, Any]) -> bool:
    return all(
        metrics.get(key) == 1.0
        for key in ("detection_score", "localization_accuracy", "rca_accuracy")
    )


def _tool_server(name: str) -> str:
    if name.startswith("frr_"):
        return "kathara_frr_mcp_server"
    if name.startswith("bmv2_"):
        return "kathara_bmv2_mcp_server"
    if name.startswith("influx_"):
        return "kathara_telemetry_mcp_server"
    return "kathara_base_mcp_server"


def _paired_primitive_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for event in events:
        if event.get("agent") != "diagnosis_agent":
            continue
        event_type = event.get("event")
        if event_type == "tool_start":
            name = (event.get("tool") or {}).get("name")
            if name in SAFE_PRIMITIVE_TOOLS:
                pending.append(
                    {
                        "tool": name,
                        "arguments": _parse_input(event.get("input")),
                        "succeeded": None,
                        "output": None,
                        "run_id": str(event.get("run_id", "")),
                    }
                )
        elif event_type in {"tool_end", "tool_error"} and pending:
            run_id = str(event.get("run_id", ""))
            match_index = next(
                (
                    index
                    for index, item in enumerate(pending)
                    if run_id and item.get("run_id") == run_id
                ),
                0,
            )
            call = pending.pop(match_index)
            call["succeeded"] = event_type == "tool_end"
            call["output"] = event.get("output") or event.get("error")
            calls.append(call)
    calls.extend(item for item in pending if item.get("succeeded") is not None)
    return calls


def _composite_outcomes(
    events: list[dict[str, Any]],
) -> tuple[list[str], list[str], int]:
    aliases = {
        str(event.get("source_name")): str(event.get("name"))
        for event in events
        if event.get("event") == "tool_evolution_candidate_verified"
        and event.get("source_name")
        and event.get("name")
    }
    pending: dict[str, list[str]] = {}
    successes: list[str] = []
    errors: list[str] = []
    reuse_count = 0
    for event in events:
        event_type = event.get("event")
        name = str(event.get("name", ""))
        if not name:
            continue
        if event_type == "tool_evolution_composite_start":
            pending.setdefault(name, []).append(str(event.get("status", "")))
            continue
        if event_type not in {
            "tool_evolution_composite_end",
            "tool_evolution_composite_error",
        }:
            continue
        statuses = pending.get(name, [])
        status = statuses.pop(0) if statuses else ""
        persisted_name = aliases.get(name, name)
        if event_type == "tool_evolution_composite_end":
            successes.append(persisted_name)
            if status != "ephemeral":
                reuse_count += 1
        else:
            errors.append(persisted_name)
    return successes, errors, reuse_count


def _curate_mastery(
    store: ToolEvolutionStore,
    calls: list[dict[str, Any]],
    *,
    devices: set[str],
    model: str,
    forbidden: set[str] | None = None,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        grouped.setdefault(call["tool"], []).append(call)

    for tool_name, tool_calls in grouped.items():
        successful = [call for call in tool_calls if call["succeeded"]]
        failed = [call for call in tool_calls if not call["succeeded"]]
        argument_fields = sorted(
            {
                str(field)
                for call in tool_calls
                for field in call["arguments"]
            }
        )
        unique_invocations = {
            json.dumps(call["arguments"], sort_keys=True, default=str)
            for call in tool_calls
        }
        preconditions = []
        if successful and failed:
            preconditions.append(
                "Observed outcomes vary by invocation; verify target availability and "
                "tool-specific preconditions before retrying."
            )
        parameter_guidance = []
        if argument_fields:
            parameter_guidance.append(
                "Observed parameter fields: " + ", ".join(argument_fields) + "."
            )
        output_interpretation = []
        if successful:
            shapes = sorted(
                {
                    _output_shape(call.get("output"))
                    for call in successful
                }
            )
            output_interpretation.append(
                "Observed successful response shapes: "
                + "; ".join(shapes)
                + ". A completed invocation is evidence collection, not proof of "
                "the root cause; interpret returned fields with topology context."
            )
        failure_semantics = []
        if failed:
            failure_semantics.append(
                "An observed invocation failed; verify parameter names, target "
                "availability, and prerequisite state before changing diagnosis."
            )
        prior = store.load().mastery.get(tool_name)
        prior_rate = (
            prior.successes / prior.calls if prior is not None and prior.calls else 0.0
        )
        current_rate = len(successful) / len(tool_calls)
        store.upsert_mastery(
            tool_name,
            preconditions=preconditions,
            parameter_guidance=parameter_guidance,
            output_interpretation=output_interpretation,
            failure_semantics=failure_semantics,
            calls=len(tool_calls),
            successes=len(successful),
            errors=len(failed),
            source_model=model,
            revision_source="rewriter",
            rationale=(
                "Explorer observed "
                f"{len(unique_invocations)} distinct invocation pattern(s); Analyzer "
                f"compared {len(successful)} success(es) and {len(failed)} failure(s)."
            ),
            utility_delta=current_rate - prior_rate,
        )
        for call in tool_calls[-6:]:
            store.upsert_mastery(
                tool_name,
                usage_example=ToolUsageExample(
                    arguments=_sanitize_value(
                        call["arguments"],
                        devices,
                        forbidden,
                    ),
                    succeeded=bool(call["succeeded"]),
                ),
                source_model=model,
                revision_source="explorer",
            )


def _minimal_successful_trace(
    calls: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Remove failed, non-composable, and exact duplicate calls in evidence order."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        if not call.get("succeeded"):
            continue
        tool_name = str(call.get("tool", ""))
        arguments = call.get("arguments", {})
        if tool_name not in COMPOSABLE_PRIMITIVE_TOOLS or not isinstance(arguments, dict):
            continue
        try:
            _validate_argument_safety(arguments, allow_placeholders=False)
            _validate_step_argument_policy(
                tool_name,
                arguments,
                allow_placeholders=False,
            )
        except ValueError:
            continue
        signature = json.dumps(
            {
                "tool": tool_name,
                "arguments": arguments,
            },
            sort_keys=True,
            default=str,
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(call)
        if len(selected) >= limit:
            break
    return selected


def _distill_trace(
    store: ToolEvolutionStore,
    calls: list[dict[str, Any]],
    *,
    scenario_name: str,
    deduplicate: bool,
    context_fingerprint: str,
    validation_enabled: bool = True,
) -> tuple[str | None, bool]:
    successful = _minimal_successful_trace(calls)
    if len(successful) < 2:
        return None, False

    parameters: list[ToolParameter] = []
    parameter_names: set[str] = set()
    value_parameters: dict[str, str] = {}
    steps: list[CompositeStep] = []
    for step_index, call in enumerate(successful):
        arguments: dict[str, Any] = {}
        for arg_name, value in call["arguments"].items():
            if isinstance(value, str) and value:
                value_key = json.dumps(value, sort_keys=True)
                if value_key in value_parameters:
                    arguments[arg_name] = f"${{{value_parameters[value_key]}}}"
                    continue
                base = re.sub(r"[^a-z0-9_]+", "_", arg_name.lower()).strip("_")
                base = base or f"value_{step_index + 1}"
                parameter_name = base
                suffix = 2
                while parameter_name in parameter_names:
                    parameter_name = f"{base}_{suffix}"
                    suffix += 1
                parameter_names.add(parameter_name)
                parameters.append(
                    ToolParameter(
                        name=parameter_name,
                        type="str",
                        description=f"Value passed to '{arg_name}' for diagnostic step {step_index + 1}.",
                    )
                )
                value_parameters[value_key] = parameter_name
                arguments[arg_name] = f"${{{parameter_name}}}"
            else:
                arguments[arg_name] = value
        steps.append(
            CompositeStep(
                tool=call["tool"],
                arguments=arguments,
                label=f"Collect evidence with {call['tool']}.",
            )
        )
        _validate_argument_safety(arguments, allow_placeholders=True)
        _validate_step_argument_policy(
            str(call["tool"]),
            arguments,
            allow_placeholders=True,
        )

    signature_seed = json.dumps(
        [{"tool": step.tool, "arguments": step.arguments} for step in steps],
        sort_keys=True,
        default=str,
    )
    short_hash = hashlib.sha256(signature_seed.encode("utf-8")).hexdigest()[:8]
    tool_names = "_".join(step.tool for step in steps[:2])
    composite = CompositeTool(
        name=f"workflow_{tool_names}_{short_hash}"[:64],
        description=(
            "Reusable read-only diagnostic workflow distilled from a successful "
            "network investigation. It collects complementary evidence in sequence."
        ),
        parameters=parameters,
        steps=steps,
        output_contract=[step.label for step in steps],
        tags=[scenario_name],
        source_trace_hash=short_hash,
        verification_reports=[
            ToolVerificationReport(
                stage="structural",
                passed=True,
                checks=[
                    "successful calls only",
                    "exact duplicate calls removed",
                    "concrete string values parameterized",
                    "read-only primitive allowlist",
                ],
                context_fingerprint=context_fingerprint,
            )
        ],
    )
    registered, created = store.register_composite(
        composite,
        deduplicate=deduplicate,
    )
    if created:
        store.record_verification(
            registered.name,
            ToolVerificationReport(
                stage="structural",
                passed=True,
                checks=["distilled artifact passed static validation"],
                context_fingerprint=context_fingerprint,
            ),
        )
    return registered.name, created


def finalize_tool_evolution_session(
    *,
    session_id: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    session = Session().load_closed_session(session_id=session_id)
    if not bool(getattr(session, "tool_evolution_enabled", False)):
        return {}

    library_id = str(getattr(session, "tool_library_id", "default"))
    mode = ToolEvolutionMode(str(getattr(session, "tool_evolution_mode", "dual")))
    store = ToolEvolutionStore(library_id)
    session_dir = Path(session.session_dir)
    artifact_path = session_dir / "tool_evolution.json"
    if artifact_path.exists():
        existing_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if existing_artifact.get("curation_version") == _CURATION_VERSION:
            return existing_artifact
    events = _load_events(session_dir / "messages.jsonl")
    calls = _paired_primitive_calls(events)
    devices: set[str] = set()
    for edge in getattr(session, "topology", []) or []:
        if not isinstance(edge, (list, tuple)):
            continue
        for item in edge:
            if not item:
                continue
            endpoint = str(item)
            devices.add(endpoint)
            devices.add(endpoint.split(":", 1)[0])
    update_enabled = bool(
        getattr(session, "tool_evolution_update_enabled", True)
    )
    if update_enabled and mode.mastery_enabled:
        forbidden = {
            str(getattr(session, "session_id", "")),
            *(
                str(item)
                for item in getattr(session, "problem_names", []) or []
            ),
        }
        _curate_mastery(
            store,
            calls,
            devices=devices,
            model=str(getattr(session, "model", "")),
            forbidden=forbidden,
        )

    incident_success = _full_incident_success(metrics)
    fingerprint = _context_fingerprint(session)
    composite_successes, composite_errors, tool_reuse_count = _composite_outcomes(
        events
    )
    called_primitives = [call["tool"] for call in calls]
    expected_servers = set(
        select_diagnosis_servers(
            str(getattr(session, "scenario_name", "")),
            list(getattr(session, "problem_names", []) or []),
            oracle=True,
        )
    )
    called_servers = {_tool_server(name) for name in called_primitives}
    tool_selection_recall = (
        len(expected_servers & called_servers) / len(expected_servers)
        if expected_servers
        else 1.0
    )
    recovered_errors = 0
    for index, call in enumerate(calls):
        if call.get("succeeded"):
            continue
        if any(later.get("succeeded") for later in calls[index + 1 :]):
            recovered_errors += 1
    promoted: list[str] = []
    regressed: list[str] = []
    if update_enabled:
        for name in [item for item in composite_successes if item]:
            composite = store.get_composite(str(name))
            semantic_valid = bool(
                composite
                and any(
                    report.passed and report.stage == "semantic"
                    for report in composite.verification_reports
                )
            )
            updated = store.record_composite_evidence(
                str(name),
                ValidationEvidence(
                    context_fingerprint=fingerprint,
                    execution_success=True,
                    incident_success=incident_success,
                    source="runtime",
                    semantic_valid=semantic_valid,
                ),
                validation_enabled=mode.validation_enabled,
            )
            if updated and updated.status == "promoted":
                promoted.append(updated.name)
        for name in [item for item in composite_errors if item]:
            updated = store.record_composite_evidence(
                str(name),
                ValidationEvidence(
                    context_fingerprint=fingerprint,
                    execution_success=False,
                    incident_success=False,
                    source="runtime",
                ),
                validation_enabled=mode.validation_enabled,
            )
            if updated and updated.status in {"candidate", "rejected"}:
                regressed.append(updated.name)

    distilled_name: str | None = None
    distilled_created = False
    if update_enabled and mode.distillation_enabled and incident_success:
        distilled_name, distilled_created = _distill_trace(
            store,
            calls,
            scenario_name=str(getattr(session, "scenario_name", "")),
            deduplicate=mode.dedup_enabled,
            context_fingerprint=fingerprint,
            validation_enabled=mode.validation_enabled,
        )

    initial_artifact_path = session_dir / "tool_evolution_session.json"
    initial = (
        json.loads(initial_artifact_path.read_text(encoding="utf-8"))
        if initial_artifact_path.exists()
        else {}
    )
    state = store.load()
    created_tools = set(initial.get("created_tools", []))
    created_tools.update(
        str(event.get("name"))
        for event in events
        if event.get("event") == "tool_evolution_candidate_verified"
        and event.get("created")
        and event.get("name")
    )
    if distilled_name and distilled_created:
        created_tools.add(distilled_name)
    ephemeral_names = {
        str(event.get("name"))
        for event in events
        if event.get("event") == "tool_evolution_ephemeral_created"
        and event.get("name")
    }
    verified_sources = {
        str(event.get("source_name"))
        for event in events
        if event.get("event") == "tool_evolution_candidate_verified"
        and event.get("source_name")
    }
    unverified_ephemeral = set(
        initial.get("unverified_ephemeral_tools", [])
    ) | (ephemeral_names - verified_sources)
    mastery_updated_tools = {call["tool"] for call in calls}
    mastery_updated_tools.update(
        str(event.get("tool_name"))
        for event in events
        if event.get("event") == "tool_evolution_mastery_recorded"
        and event.get("tool_name")
    )
    artifact = {
        **initial,
        "curation_version": _CURATION_VERSION,
        "library_id": store.library_id,
        "mode": mode.value,
        "incident_success": incident_success,
        "update_enabled": update_enabled,
        "primitive_calls": len(calls),
        "composite_calls": len(composite_successes) + len(composite_errors),
        "mastery_updates": (
            len(mastery_updated_tools)
            if update_enabled and mode.mastery_enabled
            else 0
        ),
        "argument_validity": (
            sum(bool(call.get("succeeded")) for call in calls) / len(calls)
            if calls
            else 1.0
        ),
        "error_recovery_count": recovered_errors,
        "tool_selection_recall": round(tool_selection_recall, 4),
        "created_tools": sorted(created_tools),
        "distilled_tool": distilled_name,
        "composite_successes": len(composite_successes),
        "composite_errors": len(composite_errors),
        "promoted_tools": sorted(set(promoted)),
        "regressed_tools": sorted(set(regressed)),
        "library_candidates": sum(
            item.status == "candidate" for item in state.composites.values()
        ),
        "library_promoted": sum(
            item.status == "promoted" for item in state.composites.values()
        ),
        "library_mastered_primitives": len(state.mastery),
        "tool_card_revisions": sum(
            len(item.revisions) for item in state.mastery.values()
        ),
        "capability_gaps": len(state.capability_gaps),
        "verified_composites": sum(
            any(
                report.passed and report.stage == "semantic"
                for report in item.verification_reports
            )
            for item in state.composites.values()
        ),
        "unverified_ephemeral_tools": len(unverified_ephemeral),
        "cross_model_reused_tools": len(initial.get("cross_model_mastery", [])),
        "tool_reuse_count": tool_reuse_count,
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )
    return artifact
