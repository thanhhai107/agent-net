"""Offline DRAFT training hook.

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
import math
import re
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any

from agent.training_llm import (
    format_training_error,
    training_backend,
    training_max_retries,
    training_model,
    training_timeout_seconds,
)
from agent.module_config import module_defaults
from agent.extensions.llm import load_extension_model as load_model
from agent.tool_refinement.models import (
    ComprehensionGap,
    DraftAnalyzerDraft,
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
from agent.tool_refinement.generalization import (
    generalize_tool_documentation,
    is_runtime_identifier_parameter,
)
from agent.tool_refinement.store import ToolRefinementStore
from agent.utils.tool_output import classify_tool_outcome, compact_tool_output
from agent.utils.phases import DIAGNOSIS
from nika.evaluator.result_log import MESSAGES_FILENAME
from nika.utils.session import Session


_DEFAULTS = module_defaults().tool_refinement
DRAFT_CONVERGENCE_THRESHOLD = _DEFAULTS.convergence_threshold
DRAFT_EXPLORATION_SIMILARITY_THRESHOLD = _DEFAULTS.exploration_similarity_threshold
DIAGNOSIS_AGENT_NAMES = frozenset({DIAGNOSIS, "diagnosis_agent"})
DRAFT_PROMPT_TEXT_LIMIT = 360
INTEGRATED_GUIDANCE_MARKER = "[Integrated training guidance - not evidence]"


def _short_text(value: Any, *, limit: int = 700) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    if INTEGRATED_GUIDANCE_MARKER in text:
        text = text.split(INTEGRATED_GUIDANCE_MARKER, 1)[0]
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text.strip()


def _training_updates_allowed(session: Any) -> bool:
    value = getattr(session, "allow_training_updates", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _trim_text(value: Any, *, limit: int = DRAFT_PROMPT_TEXT_LIMIT) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _stable_id(*parts: Any, prefix: str) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _tool_outcome(event: str, output: Any) -> str:
    """Classify semantic tool outcomes without treating observations as failures."""
    return classify_tool_outcome(output, event=event)


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


def _ngram_word_match(left: str, right: str, *, max_n: int = 4) -> float:
    """BLEU-like word match without an external tokenizer dependency."""
    lhs = left.lower().split()
    rhs = right.lower().split()
    if not lhs and not rhs:
        return 1.0
    if not lhs or not rhs:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        left_ngrams = {
            tuple(lhs[index : index + n]) for index in range(max(0, len(lhs) - n + 1))
        }
        right_ngrams = {
            tuple(rhs[index : index + n]) for index in range(max(0, len(rhs) - n + 1))
        }
        if not left_ngrams or not right_ngrams:
            continue
        precisions.append(len(left_ngrams & right_ngrams) / max(len(right_ngrams), 1))
    if not precisions:
        return 0.0
    brevity = min(1.0, len(rhs) / max(len(lhs), 1))
    return round(brevity * sum(precisions) / len(precisions), 6)


def _document_convergence(left: str, right: str) -> tuple[float, float, float]:
    """Return word-match, semantic proxy, and balanced convergence score."""
    word_match = _ngram_word_match(left, right)
    semantic_match = _text_similarity(left, right)
    return (
        word_match,
        semantic_match,
        round(
            (word_match + semantic_match) / 2,
            6,
        ),
    )


def _tool_usage_description(doc: ToolDocumentation) -> str:
    description = doc.description.strip() or f"diagnose with `{doc.name}`"
    summary = (
        f"{doc.name} is a primitive diagnostic tool that can {description.rstrip('.')}"
    )
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


def _source_parameter_names(doc: ToolDocumentation) -> set[str]:
    properties = doc.source_schema.get("properties")
    if isinstance(properties, dict):
        return set(properties)
    return set()


def _allowed_parameter_names(
    doc: ToolDocumentation,
    trials: list[ToolTrial],
) -> set[str]:
    if doc.source_contract_version > 0:
        return _source_parameter_names(doc)
    # Compatibility for stores built directly by the offline curator, where a
    # primitive BaseTool contract is not available. Successful tool calls are
    # the only authoritative parameter evidence in that mode.
    return set(doc.parameters).union(
        key for trial in trials for key in trial.arguments if not key.startswith("_")
    )


def _proposal_contract_errors(
    proposal: DraftRewriteProposal,
    *,
    allowed_parameter_names: set[str],
    doc: ToolDocumentation,
) -> list[str]:
    errors: list[str] = []
    unknown = sorted(set(proposal.parameters) - allowed_parameter_names)
    if unknown:
        errors.append(
            "proposed parameters are absent from the primitive schema: "
            + ", ".join(unknown)
        )
    allowed_numbers = _numeric_tokens(
        json.dumps(doc.source_schema, ensure_ascii=False, default=str)
        + " "
        + doc.description
    )
    proposed_numbers = _numeric_tokens(_proposal_contract_text(proposal))
    unsupported_numbers = sorted(proposed_numbers - allowed_numbers)
    if doc.source_contract_version > 0 and unsupported_numbers:
        errors.append(
            "proposed numeric constraints are absent from the primitive schema: "
            + ", ".join(unsupported_numbers)
        )
    return errors


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?", str(text)))


def _proposal_contract_text(proposal: DraftRewriteProposal) -> str:
    payload = proposal.model_dump(
        mode="json",
        exclude={"confidence", "rationale", "positive_examples", "negative_examples"},
    )
    return json.dumps(payload, ensure_ascii=False, default=str)


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
            output = entry.get("output") or entry.get("error") or ""
            status = _tool_outcome(str(event), output)
            trial = ToolTrial(
                trial_id=_stable_id(session_id, run_id, start, output, prefix="trial"),
                session_id=session_id,
                tool_name=start["tool_name"],
                task_description=task_description,
                arguments=start["arguments"],
                status=status,
                output_summary=(
                    compact_tool_output(output) if status != "error" else ""
                ),
                error_summary=(
                    compact_tool_output(output) if status == "error" else ""
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
        constraints.append(
            "Use an exact device identifier observed in the current environment."
        )
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
            recommendation = (
                "Clarify required parameters, expected types, and allowed values."
            )
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


def _exploration_from_trial(
    *,
    trial: ToolTrial,
    doc: ToolDocumentation,
    analyzer_suggestion: str = "",
    prior_explorations: list[DraftExploration] | None = None,
) -> DraftExploration:
    observation = trial.output_summary if trial.success else trial.error_summary
    parameter_names = ", ".join(sorted(trial.arguments)) or "no parameters"
    user_query = f"Explore `{trial.tool_name}` using its documented {parameter_names}."
    read_only = _exploration_is_read_only(
        tool_name=trial.tool_name,
        parameters=trial.arguments,
        text=user_query,
    )
    signature = _exploration_signature(
        tool_name=trial.tool_name,
        user_query=user_query,
        parameters=trial.arguments,
    )
    similarities = [
        _text_similarity(
            signature,
            _exploration_signature(
                tool_name=item.tool_name,
                user_query=item.user_query,
                parameters=item.parameters,
            ),
        )
        for item in (prior_explorations or [])[-12:]
        if item.tool_name == trial.tool_name
    ]
    max_similarity = max(similarities or [0.0])
    return DraftExploration(
        exploration_id=_stable_id(
            trial.trial_id,
            prefix="explore",
        ),
        session_id=trial.session_id,
        trial_id=trial.trial_id,
        tool_name=trial.tool_name,
        intent="tool_validation",
        user_query=user_query,
        parameters=trial.arguments,
        observation=observation,
        status=trial.status if read_only else "invalidated",
        document_hash=doc.content_hash(),
        analyzer_suggestion=analyzer_suggestion,
        diversity_score=round(1.0 - max_similarity, 6),
        reflection_count=(
            1 if max_similarity >= DRAFT_EXPLORATION_SIMILARITY_THRESHOLD else 0
        ),
        read_only=read_only,
    )


def _exploration_signature(
    *,
    tool_name: str,
    user_query: str,
    parameters: dict[str, Any],
) -> str:
    normalized_query = str(user_query)
    normalized_parameters: dict[str, Any] = {}
    for name, value in parameters.items():
        if not is_runtime_identifier_parameter(str(name)):
            normalized_parameters[name] = value
            continue
        placeholder = f"<{name}>"
        normalized_parameters[name] = placeholder
        for identifier in _nested_string_values(value):
            normalized_query = normalized_query.replace(identifier, placeholder)
    return " ".join(
        (
            tool_name,
            normalized_query,
            json.dumps(
                normalized_parameters,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ),
        )
    )


def _nested_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [
            item for value_item in value for item in _nested_string_values(value_item)
        ]
    if isinstance(value, dict):
        return [
            item
            for value_item in value.values()
            for item in _nested_string_values(value_item)
        ]
    return []


_MUTATING_OPERATION_VALUES = {
    "add",
    "apply",
    "change",
    "clear",
    "create",
    "delete",
    "disable",
    "enable",
    "flush",
    "inject",
    "install",
    "kill",
    "reload",
    "remove",
    "replace",
    "restart",
    "set",
    "shutdown",
    "start",
    "stop",
    "terminate",
    "uninstall",
    "write",
}
_MUTATING_TEXT_RE = re.compile(
    r"(?:\b(?:restart|start|stop|reload|enable|disable|kill|delete|remove|"
    r"flush|apply|inject|shutdown|reboot)\b|"
    r"\brm\s|\bmv\s|\btouch\s|\btruncate\s|\bsed\s+-i\b|"
    r"\b(?:chmod|chown|mkdir|rmdir|mkfs|mount|umount|reboot|poweroff)\b|"
    r"\b(?:cp|install|tee)\s|"
    r"\bip\s+(?:link\s+set|(?:addr|address|route)\s+(?:add|del|delete|replace|flush))\b|"
    r"\broute\s+(?:add|del|delete)\b|\bvtysh\b.*\bconfigure\b|"
    r"\bsystemctl\s+(?:restart|start|stop|reload|enable|disable)\b|"
    r"\bservice\s+\S+\s+(?:restart|start|stop|reload|enable|disable)\b|"
    r"\btc\s+qdisc\s+(?:add|change|replace|del)\b|"
    r"\b(?:iptables|nft)\s+(?:-[AIXDF]|add|insert|delete|flush)\b|"
    r"(?:^|\s)(?:>|>>)(?:\s|$))",
    re.IGNORECASE,
)
_COMMAND_PARAMETER_RE = re.compile(r"^(?:command|cmd\d*)$", re.IGNORECASE)
_SHELL_CONTROL_RE = re.compile(r"(?:[;&|`]|\$\(|\r|\n)")
_READ_ONLY_COMMAND_RE = re.compile(
    r"^(?:"
    r"show\b|info\b|display\b|"
    r"ip\s+(?:addr(?:ess)?|route|link|neigh(?:bour)?)\b|"
    r"systemctl\s+status\b|"
    r"vtysh\s+-c\s+['\"]?show\b|"
    r"(?:cat|head|tail|grep)\b|"
    r"(?:ss|netstat|ifconfig|route|arp|hostname|uname|uptime|ps|df|free|dig|nslookup|traceroute|tracepath)\b|"
    r"ethtool\b"
    r")",
    re.IGNORECASE,
)
_UNSAFE_FIXED_ARGUMENT_RE = re.compile(
    r"(?:^|\s)(?:-s|--change|-f|-D|--daemon)(?:\s|$)",
    re.IGNORECASE,
)


def _exploration_is_read_only(
    *,
    tool_name: str,
    parameters: dict[str, Any],
    text: str,
) -> bool:
    """Exclude mutating tool trials from reusable Explorer observations."""

    for key, value in parameters.items():
        normalized_key = str(key).strip().lower()
        normalized_value = str(value).strip().lower()
        if normalized_key in {"operation", "action", "verb", "mode"}:
            if normalized_value in _MUTATING_OPERATION_VALUES:
                return False
        if _COMMAND_PARAMETER_RE.fullmatch(normalized_key):
            for command in _nested_string_values(value):
                stripped = command.strip()
                if (
                    not stripped
                    or _SHELL_CONTROL_RE.search(stripped)
                    or not _READ_ONLY_COMMAND_RE.match(stripped)
                ):
                    return False
        if normalized_key in {
            "args",
            "client_args",
            "server_args",
        } and _UNSAFE_FIXED_ARGUMENT_RE.search(normalized_value):
            return False
    serialized_parameters = json.dumps(
        parameters,
        ensure_ascii=False,
        default=str,
    )
    normalized_tool_tokens = " ".join(
        token for token in re.split(r"[^a-z0-9]+", tool_name.lower()) if token
    )
    combined = " ".join(
        (normalized_tool_tokens, serialized_parameters, str(text or ""))
    )
    return _MUTATING_TEXT_RE.search(combined) is None


def _offline_analyzer_suggestion(
    *,
    tool_name: str,
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
    doc: ToolDocumentation,
) -> DraftAnalyzerSuggestion:
    error_count = sum(not trial.success for trial in trials)
    success_count = sum(trial.success for trial in trials)
    gap_text = "; ".join(gap.recommendation for gap in gaps[:4])
    if gaps:
        suggestion = (
            f"Clarify `{tool_name}` documentation using observed failures: {gap_text}"
        )
    elif success_count:
        suggestion = f"Preserve successful `{tool_name}` usage patterns and add concise positive examples."
    else:
        suggestion = (
            f"No usable `{tool_name}` execution feedback yet; keep documentation "
            "concise and request a concrete trial."
        )
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
    )


def _invoke_draft_analyzer(
    *,
    tool_name: str,
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
    doc: ToolDocumentation,
    llm_backend: str | None,
    model: str | None,
    llm: Any | None = None,
    llm_error: str = "",
) -> tuple[DraftAnalyzerSuggestion | None, str]:
    """Run DRAFT's natural-language Analyzer over feedback and revision history."""
    selected_backend = (
        training_backend(llm_backend, "tool_refinement") if llm_backend else ""
    )
    selected_model = training_model(model, "tool_refinement") if model else ""
    if llm is None and (not selected_backend or not selected_model):
        return None, ""
    if llm is None:
        if llm_error:
            return None, llm_error
        try:
            llm = load_model(
                selected_backend,
                selected_model,
                timeout=training_timeout_seconds("tool_refinement"),
                max_retries=training_max_retries("tool_refinement"),
            )
        except Exception as exc:
            return None, format_training_error(exc)
    payload = {
        "tool": tool_name,
        "immutable_source_contract": {
            "description": doc.description,
            "input_schema": doc.source_schema,
        },
        "current_documentation": doc.model_dump(
            mode="json",
            exclude={"positive_examples", "negative_examples"},
        ),
        "recent_trials": [
            {
                "arguments": trial.arguments,
                "status": trial.status,
                "output": _trim_text(trial.output_summary),
                "error": _trim_text(trial.error_summary),
            }
            for trial in trials[-8:]
        ],
        "identified_gaps": [gap.model_dump(mode="json") for gap in gaps[-8:]],
        "revision_history": doc.rewrite_history[-5:],
    }
    prompt = (
        "You are the DRAFT Analyzer. Compare current tool documentation with "
        "actual exploration arguments and feedback. Identify concrete problems "
        "in consistency, completeness, and conciseness, then propose one targeted "
        "revision. The immutable source contract "
        "is authoritative: do not invent, remove, or rename parameters, and do not "
        "interpret MCP content-block transport wrappers as changes to the primitive "
        "tool API. Do not infer benchmark answers, "
        "faulty devices, root-cause labels, or diagnosis strategies.\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False, default=str)}"
    )
    try:
        analyzer = llm.with_structured_output(DraftAnalyzerDraft)
        raw = analyzer.invoke(prompt)
        draft = (
            raw
            if isinstance(raw, DraftAnalyzerDraft)
            else DraftAnalyzerDraft.model_validate(raw)
        )
        if not draft.suggestion.strip():
            return None, "DRAFT Analyzer returned an empty suggestion"
        suggestion_text = draft.suggestion.strip()
        if draft.rationale.strip():
            suggestion_text += " Rationale: " + draft.rationale.strip()
        return (
            DraftAnalyzerSuggestion(
                suggestion_id=_stable_id(
                    tool_name,
                    [trial.trial_id for trial in trials],
                    suggestion_text,
                    prefix="suggest",
                ),
                tool_name=tool_name,
                session_id=trials[-1].session_id if trials else "",
                trial_ids=[trial.trial_id for trial in trials],
                suggestion=suggestion_text,
            ),
            "",
        )
    except Exception as exc:
        return None, format_training_error(exc)


def _recent_revision_hashes(
    revisions: list[DocumentationRevision],
    tool_name: str,
    *,
    source_signature: str = "",
    limit: int = 3,
) -> list[str]:
    return [
        revision.after_hash
        for revision in revisions
        if revision.tool_name == tool_name
        and (not source_signature or revision.source_signature == source_signature)
    ][-limit:]


def _append_unique(target: list[Any], items: list[Any], *, limit: int = 12) -> None:
    for item in items:
        if item in (None, "", [], {}) or item in target:
            continue
        target.append(item)
        if len(target) >= limit:
            break


def _unique_items(items: list[Any], *, limit: int = 12) -> list[Any]:
    result: list[Any] = []
    _append_unique(result, items, limit=limit)
    return result


def _refresh_tool_stats(
    *,
    state: Any,
    doc: ToolDocumentation,
    convergence_score: float | None = None,
    documented_path_rate: float = 0.0,
    success_path_rate: float = 0.0,
) -> None:
    tool_name = doc.name
    all_tool_trials = [trial for trial in state.trials if trial.tool_name == tool_name]
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
        mastery_score=doc.mastery_score,
        contract_mastery_score=doc.contract_mastery_score,
        diagnostic_utility_score=doc.diagnostic_utility_score,
        convergence_score=doc.last_convergence_score,
        documented_path_rate=documented_path_rate,
        success_path_rate=success_path_rate,
        mastered=(doc.published and doc.frozen and "converged" in doc.frozen_reason),
    )


def _diagnostic_outcome_score(metrics: dict[str, Any]) -> float | None:
    """Return an outcome score without coupling it to documentation mastery."""

    keys = (
        "detection_score",
        "localization_f1",
        "localization_accuracy",
        "rca_f1",
        "rca_accuracy",
    )
    if not any(key in metrics for key in keys):
        return None

    def component(prefix: str) -> float:
        for suffix in ("f1", "accuracy", "precision"):
            value = metrics.get(f"{prefix}_{suffix}")
            if value is not None:
                return max(0.0, min(1.0, float(value)))
        return 0.0

    detection = max(0.0, min(1.0, float(metrics.get("detection_score") or 0.0)))
    return (
        (0.10 * detection)
        + (0.35 * component("localization"))
        + (0.55 * component("rca"))
    )


def _update_diagnostic_utility(
    doc: ToolDocumentation,
    *,
    trials: list[ToolTrial],
    metrics: dict[str, Any],
) -> None:
    """Update once per episode, including for documentation-frozen tools."""

    outcome = _diagnostic_outcome_score(metrics)
    if outcome is None:
        return
    by_session: dict[str, list[ToolTrial]] = defaultdict(list)
    for trial in trials:
        if trial.session_id:
            by_session[trial.session_id].append(trial)
    seen = set(doc.diagnostic_utility_sessions)
    for session_id, session_trials in sorted(by_session.items()):
        if session_id in seen:
            continue
        reliability = sum(trial.success for trial in session_trials) / max(
            len(session_trials), 1
        )
        count = doc.diagnostic_utility_count
        doc.diagnostic_utility_score = round(
            ((doc.diagnostic_utility_score * count) + (outcome * reliability))
            / (count + 1),
            6,
        )
        doc.diagnostic_utility_count = count + 1
        doc.diagnostic_utility_sessions = [
            *doc.diagnostic_utility_sessions[-49:],
            session_id,
        ]
        seen.add(session_id)


def _diagnostic_utility_lcb(doc: ToolDocumentation) -> float:
    """Conservative utility support for bounded episode outcomes.

    Without counterfactual traces, diagnostic utility is associative rather
    than marginal. A worst-case standard-error term prevents two lucky
    episodes from being treated like stable evidence.
    """

    if doc.diagnostic_utility_count <= 0:
        return 0.0
    uncertainty = 0.5 / math.sqrt(doc.diagnostic_utility_count)
    return max(0.0, doc.diagnostic_utility_score - uncertainty)


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
    del metrics
    doc_payload = doc.model_dump(
        exclude={
            "rewrite_history",
            "updated_at",
            "version",
            "frozen",
            "frozen_reason",
            "trial_count",
            "success_count",
            "error_count",
            "mastery_score",
            "contract_mastery_score",
            "diagnostic_utility_score",
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
        }
        for trial in trials[-6:]
    ]
    gap_payload = [gap.model_dump(mode="json") for gap in gaps[-8:]]
    suggestion_payload = [
        {"suggestion": _trim_text(item.suggestion, limit=500)}
        for item in suggestions[-4:]
    ]
    exploration_payload = [
        {
            "trial_id": item.trial_id,
            "tool_name": item.tool_name,
            "parameters": item.parameters,
            "status": item.status,
            "observation": _trim_text(item.observation),
            "analyzer_suggestion": _trim_text(item.analyzer_suggestion, limit=500),
            "diversity_score": item.diversity_score,
            "reflection_count": item.reflection_count,
        }
        for item in explorations[-4:]
    ]
    return (
        "You are the DRAFT documentation rewriting agent for NIKA primitive "
        "network-diagnosis tools. Refine documentation only; do not invent new "
        "tools, APIs, commands, hidden labels, or benchmark answers.\n\n"
        "The current `description`, `source_schema`, and parameter names form an "
        "immutable primitive contract. Do not replace the source description or "
        "add, remove, or rename parameters. MCP content-block wrappers in traces "
        "are transport envelopes, not evidence that the primitive return contract "
        "changed. Learn only usage guidance supported by trials.\n\n"
        "DRAFT loop mapping: Explorer = self-driven and observed read-only tool trials; "
        "Analyzer = suggestions from output/error feedback; Rewriter = update "
        "the documentation.\n\n"
        f"Tool: {tool_name}\n"
        f"Current documentation:\n{json.dumps(doc_payload, indent=2, ensure_ascii=False)}\n\n"
        "Explorer observations:\n"
        f"{json.dumps(exploration_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Recent tool trials:\n{json.dumps(trial_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Analyzer suggestions:\n{json.dumps(suggestion_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Comprehension gaps:\n{json.dumps(gap_payload, indent=2, ensure_ascii=False)}\n\n"
        "Return a documentation rewrite that helps future agents understand "
        "preconditions, argument types, allowed values, constraints, common "
        "failure modes, safe usage examples, return semantics, and how to tell "
        "a valid result from a tool error. Do not write answer-pattern rules such "
        "as mapping one symptom directly to a faulty device or root-cause label. "
        "Represent runtime host, router, switch, node, source, target, and interface "
        "values with parameter placeholders such as `<host_name>`; never preserve "
        "a concrete topology identifier. Numeric ranges, defaults, and allowed "
        "values must come from the immutable source schema. "
        "Only document preconditions, constraints, failure modes, and negative "
        "examples when they are directly supported by the primitive source "
        "contract or an observed error in this batch. For successful-only batches, "
        "leave those fields empty rather than inventing possible failures or "
        "unobserved output fields. "
        "Also return `tool_usage_description` as a concise selection-time tool "
        "summary. For `parameters`, return a mapping from parameter name to a "
        "short description. Return `next_exploration_direction` as one concise "
        "aspect that a future Explorer should test; do not provide concrete "
        "runtime identifiers or a complete query."
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
        confidence=draft.confidence,
        rationale=draft.rationale,
        next_exploration_direction=draft.next_exploration_direction,
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
    llm: Any | None = None,
    llm_error: str = "",
) -> tuple[DraftRewriteProposal | None, str]:
    selected_backend = (
        training_backend(llm_backend, "tool_refinement") if llm_backend else ""
    )
    selected_model = training_model(model, "tool_refinement") if model else ""
    if llm is None and (not selected_backend or not selected_model):
        return None, ""
    if llm is None:
        if llm_error:
            return None, llm_error
        try:
            llm = load_model(
                selected_backend,
                selected_model,
                timeout=training_timeout_seconds("tool_refinement"),
                max_retries=training_max_retries("tool_refinement"),
            )
        except Exception as exc:
            return None, format_training_error(exc)
    try:
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
        return None, format_training_error(exc)


def _apply_draft_proposal(
    doc: ToolDocumentation,
    proposal: DraftRewriteProposal,
    *,
    allowed_parameter_names: set[str],
    trials: list[ToolTrial],
    gaps: list[ComprehensionGap],
) -> None:
    if proposal.tool_usage_description.strip():
        doc.tool_usage_description = proposal.tool_usage_description.strip()
    if proposal.next_exploration_direction.strip():
        doc.next_exploration_direction = proposal.next_exploration_direction.strip()
    observed_errors = [
        trial.error_summary for trial in trials if trial.status == "error"
    ]
    grounded_constraint = (
        "Tool arguments must be grounded in currently observed topology evidence."
    )
    if observed_errors:
        doc.preconditions = _unique_items(proposal.preconditions, limit=8)
        doc.constraints = _unique_items(
            [grounded_constraint, *proposal.constraints],
            limit=12,
        )
        doc.failure_modes = _unique_items(
            [*observed_errors, *proposal.failure_modes],
            limit=12,
        )
        doc.usage_notes = _unique_items(
            [*(gap.recommendation for gap in gaps), *proposal.usage_notes],
            limit=12,
        )
    else:
        doc.preconditions = []
        doc.constraints = [grounded_constraint]
        doc.failure_modes = []
        doc.usage_notes = []
    for name, param in proposal.parameters.items():
        if not name or name not in allowed_parameter_names:
            continue
        current = doc.parameters.get(name)
        if current is None:
            # This path is used only by the standalone offline curator, whose
            # source contract is reconstructed from observed valid trials.
            if doc.source_contract_version <= 0:
                doc.parameters[name] = param
            continue
        if param.type_hint != "unknown":
            current.type_hint = param.type_hint
        if param.description:
            current.description = param.description
        if doc.source_contract_version <= 0:
            _append_unique(current.constraints, param.constraints, limit=8)
        _append_unique(current.examples, param.examples, limit=5)


def rewrite_documentation(
    store: ToolRefinementStore,
    *,
    trials: list[ToolTrial],
    tool_descriptions: dict[str, str],
    metrics: dict[str, Any],
    llm_backend: str | None = None,
    model: str | None = None,
    analyzer_model: str | None = None,
    rewriter_model: str | None = None,
    documented_tools_at_start: set[str] | None = None,
    convergence_threshold: float = DRAFT_CONVERGENCE_THRESHOLD,
    publish_min_utility: float = _DEFAULTS.publish_min_utility,
    llm: Any | None = None,
) -> list[DocumentationRevision]:
    with store.exclusive():
        return _rewrite_documentation_unlocked(
            store,
            trials=trials,
            tool_descriptions=tool_descriptions,
            metrics=metrics,
            llm_backend=llm_backend,
            model=model,
            analyzer_model=analyzer_model,
            rewriter_model=rewriter_model,
            documented_tools_at_start=documented_tools_at_start,
            convergence_threshold=convergence_threshold,
            publish_min_utility=publish_min_utility,
            llm=llm,
        )


def _rewrite_documentation_unlocked(
    store: ToolRefinementStore,
    *,
    trials: list[ToolTrial],
    tool_descriptions: dict[str, str],
    metrics: dict[str, Any],
    llm_backend: str | None = None,
    model: str | None = None,
    analyzer_model: str | None = None,
    rewriter_model: str | None = None,
    documented_tools_at_start: set[str] | None = None,
    convergence_threshold: float = DRAFT_CONVERGENCE_THRESHOLD,
    publish_min_utility: float = _DEFAULTS.publish_min_utility,
    llm: Any | None = None,
) -> list[DocumentationRevision]:
    state = store.load()
    selected_backend = (
        training_backend(llm_backend, "tool_refinement") if llm_backend else ""
    )
    selected_analyzer_model = (
        analyzer_model.strip()
        if isinstance(analyzer_model, str) and analyzer_model.strip()
        else training_model(model, "tool_refinement")
    )
    selected_rewriter_model = (
        rewriter_model.strip()
        if isinstance(rewriter_model, str) and rewriter_model.strip()
        else training_model(model, "tool_refinement")
    )
    role_models: dict[str, tuple[Any | None, str]] = {}

    def load_role_model(role_model: str | None) -> tuple[Any | None, str]:
        if llm is not None:
            return llm, ""
        selected_model = (
            role_model.strip()
            if isinstance(role_model, str) and role_model.strip()
            else training_model(model, "tool_refinement")
        )
        if not selected_backend or not selected_model:
            return None, ""
        if selected_model in role_models:
            return role_models[selected_model]
        try:
            loaded = (
                load_model(
                    selected_backend,
                    selected_model,
                    timeout=training_timeout_seconds("tool_refinement"),
                    max_retries=training_max_retries("tool_refinement"),
                ),
                "",
            )
        except Exception as exc:
            loaded = (None, format_training_error(exc))
        role_models[selected_model] = loaded
        return loaded

    analyzer_llm, analyzer_llm_error = load_role_model(analyzer_model)
    rewriter_llm, rewriter_llm_error = load_role_model(rewriter_model)
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
    revisions: list[DocumentationRevision] = []
    accuracy = float(metrics.get("rca_accuracy") or 0.0)
    for tool_name, tool_trials in by_tool.items():
        doc = state.documents.setdefault(
            tool_name,
            ToolDocumentation(name=tool_name),
        )
        _update_diagnostic_utility(doc, trials=tool_trials, metrics=metrics)
        before_hash = doc.content_hash()
        before_description = doc.refined_description(max_chars=4000)
        if doc.frozen and not doc.published:
            doc.frozen = False
            doc.frozen_reason = ""
        if (
            doc.frozen
            and doc.diagnostic_utility_count >= 2
            and _diagnostic_utility_lcb(doc) < convergence_threshold
        ):
            doc.frozen = False
            doc.frozen_reason = ""
        if doc.frozen:
            _refresh_tool_stats(
                state=state,
                doc=doc,
                documented_path_rate=documented_path_rate,
                success_path_rate=success_path_rate,
            )
            state.documents[tool_name] = doc
            continue

        allowed_parameter_names = _allowed_parameter_names(doc, tool_trials)
        for trial in tool_trials:
            for key, value in trial.arguments.items():
                if key.startswith("_") or key not in allowed_parameter_names:
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

            if (
                trial.success
                and trial.arguments
                and _exploration_is_read_only(
                    tool_name=trial.tool_name,
                    parameters=trial.arguments,
                    text=trial.task_description,
                )
            ):
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
            if not any(item.gap_id == gap.gap_id for item in state.gaps):
                state.gaps.append(gap)

        grounded_constraint = (
            "Tool arguments must be grounded in currently observed topology evidence."
        )
        if grounded_constraint not in doc.constraints:
            doc.constraints.append(grounded_constraint)

        llm_suggestion, analyzer_error = _invoke_draft_analyzer(
            tool_name=tool_name,
            trials=tool_trials,
            gaps=gaps,
            doc=doc,
            llm_backend=llm_backend,
            model=analyzer_model or model,
            llm=analyzer_llm,
            llm_error=analyzer_llm_error,
        )
        suggestion = llm_suggestion
        if suggestion is None and not analyzer_error:
            suggestion = _offline_analyzer_suggestion(
                tool_name=tool_name,
                trials=tool_trials,
                gaps=gaps,
                doc=doc,
            )
        tool_suggestions = [suggestion] if suggestion is not None else []
        if suggestion is not None:
            if not any(
                item.suggestion_id == suggestion.suggestion_id
                for item in state.analyzer_suggestions
            ):
                state.analyzer_suggestions.append(suggestion)
            _append_unique(
                doc.analyzer_suggestions,
                [suggestion.suggestion],
                limit=20,
            )

        seen_explorations = {item.exploration_id for item in state.explorations}
        seen_exploration_trials = {
            item.trial_id for item in state.explorations if item.trial_id
        }
        prior_explorations = [
            item for item in state.explorations if item.tool_name == tool_name
        ]
        explorations: list[DraftExploration] = []
        for trial in tool_trials:
            exploration = _exploration_from_trial(
                trial=trial,
                doc=doc,
                analyzer_suggestion=(suggestion.suggestion if suggestion else ""),
                prior_explorations=[*prior_explorations, *explorations],
            )
            explorations.append(exploration)
        for exploration in explorations:
            if (
                exploration.exploration_id not in seen_explorations
                and exploration.trial_id not in seen_exploration_trials
            ):
                state.explorations.append(exploration)
                seen_explorations.add(exploration.exploration_id)
                seen_exploration_trials.add(exploration.trial_id)

        if analyzer_error:
            proposal, rewriter_error = None, ""
        else:
            proposal, rewriter_error = _invoke_draft_rewriter(
                tool_name=tool_name,
                doc=doc,
                trials=tool_trials,
                gaps=gaps,
                suggestions=tool_suggestions,
                explorations=explorations,
                metrics=metrics,
                llm_backend=llm_backend,
                model=rewriter_model or model,
                llm=rewriter_llm,
                llm_error=rewriter_llm_error,
            )
        all_evidence_trials = [
            trial for trial in state.trials if trial.tool_name == tool_name
        ]
        all_evidence_gaps = [gap for gap in state.gaps if gap.tool_name == tool_name]
        contract_rejected = False
        if proposal is not None:
            contract_errors = _proposal_contract_errors(
                proposal,
                allowed_parameter_names=allowed_parameter_names,
                doc=doc,
            )
            if contract_errors:
                contract_rejected = True
                contract_error = "ContractValidationError: " + "; ".join(
                    contract_errors
                )
                rewriter_error = " | ".join(
                    error for error in (rewriter_error, contract_error) if error
                )
                proposal = None
            else:
                _apply_draft_proposal(
                    doc,
                    proposal,
                    allowed_parameter_names=allowed_parameter_names,
                    trials=all_evidence_trials,
                    gaps=all_evidence_gaps,
                )
        if proposal is None and not any(
            trial.status == "error" for trial in all_evidence_trials
        ):
            doc.preconditions = []
            doc.constraints = [grounded_constraint]
            doc.failure_modes = []
            doc.usage_notes = []
        if not doc.tool_usage_description:
            doc.tool_usage_description = _tool_usage_description(doc)
        generalize_tool_documentation(doc, trials=all_evidence_trials)
        valid_rewrite = (
            proposal is not None
            and not contract_rejected
            and not analyzer_error
            and not rewriter_error
        )

        after_hash = doc.content_hash()
        after_description = doc.refined_description(max_chars=4000)
        word_match_score, semantic_match_score, convergence_score = (
            _document_convergence(before_description, after_description)
        )
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
        documentation_coverage = (
            sum(
                (
                    bool(doc.description or doc.tool_usage_description),
                    bool(doc.parameters)
                    or not any(trial.arguments for trial in all_tool_trials),
                    bool(doc.constraints or doc.preconditions),
                    bool(doc.failure_modes)
                    or not any(trial.status == "error" for trial in all_tool_trials),
                )
            )
            / 4.0
        )
        if _source_parameter_names(doc) or any(
            trial.arguments for trial in all_tool_trials
        ):
            independent_trials = len(
                {
                    json.dumps(
                        {
                            "signature": _exploration_signature(
                                tool_name=trial.tool_name,
                                user_query="",
                                parameters=trial.arguments,
                            ),
                            "status": trial.status,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    )
                    for trial in all_tool_trials
                }
            )
        else:
            independent_trials = len(
                {trial.session_id for trial in all_tool_trials if trial.session_id}
            )
        diversity_support = min(1.0, independent_trials / 2.0)
        doc.contract_mastery_score = round(
            convergence_score * documentation_coverage * diversity_support,
            6,
        )
        doc.mastery_score = doc.contract_mastery_score

        publication_utility_supported = doc.diagnostic_utility_count == 0 or (
            doc.diagnostic_utility_count >= 2
            and _diagnostic_utility_lcb(doc) >= publish_min_utility
        )
        if (
            valid_rewrite
            and publication_utility_supported
            and diversity_support >= 1.0
            and documentation_coverage >= 0.5
        ):
            doc.published = True
        elif (
            doc.diagnostic_utility_count >= 2
            and _diagnostic_utility_lcb(doc) < publish_min_utility
        ):
            doc.published = False

        recent_hashes = _recent_revision_hashes(
            state.revisions,
            tool_name,
            source_signature=doc.source_signature,
        )
        diagnostic_utility_supported = doc.diagnostic_utility_count == 0 or (
            doc.diagnostic_utility_count >= 2
            and _diagnostic_utility_lcb(doc) >= convergence_threshold
        )
        unchanged_streak = (
            valid_rewrite
            and diagnostic_utility_supported
            and len(recent_hashes) >= 1
            and len(set(recent_hashes + [after_hash])) == 1
            and diversity_support >= 1.0
            and documentation_coverage >= 0.5
        )
        converged = (
            valid_rewrite
            and diagnostic_utility_supported
            and convergence_score >= convergence_threshold
            and len(doc.rewrite_history) >= 1
            and diversity_support >= 1.0
            and doc.contract_mastery_score >= convergence_threshold * 0.5
        )
        if unchanged_streak or converged:
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
            source_signature=doc.source_signature,
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
                "llm_attempted": (
                    1.0
                    if selected_backend
                    and (selected_analyzer_model or selected_rewriter_model)
                    else 0.0
                ),
                "llm_rewrite": 1.0 if proposal is not None else 0.0,
                "llm_failed": 1.0 if (analyzer_error or rewriter_error) else 0.0,
                "llm_analyzer": 1.0 if llm_suggestion is not None else 0.0,
                "llm_analyzer_failed": 1.0 if analyzer_error else 0.0,
                "llm_contract_rejected": 1.0 if contract_rejected else 0.0,
                "convergence_score": convergence_score,
                "word_match_score": word_match_score,
                "semantic_match_score": semantic_match_score,
                "mastery_score": doc.mastery_score,
                "contract_mastery_score": doc.contract_mastery_score,
                "diagnostic_utility_score": doc.diagnostic_utility_score,
                "exploration_diversity_support": diversity_support,
                "documentation_coverage": documentation_coverage,
                "documented_path_rate": documented_path_rate,
                "success_path_rate": success_path_rate,
            },
            analyzer_suggestion_ids=(
                [suggestion.suggestion_id] if suggestion is not None else []
            ),
            llm_error=" | ".join(
                error for error in (analyzer_error, rewriter_error) if error
            ),
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

    state.library_usage_description = _library_usage_description(state.documents)
    store.save(state)
    return revisions


def finalize_tool_refinement_session(
    *,
    session_id: str,
    metrics: dict[str, Any],
    allow_training_updates: bool | None = None,
    rewrite: bool = True,
    min_new_trials: int = _DEFAULTS.min_new_trials,
    max_tools_per_update: int = _DEFAULTS.max_tools_per_update,
    publish_min_utility: float = _DEFAULTS.publish_min_utility,
) -> dict[str, Any]:
    session = Session()
    session.load_closed_session(session_id=session_id)
    updates_allowed = (
        _training_updates_allowed(session)
        if allow_training_updates is None
        else bool(allow_training_updates)
    )
    if not updates_allowed:
        report = {
            "status": "skipped",
            "reason": "training updates are disabled",
            "method": "DRAFT",
            "library_id": getattr(session, "tool_library_id", "default"),
            "draft_trials": 0,
            "draft_trials_added": 0,
            "draft_document_revisions": 0,
        }
        raw_session_dir = str(getattr(session, "session_dir", "") or "")
        if raw_session_dir:
            session_dir = Path(raw_session_dir)
            (session_dir / "tool_refinement.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return report
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
    store = ToolRefinementStore(library_id)
    documented_tools_at_start = set(store.load().documents)
    trials, tool_descriptions = extract_tool_trials(
        trace_path,
        session_id=session_id,
        task_description=str(getattr(session, "task_description", "") or ""),
    )
    with store.exclusive():
        state = store.load()
        seen_trial_ids = {trial.trial_id for trial in state.trials}
        added_trials = 0
        for trial in trials:
            if trial.trial_id in seen_trial_ids:
                continue
            state.trials.append(trial)
            seen_trial_ids.add(trial.trial_id)
            added_trials += 1
        trials_by_tool: dict[str, list[ToolTrial]] = defaultdict(list)
        for trial in trials:
            trials_by_tool[trial.tool_name].append(trial)
        for tool_name, session_trials in trials_by_tool.items():
            doc = state.documents.setdefault(
                tool_name,
                ToolDocumentation(
                    name=tool_name,
                    description=tool_descriptions.get(tool_name, "").strip(),
                ),
            )
            _update_diagnostic_utility(
                doc,
                trials=session_trials,
                metrics=metrics,
            )
            if (
                doc.published
                and doc.diagnostic_utility_count >= 2
                and _diagnostic_utility_lcb(doc) < publish_min_utility
            ):
                doc.published = False
                doc.frozen = False
                doc.frozen_reason = ""
        processed_ids = set(state.processed_trial_ids)
        pending_trials = [
            trial for trial in state.trials if trial.trial_id not in processed_ids
        ]
        pending_by_tool: dict[str, list[ToolTrial]] = defaultdict(list)
        for trial in pending_trials:
            pending_by_tool[trial.tool_name].append(trial)
        stable_trial_ids: set[str] = set()
        eligible_tools: list[str] = []
        for tool_name, tool_trials in pending_by_tool.items():
            doc = state.documents.get(tool_name)
            has_error = any(trial.status == "error" for trial in tool_trials)
            if doc is not None and (doc.published or doc.frozen) and not has_error:
                stable_trial_ids.update(trial.trial_id for trial in tool_trials)
                continue
            if len(tool_trials) >= max(1, min_new_trials) or has_error:
                eligible_tools.append(tool_name)
        if stable_trial_ids:
            state.processed_trial_ids = sorted(
                set(state.processed_trial_ids) | stable_trial_ids
            )
        selected_tools = sorted(
            eligible_tools,
            key=lambda tool_name: (
                -sum(trial.status == "error" for trial in pending_by_tool[tool_name]),
                -len(pending_by_tool[tool_name]),
                tool_name,
            ),
        )[: max(1, max_tools_per_update)]
        selected_trials = [
            trial
            for tool_name in selected_tools
            for trial in pending_by_tool[tool_name]
        ]
        store.save(state)
    documented_path_rate, success_path_rate = _path_rates(
        trials=trials,
        documented_tools_at_start=documented_tools_at_start,
    )
    revisions: list[DocumentationRevision] = []
    if rewrite and selected_trials:
        selected_descriptions = {
            tool_name: tool_descriptions.get(tool_name, "")
            for tool_name in selected_tools
        }
        revisions = rewrite_documentation(
            store,
            trials=selected_trials,
            tool_descriptions=selected_descriptions,
            metrics=metrics,
            llm_backend=getattr(session, "llm_backend", None),
            model=getattr(session, "model", None),
            analyzer_model=(getattr(session, "tool_analyzer_model", "") or None),
            rewriter_model=(getattr(session, "tool_rewriter_model", "") or None),
            documented_tools_at_start=documented_tools_at_start,
            convergence_threshold=convergence_threshold,
            publish_min_utility=publish_min_utility,
        )
        with store.exclusive():
            state = store.load()
            state.processed_trial_ids = sorted(
                set(state.processed_trial_ids)
                | {trial.trial_id for trial in selected_trials}
            )
            store.save(state)
    state = store.load()
    llm_attempts = sum(
        revision.metrics.get("llm_attempted") == 1.0 for revision in revisions
    )
    llm_failures = sum(
        revision.metrics.get("llm_failed") == 1.0 for revision in revisions
    )
    llm_errors = [revision.llm_error for revision in revisions if revision.llm_error]
    processed_ids = set(state.processed_trial_ids)
    pending_count = sum(trial.trial_id not in processed_ids for trial in state.trials)
    report = {
        "status": (
            "updated" if revisions else "collected" if not rewrite else "deferred"
        ),
        "method": "DRAFT",
        "library_id": store.library_id,
        "draft_trials": len(trials),
        "draft_trials_added": added_trials,
        "draft_pending_trials": pending_count,
        "draft_selected_tools": selected_tools if rewrite else [],
        "draft_document_revisions": sum(revision.changed for revision in revisions),
        "draft_comprehension_gaps": len(state.gaps),
        "draft_frozen_documents": sum(doc.frozen for doc in state.documents.values()),
        "draft_documented_tools": len(state.documents),
        "draft_published_documents": sum(
            doc.published for doc in state.documents.values()
        ),
        "draft_unique_trial_tools": len({trial.tool_name for trial in trials}),
        "draft_explorations": len(state.explorations),
        "draft_active_explorations": sum(
            item.session_id == session_id and item.trial_id.startswith("active_trial_")
            for item in state.explorations
        ),
        "draft_analyzer_suggestions": len(state.analyzer_suggestions),
        "draft_mastered_tools": sum(
            stat.mastered for stat in state.tool_stats.values()
        ),
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
        "draft_llm_analyzer_revisions": sum(
            revision.metrics.get("llm_analyzer") == 1.0 for revision in revisions
        ),
        "draft_llm_analyzer_failures": sum(
            revision.metrics.get("llm_analyzer_failed") == 1.0 for revision in revisions
        ),
        "draft_llm_errors": llm_errors[:5],
        "draft_config": {
            "convergence_threshold": convergence_threshold,
            "update_due": rewrite,
            "min_new_trials": min_new_trials,
            "max_tools_per_update": max_tools_per_update,
            "publish_min_utility": publish_min_utility,
            "tool_doc_chars": getattr(session, "tool_doc_chars", None),
            "exploration_similarity_threshold": getattr(
                session,
                "tool_exploration_similarity_threshold",
                DRAFT_EXPLORATION_SIMILARITY_THRESHOLD,
            ),
            "explorer_reflection_limit": getattr(
                session,
                "tool_explorer_reflection_limit",
                3,
            ),
            "explorer_model": getattr(session, "tool_explorer_model", ""),
            "analyzer_model": getattr(session, "tool_analyzer_model", ""),
            "rewriter_model": getattr(session, "tool_rewriter_model", ""),
        },
    }
    (session_dir / "tool_refinement.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
