"""Generalize learned tool documentation away from runtime identifiers."""

from __future__ import annotations

import re
from typing import Any

from agent.tool_refinement.models import ToolDocumentation, ToolTrial

_IDENTIFIER_KEY_PARTS = frozenset(
    {
        "address",
        "device",
        "dst",
        "endpoint",
        "host",
        "iface",
        "ifname",
        "interface",
        "intf",
        "ip",
        "node",
        "router",
        "source",
        "src",
        "switch",
        "target",
    }
)


def is_runtime_identifier_parameter(name: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(name).lower()))
    return bool(tokens & _IDENTIFIER_KEY_PARTS)


def generalize_tool_documentation(
    doc: ToolDocumentation,
    *,
    trials: list[ToolTrial] | None = None,
) -> bool:
    """Replace observed topology identifiers with schema-derived placeholders."""

    replacements = _identifier_replacements(doc, trials or [])
    before = doc.content_hash()
    doc.tool_usage_description = _replace_identifiers(
        doc.tool_usage_description, replacements
    )
    doc.preconditions = [
        _replace_identifiers(item, replacements) for item in doc.preconditions
    ]
    doc.constraints = [
        _replace_identifiers(item, replacements) for item in doc.constraints
    ]
    doc.failure_modes = [
        _replace_identifiers(item, replacements) for item in doc.failure_modes
    ]
    doc.usage_notes = [
        _replace_identifiers(item, replacements) for item in doc.usage_notes
    ]
    doc.rewrite_history = [
        _replace_identifiers(item, replacements) for item in doc.rewrite_history
    ]
    doc.analyzer_suggestions = [
        _replace_identifiers(item, replacements) for item in doc.analyzer_suggestions
    ]
    doc.next_exploration_direction = _replace_identifiers(
        doc.next_exploration_direction, replacements
    )
    for name, parameter in doc.parameters.items():
        parameter.description = _replace_identifiers(
            parameter.description, replacements
        )
        parameter.constraints = [
            _replace_identifiers(item, replacements) for item in parameter.constraints
        ]
        if is_runtime_identifier_parameter(name):
            parameter.examples = [f"<{name}>"] if parameter.examples else []
        else:
            parameter.examples = [
                _generalize_value(item, replacements) for item in parameter.examples
            ]
    doc.positive_examples = [
        _generalize_value(item, replacements) for item in doc.positive_examples
    ]
    doc.negative_examples = [
        _generalize_value(item, replacements) for item in doc.negative_examples
    ]
    return doc.content_hash() != before


def _identifier_replacements(
    doc: ToolDocumentation,
    trials: list[ToolTrial],
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    argument_sets = [trial.arguments for trial in trials]
    argument_sets.extend(
        example.get("arguments", {})
        for example in [*doc.positive_examples, *doc.negative_examples]
        if isinstance(example, dict)
    )
    for arguments in argument_sets:
        if not isinstance(arguments, dict):
            continue
        for name, value in arguments.items():
            if not is_runtime_identifier_parameter(str(name)):
                continue
            for identifier in _string_values(value):
                if identifier:
                    replacements[identifier] = f"<{name}>"
    for name, parameter in doc.parameters.items():
        if not is_runtime_identifier_parameter(name):
            continue
        for identifier in _string_values(parameter.examples):
            if identifier and not identifier.startswith("<"):
                replacements[identifier] = f"<{name}>"
    return replacements


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [item for value_item in value for item in _string_values(value_item)]
    return []


def _replace_identifiers(text: str, replacements: dict[str, str]) -> str:
    result = str(text or "")
    for identifier, placeholder in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        escaped = re.escape(identifier)
        if re.fullmatch(r"[A-Za-z0-9_.:-]+", identifier):
            pattern = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
        else:
            pattern = escaped
        result = re.sub(pattern, placeholder, result)
    return result


def _generalize_value(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_identifiers(value, replacements)
    if isinstance(value, list):
        return [_generalize_value(item, replacements) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if is_runtime_identifier_parameter(str(key)):
                if isinstance(item, list):
                    result[str(key)] = [f"<{key}>"] if item else []
                elif item not in (None, ""):
                    result[str(key)] = f"<{key}>"
                else:
                    result[str(key)] = item
            else:
                result[str(key)] = _generalize_value(item, replacements)
        return result
    return value
