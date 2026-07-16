"""Structured tool-output serialization and execution outcome classification."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Literal

from langchain_core.messages import ToolMessage

ToolOutcome = Literal["success", "error", "unknown"]
INTEGRATED_GUIDANCE_MARKER = "[Integrated training guidance - not evidence]"

_ERROR_STATUS_VALUES = frozenset({"error", "failed", "failure", "timeout"})
_UNKNOWN_STATUS_VALUES = frozenset({"unknown", "unavailable", "indeterminate"})
_EXECUTION_ERROR_PATTERNS = (
    re.compile(r"(?:^|\n)\s*\[timeout\]", re.IGNORECASE),
    re.compile(r"\bcommand\b.{0,240}\bexceeded\s+\d+(?:\.\d+)?s\b", re.IGNORECASE),
    re.compile(
        r"(?:^|\n)\s*(?:error|exception|traceback|fatal)\s*[:\[]", re.IGNORECASE
    ),
    re.compile(r"\b(?:command|executable)\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\bpermission\s+denied\b", re.IGNORECASE),
    re.compile(r"\binvalid\s+(?:argument|parameter|tool input)\b", re.IGNORECASE),
    re.compile(
        r"\bcannot\s+(?:get|resolve|find)\s+(?:the\s+)?(?:ip address|host|device|router|interface)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+such\s+(?:file|host|device|interface|command)\b", re.IGNORECASE
    ),
    re.compile(r"\btool\s+(?:execution\s+)?failed\b", re.IGNORECASE),
    re.compile(r"\bexit\s+(?:code|status)\s*[=:]?\s*[1-9]\d*\b", re.IGNORECASE),
)


def serialize_tool_output(output: Any) -> Any:
    """Return a JSON-serializable representation without flattening ToolMessage."""

    if isinstance(output, ToolMessage):
        return {
            "status": str(output.status or "success"),
            "content": _json_value(output.content),
            "artifact": _json_value(output.artifact),
            "name": output.name or "",
            "tool_call_id": output.tool_call_id or "",
        }
    return _json_value(output)


def tool_output_content(output: Any) -> Any:
    """Unwrap transport metadata while preserving the primitive result payload."""

    if isinstance(output, ToolMessage):
        return output.content
    if isinstance(output, tuple) and len(output) == 2:
        return output[0]
    if isinstance(output, dict) and "content" in output:
        return output["content"]
    return output


def classify_tool_outcome(
    output: Any,
    *,
    event: str = "tool_end",
) -> ToolOutcome:
    """Classify tool execution separately from valid negative network evidence."""

    if event == "tool_error":
        return "error"
    statuses: list[str] = []
    texts: list[str] = []
    _collect_output_signals(output, statuses=statuses, texts=texts)
    if any(status in _ERROR_STATUS_VALUES for status in statuses):
        return "error"
    if any(
        pattern.search(text) for text in texts for pattern in _EXECUTION_ERROR_PATTERNS
    ):
        return "error"
    meaningful_texts = [text for text in texts if text.strip()]
    if not meaningful_texts and output in (None, "", [], {}):
        return "unknown"
    semantic_statuses = [
        status
        for status in statuses
        if status and status not in {"success", "ok", "completed"}
    ]
    if semantic_statuses and all(
        status in _UNKNOWN_STATUS_VALUES for status in semantic_statuses
    ):
        return "unknown"
    return "success"


def compact_tool_output(output: Any, *, limit: int = 700) -> str:
    """Render primitive output text for training traces without guidance wrappers."""

    content = tool_output_content(output)
    if isinstance(content, str):
        text = content
    else:
        try:
            text = json.dumps(content, ensure_ascii=False, default=str)
        except TypeError:
            text = str(content)
    if INTEGRATED_GUIDANCE_MARKER in text:
        text = text.split(INTEGRATED_GUIDANCE_MARKER, 1)[0]
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_value(model_dump(mode="json"))
        except (TypeError, ValueError):
            pass
    return str(value)


def _collect_output_signals(
    value: Any,
    *,
    statuses: list[str],
    texts: list[str],
    status_scope: bool = True,
) -> None:
    if isinstance(value, ToolMessage):
        statuses.append(str(value.status or "").strip().lower())
        _collect_output_signals(
            value.content,
            statuses=statuses,
            texts=texts,
            status_scope=True,
        )
        return
    if isinstance(value, tuple) and len(value) == 2:
        _collect_output_signals(
            value[0],
            statuses=statuses,
            texts=texts,
            status_scope=status_scope,
        )
        return
    if isinstance(value, dict):
        status = value.get("status")
        if status_scope and status is not None:
            statuses.append(str(status).strip().lower())
        error = value.get("error")
        if status_scope and error not in (None, "", False, [], {}):
            statuses.append("error")
        for key, item in value.items():
            if key in {"status", "name", "tool_call_id"}:
                continue
            _collect_output_signals(
                item,
                statuses=statuses,
                texts=texts,
                status_scope=(
                    status_scope
                    and key
                    in {
                        "artifact",
                        "content",
                        "response",
                        "result",
                        "structured_content",
                    }
                ),
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict) and item.get("type") in {
                "json",
                "text",
            }:
                block_content = item.get("text", item.get("content"))
                _collect_output_signals(
                    block_content,
                    statuses=statuses,
                    texts=texts,
                    status_scope=status_scope,
                )
                continue
            _collect_output_signals(
                item,
                statuses=statuses,
                texts=texts,
                status_scope=False,
            )
        return
    if value is None:
        return
    if isinstance(value, (bool, int, float)):
        texts.append(json.dumps(value))
        return
    text = str(value).strip()
    if not text:
        return
    if INTEGRATED_GUIDANCE_MARKER in text:
        text = text.split(INTEGRATED_GUIDANCE_MARKER, 1)[0].strip()
    parsed = _parse_embedded_payload(text)
    if parsed is not None and parsed != text:
        _collect_output_signals(
            parsed,
            statuses=statuses,
            texts=texts,
            status_scope=status_scope,
        )
        return
    texts.append(text)


def _parse_embedded_payload(text: str) -> Any | None:
    candidates = [text]
    if text.startswith("content="):
        candidates.insert(0, text.partition("=")[2].strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, ValueError):
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
    return None
