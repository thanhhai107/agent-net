"""Runtime injection for DRAFT-refined primitive tool documentation."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import BaseTool

from agent.tool_evolution.models import ToolDocumentation
from agent.tool_evolution.store import ToolEvolutionStore


def _planned_parameters_match(
    planned_parameters: dict[str, Any],
    tool_arguments: dict[str, Any],
) -> bool:
    if not planned_parameters:
        return True
    for key, value in planned_parameters.items():
        if key not in tool_arguments or tool_arguments[key] != value:
            return False
    return True


def _clip(text: Any, *, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _format_checks(checks: list[str], *, item_limit: int = 120) -> str:
    return "; ".join(_clip(check, limit=item_limit) for check in checks if check)


_DIAGNOSIS_EXPLORATION_INTENTS = {"diagnosis_check"}
_DIAGNOSIS_SUGGESTION_MARKERS = (
    "diagnos",
    "localization",
    "localisation",
    "rca",
    "root cause",
    "fault",
    "anomaly",
    "hypothesis",
    "evidence",
    "reachability",
    "reachable",
    "connectivity",
    "packet loss",
    "latency",
    "interface",
    "link",
    "route",
    "routing",
    "bgp",
    "ospf",
    "neighbor",
    "service",
    "endpoint",
)
_TOOL_LEARNING_SUGGESTION_MARKERS = (
    "invalid",
    "boundary",
    "schema",
    "validation behavior",
    "parameter semantics",
    "automatic sampling",
    "minimal valid call",
    "topology-grounded call",
    "topology identifiers",
    "lab with",
    "api behavior",
    "error case",
    "stress",
)


def _normalize_check_text(text: Any) -> str:
    return (
        str(text or "")
        .replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u202f", " ")
        .strip()
        .lower()
    )


def _diagnosis_relevant_check(text: Any) -> bool:
    """Return whether a DRAFT exploration should guide live diagnosis."""
    return bool(_normalize_check_text(text))


def _diagnosis_relevant_doc_suggestion_text(text: Any) -> bool:
    normalized = _normalize_check_text(text)
    if not normalized:
        return False
    if any(marker in normalized for marker in _TOOL_LEARNING_SUGGESTION_MARKERS):
        return False
    if any(marker in normalized for marker in _DIAGNOSIS_SUGGESTION_MARKERS):
        return True
    return bool(_extract_topology_hosts(normalized))


def _extract_topology_hosts(text: str) -> set[str]:
    hosts: set[str] = set()
    for pattern in (
        r"\bpc_\d+_\d+\b",
        r"\bpc\d+\b",
        r"\bh\d+\b",
        r"\br\d+\b",
        r"\bhost\d+\b",
        r"\bhost[-_][A-Za-z0-9_.-]+\b",
        r"\b(?:leaf|spine|border|core|edge)[-_]router[-_][A-Za-z0-9_.-]+\b",
        r"\b(?:leaf|spine|border|core|edge)[-_]switch[-_][A-Za-z0-9_.-]+\b",
        r"\b(?:dns|dhcp|web|vpn)[-_]server\b",
    ):
        hosts.update(match.group(0).lower() for match in re.finditer(pattern, text))
    return hosts


def _extract_topology_hosts_from_value(value: Any) -> set[str]:
    if value in (None, "", [], {}):
        return set()
    if isinstance(value, dict):
        hosts: set[str] = set()
        for item in value.values():
            hosts.update(_extract_topology_hosts_from_value(item))
        return hosts
    if isinstance(value, (list, tuple, set)):
        hosts: set[str] = set()
        for item in value:
            hosts.update(_extract_topology_hosts_from_value(item))
        return hosts
    text = str(value).lower()
    hosts = _extract_topology_hosts(text)
    for token in re.findall(r"\b[a-z][a-z0-9_.-]*(?::eth\d+)?\b", text):
        name = token.split(":", 1)[0]
        if re.search(r"\d", name) or "_" in name or "-" in name:
            hosts.add(name)
    return hosts


class ToolEvolutionRuntime:
    """Inject refined documentation while keeping the primitive tool surface fixed."""

    def __init__(
        self,
        *,
        session: Any,
        primitive_tools: list[BaseTool],
        library_id: str,
        model: str = "",
        task_description: str = "",
        store: ToolEvolutionStore | None = None,
    ) -> None:
        self.session = session
        self.primitive_tools = list(primitive_tools)
        self.library_id = library_id
        self.model = model
        self.task_description = task_description
        self.store = store or ToolEvolutionStore(library_id)
        self._claimed_exploration_ids: set[str] = set()
        self._base_descriptions = {
            tool.name: (getattr(tool, "description", "") or "").strip()
            for tool in self.primitive_tools
        }
        self._docs = self._ensure_primitive_documents()

    def _ensure_primitive_documents(self) -> dict[str, ToolDocumentation]:
        state = self.store.load()
        changed = False
        for tool in self.primitive_tools:
            if tool.name in state.documents:
                continue
            state.documents[tool.name] = make_document_from_tool(tool)
            changed = True
        if changed:
            self.store.save(state)
        return state.documents

    def build_tools(self, *, append_docs: bool = True) -> list[BaseTool]:
        """Return the same primitive tools with DRAFT docs appended to descriptions."""
        for tool in self.primitive_tools:
            base_description = self._base_descriptions.get(
                tool.name,
                (getattr(tool, "description", "") or "").strip(),
            )
            if not append_docs:
                tool.description = base_description
                continue
            doc = self._docs.get(tool.name)
            if doc is None:
                tool.description = base_description
                continue
            refined = self._doc_runtime_text(tool.name, max_chars=500)
            if refined and refined not in base_description:
                tool.description = (
                    f"{base_description}\n\nDRAFT refined guidance:\n{refined}"
                )
        return self.primitive_tools

    def prompt_suffix(
        self,
        *,
        tool_names: list[str] | None = None,
        diagnosis_only: bool = True,
    ) -> str:
        if not self._docs:
            return ""
        tool_filter = {name for name in (tool_names or []) if name}
        active_docs = [doc for doc in self._docs.values() if not doc.frozen]
        if not active_docs:
            active_docs = list(self._docs.values())
        if tool_filter:
            active_docs = [doc for doc in active_docs if doc.name in tool_filter]
            if not active_docs:
                return ""
        state = self.store.load()
        planned_queue = self.planned_explorations(
            limit=4,
            diagnosis_only=diagnosis_only,
        )
        if tool_filter:
            planned_queue = [
                item for item in planned_queue if item.get("tool_name") in tool_filter
            ]
        planned_lines = [
            (
                f"- {item['tool_name']} [{item['exploration_id']}]: "
                f"{_clip(item['next_exploration'], limit=160)}"
                + (
                    f" Suggested parameters: {_clip(item['parameters'], limit=120)}"
                    if item.get("parameters")
                    else ""
                )
            )
            for item in planned_queue
        ]
        snippets = []
        doc_limit = 4 if tool_filter else 6
        for doc in sorted(active_docs, key=lambda item: item.name)[:doc_limit]:
            planned_checks = self.next_checks(
                doc.name,
                limit=2,
                diagnosis_only=diagnosis_only,
            )
            suffix = (
                "\n  Next active checks: " + _format_checks(planned_checks)
                if planned_checks
                else ""
            )
            snippets.append(
                f"- {doc.name}: "
                f"{self._refined_description(doc, max_chars=220, diagnosis_only=diagnosis_only)}"
                f"{suffix}"
            )
        return (
            "\n\nDRAFT tool documentation memory:\n"
            "The primitive tool surface is fixed. Use the following refined docs "
            "to choose valid arguments, avoid known failure modes, and follow "
            "DRAFT Explorer/Analyzer/Rewriter next-check suggestions when more "
            "evidence is needed. Treat DRAFT guidance as tool-use guidance, not "
            "as evidence by itself.\n"
            + (
                f"{state.library_usage_description}\n"
                if state.library_usage_description and not tool_filter
                else ""
            )
            + (
                "DRAFT active exploration queue. Work these checks into the "
                "investigation plan when relevant; they are hypotheses to test, "
                "not observations:\n"
                + "\n".join(planned_lines)
                + "\n"
                if planned_lines
                else ""
            )
            + "\n".join(snippets)
        )

    def tool_runtime_guidance(self, tool_name: str, *, max_chars: int = 500) -> str:
        """Return DRAFT guidance for one primitive tool."""
        return self._doc_runtime_text(tool_name, max_chars=max_chars)

    def _doc_runtime_text(
        self,
        tool_name: str,
        *,
        max_chars: int = 500,
        diagnosis_only: bool = True,
    ) -> str:
        doc = self._docs.get(tool_name)
        if doc is None:
            return ""
        planned = self.next_checks(
            tool_name,
            limit=2,
            diagnosis_only=diagnosis_only,
        )
        if planned:
            planned_text = (
                "DRAFT planned active checks: " + _format_checks(planned)
            )[: max(160, max_chars // 2)]
            base_budget = max(200, max_chars - len(planned_text) - 1)
            text = self._refined_description(
                doc,
                max_chars=base_budget,
                diagnosis_only=diagnosis_only,
            )
            text = (
                f"{text}\n"
                f"{planned_text}"
            )
            return text[:max_chars]
        return self._refined_description(
            doc,
            max_chars=max_chars,
            diagnosis_only=diagnosis_only,
        )

    def _refined_description(
        self,
        doc: ToolDocumentation,
        *,
        max_chars: int,
        diagnosis_only: bool = True,
    ) -> str:
        if not diagnosis_only:
            return doc.refined_description(max_chars=max_chars)
        suggestions = [
            item
            for item in doc.exploration_suggestions
            if self._diagnosis_relevant_doc_suggestion(
                tool_name=doc.name,
                text=item,
            )
        ]
        if len(suggestions) != len(doc.exploration_suggestions):
            doc = doc.model_copy(update={"exploration_suggestions": suggestions})
        return doc.refined_description(max_chars=max_chars)

    def _diagnosis_relevant_exploration(
        self,
        *,
        tool_name: str,
        text: Any,
        parameters: dict[str, Any],
        intent: str = "",
    ) -> bool:
        if str(intent or "unknown") not in _DIAGNOSIS_EXPLORATION_INTENTS:
            return False
        if not _diagnosis_relevant_check(text):
            return False
        known_hosts = self._known_hosts()
        mentioned_hosts = _extract_topology_hosts(_normalize_check_text(text))
        if known_hosts and any(host not in known_hosts for host in mentioned_hosts):
            return False
        return self._diagnosis_relevant_parameters(
            tool_name=tool_name,
            parameters=parameters,
            known_hosts=known_hosts,
        )

    def _diagnosis_relevant_doc_suggestion(
        self,
        *,
        tool_name: str,
        text: Any,
    ) -> bool:
        if not _diagnosis_relevant_doc_suggestion_text(text):
            return False
        known_hosts = self._known_hosts()
        mentioned_hosts = _extract_topology_hosts(_normalize_check_text(text))
        if mentioned_hosts and not known_hosts:
            return False
        if known_hosts and any(host not in known_hosts for host in mentioned_hosts):
            return False
        return self._diagnosis_relevant_parameters(
            tool_name=tool_name,
            parameters={},
            known_hosts=known_hosts,
        )

    def _known_hosts(self) -> set[str]:
        text = " ".join(
            str(item or "")
            for item in [
                self.task_description,
                getattr(self.session, "task_description", ""),
            ]
        ).lower()
        hosts = _extract_topology_hosts(text)
        hosts.update(
            _extract_topology_hosts_from_value(getattr(self.session, "topology", []))
        )
        return hosts

    @staticmethod
    def _diagnosis_relevant_parameters(
        *,
        tool_name: str,
        parameters: dict[str, Any],
        known_hosts: set[str],
    ) -> bool:
        if not parameters:
            return True
        lower_tool = tool_name.lower()
        if lower_tool in {"get_host_net_config", "ip_addr_statistics"}:
            host_name = str(
                parameters.get("host_name") or parameters.get("host") or ""
            ).lower()
            if known_hosts and host_name and host_name not in known_hosts:
                return False
        if lower_tool == "ping_pair":
            for key in ("host_a", "host_b"):
                value = str(parameters.get(key) or "").lower()
                if known_hosts and value not in known_hosts:
                    return False
        return True

    def next_checks(
        self,
        tool_name: str,
        *,
        limit: int = 3,
        diagnosis_only: bool = True,
    ) -> list[str]:
        """Return active DRAFT exploration directions for one tool."""
        doc = self._docs.get(tool_name)
        if doc is None:
            return []
        state = self.store.load()
        planned = [
            (item.next_exploration or item.user_query).strip()
            for item in state.explorations
            if item.tool_name == tool_name
            and item.status == "planned"
            and item.exploration_id not in self._claimed_exploration_ids
            and (item.next_exploration or item.user_query).strip()
            and (
                not diagnosis_only
                or self._diagnosis_relevant_exploration(
                    tool_name=item.tool_name,
                    text=item.next_exploration or item.user_query,
                    parameters=item.parameters,
                    intent=getattr(item, "intent", "unknown"),
                )
            )
        ]
        checks = planned[-limit:]
        checks.extend(
            item.strip()
            for item in doc.exploration_suggestions[-limit:]
            if item.strip()
            and (
                not diagnosis_only
                or self._diagnosis_relevant_doc_suggestion(
                    tool_name=tool_name,
                    text=item,
                )
            )
        )
        deduped: list[str] = []
        for check in checks:
            if check not in deduped:
                deduped.append(check)
            if len(deduped) >= limit:
                break
        return deduped

    def match_planned_exploration(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any] | None = None,
        *,
        diagnosis_only: bool = True,
    ) -> dict[str, Any] | None:
        """Return the planned DRAFT Explorer check a tool call is trying."""
        arguments = tool_arguments or {}
        state = self.store.load()
        fallback: dict[str, Any] | None = None
        for item in reversed(state.explorations):
            if item.status != "planned" or item.tool_name != tool_name:
                continue
            if item.exploration_id in self._claimed_exploration_ids:
                continue
            if diagnosis_only and not self._diagnosis_relevant_exploration(
                tool_name=item.tool_name,
                text=item.next_exploration or item.user_query,
                parameters=item.parameters,
                intent=getattr(item, "intent", "unknown"),
            ):
                continue
            row = {
                "exploration_id": item.exploration_id,
                "tool_name": item.tool_name,
                "intent": getattr(item, "intent", "unknown"),
                "next_exploration": item.next_exploration or item.user_query,
                "parameters": item.parameters,
                "analyzer_suggestion": item.analyzer_suggestion,
                "session_id": item.session_id,
            }
            if item.parameters and _planned_parameters_match(
                item.parameters,
                arguments,
            ):
                self._claimed_exploration_ids.add(item.exploration_id)
                return row
            if not item.parameters and fallback is None:
                fallback = row
        if fallback is not None:
            self._claimed_exploration_ids.add(fallback["exploration_id"])
        return fallback

    def planned_explorations(
        self,
        *,
        limit: int = 8,
        diagnosis_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return structured DRAFT Explorer checks waiting to be tried."""
        state = self.store.load()
        rows: list[dict[str, Any]] = []
        for item in state.explorations:
            if item.status != "planned":
                continue
            if item.exploration_id in self._claimed_exploration_ids:
                continue
            if diagnosis_only and not self._diagnosis_relevant_exploration(
                tool_name=item.tool_name,
                text=item.next_exploration or item.user_query,
                parameters=item.parameters,
                intent=getattr(item, "intent", "unknown"),
            ):
                continue
            rows.append(
                {
                    "exploration_id": item.exploration_id,
                    "tool_name": item.tool_name,
                    "intent": getattr(item, "intent", "unknown"),
                    "next_exploration": item.next_exploration or item.user_query,
                    "parameters": item.parameters,
                    "analyzer_suggestion": item.analyzer_suggestion,
                    "session_id": item.session_id,
                }
            )
        return rows[-limit:]

    def snapshot(self) -> dict[str, Any]:
        state = self.store.load()
        return {
            "library_id": self.library_id,
            "model": self.model,
            "task_description": self.task_description,
            "available_documents": sorted(self._docs),
            "library_usage_description": state.library_usage_description,
            "tool_stats": {
                name: stat.model_dump(mode="json")
                for name, stat in sorted(state.tool_stats.items())
            },
            "explorations": len(state.explorations),
            "planned_explorations": sum(
                exploration.status == "planned"
                for exploration in state.explorations
            ),
            "consumed_explorations": sum(
                exploration.status == "consumed"
                for exploration in state.explorations
            ),
            "planned_queue": self.planned_explorations(limit=8),
            "claimed_exploration_ids": sorted(self._claimed_exploration_ids),
            "analyzer_suggestions": len(state.analyzer_suggestions),
            "primitive_tools": [tool.name for tool in self.primitive_tools],
        }


def make_document_from_tool(tool: BaseTool) -> ToolDocumentation:
    description = getattr(tool, "description", "") or ""
    return ToolDocumentation(name=tool.name, description=description.strip())
