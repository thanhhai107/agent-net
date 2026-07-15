"""Runtime injection for DRAFT-refined primitive tool documentation."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from langchain_core.tools import BaseTool

from agent.module_config import module_defaults
from agent.tool_refinement.generalization import generalize_tool_documentation
from agent.tool_refinement.models import ToolDocumentation, ToolParameterDoc, utc_now
from agent.tool_refinement.store import ToolRefinementStore


SOURCE_CONTRACT_VERSION = 1
_DEFAULTS = module_defaults().tool_refinement


def _primitive_description(tool: BaseTool) -> str:
    description = (getattr(tool, "description", "") or "").strip()
    for marker in (
        "\n\nDRAFT refined guidance:",
        "\n\n[Integrated learning guidance - not evidence]",
    ):
        if marker in description:
            description = description.split(marker, 1)[0].strip()
    return description


def _tool_args_schema(tool: BaseTool) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    schema: Any = {}
    if args_schema is not None:
        try:
            schema = args_schema.model_json_schema()
        except (AttributeError, TypeError, ValueError):
            if isinstance(args_schema, dict):
                schema = args_schema
    if not isinstance(schema, dict):
        return {}
    return schema


def _tool_source_payload(tool: BaseTool) -> dict[str, Any]:
    """Return the stable primitive contract that learned docs depend on."""
    return {
        "name": tool.name,
        "description": _primitive_description(tool),
        "args_schema": _tool_args_schema(tool),
    }


def _tool_source_signature(tool: BaseTool) -> str:
    encoded = json.dumps(
        _tool_source_payload(tool),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _schema_type_hint(schema: dict[str, Any]) -> str:
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        return raw_type
    if isinstance(raw_type, list):
        return " | ".join(str(item) for item in raw_type)
    for union_key in ("anyOf", "oneOf"):
        options = schema.get(union_key)
        if not isinstance(options, list):
            continue
        types = [
            str(option.get("type"))
            for option in options
            if isinstance(option, dict) and option.get("type")
        ]
        if types:
            return " | ".join(dict.fromkeys(types))
    return "unknown"


def _source_parameter_docs(tool: BaseTool) -> dict[str, ToolParameterDoc]:
    schema = _tool_args_schema(tool)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    required = set(schema.get("required") or [])
    docs: dict[str, ToolParameterDoc] = {}
    for name, raw in properties.items():
        parameter_schema = raw if isinstance(raw, dict) else {}
        constraints: list[str] = []
        if name in required:
            constraints.append("Required by the primitive tool contract.")
        if "enum" in parameter_schema:
            constraints.append(
                "Allowed values: "
                + ", ".join(str(item) for item in parameter_schema["enum"])
                + "."
            )
        docs[name] = ToolParameterDoc(
            name=name,
            type_hint=_schema_type_hint(parameter_schema),
            description=str(parameter_schema.get("description") or "").strip(),
            constraints=constraints,
        )
    return docs


def _clip(text: Any, *, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


class ToolRefinementRuntime:
    """Inject refined documentation while keeping the primitive tool surface fixed."""

    def __init__(
        self,
        *,
        session: Any,
        primitive_tools: list[BaseTool],
        library_id: str,
        store: ToolRefinementStore | None = None,
        tool_doc_chars: int = _DEFAULTS.tool_doc_chars,
        explorer_llm: Any | None = None,
        llm_backend: str = "",
        model: str = "",
        convergence_threshold: float = _DEFAULTS.convergence_threshold,
        exploration_similarity_threshold: float = _DEFAULTS.exploration_similarity_threshold,
        explorer_reflection_limit: int = _DEFAULTS.explorer_reflection_limit,
        max_tools_per_update: int = _DEFAULTS.max_tools_per_update,
        explorer_model: str = "",
        analyzer_model: str = "",
        rewriter_model: str = "",
    ) -> None:
        self.session = session
        self.primitive_tools = list(primitive_tools)
        self.library_id = library_id
        self.store = store or ToolRefinementStore(library_id)
        self.tool_doc_chars = max(100, int(tool_doc_chars))
        self.explorer_llm = explorer_llm
        self.llm_backend = llm_backend
        self.model = model
        self.convergence_threshold = float(convergence_threshold)
        self.exploration_similarity_threshold = float(exploration_similarity_threshold)
        self.explorer_reflection_limit = max(0, int(explorer_reflection_limit))
        self.max_tools_per_update = max(1, int(max_tools_per_update))
        self.explorer_model = explorer_model.strip()
        self.analyzer_model = analyzer_model.strip()
        self.rewriter_model = rewriter_model.strip()
        self._explorer_report: dict[str, Any] = {}
        self._explorer_duration = 0.0
        self._base_descriptions = {
            tool.name: _primitive_description(tool) for tool in self.primitive_tools
        }
        self._docs = self._ensure_primitive_documents()
        self._guidance_limits = self._allocate_guidance_limits()

    def _allocate_guidance_limits(self) -> dict[str, int]:
        """Bound DRAFT additions across the whole tool catalog.

        Primitive descriptions remain available for every tool. Refined notes are
        reserved for the best-supported tools so they cannot crowd out runtime
        evidence when both learning modules are enabled.
        """

        budget = min(
            _DEFAULTS.guidance_total_token_budget,
            max(_DEFAULTS.guidance_min_token_budget, self.tool_doc_chars * 4),
        )
        ranked = sorted(
            (doc for doc in self._docs.values() if doc.published),
            key=lambda doc: (
                doc.diagnostic_utility_score,
                doc.trial_count,
                doc.contract_mastery_score,
                doc.name,
            ),
            reverse=True,
        )
        limits: dict[str, int] = {}
        for doc in ranked:
            if budget < 60:
                break
            limit = min(_DEFAULTS.guidance_per_tool_token_budget, budget)
            limits[doc.name] = limit
            budget -= limit
        return limits

    def _ensure_primitive_documents(self) -> dict[str, ToolDocumentation]:
        with self.store.exclusive():
            state = self.store.load()
            changed = False
            for tool in self.primitive_tools:
                source_signature = _tool_source_signature(tool)
                source_schema = _tool_args_schema(tool)
                existing = state.documents.get(tool.name)
                if existing is None:
                    state.documents[tool.name] = make_document_from_tool(tool)
                    changed = True
                    continue
                contract_changed = bool(
                    existing.source_signature
                    and existing.source_signature != source_signature
                )
                legacy_contract = (
                    existing.source_contract_version < SOURCE_CONTRACT_VERSION
                )
                if contract_changed or legacy_contract:
                    replacement = make_document_from_tool(tool)
                    replacement.version = existing.version + 1
                    replacement.trial_count = existing.trial_count
                    replacement.success_count = existing.success_count
                    replacement.error_count = existing.error_count
                    reason = (
                        "Primitive tool contract changed; DRAFT documentation reset and reopened."
                        if contract_changed
                        else "Legacy DRAFT documentation reset against the immutable primitive contract."
                    )
                    # Old revisions remain in state.revisions for audit, but must
                    # not feed Analyzer/Rewriter under a different source contract.
                    replacement.rewrite_history = [reason]
                    state.documents[tool.name] = replacement
                    for exploration in state.explorations:
                        if exploration.tool_name == tool.name:
                            exploration.status = "invalidated"
                    changed = True
                    continue

                source_parameters = _source_parameter_docs(tool)
                repaired_parameters: dict[str, ToolParameterDoc] = {}
                for name, source_parameter in source_parameters.items():
                    learned_parameter = existing.parameters.get(name)
                    if learned_parameter is not None:
                        if not source_parameter.description:
                            source_parameter.description = learned_parameter.description
                        source_parameter.examples = learned_parameter.examples[-5:]
                    repaired_parameters[name] = source_parameter
                needs_repair = (
                    existing.description != self._base_descriptions.get(tool.name, "")
                    or existing.source_signature != source_signature
                    or existing.source_schema != source_schema
                    or existing.parameters != repaired_parameters
                )
                if needs_repair:
                    existing.description = self._base_descriptions.get(tool.name, "")
                    existing.source_signature = source_signature
                    existing.source_schema = source_schema
                    existing.parameters = repaired_parameters
                    existing.updated_at = utc_now()
                    changed = True
                historical_trials = [
                    trial for trial in state.trials if trial.tool_name == tool.name
                ]
                if generalize_tool_documentation(existing, trials=historical_trials):
                    existing.updated_at = utc_now()
                    changed = True
                state.documents[tool.name] = existing
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
            refined = self.tool_runtime_guidance(
                tool.name,
                max_chars=self.tool_doc_chars,
            )
            if refined and refined not in base_description:
                tool.description = (
                    f"{base_description}\n\nDRAFT refined guidance:\n{refined}"
                )
        return self.primitive_tools

    async def explore(self, task_description: str) -> dict[str, Any]:
        """Run the self-driven DRAFT Explorer before the submission phase."""

        if self.explorer_llm is None:
            self._explorer_report = {
                "status": "skipped",
                "reason": "explorer model unavailable",
            }
            return self._explorer_report
        from agent.tool_refinement.explorer import run_active_exploration

        started = time.perf_counter()
        try:
            self._explorer_report = await run_active_exploration(
                session_id=str(getattr(self.session, "session_id", "") or ""),
                session_dir=str(getattr(self.session, "session_dir", "") or ""),
                task_description=task_description,
                tools=self.primitive_tools,
                store=self.store,
                llm=self.explorer_llm,
                model=self.model,
                exploration_similarity_threshold=(
                    self.exploration_similarity_threshold
                ),
                explorer_reflection_limit=self.explorer_reflection_limit,
                max_tools=self.max_tools_per_update,
                analyzer_model=self.analyzer_model,
                rewriter_model=self.rewriter_model,
            )
        except Exception as exc:
            self._explorer_report = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self._explorer_duration += time.perf_counter() - started
        self._docs = self.store.load().documents
        self._guidance_limits = self._allocate_guidance_limits()
        return self._explorer_report

    def tool_runtime_guidance(
        self,
        tool_name: str,
        *,
        max_chars: int = _DEFAULTS.guidance_total_token_budget,
    ) -> str:
        """Return only learned contract deltas for one primitive tool."""

        doc = self._docs.get(tool_name)
        if doc is None or not doc.published:
            return ""
        allocated_tokens = self._guidance_limits.get(tool_name, 0)
        if allocated_tokens <= 0:
            return ""
        max_chars = min(max_chars, allocated_tokens * 4)
        parts: list[str] = []
        usage = doc.tool_usage_description.strip()
        if usage and usage != doc.description.strip():
            parts.append(usage)
        if doc.preconditions:
            parts.append("Preconditions: " + "; ".join(doc.preconditions[:2]))
        if doc.constraints:
            parts.append("Constraints: " + "; ".join(doc.constraints[:3]))
        if doc.failure_modes:
            parts.append("Failure modes: " + "; ".join(doc.failure_modes[:2]))
        if doc.usage_notes:
            parts.append("Usage notes: " + "; ".join(doc.usage_notes[:2]))
        return _clip("\n".join(parts), limit=max_chars)

    def snapshot(self) -> dict[str, Any]:
        state = self.store.load()
        return {
            "library_id": self.library_id,
            "available_documents": sorted(self._docs),
            "library_usage_description": state.library_usage_description,
            "tool_stats": {
                name: stat.model_dump(mode="json")
                for name, stat in sorted(state.tool_stats.items())
            },
            "explorations": len(state.explorations),
            "config": {
                "tool_doc_chars": self.tool_doc_chars,
                "convergence_threshold": self.convergence_threshold,
                "exploration_similarity_threshold": (
                    self.exploration_similarity_threshold
                ),
                "explorer_reflection_limit": self.explorer_reflection_limit,
                "max_tools_per_update": self.max_tools_per_update,
                "explorer_model": self.explorer_model or self.model,
                "analyzer_model": self.analyzer_model or self.model,
                "rewriter_model": self.rewriter_model or self.model,
            },
            "explorer": self._explorer_report,
            "explorer_duration": round(self._explorer_duration, 6),
            "analyzer_suggestions": len(state.analyzer_suggestions),
            "published_documents": sorted(
                name for name, doc in state.documents.items() if doc.published
            ),
            "primitive_tools": [tool.name for tool in self.primitive_tools],
        }


def make_document_from_tool(tool: BaseTool) -> ToolDocumentation:
    description = _primitive_description(tool)
    return ToolDocumentation(
        name=tool.name,
        description=description,
        source_signature=_tool_source_signature(tool),
        source_schema=_tool_args_schema(tool),
        source_contract_version=SOURCE_CONTRACT_VERSION,
        parameters=_source_parameter_docs(tool),
    )
