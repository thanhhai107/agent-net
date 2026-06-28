"""Concurrent-safe persistent storage for one diagnostic tool library."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import fcntl

from agent.tool_evolution.models import (
    CapabilityGap,
    CompositeTool,
    GeneratedTool,
    ToolLibraryState,
    ToolCardRevision,
    ToolMastery,
    ToolUsageExample,
    ToolVerificationReport,
    ValidationEvidence,
    utc_now,
)
from nika.config import TOOL_EVOLUTION_DIR


_SAFE_LIBRARY_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


def normalize_library_id(library_id: str) -> str:
    normalized = _SAFE_LIBRARY_ID.sub("-", library_id.strip()).strip(".-")
    if not normalized:
        raise ValueError("tool library id must contain at least one letter or number")
    return normalized[:96]


def _append_unique(target: list[str], values: list[str], *, limit: int = 12) -> None:
    for raw in values:
        value = " ".join(str(raw).split()).strip()
        if value and value not in target:
            target.append(value)
    if len(target) > limit:
        del target[:-limit]


class ToolEvolutionStore:
    """JSON-backed library isolated by ``library_id``."""

    def __init__(
        self,
        library_id: str,
        root: str | Path | None = None,
        *,
        capacity: int | None = None,
    ) -> None:
        self.library_id = normalize_library_id(library_id)
        self.root = Path(root or TOOL_EVOLUTION_DIR)
        configured_capacity = capacity or int(
            os.environ.get("NIKA_TOOL_LIBRARY_CAPACITY", "250")
        )
        self.capacity = max(configured_capacity, 10)
        self.library_dir = self.root / self.library_id
        self.state_path = self.library_dir / "state.json"
        self.lock_path = self.library_dir / ".lock"
        self.events_path = self.library_dir / "events.jsonl"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> ToolLibraryState:
        if not self.state_path.exists():
            return ToolLibraryState(library_id=self.library_id)
        state = ToolLibraryState.model_validate_json(
            self.state_path.read_text(encoding="utf-8")
        )
        if state.schema_version < 3:
            state.schema_version = 3
        return state

    def _write_unlocked(self, state: ToolLibraryState) -> None:
        state.updated_at = utc_now()
        self.library_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".state-",
            suffix=".json",
            dir=self.library_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(state.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.state_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def load(self) -> ToolLibraryState:
        with self._lock():
            return self._read_unlocked()

    def reset(self) -> None:
        with self._lock():
            self._write_unlocked(ToolLibraryState(library_id=self.library_id))
            self.events_path.unlink(missing_ok=True)

    def append_event(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock():
            row = {"timestamp": utc_now(), "event": event, **payload}
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def upsert_mastery(
        self,
        tool_name: str,
        *,
        preconditions: list[str] | None = None,
        parameter_guidance: list[str] | None = None,
        output_interpretation: list[str] | None = None,
        failure_semantics: list[str] | None = None,
        usage_example: ToolUsageExample | None = None,
        calls: int = 0,
        successes: int = 0,
        errors: int = 0,
        source_model: str | None = None,
        revision_source: Literal[
            "runtime",
            "explorer",
            "analyzer",
            "rewriter",
        ] = "rewriter",
        rationale: str = "",
        utility_delta: float = 0.0,
    ) -> ToolMastery:
        with self._lock():
            state = self._read_unlocked()
            mastery = state.mastery.get(tool_name) or ToolMastery(tool_name=tool_name)
            before = self._mastery_semantic_hash(mastery)
            _append_unique(mastery.preconditions, preconditions or [])
            _append_unique(mastery.parameter_guidance, parameter_guidance or [])
            _append_unique(mastery.output_interpretation, output_interpretation or [])
            _append_unique(mastery.failure_semantics, failure_semantics or [])
            if usage_example is not None:
                serialized = usage_example.model_dump()
                existing = [item.model_dump() for item in mastery.usage_examples]
                if serialized not in existing:
                    mastery.usage_examples.append(usage_example)
                mastery.usage_examples = mastery.usage_examples[-6:]
            mastery.calls += calls
            mastery.successes += successes
            mastery.errors += errors
            if source_model and source_model not in mastery.source_models:
                mastery.source_models.append(source_model)
            after = self._mastery_semantic_hash(mastery)
            if after != before:
                if mastery.revisions:
                    mastery.version += 1
                mastery.revisions.append(
                    ToolCardRevision(
                        version=mastery.version,
                        source=revision_source,
                        rationale=rationale,
                        evidence_hash=after,
                        utility_delta=utility_delta,
                    )
                )
                mastery.revisions = mastery.revisions[-20:]
                mastery.convergence_count = 0
            elif any(
                (
                    preconditions,
                    parameter_guidance,
                    output_interpretation,
                    failure_semantics,
                )
            ):
                mastery.convergence_count += 1
            mastery.updated_at = utc_now()
            state.mastery[tool_name] = mastery
            self._write_unlocked(state)
            return mastery

    @staticmethod
    def _mastery_semantic_hash(mastery: ToolMastery) -> str:
        payload = {
            "preconditions": mastery.preconditions,
            "parameter_guidance": mastery.parameter_guidance,
            "output_interpretation": mastery.output_interpretation,
            "failure_semantics": mastery.failure_semantics,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def record_capability_gap(self, gap: CapabilityGap) -> CapabilityGap:
        with self._lock():
            state = self._read_unlocked()
            existing = state.capability_gaps.get(gap.gap_id)
            if existing is not None:
                return existing
            state.capability_gaps[gap.gap_id] = gap
            if len(state.capability_gaps) > 100:
                oldest = sorted(
                    state.capability_gaps.values(),
                    key=lambda item: item.created_at,
                )
                for item in oldest[:-100]:
                    state.capability_gaps.pop(item.gap_id, None)
            self._write_unlocked(state)
            return gap

    def resolve_capability_gap(
        self,
        gap_id: str,
        *,
        proposed_tool: str,
        resolved: bool = False,
    ) -> CapabilityGap | None:
        with self._lock():
            state = self._read_unlocked()
            gap = state.capability_gaps.get(gap_id)
            if gap is None:
                return None
            gap.proposed_tool = proposed_tool
            gap.status = "resolved" if resolved else "synthesized"
            gap.updated_at = utc_now()
            state.capability_gaps[gap_id] = gap
            self._write_unlocked(state)
            return gap

    def register_composite(
        self,
        composite: CompositeTool,
        *,
        deduplicate: bool = True,
    ) -> tuple[CompositeTool, bool]:
        composite = composite.model_copy(deep=True).ensure_signature()
        with self._lock():
            state = self._read_unlocked()
            if deduplicate:
                for existing in state.composites.values():
                    if (
                        existing.status != "rejected"
                        and existing.signature == composite.signature
                    ):
                        return existing, False

            base_name = composite.name
            suffix = 2
            while composite.name in state.composites:
                composite.name = f"{base_name[:58]}_{suffix}"
                suffix += 1
            composite.status = (
                "promoted" if composite.status == "promoted" else "candidate"
            )
            composite.updated_at = utc_now()
            state.composites[composite.name] = composite
            self._prune_to_capacity(state)
            self._write_unlocked(state)
            return composite, True

    def register_generated_tool(
        self,
        tool: GeneratedTool,
        *,
        deduplicate: bool = True,
    ) -> tuple[GeneratedTool, bool]:
        tool = tool.model_copy(deep=True).ensure_signature()
        with self._lock():
            state = self._read_unlocked()
            if deduplicate:
                for existing in state.generated_tools.values():
                    if (
                        existing.status != "rejected"
                        and existing.signature == tool.signature
                    ):
                        return existing, False

            base_name = tool.name
            suffix = 2
            while tool.name in state.generated_tools or tool.name in state.composites:
                tool.name = f"{base_name[:58]}_{suffix}"
                suffix += 1
            tool.status = "promoted" if tool.status == "promoted" else "candidate"
            tool.updated_at = utc_now()
            state.generated_tools[tool.name] = tool
            self._prune_generated_to_capacity(state)
            self._write_unlocked(state)
            return tool, True

    def _prune_to_capacity(self, state: ToolLibraryState) -> None:
        overflow = len(state.composites) - self.capacity
        if overflow <= 0:
            return
        ranked = sorted(
            state.composites.values(),
            key=lambda item: (
                item.status != "rejected",
                item.status == "promoted",
                item.utility_score(),
                item.updated_at,
            ),
        )
        for composite in ranked[:overflow]:
            state.composites.pop(composite.name, None)

    def _prune_generated_to_capacity(self, state: ToolLibraryState) -> None:
        overflow = len(state.generated_tools) - self.capacity
        if overflow <= 0:
            return
        ranked = sorted(
            state.generated_tools.values(),
            key=lambda item: (
                item.status != "rejected",
                item.status == "promoted",
                item.utility_score(),
                item.updated_at,
            ),
        )
        for tool in ranked[:overflow]:
            state.generated_tools.pop(tool.name, None)

    def get_composite(self, name: str) -> CompositeTool | None:
        return self.load().composites.get(name)

    def get_generated_tool(self, name: str) -> GeneratedTool | None:
        return self.load().generated_tools.get(name)

    def record_composite_evidence(
        self,
        name: str,
        evidence: ValidationEvidence,
        *,
        validation_enabled: bool = True,
        min_distinct_contexts: int = 2,
    ) -> CompositeTool | None:
        with self._lock():
            state = self._read_unlocked()
            composite = state.composites.get(name)
            if composite is None:
                return None
            duplicate = any(
                item.context_fingerprint == evidence.context_fingerprint
                and item.source == evidence.source
                and item.incident_success == evidence.incident_success
                and item.execution_success == evidence.execution_success
                and item.structural_valid == evidence.structural_valid
                and item.semantic_valid == evidence.semantic_valid
                for item in composite.evidence
            )
            if not duplicate:
                composite.evidence.append(evidence)
                composite.execution_count += 1
                if evidence.execution_success:
                    composite.success_count += 1
                if evidence.source == "replay":
                    replay_passed = (
                        evidence.execution_success
                        and evidence.incident_success
                        and evidence.structural_valid
                        and evidence.semantic_valid
                    )
                    composite.verification_reports.append(
                        ToolVerificationReport(
                            stage="replay",
                            passed=replay_passed,
                            checks=(
                                ["hidden replay incident solved"]
                                if replay_passed
                                else []
                            ),
                            error=(
                                None
                                if replay_passed
                                else "hidden replay validation failed"
                            ),
                            context_fingerprint=evidence.context_fingerprint,
                        )
                    )

            successful_contexts = {
                item.context_fingerprint
                for item in composite.evidence
                if item.execution_success
                and item.incident_success
                and item.structural_valid
                and item.semantic_valid
                and item.source in {"runtime", "replay"}
            }
            consecutive_failures = 0
            for item in reversed(composite.evidence):
                if item.execution_success:
                    break
                consecutive_failures += 1
            latest_succeeded = bool(
                composite.evidence and composite.evidence[-1].execution_success
            )
            if composite.status == "rejected":
                pass
            elif consecutive_failures >= 2:
                composite.status = "rejected"
            elif not latest_succeeded:
                composite.status = "candidate"
            elif (
                not validation_enabled
                or len(successful_contexts) >= min_distinct_contexts
            ):
                composite.status = "promoted"
            elif composite.status != "rejected":
                composite.status = "candidate"
            composite.updated_at = utc_now()
            state.composites[name] = composite
            self._write_unlocked(state)
            return composite

    def record_verification(
        self,
        name: str,
        report: ToolVerificationReport,
    ) -> CompositeTool | None:
        with self._lock():
            state = self._read_unlocked()
            composite = state.composites.get(name)
            if composite is None:
                return None
            composite.verification_reports.append(report)
            composite.verification_reports = composite.verification_reports[-30:]
            composite.updated_at = utc_now()
            state.composites[name] = composite
            self._write_unlocked(state)
            return composite

    def record_generated_evidence(
        self,
        name: str,
        evidence: ValidationEvidence,
        *,
        validation_enabled: bool = True,
        min_distinct_contexts: int = 2,
    ) -> GeneratedTool | None:
        with self._lock():
            state = self._read_unlocked()
            tool = state.generated_tools.get(name)
            if tool is None:
                return None
            duplicate = any(
                item.context_fingerprint == evidence.context_fingerprint
                and item.source == evidence.source
                and item.incident_success == evidence.incident_success
                and item.execution_success == evidence.execution_success
                and item.structural_valid == evidence.structural_valid
                and item.semantic_valid == evidence.semantic_valid
                for item in tool.evidence
            )
            if not duplicate:
                tool.evidence.append(evidence)
                tool.execution_count += 1
                if evidence.execution_success:
                    tool.success_count += 1

            successful_contexts = {
                item.context_fingerprint
                for item in tool.evidence
                if item.execution_success
                and item.incident_success
                and item.structural_valid
                and item.semantic_valid
                and item.source in {"runtime", "replay"}
            }
            consecutive_failures = 0
            for item in reversed(tool.evidence):
                if item.execution_success:
                    break
                consecutive_failures += 1
            latest_succeeded = bool(
                tool.evidence and tool.evidence[-1].execution_success
            )
            if tool.status == "rejected":
                pass
            elif consecutive_failures >= 2:
                tool.status = "rejected"
            elif not latest_succeeded:
                tool.status = "candidate"
            elif (
                not validation_enabled
                or len(successful_contexts) >= min_distinct_contexts
            ):
                tool.status = "promoted"
            elif tool.status != "rejected":
                tool.status = "candidate"
            tool.updated_at = utc_now()
            state.generated_tools[name] = tool
            self._write_unlocked(state)
            return tool

    def record_generated_verification(
        self,
        name: str,
        report: ToolVerificationReport,
    ) -> GeneratedTool | None:
        with self._lock():
            state = self._read_unlocked()
            tool = state.generated_tools.get(name)
            if tool is None:
                return None
            tool.verification_reports.append(report)
            tool.verification_reports = tool.verification_reports[-30:]
            tool.updated_at = utc_now()
            state.generated_tools[name] = tool
            self._write_unlocked(state)
            return tool

    @staticmethod
    def _composite_search_text(composite: CompositeTool) -> str:
        step_parts: list[str] = []
        for step in composite.steps:
            step_parts.extend(
                [
                    step.tool,
                    step.label,
                    json.dumps(step.arguments, sort_keys=True, default=str),
                ]
            )
        parameter_parts = [
            f"{parameter.name} {parameter.description} {parameter.type}"
            for parameter in composite.parameters
        ]
        return " ".join(
            [
                composite.name,
                composite.description,
                *composite.tags,
                *composite.output_contract,
                *parameter_parts,
                *step_parts,
            ]
        ).lower()

    @staticmethod
    def _generated_search_text(tool: GeneratedTool) -> str:
        parameter_parts = [
            f"{parameter.name} {parameter.description} {parameter.type}"
            for parameter in tool.parameters
        ]
        return " ".join(
            [
                tool.name,
                tool.description,
                tool.output_description,
                *tool.tags,
                *parameter_parts,
            ]
        ).lower()

    def search_composites(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        top_k: int = 5,
        include_candidates: bool = True,
        record_usage: bool = True,
    ) -> list[CompositeTool]:
        with self._lock():
            state = self._read_unlocked()
            query_tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
            query_tokens.update(str(tag).lower() for tag in tags or [])
            query_ngrams = self._character_ngrams(" ".join(sorted(query_tokens)))
            scored: list[tuple[float, CompositeTool]] = []
            for composite in state.composites.values():
                if composite.status == "rejected":
                    continue
                if composite.status == "candidate" and not include_candidates:
                    continue
                text = self._composite_search_text(composite)
                tokens = set(re.findall(r"[a-z0-9_]+", text))
                overlap = len(query_tokens & tokens)
                candidate_ngrams = self._character_ngrams(text)
                union = query_ngrams | candidate_ngrams
                fuzzy = (
                    len(query_ngrams & candidate_ngrams) / len(union)
                    if union and query_ngrams
                    else 0.0
                )
                score = float(overlap) + fuzzy * 2.0
                if composite.status == "promoted":
                    score += 2.0
                score += min(composite.utility_score(), 10.0) * 0.1
                if score > 0 or not query_tokens:
                    scored.append((score, composite))
            scored.sort(
                key=lambda item: (item[0], item[1].utility_score()),
                reverse=True,
            )
            selected = [item[1] for item in scored[:top_k]]
            now = utc_now()
            for composite in selected:
                stored = state.composites[composite.name]
                stored.retrieval_count += 1
                stored.last_used_at = now
            if selected and record_usage:
                self._write_unlocked(state)
            return selected

    def search_generated_tools(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        top_k: int = 5,
        include_candidates: bool = True,
        record_usage: bool = True,
    ) -> list[GeneratedTool]:
        with self._lock():
            state = self._read_unlocked()
            query_tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
            query_tokens.update(str(tag).lower() for tag in tags or [])
            query_ngrams = self._character_ngrams(" ".join(sorted(query_tokens)))
            scored: list[tuple[float, GeneratedTool]] = []
            for tool in state.generated_tools.values():
                if tool.status == "rejected":
                    continue
                if tool.status == "candidate" and not include_candidates:
                    continue
                text = self._generated_search_text(tool)
                tokens = set(re.findall(r"[a-z0-9_]+", text))
                overlap = len(query_tokens & tokens)
                candidate_ngrams = self._character_ngrams(text)
                union = query_ngrams | candidate_ngrams
                fuzzy = (
                    len(query_ngrams & candidate_ngrams) / len(union)
                    if union and query_ngrams
                    else 0.0
                )
                score = float(overlap) + fuzzy * 2.0
                if tool.status == "promoted":
                    score += 2.0
                score += min(tool.utility_score(), 10.0) * 0.1
                if score > 0 or not query_tokens:
                    scored.append((score, tool))
            scored.sort(
                key=lambda item: (item[0], item[1].utility_score()),
                reverse=True,
            )
            selected = [item[1] for item in scored[:top_k]]
            now = utc_now()
            for tool in selected:
                stored = state.generated_tools[tool.name]
                stored.retrieval_count += 1
                stored.last_used_at = now
            if selected and record_usage:
                self._write_unlocked(state)
            return selected

    @staticmethod
    def _character_ngrams(value: str, size: int = 3) -> set[str]:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if len(normalized) < size:
            return {normalized} if normalized else set()
        return {
            normalized[index : index + size]
            for index in range(len(normalized) - size + 1)
        }
