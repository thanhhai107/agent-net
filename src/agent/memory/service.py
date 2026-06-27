"""Extraction, retrieval, and LightMem-style consolidation orchestration."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm.model_factory import load_model
from agent.memory.models import (
    EvaluationEvidence,
    MemoryCandidate,
    MemoryExtraction,
    MemoryLinkType,
    MemoryQuery,
    MemoryRelationDecision,
    MemoryStatus,
    RetrievedMemory,
    StoredMemory,
)
from agent.memory.store import create_memory_store
from agent.memory.vector_index import QdrantMemoryIndex
from nika.config import MEMORY_DIR

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You extract reusable procedural memory from a network-troubleshooting trajectory.
Return at most six A-Mem-style atomic notes. Each note must contain exactly one
general procedural lesson that can improve a future investigation.

Strict safety requirements:
- Generalize device names, addresses, interface identifiers, and topology-specific
  values into roles such as source host, edge router, or affected interface.
- Do not store a benchmark problem identifier or a scenario-to-answer mapping.
- Treat proposed root causes as hypotheses requiring listed evidence.
- Do not invent observations that are absent from the trajectory.
- Keep tool names when they make the procedure reproducible.
- Ground truth and benchmark scores are intentionally unavailable.
- Do not classify notes into observation/error/learning/instruction types.
"""

RELATION_PROMPT = """\
Compare two reusable network-diagnosis memories. Return one relation only when it
is clearly supported:
- supports: the new memory provides compatible additional evidence;
- refines: the new memory is a more specific or better-scoped version;
- contradicts: both apply under the same conditions but recommend incompatible
  conclusions or actions;
- same_pattern: they describe the same diagnostic pattern without one replacing
  the other.
Return relation=null when no meaningful link exists.
"""

_DEVICE_TOKEN = re.compile(
    r"\b(?:pc|host|router|switch|server|client|leaf|spine|node|r|s|h)[_-]?\d+\b",
    re.IGNORECASE,
)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")
_MAC = re.compile(r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b", re.IGNORECASE)
_INTERFACE_TOKEN = re.compile(
    r"\b(?:eth|ens|enp|eno|ge|xe|swp)[A-Za-z0-9_.:/-]*\d+\b",
    re.IGNORECASE,
)


def _redact_episode_entities(text: str) -> str:
    text = _IPV4.sub("<network-address>", text)
    text = _MAC.sub("<mac-address>", text)
    text = _INTERFACE_TOKEN.sub("<interface>", text)
    return _DEVICE_TOKEN.sub("<device>", text)


def _token_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _jaccard(left: str, right: str) -> float:
    left_tokens = _terms(left)
    right_tokens = _terms(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_-]{3,}", text.lower()))


class ProceduralMemoryModule:
    """Atomic procedural memory backed by PostgreSQL and optional Qdrant."""

    def __init__(
        self,
        *,
        bank_id: str,
        llm_backend: str | None = None,
        model: str | None = None,
        store_path: str | Path | None = None,
        vector_index: QdrantMemoryIndex | None = None,
    ) -> None:
        safe_bank = re.sub(r"[^A-Za-z0-9_.-]+", "_", bank_id).strip("._")
        if not safe_bank:
            raise ValueError("memory bank id must contain at least one safe character")
        self.bank_id = safe_bank
        self.store = create_memory_store(
            sqlite_path=store_path or (Path(MEMORY_DIR) / f"{safe_bank}.sqlite3"),
            database_url=os.getenv("MEMORY_DATABASE_URL", "").strip() or None,
            force_sqlite=store_path is not None,
        )
        self.vector_index = vector_index or QdrantMemoryIndex()
        self.llm_backend = llm_backend
        self.model = model
        self._llm = None

    def _load_llm(self):
        if self._llm is None:
            if not self.llm_backend or not self.model:
                raise ValueError(
                    "llm_backend and model are required for memory extraction"
                )
            self._llm = load_model(self.llm_backend, self.model)
        return self._llm

    @staticmethod
    def compact_trace(trace_path: str | Path, max_chars: int = 32000) -> str:
        """Keep diagnosis evidence while excluding submission and oracle artifacts."""
        entries: list[dict[str, Any]] = []
        path = Path(trace_path)
        if not path.exists():
            return "[]"
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if row.get("agent") not in {"diagnosis_agent", "diagnosis_agent_cli"}:
                continue
            event = row.get("event")
            compact: dict[str, Any] = {"event": event}
            if row.get("phase"):
                compact["phase"] = row["phase"]
            if event == "tool_start":
                tool = row.get("tool") or {}
                compact["tool"] = (
                    tool.get("name") if isinstance(tool, dict) else str(tool)
                )
                compact["input"] = str(row.get("input", ""))[:1200]
            elif event in {"tool_end", "tool_error"}:
                compact["result"] = str(row.get("output") or row.get("error") or "")[
                    :1800
                ]
            elif event in {"llm_end", "item.completed", "turn.completed"}:
                compact["text"] = str(row.get("text") or row.get("codex_event") or "")[
                    :2400
                ]
            else:
                continue
            entries.append(compact)

        while entries and len(json.dumps(entries, default=str)) > max_chars:
            entries.pop(0)
        return _redact_episode_entities(
            json.dumps(entries, ensure_ascii=False, default=str)
        )

    async def extract(
        self,
        *,
        task_description: str,
        trace: str,
        scenario: str,
        topology_class: str,
    ) -> list[MemoryCandidate]:
        extractor = self._load_llm().with_structured_output(MemoryExtraction)
        payload = {
            "task": _redact_episode_entities(task_description)[:8000],
            "scenario_family": scenario,
            "topology_class": topology_class,
            "diagnosis_trajectory": trace,
        }
        raw = await extractor.ainvoke(
            [
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(
                    content=json.dumps(payload, ensure_ascii=False, default=str)
                ),
            ]
        )
        extraction = MemoryExtraction.model_validate(raw)
        result: list[MemoryCandidate] = []
        for candidate in extraction.memories:
            data = candidate.model_dump()
            data["content"] = _redact_episode_entities(candidate.content)
            data["applicability"] = [
                _redact_episode_entities(item) for item in candidate.applicability
            ]
            data["evidence_required"] = [
                _redact_episode_entities(item) for item in candidate.evidence_required
            ]
            data["avoid"] = [_redact_episode_entities(item) for item in candidate.avoid]
            attrs = candidate.attributes.model_dump()
            attrs["scenarios"] = sorted(
                set(attrs["scenarios"] + ([scenario] if scenario else []))
            )
            attrs["topology_classes"] = sorted(
                set(
                    attrs["topology_classes"]
                    + ([topology_class] if topology_class else [])
                )
            )
            data["attributes"] = attrs
            result.append(MemoryCandidate.model_validate(data))
        return result

    def validate(
        self,
        candidates: list[MemoryCandidate],
        evidence: EvaluationEvidence,
    ) -> list[tuple[MemoryCandidate, MemoryStatus, float]]:
        """LightMem-style numeric score gates without oracle text."""
        accepted: list[tuple[MemoryCandidate, MemoryStatus, float]] = []
        if evidence.fully_successful:
            for candidate in candidates:
                confidence = 0.72 + 0.18 * evidence.aggregate_score
                accepted.append(
                    (
                        candidate,
                        MemoryStatus.VALIDATED,
                        max(0.0, min(1.0, confidence)),
                    )
                )
            return accepted

        for candidate in candidates:
            # Failed/partial episodes can still teach cautious notes, but not
            # validated procedural rules. Require explicit evidence or avoid
            # clauses so the staged note remains checkable in future episodes.
            if not (candidate.evidence_required or candidate.avoid):
                continue
            confidence = 0.30 + 0.25 * evidence.aggregate_score
            accepted.append((candidate, MemoryStatus.STAGED, confidence))
        return accepted

    async def consolidate(
        self,
        *,
        source_session_id: str,
        validated: list[tuple[MemoryCandidate, MemoryStatus, float]],
        successful_episode: bool,
    ) -> list[StoredMemory]:
        persisted: list[StoredMemory] = []
        created: list[StoredMemory] = []
        for candidate, status, confidence in validated:
            memory, is_new = self.store.add_or_corroborate(
                bank_id=self.bank_id,
                candidate=candidate,
                status=status,
                confidence=confidence,
                source_session_id=source_session_id,
                successful_episode=successful_episode,
            )
            persisted.append(memory)
            if is_new:
                created.append(memory)
            try:
                self.vector_index.upsert(memory)
            except Exception as exc:
                logger.warning("Qdrant memory indexing skipped: %s", exc)

        if not created:
            return persisted

        relation_model = None
        for memory in created:
            related = self.store.search_fts(
                bank_id=self.bank_id,
                query=memory.embedding_text(),
                limit=3,
                statuses=(MemoryStatus.VALIDATED, MemoryStatus.STAGED),
                exclude_id=memory.memory_id,
                fallback=False,
            )
            if related and relation_model is None:
                relation_model = self._load_llm().with_structured_output(
                    MemoryRelationDecision
                )
            for other, lexical_score in related:
                if lexical_score <= 0 and not (
                    memory.attributes.flat_values() & other.attributes.flat_values()
                ):
                    continue
                try:
                    raw = await relation_model.ainvoke(
                        [
                            SystemMessage(content=RELATION_PROMPT),
                            HumanMessage(
                                content=json.dumps(
                                    {
                                        "new_memory": memory.model_dump(),
                                        "existing_memory": other.model_dump(),
                                    },
                                    ensure_ascii=False,
                                    default=str,
                                )
                            ),
                        ]
                    )
                    decision = MemoryRelationDecision.model_validate(raw)
                except Exception as exc:
                    logger.warning("Memory link classification skipped: %s", exc)
                    continue
                if decision.relation is None:
                    continue
                self.store.add_link(
                    bank_id=self.bank_id,
                    source_id=memory.memory_id,
                    target_id=other.memory_id,
                    relation=decision.relation,
                    reason=decision.reason,
                )
                if (
                    decision.relation == MemoryLinkType.REFINES
                    and memory.status == MemoryStatus.VALIDATED
                    and memory.confidence >= other.confidence
                ):
                    self.store.supersede(other.memory_id, memory.memory_id)
        return persisted

    def retrieve(
        self,
        *,
        query: MemoryQuery,
        session_id: str,
    ) -> list[RetrievedMemory]:
        self.store.record_episode_start(self.bank_id, session_id)
        lexical = self.store.search_fts(
            bank_id=self.bank_id,
            query=query.text,
            limit=query.candidate_limit,
        )
        lexical_scores = {memory.memory_id: score for memory, score in lexical}
        candidates = {memory.memory_id: memory for memory, _ in lexical}

        semantic_scores: dict[str, float] = {}
        try:
            semantic = self.vector_index.search(
                bank_id=self.bank_id,
                query=query.text,
                limit=query.candidate_limit,
            )
            semantic_scores = dict(semantic)
            for memory in self.store.get_many([item[0] for item in semantic]):
                if memory.status == MemoryStatus.VALIDATED:
                    candidates[memory.memory_id] = memory
        except Exception as exc:
            logger.warning("Qdrant memory retrieval skipped: %s", exc)

        query_attrs = query.attributes().flat_values()
        link_counts = self.store.link_counts(self.bank_id, list(candidates))
        max_links = max(link_counts.values(), default=1) or 1

        ranked: list[RetrievedMemory] = []
        for memory in candidates.values():
            memory_attrs = memory.attributes.flat_values()
            attribute_score = (
                len(query_attrs & memory_attrs) / len(query_attrs)
                if query_attrs
                else 0.0
            )
            structural_score = min(
                1.0,
                (
                    len(memory.evidence_required)
                    + len(memory.applicability)
                    + len(memory.avoid)
                )
                / 8,
            )
            graph_score = (
                0.65 * min(1.0, link_counts.get(memory.memory_id, 0) / max_links)
                + 0.35 * structural_score
            )
            lexical_score = lexical_scores.get(memory.memory_id, 0.0)
            semantic_score = semantic_scores.get(memory.memory_id, 0.0)
            relevance = max(lexical_score, semantic_score)
            score = (
                0.40 * relevance
                + 0.30 * attribute_score
                + 0.20 * memory.confidence
                + 0.10 * graph_score
            )
            ranked.append(
                RetrievedMemory(
                    memory=memory,
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    attribute_score=attribute_score,
                    graph_score=graph_score,
                )
            )

        selected: list[RetrievedMemory] = []
        used_tokens = 0
        remaining = ranked[:]
        while remaining and len(selected) < query.top_k:
            best: RetrievedMemory | None = None
            best_adjusted = float("-inf")
            for candidate in remaining:
                diversity_penalty = max(
                    (
                        _jaccard(
                            candidate.memory.embedding_text(),
                            prior.memory.embedding_text(),
                        )
                        for prior in selected
                    ),
                    default=0.0,
                )
                adjusted = candidate.score - 0.20 * diversity_penalty
                if adjusted > best_adjusted:
                    best = candidate
                    best_adjusted = adjusted
            if best is None:
                break
            remaining.remove(best)
            memory_tokens = _token_estimate(best.memory.embedding_text())
            if used_tokens + memory_tokens > query.token_budget:
                continue
            best.score = best_adjusted
            selected.append(best)
            used_tokens += memory_tokens

        self.store.record_retrieval(
            bank_id=self.bank_id,
            session_id=session_id,
            query_text=_redact_episode_entities(query.text),
            memory_ids=[item.memory.memory_id for item in selected],
            scores=[round(item.score, 6) for item in selected],
        )
        return selected

    def snapshot(self, *, session_id: str, output_path: str | Path) -> Path:
        memories, links = self.store.export_bank(self.bank_id)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "kind": "snapshot",
            "bank_id": self.bank_id,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for memory in memories:
                handle.write(
                    json.dumps(
                        {"kind": "memory", **memory},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
            for link in links:
                handle.write(
                    json.dumps(
                        {"kind": "link", **link},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        return path

    def clear(self) -> None:
        self.store.clear_bank(self.bank_id)
        try:
            self.vector_index.delete_bank(self.bank_id)
        except Exception as exc:
            logger.warning("Qdrant bank cleanup skipped: %s", exc)

    @staticmethod
    def format_context(memories: list[RetrievedMemory]) -> str:
        if not memories:
            return ""
        lines = [
            "Prior procedural memories (guidance only; verify every item with current tools):"
        ]
        for index, item in enumerate(memories, start=1):
            memory = item.memory
            lines.append(
                f"{index}. [confidence={memory.confidence:.2f}] {memory.content}"
            )
            if memory.applicability:
                lines.append("   Applies when: " + "; ".join(memory.applicability))
            if memory.evidence_required:
                lines.append("   Verify with: " + "; ".join(memory.evidence_required))
            if memory.avoid:
                lines.append("   Avoid: " + "; ".join(memory.avoid))
        return "\n".join(lines)
