"""Unit tests for the persistent hybrid procedural-memory module."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from qdrant_client import QdrantClient

from agent.memory.adapter import MemoryAugmentedAgent
from agent.memory.models import (
    EvaluationEvidence,
    MemoryAttributes,
    MemoryCandidate,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
)
from agent.memory.service import HybridMemoryModule
from agent.memory.store import SQLiteMemoryStore
from agent.memory.vector_index import QdrantMemoryIndex
from agent.memory.workflow import evolve_session_memory
from nika.workflows.benchmark.run import run_benchmark_from_csv


class DisabledVectorIndex:
    enabled = False

    def upsert(self, memory) -> None:
        return None

    def search(self, **kwargs) -> list:
        return []

    def delete_bank(self, bank_id: str) -> None:
        return None


class FixedEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float("bgp" in text.lower()),
                float("routing" in text.lower()),
                1.0,
            ]
            for text in texts
        ]


class FakeWorkflow:
    def __init__(self, session_dir: str) -> None:
        self.session_id = "episode-1"
        self.session = MagicMock(
            session_dir=session_dir,
            scenario_name="dc_clos_bgp",
            scenario_topo_size="s",
        )
        self.diagnosis_tool_names = ["ping_pair", "frr_show_ip_route"]
        self.received_task = ""

    async def run(self, task_description: str) -> dict:
        self.received_task = task_description
        return {"ok": True}


def learning_candidate(content: str | None = None) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=MemoryType.LEARNING,
        content=content
        or (
            "Check routing adjacency and route propagation before concluding "
            "that an inter-router link has failed."
        ),
        applicability=["Reachability is asymmetric across routing domains."],
        evidence_required=[
            "Compare neighbor state and advertised routes at both endpoints."
        ],
        avoid=["Do not localize from one failed ping alone."],
        attributes=MemoryAttributes(
            protocols=["bgp"],
            task_stages=["diagnosis", "localization"],
            symptoms=["asymmetric reachability"],
            tools=["frr_show_ip_route"],
        ),
    )


class SQLiteMemoryStoreTest(unittest.TestCase):
    def test_atomic_memory_is_canonical_and_fts_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            memory, created = store.add_or_corroborate(
                bank_id="experiment",
                candidate=learning_candidate(),
                status=MemoryStatus.VALIDATED,
                confidence=0.9,
                source_session_id="episode-1",
                successful_episode=True,
            )

            results = store.search_fts(
                bank_id="experiment",
                query="BGP neighbor routing adjacency",
                limit=10,
            )

        self.assertTrue(created)
        self.assertEqual(memory.status, MemoryStatus.VALIDATED)
        self.assertEqual(results[0][0].memory_id, memory.memory_id)

    def test_repeated_validation_promotes_learning_to_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            memory = None
            for episode in range(3):
                memory, _ = store.add_or_corroborate(
                    bank_id="experiment",
                    candidate=learning_candidate(),
                    status=MemoryStatus.VALIDATED,
                    confidence=0.8,
                    source_session_id=f"episode-{episode}",
                    successful_episode=True,
                )

        self.assertIsNotNone(memory)
        self.assertEqual(memory.memory_type, MemoryType.INSTRUCTION)
        self.assertEqual(memory.validation_count, 3)

    def test_success_can_promote_matching_staged_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            observation = learning_candidate().model_copy(
                update={"memory_type": MemoryType.OBSERVATION}
            )
            staged, _ = store.add_or_corroborate(
                bank_id="experiment",
                candidate=observation,
                status=MemoryStatus.STAGED,
                confidence=0.4,
                source_session_id="failed-episode",
                successful_episode=False,
            )
            promoted, created = store.add_or_corroborate(
                bank_id="experiment",
                candidate=learning_candidate(),
                status=MemoryStatus.VALIDATED,
                confidence=0.9,
                source_session_id="successful-episode",
                successful_episode=True,
            )

        self.assertFalse(created)
        self.assertEqual(promoted.memory_id, staged.memory_id)
        self.assertEqual(promoted.status, MemoryStatus.VALIDATED)
        self.assertEqual(promoted.memory_type, MemoryType.LEARNING)


class MemoryPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.module = HybridMemoryModule(
            bank_id="test",
            store_path=Path(self.tmp.name) / "memory.sqlite3",
            vector_index=DisabledVectorIndex(),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_success_promotes_observation_to_validated_learning(self) -> None:
        observation = learning_candidate().model_copy(
            update={"memory_type": MemoryType.OBSERVATION}
        )
        gated = self.module.validate(
            [observation],
            EvaluationEvidence(
                detection_score=1,
                localization_f1=1,
                rca_f1=1,
            ),
        )

        candidate, status, confidence = gated[0]
        self.assertEqual(candidate.memory_type, MemoryType.LEARNING)
        self.assertEqual(status, MemoryStatus.VALIDATED)
        self.assertGreater(confidence, 0.8)

    def test_single_episode_cannot_create_instruction_directly(self) -> None:
        instruction = learning_candidate().model_copy(
            update={"memory_type": MemoryType.INSTRUCTION}
        )
        gated = self.module.validate(
            [instruction],
            EvaluationEvidence(
                detection_score=1,
                localization_f1=1,
                rca_f1=1,
            ),
        )

        self.assertEqual(gated[0][0].memory_type, MemoryType.LEARNING)

    def test_failure_only_stages_observations_and_errors(self) -> None:
        observation = learning_candidate().model_copy(
            update={"memory_type": MemoryType.OBSERVATION}
        )
        unsafe_learning = learning_candidate(
            "Always classify this symptom as one fixed benchmark answer."
        )
        gated = self.module.validate(
            [observation, unsafe_learning],
            EvaluationEvidence(
                detection_score=1,
                localization_f1=0,
                rca_f1=0,
            ),
        )

        self.assertEqual(len(gated), 1)
        self.assertEqual(gated[0][1], MemoryStatus.STAGED)

    def test_retrieval_respects_top_k_and_token_budget(self) -> None:
        for index in range(4):
            self.module.store.add_or_corroborate(
                bank_id="test",
                candidate=learning_candidate(
                    f"Check BGP neighbor evidence using diagnostic sequence {index} "
                    "before localizing the affected routing role."
                ),
                status=MemoryStatus.VALIDATED,
                confidence=0.8,
                source_session_id=f"episode-{index}",
                successful_episode=True,
            )

        results = self.module.retrieve(
            query=MemoryQuery(
                text="Investigate BGP neighbor reachability and routing state.",
                scenario="dc_clos_bgp",
                protocols=["bgp"],
                top_k=2,
                token_budget=300,
            ),
            session_id="next-episode",
        )

        self.assertLessEqual(len(results), 2)
        self.assertTrue(
            all(item.memory.status == MemoryStatus.VALIDATED for item in results)
        )

    def test_compact_trace_excludes_submission_and_redacts_entities(self) -> None:
        trace_path = Path(self.tmp.name) / "messages.jsonl"
        rows = [
            {
                "agent": "diagnosis_agent",
                "event": "tool_start",
                "tool": {"name": "ping_pair"},
                "input": '{"source": "pc1", "target": "r2", "ip": "10.0.0.1"}',
            },
            {
                "agent": "submission_agent",
                "event": "llm_end",
                "text": "ground_truth_like_answer",
            },
        ]
        trace_path.write_text(
            "\n".join(json.dumps(row) for row in rows),
            encoding="utf-8",
        )

        compact = self.module.compact_trace(trace_path)

        self.assertNotIn("ground_truth_like_answer", compact)
        self.assertNotIn("pc1", compact)
        self.assertNotIn("10.0.0.1", compact)
        self.assertIn("<device>", compact)

    def test_snapshot_is_jsonl_audit_artifact(self) -> None:
        self.module.store.add_or_corroborate(
            bank_id="test",
            candidate=learning_candidate(),
            status=MemoryStatus.VALIDATED,
            confidence=0.9,
            source_session_id="episode-1",
            successful_episode=True,
        )
        output = self.module.snapshot(
            session_id="episode-1",
            output_path=Path(self.tmp.name) / "snapshot.jsonl",
        )
        rows = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]

        self.assertEqual(rows[0]["kind"], "snapshot")
        self.assertTrue(any(row["kind"] == "memory" for row in rows[1:]))


class MemoryAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_wrapper_composes_with_existing_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = FakeWorkflow(tmp)
            memory = MagicMock()
            memory.bank_id = "experiment"
            memory.retrieve.return_value = []
            memory.format_context.return_value = (
                "Prior procedural memories:\n1. Verify both link endpoints."
            )
            adapter = MemoryAugmentedAgent(
                workflow,
                memory,
                memory_mode="read",
                memory_top_k=3,
                memory_token_budget=700,
            )

            result = await adapter.run("Diagnose the network.")

        self.assertEqual(result, {"ok": True})
        self.assertIn("Verify both link endpoints", workflow.received_task)
        query = memory.retrieve.call_args.kwargs["query"]
        self.assertEqual(query.top_k, 3)
        self.assertEqual(query.token_budget, 700)
        self.assertIn("frr_show_ip_route", query.tools)


class QdrantMemoryIndexTest(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "http://unused",
            "QDRANT_COLLECTION": "memory-test",
            "EMBEDDING_DIMENSION": "3",
        },
        clear=False,
    )
    def test_qdrant_stores_vectors_as_secondary_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            memory, _ = store.add_or_corroborate(
                bank_id="experiment",
                candidate=learning_candidate(),
                status=MemoryStatus.VALIDATED,
                confidence=0.9,
                source_session_id="episode-1",
                successful_episode=True,
            )
            index = QdrantMemoryIndex(provider=FixedEmbeddingProvider())
            index._client = QdrantClient(":memory:")

            index.upsert(memory)
            results = index.search(
                bank_id="experiment",
                query="BGP routing diagnosis",
                limit=5,
            )

            self.assertEqual(results[0][0], memory.memory_id)
            index.delete_bank("experiment")
            self.assertEqual(
                index.search(
                    bank_id="experiment",
                    query="BGP routing diagnosis",
                    limit=5,
                ),
                [],
            )


class OnlineEvolutionWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_extractor_never_receives_problem_names_or_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            (session_dir / "messages.jsonl").write_text("", encoding="utf-8")
            module = MagicMock()
            module.bank_id = "experiment"
            module.store.episode_is_evaluated.return_value = False
            module.compact_trace.return_value = "[]"
            module.extract = AsyncMock(return_value=[])
            module.validate.return_value = []
            module.consolidate = AsyncMock(return_value=[])
            module.snapshot.return_value = session_dir / "memory_snapshot.jsonl"

            with patch(
                "agent.memory.workflow.HybridMemoryModule",
                return_value=module,
            ):
                await evolve_session_memory(
                    run_meta={
                        "agent_type": "plan-execute",
                        "memory_mode": "evolve",
                        "memory_bank": "experiment",
                        "session_id": "episode-1",
                        "llm_backend": "openai",
                        "model": "model",
                        "task_description": "Public network description",
                        "scenario_name": "dc_clos_bgp",
                        "scenario_topo_size": "s",
                        "problem_names": ["hidden_fault_id"],
                        "ground_truth": {"root_cause_name": ["hidden_fault_id"]},
                    },
                    metrics={
                        "detection_score": 1,
                        "localization_f1": 1,
                        "rca_f1": 1,
                    },
                    session_dir=session_dir,
                )

        extract_kwargs = module.extract.await_args.kwargs
        self.assertEqual(
            set(extract_kwargs),
            {
                "task_description",
                "trace",
                "scenario",
                "topology_class",
            },
        )
        self.assertNotIn("hidden_fault_id", str(extract_kwargs))


class ContinualBenchmarkPolicyTest(unittest.TestCase):
    def test_online_evolution_rejects_parallel_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --parallel 1"):
            run_benchmark_from_csv(
                benchmark_file="does-not-matter.csv",
                agent_type="react",
                llm_backend="openai",
                model="model",
                max_steps=1,
                parallel=2,
                memory_mode="evolve",
            )
