"""Unit tests for the persistent atomic procedural-memory module."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

from qdrant_client import QdrantClient

from agent.composition import MemoryConfig
from agent.memory.adapter import MemoryAugmentedAgent
from agent.memory.attributes import infer_memory_attributes
from agent.memory.models import (
    EvaluationEvidence,
    MemoryAttributes,
    MemoryCandidate,
    MemoryExtraction,
    MemoryLinkType,
    MemoryQuery,
    MemoryStatus,
)
from agent.memory.safety import MemoryOracleLeakageError, assert_no_oracle_leakage
from agent.memory.service import ProceduralMemoryModule
from agent.memory.store import SQLiteMemoryStore, create_memory_store
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


class RecordingVectorIndex(DisabledVectorIndex):
    enabled = True

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, **kwargs) -> list:
        self.queries.append(kwargs["query"])
        return []


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


class FakeExtractor:
    def __init__(self, extraction: MemoryExtraction) -> None:
        self.extraction = extraction

    async def ainvoke(self, messages) -> MemoryExtraction:
        return self.extraction


class FakeLLM:
    def __init__(self, extraction: MemoryExtraction) -> None:
        self.extraction = extraction

    def with_structured_output(self, schema):
        return FakeExtractor(self.extraction)


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


def atomic_note(content: str | None = None) -> MemoryCandidate:
    return MemoryCandidate(
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
    def test_store_factory_defaults_to_postgres_unless_sqlite_is_forced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sqlite_path = Path(tmp) / "memory.sqlite3"
            with patch("agent.memory.store.PostgreSQLMemoryStore") as postgres_store:
                selected_default = create_memory_store(
                    sqlite_path=sqlite_path,
                )
                selected_custom = create_memory_store(
                    sqlite_path=sqlite_path,
                    database_url="postgresql://example/db",
                )
                forced = create_memory_store(
                    sqlite_path=sqlite_path,
                    database_url="postgresql://example/db",
                    force_sqlite=True,
                )

        self.assertEqual(
            postgres_store.call_args_list,
            [
                call("postgresql://nika:nika@localhost:5432/nika_memory"),
                call("postgresql://example/db"),
            ],
        )
        self.assertEqual(selected_default, postgres_store.return_value)
        self.assertEqual(selected_custom, postgres_store.return_value)
        self.assertIsInstance(forced, SQLiteMemoryStore)

    def test_atomic_memory_is_canonical_and_fts_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            memory, created = store.add_or_corroborate(
                bank_id="experiment",
                candidate=atomic_note(),
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

    def test_repeated_validation_corroborates_atomic_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            memory = None
            for episode in range(3):
                memory, _ = store.add_or_corroborate(
                    bank_id="experiment",
                    candidate=atomic_note(),
                    status=MemoryStatus.VALIDATED,
                    confidence=0.8,
                    source_session_id=f"episode-{episode}",
                    successful_episode=True,
                )

        self.assertIsNotNone(memory)
        self.assertEqual(memory.validation_count, 3)
        self.assertGreaterEqual(memory.confidence, 0.9)
        self.assertNotIn("memory_type", memory.model_dump())

    def test_success_can_validate_matching_staged_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            staged, _ = store.add_or_corroborate(
                bank_id="experiment",
                candidate=atomic_note(),
                status=MemoryStatus.STAGED,
                confidence=0.4,
                source_session_id="failed-episode",
                successful_episode=False,
            )
            promoted, created = store.add_or_corroborate(
                bank_id="experiment",
                candidate=atomic_note(),
                status=MemoryStatus.VALIDATED,
                confidence=0.9,
                source_session_id="successful-episode",
                successful_episode=True,
            )

        self.assertFalse(created)
        self.assertEqual(promoted.memory_id, staged.memory_id)
        self.assertEqual(promoted.status, MemoryStatus.VALIDATED)

    def test_relation_counts_preserve_link_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            left, _ = store.add_or_corroborate(
                bank_id="experiment",
                candidate=atomic_note("Check BGP neighbor state before RCA."),
                status=MemoryStatus.VALIDATED,
                confidence=0.8,
                source_session_id="episode-1",
                successful_episode=True,
            )
            right, _ = store.add_or_corroborate(
                bank_id="experiment",
                candidate=atomic_note("Compare BGP advertised routes before RCA."),
                status=MemoryStatus.VALIDATED,
                confidence=0.8,
                source_session_id="episode-2",
                successful_episode=True,
            )

            store.add_link(
                bank_id="experiment",
                source_id=left.memory_id,
                target_id=right.memory_id,
                relation=MemoryLinkType.SUPPORTS,
            )
            counts = store.relation_counts(
                "experiment",
                [left.memory_id, right.memory_id],
            )

        self.assertEqual(counts[left.memory_id][MemoryLinkType.SUPPORTS.value], 1)
        self.assertEqual(counts[right.memory_id][MemoryLinkType.SUPPORTS.value], 1)


class MemoryPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.module = ProceduralMemoryModule(
            bank_id="test",
            store_path=Path(self.tmp.name) / "memory.sqlite3",
            vector_index=DisabledVectorIndex(),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_success_validates_atomic_note(self) -> None:
        gated = self.module.validate(
            [atomic_note()],
            EvaluationEvidence(
                detection_score=1,
                localization_f1=1,
                rca_f1=1,
            ),
        )

        candidate, status, confidence = gated[0]
        self.assertNotIn("memory_type", candidate.model_dump())
        self.assertEqual(status, MemoryStatus.VALIDATED)
        self.assertGreater(confidence, 0.8)

    def test_atomic_note_schema_has_no_instruction_type(self) -> None:
        gated = self.module.validate(
            [atomic_note()],
            EvaluationEvidence(
                detection_score=1,
                localization_f1=1,
                rca_f1=1,
            ),
        )

        self.assertNotIn("memory_type", gated[0][0].model_dump())

    def test_failure_only_stages_cautious_checkable_notes(self) -> None:
        cautious_note = atomic_note()
        unsafe_claim = MemoryCandidate(
            content="Always classify this symptom as one fixed benchmark answer.",
            applicability=["A vague symptom resembles a previous episode."],
            attributes=MemoryAttributes(protocols=["bgp"]),
        )
        gated = self.module.validate(
            [cautious_note, unsafe_claim],
            EvaluationEvidence(
                detection_score=1,
                localization_f1=0,
                rca_f1=0,
            ),
        )

        self.assertEqual(len(gated), 1)
        self.assertEqual(gated[0][0].content, cautious_note.content)
        self.assertEqual(gated[0][1], MemoryStatus.STAGED)

    def test_retrieval_respects_top_k_and_token_budget(self) -> None:
        for index in range(4):
            self.module.store.add_or_corroborate(
                bank_id="test",
                candidate=atomic_note(
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

    def test_vector_retrieval_uses_compact_semantic_query(self) -> None:
        index = RecordingVectorIndex()
        module = ProceduralMemoryModule(
            bank_id="test",
            store_path=Path(self.tmp.name) / "recording.sqlite3",
            vector_index=index,
        )
        module.retrieve(
            query=MemoryQuery(
                text=" ".join(["BGP 10.0.0.1 pc_0_0 eth0"] * 400),
                scenario="dc_clos_bgp",
                protocols=["bgp"],
                services=["frr"],
                top_k=2,
            ),
            session_id="next-episode",
        )

        self.assertEqual(len(index.queries), 1)
        self.assertLessEqual(len(index.queries[0]), 600)
        self.assertIn("bgp", index.queries[0].lower())
        self.assertIn("frr", index.queries[0].lower())
        self.assertNotIn("10.0.0.1", index.queries[0])
        self.assertNotIn("pc_0_0", index.queries[0])

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
        payload = json.loads(compact)
        self.assertEqual(payload["events"][0]["topic"], "connectivity_probe")
        self.assertTrue(payload["topics"])

    def test_extract_scopes_attributes_to_atomic_note_text(self) -> None:
        broad_note = MemoryCandidate(
            content="Check DNS resolver configuration with dig before changing routing.",
            applicability=["Name resolution fails while ICMP still works."],
            evidence_required=["Inspect resolv.conf and run a DNS lookup."],
            attributes=MemoryAttributes(
                protocols=["bgp", "rip"],
                services=["bgp", "http"],
                tools=["dig_lookup"],
            ),
        )
        self.module._llm = FakeLLM(MemoryExtraction(memories=[broad_note]))

        extracted = asyncio.run(
            self.module.extract(
                task_description="The task mentions BGP, HTTP, and RIP elsewhere.",
                trace="A long trace mentions BGP neighbors, HTTP checks, and RIP routes.",
                scenario="ospf_enterprise_dhcp",
                topology_class="s",
            )
        )

        attrs = extracted[0].attributes
        self.assertIn("dns", attrs.protocols)
        self.assertIn("dns", attrs.services)
        self.assertIn("icmp", attrs.protocols)
        self.assertIn("dig_lookup", attrs.tools)
        self.assertNotIn("bgp", attrs.protocols)
        self.assertNotIn("rip", attrs.protocols)
        self.assertNotIn("http", attrs.services)

    def test_compact_trace_groups_routing_topic_evidence(self) -> None:
        trace_path = Path(self.tmp.name) / "messages.jsonl"
        rows = [
            {
                "agent": "diagnosis_agent",
                "event": "tool_start",
                "tool": {"name": "frr_show_ip_route"},
                "input": '{"node": "r1"}',
            },
            {
                "agent": "diagnosis_agent",
                "event": "tool_end",
                "output": "BGP neighbor is idle and route is missing.",
            },
        ]
        trace_path.write_text(
            "\n".join(json.dumps(row) for row in rows),
            encoding="utf-8",
        )

        payload = json.loads(self.module.compact_trace(trace_path))
        topics = {item["topic"]: item for item in payload["topics"]}

        self.assertIn("routing_inspection", topics)
        self.assertIn("frr_show_ip_route", topics["routing_inspection"]["tools"])

    def test_oracle_guard_rejects_hidden_payload_keys(self) -> None:
        with self.assertRaises(MemoryOracleLeakageError):
            assert_no_oracle_leakage({"problem_names": ["hidden_fault_id"]})

    def test_attribute_miner_extracts_services_and_symptoms(self) -> None:
        attrs = infer_memory_attributes(
            "DNS resolver timeout after HTTP check failed.",
            scenario="ospf_enterprise_dns",
            topology_class="s",
            tools=["curl_url", "dig_lookup"],
        )

        self.assertIn("dns", attrs.services)
        self.assertIn("timeout", attrs.symptoms)
        self.assertIn("ospf", attrs.protocols)

    def test_snapshot_is_jsonl_audit_artifact(self) -> None:
        self.module.store.add_or_corroborate(
            bank_id="test",
            candidate=atomic_note(),
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

    async def test_wrapper_mines_service_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = FakeWorkflow(tmp)
            memory = MagicMock()
            memory.bank_id = "experiment"
            memory.retrieve.return_value = []
            memory.format_context.return_value = ""
            adapter = MemoryAugmentedAgent(
                workflow,
                memory,
                memory_mode="read",
            )

            await adapter.run("Diagnose DNS resolver timeout.")

        query = memory.retrieve.call_args.kwargs["query"]
        self.assertIn("dns", query.services)
        self.assertIn("timeout", query.symptoms)


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
    def test_qdrant_readiness_reports_server_and_index_config(self) -> None:
        index = QdrantMemoryIndex(provider=FixedEmbeddingProvider())
        index._client = QdrantClient(":memory:")

        report = index.readiness()

        self.assertTrue(report["server_reachable"])
        self.assertTrue(report["ready"])
        self.assertFalse(report["collection_exists"])

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
                candidate=atomic_note(),
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

    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "http://unused",
            "QDRANT_COLLECTION": "memory-test",
            "EMBEDDING_DIMENSION": "3",
        },
        clear=False,
    )
    def test_qdrant_search_filters_by_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp) / "memory.sqlite3")
            bgp_memory, _ = store.add_or_corroborate(
                bank_id="experiment",
                candidate=atomic_note("Check BGP routing evidence first."),
                status=MemoryStatus.VALIDATED,
                confidence=0.9,
                source_session_id="episode-1",
                successful_episode=True,
            )
            ospf_memory, _ = store.add_or_corroborate(
                bank_id="experiment",
                candidate=MemoryCandidate(
                    content="Check OSPF adjacency evidence before RCA.",
                    applicability=["OSPF reachability is unstable."],
                    evidence_required=["Inspect OSPF neighbors."],
                    attributes=MemoryAttributes(protocols=["ospf"]),
                ),
                status=MemoryStatus.VALIDATED,
                confidence=0.9,
                source_session_id="episode-2",
                successful_episode=True,
            )
            index = QdrantMemoryIndex(provider=FixedEmbeddingProvider())
            index._client = QdrantClient(":memory:")
            index.upsert(bgp_memory)
            index.upsert(ospf_memory)

            results = index.search(
                bank_id="experiment",
                query="routing diagnosis",
                limit=5,
                protocols=["ospf"],
            )

        self.assertEqual(results, [(ospf_memory.memory_id, results[0][1])])


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
                "agent.memory.workflow.ProceduralMemoryModule",
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
                memory=MemoryConfig(mode="evolve"),
            )
