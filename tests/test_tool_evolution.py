"""Tests for DRAFT-style tool documentation refinement."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.tools import StructuredTool
from unittest.mock import patch

from agent.tool_evolution.curator import (
    extract_tool_trials,
    identify_comprehension_gaps,
    rewrite_documentation,
)
from agent.tool_evolution.models import (
    DraftAnalyzerDraft,
    DraftExploration,
    DraftRewriteProposal,
    ToolDocumentation,
    ToolParameterDoc,
)
from agent.tool_evolution.runtime import ToolEvolutionRuntime
from agent.tool_evolution.store import ToolEvolutionStore
from nika.evaluator.result_log import build_eval_result_from_session_dir
from nika.workflows.eval.session import run_eval_metrics


class DraftToolEvolutionTest(unittest.TestCase):
    def test_primitive_contract_change_reopens_frozen_documentation(self) -> None:
        def ping(host: str) -> str:
            return host

        def ping_with_count(host: str, count: int = 1) -> str:
            return host * count

        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            first_tool = StructuredTool.from_function(
                ping,
                name="ping_host",
                description="Ping one host.",
            )
            ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[first_tool],
                library_id="draft",
                store=store,
            )
            before = store.get_document("ping_host")
            assert before is not None
            before.frozen = True
            before.frozen_reason = "converged"
            before.usage_notes.append("Preserve learned interpretation.")
            before.rewrite_history.append("Stale contract-specific rewrite.")
            store.upsert_document(before)
            state = store.load()
            state.explorations.append(
                DraftExploration(
                    exploration_id="stale-plan",
                    session_id="s1",
                    tool_name="ping_host",
                    intent="tool_validation",
                    user_query="Ping a current host.",
                    observation="host reachable",
                    status="success",
                    document_hash=before.content_hash(),
                )
            )
            store.save(state)

            second_tool = StructuredTool.from_function(
                ping_with_count,
                name="ping_host",
                description="Ping one host one or more times.",
            )
            ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[second_tool],
                library_id="draft",
                store=store,
            )
            after = store.get_document("ping_host")
            after_state = store.load()

        assert after is not None
        self.assertNotEqual(after.source_signature, before.source_signature)
        self.assertFalse(after.frozen)
        self.assertEqual(after.frozen_reason, "")
        self.assertGreater(after.version, before.version)
        self.assertEqual(after.description, "Ping one host one or more times.")
        self.assertNotIn("Preserve learned interpretation.", after.usage_notes)
        self.assertEqual(set(after.parameters), {"host", "count"})
        self.assertEqual(after.source_contract_version, 1)
        self.assertTrue(
            any("reset and reopened" in item for item in after.rewrite_history)
        )
        self.assertNotIn("Stale contract-specific rewrite.", after.rewrite_history)
        self.assertEqual(after_state.explorations, [])

    def test_legacy_exploration_is_kept_only_when_it_matches_one_real_trial(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "show_route", "{'router': 'r1'}", ""),
                        ("tool_end", "1", "show_route", "", "route present"),
                    ],
                ),
                session_id="s1",
            )
            state = store.load()
            state.trials.extend(trials)
            state.explorations.append(
                DraftExploration(
                    exploration_id="legacy-consumed",
                    session_id="s1",
                    tool_name="show_route",
                    parameters={"router": "r1"},
                    observation="route present",
                    status="consumed",
                )
            )
            store.save(state)

            loaded = store.load()

        self.assertEqual(len(loaded.explorations), 1)
        self.assertEqual(loaded.explorations[0].trial_id, trials[0].trial_id)
        self.assertEqual(loaded.explorations[0].status, "success")

    def test_draft_explorer_records_only_observed_trials_and_scores_diversity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "show_route", "{'router': 'r1'}", ""),
                        ("tool_end", "1", "show_route", "", "route missing"),
                        ("tool_start", "2", "show_route", "{'router': 'r1'}", ""),
                        ("tool_end", "2", "show_route", "", "route missing"),
                    ],
                ),
                session_id="s1",
                task_description="Investigate route failure.",
            )
            rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={"show_route": "Show routes."},
                metrics={"localization_f1": 0.0, "rca_f1": 0.0},
            )
            rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={"show_route": "Show routes."},
                metrics={"localization_f1": 0.0, "rca_f1": 0.0},
            )
            state = store.load()

        self.assertEqual(len(state.explorations), 2)
        self.assertEqual(
            {item.trial_id for item in state.explorations},
            {trial.trial_id for trial in trials},
        )
        self.assertTrue(all(item.status == "success" for item in state.explorations))
        self.assertTrue(all(item.observation == "route missing" for item in state.explorations))
        self.assertEqual(state.explorations[0].diversity_score, 1.0)
        self.assertEqual(state.explorations[1].diversity_score, 0.0)
        self.assertEqual(state.explorations[1].reflection_count, 1)

    def test_documentation_mastery_is_independent_of_rca_score(self) -> None:
        def evolve(root: str, rca_f1: float) -> float:
            store = ToolEvolutionStore("draft", root=root)
            trials, _ = extract_tool_trials(
                self._trace(
                    root,
                    [
                        ("tool_start", "1", "show_route", "{'router': 'r1'}", ""),
                        ("tool_end", "1", "show_route", "", "route present"),
                    ],
                ),
                session_id="s1",
            )
            rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={"show_route": "Show routes."},
                metrics={"rca_f1": rca_f1},
            )
            doc = store.get_document("show_route")
            assert doc is not None
            return doc.mastery_score

        with (
            tempfile.TemporaryDirectory() as left,
            tempfile.TemporaryDirectory() as right,
        ):
            low_rca = evolve(left, 0.0)
            high_rca = evolve(right, 1.0)

        self.assertEqual(low_rca, high_rca)

    def test_extracts_trials_and_argument_gaps_from_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
            rows = [
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_start",
                    "run_id": "1",
                    "tool": {
                        "name": "frr_show_ip_route",
                        "description": "Show routes on a router.",
                    },
                    "input": "{'router': 'router_x'}",
                },
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_error",
                    "run_id": "1",
                    "output": "router_x not found",
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            trials, docs = extract_tool_trials(trace, session_id="s1")
            gaps = identify_comprehension_gaps(trials)

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].tool_name, "frr_show_ip_route")
        self.assertEqual(docs["frr_show_ip_route"], "Show routes on a router.")
        self.assertEqual(gaps[0].gap_type, "environment_reference")

    def test_extracts_trials_without_run_ids_in_fifo_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
            rows = [
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_start",
                    "tool": {"name": "show_iface", "description": "Inspect interface."},
                    "input": "{'router': 'r1'}",
                },
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_end",
                    "output": "eth0 up",
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            trials, docs = extract_tool_trials(
                trace,
                session_id="s1",
                task_description="Check interface state",
            )

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].tool_name, "show_iface")
        self.assertEqual(trials[0].status, "success")
        self.assertEqual(trials[0].task_description, "Check interface state")
        self.assertEqual(docs["show_iface"], "Inspect interface.")

    def test_trial_extraction_strips_integrated_runtime_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
            rows = [
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_start",
                    "run_id": "1",
                    "tool": {"name": "show_iface", "description": "Inspect interface."},
                    "input": "{'router': 'r1'}",
                },
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_end",
                    "run_id": "1",
                    "output": (
                        "eth0 up\n\n"
                        "[Integrated learning guidance - not evidence]\n"
                        "Active Skill-MDP option: seed."
                    ),
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            trials, _ = extract_tool_trials(trace, session_id="s1")

        self.assertIn("eth0 up", trials[0].output_summary)
        self.assertNotIn("Integrated learning guidance", trials[0].output_summary)

    def test_extracts_react_diagnosis_phase_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
            rows = [
                {
                    "agent": "diagnosis",
                    "event": "tool_start",
                    "run_id": "1",
                    "tool": {"name": "ping_pair", "description": "Ping hosts."},
                    "input": "{'host_a': 'pc1', 'host_b': 'dns'}",
                },
                {
                    "agent": "diagnosis",
                    "event": "tool_end",
                    "run_id": "1",
                    "output": "2 packets transmitted, 2 received",
                },
                {
                    "agent": "submission",
                    "event": "tool_start",
                    "run_id": "2",
                    "tool": {"name": "submit"},
                    "input": "{}",
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            trials, docs = extract_tool_trials(trace, session_id="s1")

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].tool_name, "ping_pair")
        self.assertEqual(trials[0].status, "success")
        self.assertEqual(docs["ping_pair"], "Ping hosts.")

    def test_rewrites_documentation_with_preconditions_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        (
                            "tool_start",
                            "1",
                            "show_iface",
                            "{'router': 'r1', 'interface': 'eth9'}",
                            "",
                        ),
                        (
                            "tool_error",
                            "1",
                            "show_iface",
                            "",
                            "interface eth9 not found",
                        ),
                    ],
                ),
                session_id="s1",
            )
            revisions = rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={"show_iface": "Inspect one router interface."},
                metrics={"rca_accuracy": 0.0},
            )
            doc = store.get_document("show_iface")
            state = store.load()

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertTrue(revisions[0].changed)
        self.assertIn("router", doc.parameters)
        self.assertIn("interface", doc.parameters)
        self.assertTrue(doc.failure_modes)
        self.assertIn("Tool arguments must be grounded", doc.constraints[0])
        self.assertEqual(len(state.explorations), 1)
        self.assertEqual(state.explorations[0].status, "error")
        self.assertIn("interface eth9 not found", state.explorations[0].observation)
        self.assertEqual(len(state.analyzer_suggestions), 1)
        self.assertIn("show_iface", state.tool_stats)
        self.assertGreaterEqual(state.tool_stats["show_iface"].trials, 1)
        self.assertGreaterEqual(revisions[0].metrics["convergence_score"], 0.0)
        self.assertIn(
            state.analyzer_suggestions[0].suggestion_id,
            revisions[0].analyzer_suggestion_ids,
        )
        self.assertTrue(doc.tool_usage_description)
        self.assertEqual(revisions[0].metrics["documented_path_rate"], 0.0)
        self.assertEqual(revisions[0].metrics["success_path_rate"], 0.0)
        self.assertTrue(state.library_usage_description)

    def test_final_diagnosis_metrics_do_not_create_tool_contract_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        (
                            "tool_start",
                            "1",
                            "frr_show_bgp_summary",
                            "{'router': 'leaf1'}",
                            "",
                        ),
                        (
                            "tool_end",
                            "1",
                            "frr_show_bgp_summary",
                            "",
                            "Neighbor idle; no advertised prefixes",
                        ),
                    ],
                ),
                session_id="s1",
                task_description="BGP route missing",
            )
            rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={
                    "frr_show_bgp_summary": "Show BGP neighbor summary."
                },
                metrics={
                    "localization_accuracy": 0.0,
                    "localization_f1": 0.0,
                    "rca_accuracy": 0.0,
                    "rca_f1": 0.0,
                },
            )
            state = store.load()

        self.assertFalse(
            any(gap.gap_type == "diagnostic_semantic_gap" for gap in state.gaps)
        )
        self.assertEqual(len(state.explorations), 1)
        self.assertEqual(state.explorations[0].tool_name, "frr_show_bgp_summary")
        self.assertEqual(state.explorations[0].status, "success")
        self.assertNotIn("localization/RCA", state.explorations[0].observation)

    def test_poor_diagnosis_does_not_create_tool_contract_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "ping_pair", "{'host_a': 'pc1'}", ""),
                        ("tool_end", "1", "ping_pair", "", "packet loss"),
                        (
                            "tool_start",
                            "2",
                            "frr_show_ip_route",
                            "{'router': 'r1'}",
                            "",
                        ),
                        (
                            "tool_end",
                            "2",
                            "frr_show_ip_route",
                            "",
                            "missing route to pc1",
                        ),
                    ],
                ),
                session_id="s1",
            )

        gaps = identify_comprehension_gaps(trials)
        self.assertFalse(
            any(gap.gap_type == "diagnostic_semantic_gap" for gap in gaps)
        )

    def test_draft_does_not_learn_from_documented_tool_without_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping two topology hosts."},
                metrics={},
            )
            state = store.load()
            doc = state.documents["ping_pair"]

        self.assertEqual(state.explorations, [])
        self.assertEqual(state.analyzer_suggestions, [])
        self.assertEqual(state.trials, [])
        self.assertEqual(doc.version, 1)

    def _legacy_eval_metrics_and_summary_include_draft_core_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "session"
            session_dir.mkdir()
            (session_dir / "run.json").write_text(
                json.dumps(
                    {
                        "session_id": "s-draft",
                        "status": "finished",
                        "agent_type": "react",
                        "model": "test-model",
                        "scenario_name": "simple_bgp",
                        "scenario_topo_size": "small",
                        "problem_names": ["link_down"],
                        "root_cause_name": "link_down",
                        "tool_evolution_enabled": True,
                        "tool_library_id": "draft",
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "ground_truth.json").write_text(
                json.dumps(
                    {
                        "is_anomaly": True,
                        "faulty_devices": ["r1"],
                        "root_cause_name": ["link_down"],
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "submission.json").write_text(
                json.dumps(
                    {
                        "is_anomaly": True,
                        "faulty_devices": ["r1"],
                        "root_cause_name": ["link_down"],
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "messages.jsonl").write_text("", encoding="utf-8")
            updates: list[tuple[str, object]] = []

            class FakeSession:
                def __init__(self) -> None:
                    self.session_dir = str(session_dir)
                    self.session_id = "s-draft"
                    self.tool_evolution_enabled = True
                    self.memory_mode = "off"
                    self.store = None

                def load_closed_session(self, *, session_id=None) -> None:
                    self.session_id = session_id or self.session_id

                def update_run_meta(self, key: str, value: object) -> None:
                    updates.append((key, value))
                    setattr(self, key, value)

            draft_report = {
                "library_id": "draft",
                "draft_trials": 1,
                "draft_trials_added": 1,
                "draft_document_revisions": 1,
                "draft_comprehension_gaps": 1,
                "draft_frozen_documents": 0,
                "draft_documented_tools": 1,
                "draft_unique_trial_tools": 1,
                "draft_explorations": 2,
                "draft_analyzer_suggestions": 1,
                "draft_mastered_tools": 0,
                "draft_documented_path_rate": 1.0,
                "draft_success_path_rate": 1.0,
                "draft_converged_documents": 0,
                "draft_llm_attempts": 0,
                "draft_llm_failures": 0,
                "draft_llm_revisions": 0,
                "draft_llm_analyzer_revisions": 1,
                "draft_llm_analyzer_failures": 0,
                "draft_llm_errors": [],
            }
            with (
                patch("nika.workflows.eval.session.Session", FakeSession),
                patch(
                    "agent.tool_evolution.curator.finalize_tool_evolution_session",
                    return_value=draft_report,
                ),
            ):
                run_eval_metrics(session_id="s-draft")

            metrics = json.loads(
                (session_dir / "eval_metrics.json").read_text(encoding="utf-8")
            )
            result = build_eval_result_from_session_dir(session_dir)

        self.assertEqual(metrics["draft_explorations"], 2)
        self.assertEqual(result.draft_explorations, 2)
        self.assertEqual(metrics["draft_analyzer_suggestions"], 1)
        self.assertEqual(result.draft_analyzer_suggestions, 1)
        self.assertEqual(metrics["draft_llm_analyzer_revisions"], 1)
        self.assertEqual(result.draft_llm_analyzer_revisions, 1)
        self.assertEqual(result.draft_llm_analyzer_failures, 0)
        self.assertIn(("eval_metrics", metrics), updates)

    def test_runtime_appends_refined_docs_without_adding_tools(self) -> None:
        def ping(host: str) -> str:
            return host

        tool = StructuredTool.from_function(
            ping,
            name="ping_pair",
            description="Ping a host.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping a host."},
                metrics={},
            )
            doc = store.get_document("ping_pair")
            assert doc is not None
            doc.usage_notes.append("Use exact host names from the active topology.")
            store.upsert_document(doc)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            tools = runtime.build_tools()
            snapshot = runtime.snapshot()
            seeded = store.get_document("ping_pair")

        self.assertEqual([item.name for item in tools], ["ping_pair"])
        self.assertIn("DRAFT refined guidance", tools[0].description)
        self.assertNotIn("Ping pc1", tools[0].description)
        self.assertEqual(snapshot["available_documents"], ["ping_pair"])
        self.assertFalse(hasattr(runtime, "prompt_suffix"))
        self.assertIsNotNone(seeded)

    def test_runtime_can_restore_base_descriptions_for_scoped_memory_runtime(
        self,
    ) -> None:
        def ping(host: str) -> str:
            return host

        tool = StructuredTool.from_function(
            ping,
            name="ping_pair",
            description="Ping a host.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            state = store.load()
            state.documents["ping_pair"] = ToolDocumentation(
                name="ping_pair",
                description="Ping a host.",
                usage_notes=["Use exact host names from the active topology."],
            )
            store.save(state)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )

            enriched = runtime.build_tools()
            enriched_description = enriched[0].description
            restored = runtime.build_tools(append_docs=False)
            restored_description = restored[0].description

        self.assertEqual(enriched_description, "Ping a host.")
        self.assertEqual(restored_description, "Ping a host.")

    def test_runtime_seeds_primitive_tool_documents(self) -> None:
        def ping(host: str) -> str:
            return host

        def route(router: str) -> str:
            return router

        tools = [
            StructuredTool.from_function(
                ping,
                name="ping_pair",
                description="Ping topology hosts.",
            ),
            StructuredTool.from_function(
                route,
                name="show_route",
                description="Inspect routing table.",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=tools,
                library_id="draft",
                store=store,
            )
            state = store.load()
            snapshot = runtime.snapshot()

        self.assertEqual(sorted(state.documents), ["ping_pair", "show_route"])
        self.assertEqual(snapshot["available_documents"], ["ping_pair", "show_route"])
        self.assertIn("Ping topology hosts", state.documents["ping_pair"].description)
        self.assertEqual(set(state.documents["ping_pair"].parameters), {"host"})
        self.assertEqual(state.documents["ping_pair"].source_contract_version, 1)

    def test_llm_rewrite_proposal_is_merged_into_documentation(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
                return DraftRewriteProposal(
                    tool_name="show_iface",
                    description="Inspect one router interface with verified names.",
                    tool_usage_description=(
                        "show_iface is a tool that can inspect one verified router interface."
                    ),
                    preconditions=["Discover router and interface names first."],
                    parameters={
                        "router": ToolParameterDoc(
                            name="router",
                            type_hint="str",
                            description="Exact router name from topology.",
                        )
                    },
                    constraints=["Never guess interface names."],
                    failure_modes=["Unknown interface returns not found."],
                    usage_notes=["Call topology discovery before interface checks."],
                    rationale="Observed interface-name confusion.",
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "show_iface", "{'router': 'r1'}", ""),
                        ("tool_error", "1", "show_iface", "", "interface missing"),
                    ],
                ),
                session_id="s1",
            )
            with patch(
                "agent.tool_evolution.curator.load_model", return_value=FakeModel()
            ):
                revisions = rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={"show_iface": "Inspect interface."},
                    metrics={"rca_accuracy": 0.0},
                    llm_backend="openai",
                    model="test-model",
                )
            doc = store.get_document("show_iface")

        assert doc is not None
        self.assertEqual(doc.description, "Inspect interface.")
        self.assertEqual(
            doc.tool_usage_description,
            "show_iface is a tool that can inspect one verified router interface.",
        )
        self.assertIn("Never guess interface names.", doc.constraints)
        self.assertTrue(any("DRAFT Analyzer" in prompt for prompt in prompts))
        rewrite_prompt = next(
            prompt for prompt in prompts if "Explorer observations" in prompt
        )
        self.assertIn("Analyzer suggestions", rewrite_prompt)
        self.assertIn("tool_usage_description", rewrite_prompt)
        self.assertEqual(revisions[0].metrics["llm_rewrite"], 1.0)

    def test_success_only_rewrite_drops_unsupported_negative_knowledge_and_converges(
        self,
    ) -> None:
        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, _prompt):
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(
                        suggestion="Clarify the observed successful return contract."
                    )
                return DraftRewriteProposal(
                    tool_name="inspect_state",
                    tool_usage_description=(
                        "Call inspect_state with no arguments and inspect its JSON result."
                    ),
                    preconditions=["The controller must have elevated privileges."],
                    constraints=["A hidden mode must be enabled."],
                    failure_modes=["An unobserved timeout returns an error field."],
                    usage_notes=["Check an unobserved truncated flag."],
                    rationale="This rationale is audit metadata, not runtime guidance.",
                )

        def inspect_state() -> str:
            return "{}"

        tool = StructuredTool.from_function(
            inspect_state,
            name="inspect_state",
            description="Return the current state as JSON.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            with patch(
                "agent.tool_evolution.curator.load_model", return_value=FakeModel()
            ):
                for session_id in ("s1", "s2"):
                    trials, _ = extract_tool_trials(
                        self._trace(
                            tmp,
                            [
                                ("tool_start", "1", "inspect_state", "{}", ""),
                                ("tool_end", "1", "inspect_state", "", '{"ok": true}'),
                            ],
                        ),
                        session_id=session_id,
                    )
                    rewrite_documentation(
                        store,
                        trials=trials,
                        tool_descriptions={
                            "inspect_state": "Return the current state as JSON."
                        },
                        metrics={},
                        llm_backend="custom",
                        model="test-model",
                    )
            doc = store.get_document("inspect_state")

        assert doc is not None
        self.assertEqual(doc.preconditions, [])
        self.assertEqual(
            doc.constraints,
            ["Tool arguments must be grounded in currently observed topology evidence."],
        )
        self.assertEqual(doc.failure_modes, [])
        self.assertEqual(doc.usage_notes, [])
        self.assertNotIn("audit metadata", doc.refined_description(max_chars=4000))
        self.assertTrue(doc.frozen)
        self.assertIn("adaptive termination", doc.frozen_reason)

    def test_llm_rewrite_cannot_expand_primitive_parameter_schema(self) -> None:
        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, _prompt):
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(
                        suggestion="Add a hosts filter not present in the source API.",
                    )
                return DraftRewriteProposal(
                    tool_name="get_reachability",
                    description="Reachability with an optional hosts filter.",
                    tool_usage_description="Call with hosts to restrict the matrix.",
                    parameters={
                        "hosts": ToolParameterDoc(
                            name="hosts",
                            type_hint="array",
                            description="Optional host filter.",
                        )
                    },
                )

        def reachability() -> str:
            return "{}"

        tool = StructuredTool.from_function(
            reachability,
            name="get_reachability",
            description="Collect the current reachability matrix.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "get_reachability", "{}", ""),
                        ("tool_end", "1", "get_reachability", "", "{}"),
                    ],
                ),
                session_id="s1",
                task_description="Investigate reachability loss.",
            )
            with patch(
                "agent.tool_evolution.curator.load_model",
                return_value=FakeModel(),
            ):
                revisions = rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={
                        "get_reachability": "Collect the current reachability matrix."
                    },
                    metrics={"localization_f1": 0.0, "rca_f1": 0.0},
                    llm_backend="custom",
                    model="test-model",
                )
            doc = store.get_document("get_reachability")
            state = store.load()

        assert doc is not None
        self.assertEqual(doc.description, "Collect the current reachability matrix.")
        self.assertEqual(doc.parameters, {})
        self.assertIn(
            "get_reachability is a primitive diagnostic tool",
            doc.tool_usage_description,
        )
        self.assertNotIn("hosts filter", doc.tool_usage_description)
        self.assertEqual(revisions[0].metrics["llm_rewrite"], 0.0)
        self.assertEqual(revisions[0].metrics["llm_contract_rejected"], 1.0)
        self.assertEqual(revisions[0].source_signature, doc.source_signature)
        self.assertIn("ContractValidationError", revisions[0].llm_error)
        self.assertEqual(len(state.explorations), 1)
        self.assertEqual(state.explorations[0].status, "success")
        self.assertEqual(state.explorations[0].parameters, {})

    def test_llm_rewrite_failure_is_reported(self) -> None:
        class FailingModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                raise TimeoutError("draft timeout")

        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "show_iface", "{'router': 'r1'}", ""),
                        ("tool_end", "1", "show_iface", "", "eth0 up"),
                    ],
                ),
                session_id="s1",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "NIKA_LEARNING_LLM_BACKEND": "custom",
                        "NIKA_LEARNING_LLM_MODEL": "learning-model",
                    },
                ),
                patch(
                    "agent.tool_evolution.curator.load_model",
                    return_value=FailingModel(),
                ) as load_model,
            ):
                revisions = rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={"show_iface": "Inspect interface."},
                    metrics={"rca_accuracy": 1.0},
                    llm_backend="custom",
                    model="test-model",
                )

        load_model.assert_called_once()
        args, kwargs = load_model.call_args
        self.assertEqual(args[:2], ("custom", "learning-model"))
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(revisions[0].metrics["llm_attempted"], 1.0)
        self.assertEqual(revisions[0].metrics["llm_failed"], 1.0)
        self.assertEqual(revisions[0].metrics["llm_rewrite"], 0.0)
        self.assertIn("TimeoutError", revisions[0].llm_error)

    def test_draft_rewrite_prompt_compacts_large_outputs(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
                return DraftRewriteProposal(
                    tool_name="get_reachability",
                    tool_usage_description="Summarize reachability matrix safely.",
                )

        huge_output = "reachable " + ("x" * 8000)
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "get_reachability", "{}", ""),
                        ("tool_end", "1", "get_reachability", "", huge_output),
                    ],
                ),
                session_id="s1",
            )
            with patch(
                "agent.tool_evolution.curator.load_model", return_value=FakeModel()
            ):
                rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={"get_reachability": "Reachability matrix."},
                    metrics={"rca_accuracy": 0.0},
                    llm_backend="custom",
                    model="test-model",
                )

        self.assertEqual(len(prompts), 2)
        self.assertTrue(all(len(prompt) < 8000 for prompt in prompts))
        self.assertNotIn("x" * 1000, "\n".join(prompts))

    def test_path_rate_counts_tools_documented_before_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"show_iface": "Inspect interface."},
                metrics={},
            )
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "show_iface", "{'router': 'r1'}", ""),
                        ("tool_end", "1", "show_iface", "", "eth0 up"),
                        ("tool_start", "2", "new_tool", "{}", ""),
                        ("tool_error", "2", "new_tool", "", "unknown failure"),
                    ],
                ),
                session_id="s2",
            )
            revisions = rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={
                    "show_iface": "Inspect interface.",
                    "new_tool": "New.",
                },
                metrics={"rca_accuracy": 1.0},
                documented_tools_at_start={"show_iface"},
            )

        by_tool = {revision.tool_name: revision for revision in revisions}
        self.assertEqual(by_tool["show_iface"].metrics["documented_path_rate"], 0.5)
        self.assertEqual(by_tool["show_iface"].metrics["success_path_rate"], 0.5)

    @staticmethod
    def _trace(tmp: str, rows: list[tuple[str, str, str, str, str]]) -> Path:
        path = Path(tmp) / "messages.jsonl"
        payloads = []
        for event, run_id, tool, input_value, output in rows:
            item = {
                "agent": "diagnosis_agent",
                "event": event,
                "run_id": run_id,
            }
            if event == "tool_start":
                item["tool"] = {"name": tool, "description": tool}
                item["input"] = input_value
            else:
                item["output"] = output
            payloads.append(item)
        path.write_text(
            "\n".join(json.dumps(row) for row in payloads), encoding="utf-8"
        )
        return path
