"""Tests for DRAFT-style tool documentation refinement."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.tools import StructuredTool
from unittest.mock import patch

from agent.tool_evolution.curator import (
    extract_tool_trials,
    identify_comprehension_gaps,
    identify_diagnostic_semantic_gaps,
    rewrite_documentation,
)
from agent.tool_evolution.models import (
    DraftAnalyzerSuggestion,
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
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

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
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

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
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            trials, _ = extract_tool_trials(trace, session_id="s1")

        self.assertIn("eth0 up", trials[0].output_summary)
        self.assertNotIn("Integrated learning guidance", trials[0].output_summary)

    def test_runtime_draft_hint_consumes_planned_exploration_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "messages.jsonl"
            rows = [
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_start",
                    "run_id": "1",
                    "tool": {"name": "ping_host", "description": "Ping one host."},
                    "input": "{'host': 'pc1'}",
                },
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_end",
                    "run_id": "1",
                    "output": "pc1 reachable",
                },
                {
                    "agent": "memory_agent",
                    "phase": "skill_mdp_runtime",
                    "event": "skill_transition",
                    "tool": "ping_host",
                    "tool_input": {"host": "pc1"},
                    "status": "success",
                    "observation_summary": "pc1 reachable",
                    "draft_exploration_id": "explore_ping_pc1",
                    "draft_next_exploration": (
                        "Ping pc1 to verify endpoint reachability."
                    ),
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            store = ToolEvolutionStore("draft", root=tmp)
            state = store.load()
            state.documents["ping_host"] = ToolDocumentation(
                name="ping_host",
                description="Ping one host.",
            )
            state.explorations.append(
                DraftExploration(
                    exploration_id="explore_ping_pc1",
                    session_id="s0",
                    tool_name="ping_host",
                    intent="diagnosis_check",
                    user_query="Check pc1 reachability.",
                    parameters={"host": "pc1"},
                    status="planned",
                    next_exploration="Ping pc1 to verify endpoint reachability.",
                )
            )
            store.save(state)

            trials, _ = extract_tool_trials(trace, session_id="s1")
            revisions = rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={"ping_host": "Ping one host."},
                metrics={"rca_accuracy": 1.0},
            )
            state = store.load()

        self.assertEqual(trials[0].planned_exploration_id, "explore_ping_pc1")
        self.assertEqual(state.explorations[0].status, "consumed")
        self.assertEqual(
            state.explorations[0].consumed_by_trial_id,
            trials[0].trial_id,
        )
        self.assertTrue(
            any(
                revision.metrics.get("planned_explorations_consumed") == 1.0
                for revision in revisions
            )
        )

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
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

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
                        ("tool_error", "1", "show_iface", "", "interface eth9 not found"),
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
        self.assertGreaterEqual(len(state.explorations), 2)
        self.assertTrue(
            any(exploration.status == "planned" for exploration in state.explorations)
        )
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

    def test_successful_trials_with_poor_diagnosis_create_semantic_gap(self) -> None:
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
            doc = state.documents["frr_show_bgp_summary"]

        self.assertTrue(
            any(gap.gap_type == "diagnostic_semantic_gap" for gap in state.gaps)
        )
        self.assertTrue(
            any("interpret the output" in note for note in doc.usage_notes)
        )
        self.assertTrue(
            any(
                exploration.status == "planned"
                and exploration.tool_name == "frr_show_bgp_summary"
                for exploration in state.explorations
            )
        )
        self.assertTrue(
            any(
                "localization/RCA alternatives" in item
                for item in doc.exploration_suggestions
            )
        )

    def test_poor_diagnosis_creates_semantic_gap_per_successful_tool(self) -> None:
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

        gaps = identify_diagnostic_semantic_gaps(
            trials,
            metrics={
                "localization_accuracy": 0.0,
                "localization_f1": 0.0,
                "rca_accuracy": 0.0,
                "rca_f1": 0.0,
            },
        )

        self.assertEqual(
            {gap.tool_name for gap in gaps},
            {"ping_pair", "frr_show_ip_route"},
        )
        self.assertTrue(
            all(gap.gap_type == "diagnostic_semantic_gap" for gap in gaps)
        )

    def test_draft_plans_exploration_for_documented_tool_without_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping two topology hosts."},
                metrics={},
                session_id="s1",
                task_description="Investigate reachability loss",
            )
            state = store.load()
            doc = state.documents["ping_pair"]

        planned = [
            exploration
            for exploration in state.explorations
            if exploration.tool_name == "ping_pair"
            and exploration.status == "planned"
        ]
        self.assertEqual(len(planned), 1)
        self.assertIn("Investigate reachability loss", planned[0].user_query)
        self.assertTrue(doc.exploration_suggestions)
        self.assertEqual(planned[0].intent, "diagnosis_check")
        self.assertIn("localization/RCA", planned[0].next_exploration)

    def test_diagnosis_planned_exploration_reaches_runtime_prompt(self) -> None:
        def ping(host: str) -> str:
            return host

        tool = StructuredTool.from_function(
            ping,
            name="ping_pair",
            description="Ping one topology host.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping one topology host."},
                metrics={},
                session_id="s1",
                task_description="Investigate reachability loss",
            )
            runtime = ToolEvolutionRuntime(
                session=SimpleNamespace(
                    task_description="Investigate reachability loss",
                    topology=[("pc1:eth0", "r1:eth0")],
                ),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )

            prompt = runtime.prompt_suffix(tool_names=["ping_pair"])
            queue = runtime.planned_explorations(diagnosis_only=True)

        self.assertEqual([item["intent"] for item in queue], ["diagnosis_check"])
        self.assertIn("DRAFT active exploration queue", prompt)
        self.assertIn("localization/RCA", prompt)

    def test_planned_exploration_is_consumed_by_matching_trial(self) -> None:
        def ping(host_a: str, host_b: str) -> str:
            return f"{host_a}->{host_b}"

        tool = StructuredTool.from_function(
            ping,
            name="ping_pair",
            description="Ping two topology hosts.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping two topology hosts."},
                metrics={},
                session_id="s1",
                task_description="Investigate reachability loss",
            )
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        (
                            "tool_start",
                            "1",
                            "ping_pair",
                            "{'host_a': 'pc1', 'host_b': 'dns'}",
                            "",
                        ),
                        (
                            "tool_end",
                            "1",
                            "ping_pair",
                            "",
                            "2 packets transmitted, 2 received",
                        ),
                    ],
                ),
                session_id="s2",
                task_description="Investigate reachability loss",
            )
            revisions = rewrite_documentation(
                store,
                trials=trials,
                tool_descriptions={"ping_pair": "Ping two topology hosts."},
                metrics={"rca_accuracy": 1.0},
            )
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            state = store.load()
            snapshot = runtime.snapshot()

        consumed = [
            exploration
            for exploration in state.explorations
            if exploration.tool_name == "ping_pair"
            and exploration.status == "consumed"
        ]
        self.assertEqual(len(consumed), 1)
        self.assertEqual(consumed[0].consumed_by_trial_id, trials[0].trial_id)
        self.assertIn("2 packets transmitted", consumed[0].observation)
        self.assertEqual(snapshot["consumed_explorations"], 1)
        self.assertTrue(
            all(item["session_id"] != "s1" for item in snapshot["planned_queue"])
        )
        self.assertEqual(
            revisions[0].metrics["planned_explorations_consumed"],
            1.0,
        )

    def test_consumed_planned_exploration_is_not_replanned_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping two topology hosts."},
                metrics={},
                session_id="s1",
                task_description="Investigate reachability loss",
            )
            first_state = store.load()
            original_plan = next(
                exploration
                for exploration in first_state.explorations
                if exploration.tool_name == "ping_pair"
                and exploration.status == "planned"
            )
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        (
                            "tool_start",
                            "1",
                            "ping_pair",
                            "{'host_a': 'pc1', 'host_b': 'dns'}",
                            "",
                        ),
                        (
                            "tool_end",
                            "1",
                            "ping_pair",
                            "",
                            "2 packets transmitted, 2 received",
                        ),
                    ],
                ),
                session_id="s2",
                task_description="Investigate reachability loss",
            )
            repeated_suggestion = DraftAnalyzerSuggestion(
                suggestion_id="repeat-plan",
                tool_name="ping_pair",
                session_id="s2",
                trial_ids=[trials[0].trial_id],
                suggestion="Do not replan the same consumed check.",
                next_exploration=original_plan.next_exploration,
            )
            with patch(
                "agent.tool_evolution.curator._analyzer_suggestion_for_tool",
                return_value=repeated_suggestion,
            ):
                rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={"ping_pair": "Ping two topology hosts."},
                    metrics={"rca_accuracy": 1.0},
                )
            state = store.load()
            repeated_plans = [
                exploration
                for exploration in state.explorations
                if exploration.tool_name == "ping_pair"
                and exploration.status == "planned"
                and exploration.next_exploration == original_plan.next_exploration
            ]

        self.assertEqual(repeated_plans, [])

    def test_similar_consumed_planned_exploration_is_not_replanned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping two topology hosts."},
                metrics={},
                session_id="s1",
                task_description="Investigate reachability loss",
            )
            first_state = store.load()
            original_plan = next(
                exploration
                for exploration in first_state.explorations
                if exploration.tool_name == "ping_pair"
                and exploration.status == "planned"
            )
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        (
                            "tool_start",
                            "1",
                            "ping_pair",
                            "{'host_a': 'pc1', 'host_b': 'dns'}",
                            "",
                        ),
                        (
                            "tool_end",
                            "1",
                            "ping_pair",
                            "",
                            "2 packets transmitted, 2 received",
                        ),
                    ],
                ),
                session_id="s2",
                task_description="Investigate reachability loss",
            )
            similar_suggestion = DraftAnalyzerSuggestion(
                suggestion_id="near-repeat-plan",
                tool_name="ping_pair",
                session_id="s2",
                trial_ids=[trials[0].trial_id],
                suggestion="Avoid near-duplicate planned checks.",
                next_exploration=original_plan.next_exploration + " now.",
            )
            with patch(
                "agent.tool_evolution.curator._analyzer_suggestion_for_tool",
                return_value=similar_suggestion,
            ):
                rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={"ping_pair": "Ping two topology hosts."},
                    metrics={"rca_accuracy": 1.0},
                )
            state = store.load()
            near_repeated_plans = [
                exploration
                for exploration in state.explorations
                if exploration.tool_name == "ping_pair"
                and exploration.status == "planned"
                and exploration.next_exploration == similar_suggestion.next_exploration
            ]

        self.assertEqual(near_repeated_plans, [])

    def test_eval_metrics_and_summary_include_planned_explorations(self) -> None:
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
                "draft_planned_explorations": 1,
                "draft_consumed_explorations": 1,
                "draft_analyzer_suggestions": 1,
                "draft_mastered_tools": 0,
                "draft_documented_path_rate": 1.0,
                "draft_success_path_rate": 1.0,
                "draft_converged_documents": 0,
                "draft_llm_attempts": 0,
                "draft_llm_failures": 0,
                "draft_llm_revisions": 0,
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

        self.assertEqual(metrics["draft_planned_explorations"], 1)
        self.assertEqual(metrics["draft_consumed_explorations"], 1)
        self.assertEqual(result.draft_planned_explorations, 1)
        self.assertEqual(result.draft_consumed_explorations, 1)
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
            rewrite_documentation(
                store,
                trials=[],
                tool_descriptions={"ping_pair": "Ping a host."},
                metrics={},
            )
            doc = store.get_document("ping_pair")
            assert doc is not None
            doc.usage_notes.append("Use exact host names from the active topology.")
            doc.exploration_suggestions.append(
                "Ping pc1 to verify endpoint reachability."
            )
            store.upsert_document(doc)
            state = store.load()
            state.explorations.append(
                DraftExploration(
                    exploration_id="explore_ping_pc1",
                    session_id="s1",
                    tool_name="ping_pair",
                    user_query="Ping pc1.",
                    status="planned",
                    next_exploration="Ping pc1 to verify endpoint reachability.",
                )
            )
            store.save(state)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                model="test",
                task_description="",
                store=store,
            )
            tools = runtime.build_tools()
            snapshot = runtime.snapshot()
            prompt_suffix = runtime.prompt_suffix()
            tool_learning_prompt_suffix = runtime.prompt_suffix(diagnosis_only=False)
            seeded = store.get_document("ping_pair")

        self.assertEqual([item.name for item in tools], ["ping_pair"])
        self.assertIn("DRAFT refined guidance", tools[0].description)
        self.assertNotIn("DRAFT planned active checks", tools[0].description)
        self.assertEqual(snapshot["available_documents"], ["ping_pair"])
        self.assertGreaterEqual(snapshot["planned_explorations"], 1)
        self.assertTrue(snapshot["planned_queue"])
        self.assertNotIn("DRAFT active exploration queue", prompt_suffix)
        self.assertIn("DRAFT active exploration queue", tool_learning_prompt_suffix)
        self.assertIsNotNone(seeded)

    def test_runtime_can_restore_base_descriptions_for_scoped_memory_runtime(self) -> None:
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
            state.explorations.append(
                DraftExploration(
                    exploration_id="explore_ping_pair",
                    session_id="s1",
                    tool_name="ping_pair",
                    intent="diagnosis_check",
                    user_query="Ping pc1.",
                    status="planned",
                    next_exploration="Ping pc1 to verify endpoint reachability.",
                )
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

        self.assertIn("DRAFT refined guidance", enriched_description)
        self.assertIn("DRAFT planned active checks", enriched_description)
        self.assertEqual(restored_description, "Ping a host.")

    def test_prompt_suffix_filters_global_library_description_when_scoped(self) -> None:
        def ping(host: str) -> str:
            return host

        def cat(host: str) -> str:
            return host

        tools = [
            StructuredTool.from_function(
                ping,
                name="ping_pair",
                description="Ping a host.",
            ),
            StructuredTool.from_function(
                cat,
                name="cat_file",
                description="Read a file.",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            state = store.load()
            state.library_usage_description = (
                "DRAFT-refined primitive diagnostic tools:\n"
                "- ping_pair: Ping a host.\n"
                "- cat_file: Read a file."
            )
            state.documents["ping_pair"] = ToolDocumentation(
                name="ping_pair",
                description="Ping a host.",
            )
            state.documents["cat_file"] = ToolDocumentation(
                name="cat_file",
                description="Read a file.",
            )
            store.save(state)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=tools,
                library_id="draft",
                store=store,
            )

            scoped_prompt = runtime.prompt_suffix(tool_names=["ping_pair"])
            global_prompt = runtime.prompt_suffix()

        self.assertIn("ping_pair", scoped_prompt)
        self.assertNotIn("cat_file", scoped_prompt)
        self.assertIn("cat_file", global_prompt)

    def test_runtime_filters_tool_learning_explorations_from_diagnosis_prompt(self) -> None:
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
                exploration_suggestions=[
                    "Run ping_pair on a lab with >20 hosts to verify automatic sampling.",
                    "Ping pc_0_0 to verify endpoint reachability.",
                    "Repeat with invalid host to document validation behavior.",
                ],
            )
            state.explorations.extend(
                [
                    DraftExploration(
                        exploration_id="explore_api_boundary",
                        session_id="s1",
                        tool_name="ping_pair",
                        intent="tool_validation",
                        status="planned",
                        next_exploration=(
                            "Explore one minimal valid call using observed topology "
                            "identifiers and record how output affects localization/RCA."
                        ),
                    ),
                    DraftExploration(
                        exploration_id="explore_ping_pc",
                        session_id="s1",
                        tool_name="ping_pair",
                        intent="diagnosis_check",
                        status="planned",
                        next_exploration="Ping pc_0_0 to verify endpoint reachability.",
                    ),
                ]
            )
            store.save(state)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )

            prompt = runtime.prompt_suffix(tool_names=["ping_pair"])
            checks = runtime.next_checks("ping_pair")
            diagnosis_queue = runtime.planned_explorations(
                diagnosis_only=True,
            )
            snapshot_queue = runtime.planned_explorations()

        self.assertIn("Ping pc_0_0", prompt)
        self.assertIn("Ping pc_0_0", " ".join(checks))
        self.assertEqual(
            [item["exploration_id"] for item in diagnosis_queue],
            ["explore_ping_pc"],
        )
        self.assertEqual(len(snapshot_queue), 2)
        self.assertNotIn("minimal valid call", prompt)
        self.assertNotIn(">20 hosts", prompt)
        self.assertNotIn("invalid host", prompt)

    def test_runtime_validates_diagnosis_explorations_against_generic_topology_hosts(self) -> None:
        def net_config(host_name: str) -> str:
            return host_name

        tool = StructuredTool.from_function(
            net_config,
            name="get_host_net_config",
            description="Inspect host network config.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            state = store.load()
            state.documents["get_host_net_config"] = ToolDocumentation(
                name="get_host_net_config",
                description="Inspect host network config.",
            )
            state.explorations.extend(
                [
                    DraftExploration(
                        exploration_id="explore_host_1",
                        session_id="s1",
                        tool_name="get_host_net_config",
                        intent="diagnosis_check",
                        status="planned",
                        parameters={"host_name": "host-1"},
                        next_exploration="Inspect host-1 interface state.",
                    ),
                    DraftExploration(
                        exploration_id="explore_host_3",
                        session_id="s1",
                        tool_name="get_host_net_config",
                        intent="diagnosis_check",
                        status="planned",
                        parameters={"host_name": "host-3"},
                        next_exploration="Inspect host-3 interface state.",
                    ),
                ]
            )
            store.save(state)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                task_description="Topology hosts: host-1, host-2.",
                store=store,
            )

            prompt = runtime.prompt_suffix(tool_names=["get_host_net_config"])
            diagnosis_queue = runtime.planned_explorations(diagnosis_only=True)

        self.assertIn("host-1", prompt)
        self.assertNotIn("host-3", prompt)
        self.assertEqual(
            [item["exploration_id"] for item in diagnosis_queue],
            ["explore_host_1"],
        )

    def test_runtime_filters_document_suggestions_by_current_topology(self) -> None:
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
                exploration_suggestions=[
                    "Ping pc1 to verify endpoint reachability.",
                    "Ping pc3 to verify endpoint reachability.",
                    "Repeat with invalid host to document validation behavior.",
                ],
            )
            store.save(state)
            runtime = ToolEvolutionRuntime(
                session=SimpleNamespace(
                    task_description="Investigate reachability loss",
                    topology=[("pc1:eth0", "r1:eth0")],
                ),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )

            prompt = runtime.prompt_suffix(tool_names=["ping_pair"])
            checks = runtime.next_checks("ping_pair")

        joined_checks = " ".join(checks)
        self.assertIn("pc1", prompt)
        self.assertIn("pc1", joined_checks)
        self.assertNotIn("pc3", prompt)
        self.assertNotIn("pc3", joined_checks)
        self.assertNotIn("invalid host", prompt)

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

    def test_runtime_claims_planned_exploration_once_per_session(self) -> None:
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
            )
            state.explorations.append(
                DraftExploration(
                    exploration_id="explore_ping_once",
                    session_id="s1",
                    tool_name="ping_pair",
                    intent="diagnosis_check",
                    user_query="Ping pc1 once.",
                    parameters={"host": "pc1"},
                    status="planned",
                    next_exploration="Ping pc1 to verify endpoint reachability.",
                )
            )
            store.save(state)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )

            first = runtime.match_planned_exploration(
                "ping_pair",
                {"host": "pc1"},
            )
            second = runtime.match_planned_exploration(
                "ping_pair",
                {"host": "pc1"},
            )
            snapshot = runtime.snapshot()
            next_checks = runtime.next_checks("ping_pair")

        self.assertIsNotNone(first)
        self.assertEqual(first["exploration_id"], "explore_ping_once")
        self.assertIsNone(second)
        self.assertEqual(snapshot["planned_queue"], [])
        self.assertEqual(snapshot["claimed_exploration_ids"], ["explore_ping_once"])
        self.assertNotIn("Ping pc1", " ".join(next_checks))

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
                    suggestions_for_exploring=(
                        "Try a valid discovered interface and one invalid boundary case."
                    ),
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
            with patch("agent.tool_evolution.curator.load_model", return_value=FakeModel()):
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
        self.assertEqual(doc.description, "Inspect one router interface with verified names.")
        self.assertEqual(
            doc.tool_usage_description,
            "show_iface is a tool that can inspect one verified router interface.",
        )
        self.assertIn("Never guess interface names.", doc.constraints)
        self.assertTrue(
            any(
                "Try a valid discovered interface" in item
                for item in doc.exploration_suggestions
            )
        )
        self.assertIn("Explorer observations", prompts[0])
        self.assertIn("Analyzer suggestions", prompts[0])
        self.assertIn("tool_usage_description", prompts[0])
        self.assertEqual(revisions[0].metrics["llm_rewrite"], 1.0)

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
            with patch("agent.tool_evolution.curator.load_model", return_value=FakeModel()):
                rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={"get_reachability": "Reachability matrix."},
                    metrics={"rca_accuracy": 0.0},
                    llm_backend="custom",
                    model="test-model",
                )

        self.assertEqual(len(prompts), 1)
        self.assertLess(len(prompts[0]), 8000)
        self.assertNotIn("x" * 1000, prompts[0])

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
                tool_descriptions={"show_iface": "Inspect interface.", "new_tool": "New."},
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
        path.write_text("\n".join(json.dumps(row) for row in payloads), encoding="utf-8")
        return path
