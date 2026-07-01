"""Tests for DRAFT-style tool documentation refinement."""

from __future__ import annotations

import json
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
from agent.tool_evolution.models import DraftRewriteProposal, ToolParameterDoc
from agent.tool_evolution.runtime import ToolEvolutionRuntime
from agent.tool_evolution.store import ToolEvolutionStore


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
        self.assertEqual(len(state.explorations), 1)
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
            store.upsert_document(doc)
            runtime = ToolEvolutionRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                model="test",
                task_description="",
            )
            runtime.store = store
            runtime._docs = store.load().documents
            tools = runtime.build_tools()
            snapshot = runtime.snapshot()

        self.assertEqual([item.name for item in tools], ["ping_pair"])
        self.assertIn("DRAFT refined guidance", tools[0].description)
        self.assertEqual(snapshot["available_documents"], ["ping_pair"])

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
