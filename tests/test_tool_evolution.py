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

    def test_rewrites_documentation_with_preconditions_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("draft", root=tmp)
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        ("tool_start", "1", "show_iface", "{'router': 'r1', 'interface': 'eth9'}", ""),
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

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertTrue(revisions[0].changed)
        self.assertIn("router", doc.parameters)
        self.assertIn("interface", doc.parameters)
        self.assertTrue(doc.failure_modes)
        self.assertIn("Tool arguments must be grounded", doc.constraints[0])

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

        self.assertEqual([item.name for item in tools], ["ping_pair"])
        self.assertIn("DRAFT refined guidance", tools[0].description)

    def test_llm_rewrite_proposal_is_merged_into_documentation(self) -> None:
        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                return DraftRewriteProposal(
                    tool_name="show_iface",
                    description="Inspect one router interface with verified names.",
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
        self.assertIn("Never guess interface names.", doc.constraints)
        self.assertEqual(revisions[0].metrics["llm_rewrite"], 1.0)

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
