"""Tests for DRAFT-style tool documentation refinement."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from unittest.mock import patch

from agent.tool_refinement.curator import (
    _diagnostic_utility_lcb,
    _exploration_is_read_only,
    _exploration_signature,
    extract_tool_trials,
    identify_comprehension_gaps,
    rewrite_documentation,
)
from agent.tool_refinement.explorer import _validate_parameters, run_active_exploration
from agent.tool_refinement.generalization import generalize_tool_documentation
from agent.tool_refinement.models import (
    DraftAnalyzerDraft,
    DraftExploration,
    DraftExplorerDraft,
    DraftRewriteProposal,
    ToolDocumentation,
    ToolParameterDoc,
    ToolTrial,
)
from agent.tool_refinement.runtime import ToolRefinementRuntime
from agent.tool_refinement.store import ToolRefinementStore
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.tool_output import classify_tool_outcome
from nika.evaluator.result_log import build_eval_result_from_session_dir
from nika.workflows.eval.session import run_eval_metrics




class DraftToolRefinementTestPart1(unittest.TestCase):
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
                        "tool_refinement_enabled": True,
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
                    self.tool_refinement_enabled = True
                    self.procedural_memory_enabled = False
                    self.allow_training_updates = False
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
                    "agent.tool_refinement.curator.finalize_tool_refinement_session",
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


    def test_evaluation_runtime_is_read_only_and_explorer_is_not_started(self) -> None:
        def ping(host: str) -> str:
            return host

        tool = StructuredTool.from_function(
            ping,
            name="ping_host",
            description="Ping one host.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("frozen", root=Path(tmp) / "library")
            training = ToolRefinementRuntime(
                session=SimpleNamespace(session_id="training"),
                primitive_tools=[tool],
                library_id="frozen",
                store=store,
                allow_training_updates=True,
            )
            before = store.state_hash()
            evaluation = ToolRefinementRuntime(
                session=SimpleNamespace(session_id="evaluation"),
                primitive_tools=[tool],
                library_id="frozen",
                store=store,
                explorer_llm=object(),
                allow_training_updates=False,
            )
            report = asyncio.run(evaluation.explore("Inspect reachability."))
            after = store.state_hash()
            snapshot = evaluation.snapshot()

            with self.assertRaises(PermissionError):
                evaluation.store.save(evaluation.store.load())

        self.assertIsNotNone(training)
        self.assertEqual(report["status"], "skipped")
        self.assertIn("disabled", report["reason"])
        self.assertEqual(before, after)
        self.assertTrue(snapshot["store_read_only"])
        self.assertTrue(snapshot["state_unchanged"])
        self.assertEqual(snapshot["state_hash"], before)


    def test_store_serializes_concurrent_trial_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"

            def record(index: int) -> None:
                ToolRefinementStore("draft", root=root).record_trials(
                    [
                        ToolTrial(
                            trial_id=f"trial-{index}",
                            session_id=f"session-{index}",
                            tool_name="inspect_state",
                            status="success",
                        )
                    ]
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(record, range(40)))
            state = ToolRefinementStore("draft", root=root).load()

        self.assertEqual(len(state.trials), 40)
        self.assertEqual(len({trial.trial_id for trial in state.trials}), 40)


    def test_explorer_read_only_gate_blocks_network_mutation_commands(self) -> None:
        for command in (
            "ip link set eth0 down",
            "ip route add default via 192.0.2.1",
            "vtysh -c 'configure terminal'",
            "service frr restart",
            "tee /etc/network/interfaces",
        ):
            self.assertFalse(
                _exploration_is_read_only(
                    tool_name="run_command",
                    parameters={"command": command},
                    text="Inspect network state.",
                ),
                command,
            )
        self.assertTrue(
            _exploration_is_read_only(
                tool_name="run_command",
                parameters={"command": "ip route show"},
                text="Inspect routing state.",
            )
        )
        self.assertTrue(
            _exploration_is_read_only(
                tool_name="frr_exec",
                parameters={"router_name": "r1", "command": "show ip bgp summary"},
                text="Inspect BGP state.",
            )
        )
        self.assertFalse(
            _exploration_is_read_only(
                tool_name="exec_shell",
                parameters={"host_name": "h1", "command": "echo changed"},
                text="Inspect state.",
            )
        )
        self.assertFalse(
            _exploration_is_read_only(
                tool_name="exec_shell",
                parameters={
                    "host_name": "h1",
                    "command": "ip route show; touch /tmp/probe",
                },
                text="Inspect state.",
            )
        )
        self.assertFalse(
            _exploration_is_read_only(
                tool_name="ethtool",
                parameters={"host_name": "h1", "interface": "eth0", "args": "-s"},
                text="Inspect link state.",
            )
        )


    def test_exploration_diversity_ignores_runtime_identifier_changes(self) -> None:
        first = _exploration_signature(
            tool_name="show_route",
            user_query="Inspect routes on router_a.",
            parameters={"router_name": "router_a"},
        )
        second = _exploration_signature(
            tool_name="show_route",
            user_query="Inspect routes on router_b.",
            parameters={"router_name": "router_b"},
        )

        self.assertEqual(first, second)


    def test_explorer_rejects_unbounded_numeric_probe_values(self) -> None:
        def ping_host(host_name: str, count: int = 4) -> str:
            return f"{host_name}:{count}"

        tool = StructuredTool.from_function(
            ping_host,
            name="ping_host",
            description="Check reachability.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=Path(tmp) / "library")
            ToolRefinementRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            doc = store.get_document("ping_host")
        assert doc is not None
        observed = [
            ToolTrial(
                trial_id="trial",
                session_id="s1",
                tool_name="ping_host",
                arguments={"host_name": "host_a"},
                status="success",
            )
        ]

        rejected, error = _validate_parameters(
            tool,
            doc,
            {"host_name": "host_a", "count": 999999},
            grounded_identifiers={"host_name": {"host_a"}},
            observed_trials=observed,
        )
        accepted, accepted_error = _validate_parameters(
            tool,
            doc,
            {"host_name": "host_a", "count": 4},
            grounded_identifiers={"host_name": {"host_a"}},
            observed_trials=observed,
        )

        self.assertIsNone(rejected)
        self.assertIn("source default", error)
        self.assertEqual(accepted, {"host_name": "host_a", "count": 4})
        self.assertEqual(accepted_error, "")


    def test_self_driven_explorer_collects_grounded_probe_for_checkpoint(self) -> None:
        calls: list[str] = []
        explorer_prompts: list[str] = []

        async def ping_host(host_name: str) -> str:
            calls.append(host_name)
            return f"{host_name} reachable"

        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            async def ainvoke(self, prompt):
                explorer_prompts.append(prompt)
                return DraftExplorerDraft(
                    user_query="Check current reachability for the observed endpoint.",
                    parameters={"host_name": "host_a"},
                )

            def invoke(self, _prompt):
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(
                        suggestion="Clarify successful reachability output."
                    )
                return DraftRewriteProposal(
                    tool_name="ping_host",
                    tool_usage_description="Check reachability for an observed host.",
                    next_exploration_direction="Explore another valid observed endpoint.",
                )

        tool = StructuredTool.from_function(
            coroutine=ping_host,
            name="ping_host",
            description="Check whether one host is reachable.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            session = SimpleNamespace(session_id="s1", session_dir=tmp)
            store = ToolRefinementStore("draft", root=Path(tmp) / "library")
            ToolRefinementRuntime(
                session=session,
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            trace = self._trace(
                tmp,
                [
                    ("tool_start", "1", "ping_host", "{'host_name': 'host_a'}", ""),
                    ("tool_end", "1", "ping_host", "", "host_a reachable"),
                ],
            )
            before_trace = trace.read_text(encoding="utf-8")
            report = asyncio.run(
                run_active_exploration(
                    session_id="s1",
                    session_dir=tmp,
                    task_description="Inspect network health.",
                    tools=[tool],
                    store=store,
                    llm=FakeModel(),
                    model="test-model",
                )
            )
            state = store.load()
            doc = state.documents["ping_host"]
            after_trace = trace.read_text(encoding="utf-8")

        self.assertEqual(report["active_explorations"], 1)
        self.assertEqual(calls, ["host_a"])
        self.assertEqual(len(state.explorations), 1)
        self.assertTrue(state.explorations[0].trial_id.startswith("active_trial_"))
        self.assertEqual(doc.next_exploration_direction, "")
        self.assertEqual(len(state.revisions), 0)
        self.assertIn("Grounded identifier values", explorer_prompts[0])
        self.assertEqual(before_trace, after_trace)


    def test_self_driven_explorer_rejects_ungrounded_identifier(self) -> None:
        calls: list[str] = []
        prompts: list[str] = []

        async def ping_host(host_name: str) -> str:
            calls.append(host_name)
            return "reachable"

        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            async def ainvoke(self, prompt):
                prompts.append(prompt)
                return DraftExplorerDraft(
                    user_query="Check a different endpoint.",
                    parameters={"host_name": "invented_host"},
                )

        tool = StructuredTool.from_function(
            coroutine=ping_host,
            name="ping_host",
            description="Check whether one host is reachable.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=Path(tmp) / "library")
            ToolRefinementRuntime(
                session=SimpleNamespace(session_id="s1", session_dir=tmp),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            state = store.load()
            state.trials.append(
                ToolTrial(
                    trial_id="old-trial",
                    session_id="old-session",
                    tool_name="ping_host",
                    arguments={"host_name": "invented_host"},
                    status="success",
                    output_summary="reachable",
                )
            )
            state.explorations.append(
                DraftExploration(
                    exploration_id="old-exploration",
                    session_id="old-session",
                    trial_id="old-trial",
                    tool_name="ping_host",
                    user_query="Check invented_host.",
                    parameters={"host_name": "invented_host"},
                    observation="reachable",
                    status="success",
                )
            )
            store.save(state)
            self._trace(
                tmp,
                [
                    ("tool_start", "1", "ping_host", "{'host_name': 'host_a'}", ""),
                    ("tool_end", "1", "ping_host", "", "host_a reachable"),
                ],
            )
            report = asyncio.run(
                run_active_exploration(
                    session_id="s1",
                    session_dir=tmp,
                    task_description="Inspect network health.",
                    tools=[tool],
                    store=store,
                    llm=FakeModel(),
                    model="test-model",
                )
            )

        self.assertEqual(calls, [])
        self.assertEqual(report["active_explorations"], 0)
        self.assertIn("must reuse values", report["skipped"]["ping_host"])
        self.assertTrue(prompts)
        self.assertNotIn("invented_host", "\n".join(prompts))
        self.assertIn("<host_name>", prompts[0])


    def test_self_driven_explorer_schedules_one_underexplored_tool(self) -> None:
        calls: list[str] = []

        async def ping_host(host_name: str) -> str:
            calls.append(f"ping:{host_name}")
            return "reachable"

        async def inspect_state() -> str:
            calls.append("inspect")
            return "healthy"

        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            async def ainvoke(self, prompt):
                if "`inspect_state`" in prompt:
                    return DraftExplorerDraft(
                        user_query="Inspect the current state format.",
                        parameters={},
                    )
                return DraftExplorerDraft(
                    user_query="Check the observed endpoint from another angle.",
                    parameters={"host_name": "host_a"},
                )

            def invoke(self, prompt):
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(suggestion="Clarify observed output.")
                return DraftRewriteProposal(
                    tool_name=(
                        "inspect_state"
                        if "Tool: inspect_state" in prompt
                        else "ping_host"
                    )
                )

        tools = [
            StructuredTool.from_function(
                coroutine=ping_host,
                name="ping_host",
                description="Check whether one host is reachable.",
            ),
            StructuredTool.from_function(
                coroutine=inspect_state,
                name="inspect_state",
                description="Inspect the current network state.",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=Path(tmp) / "library")
            ToolRefinementRuntime(
                session=SimpleNamespace(session_id="s1", session_dir=tmp),
                primitive_tools=tools,
                library_id="draft",
                store=store,
            )
            self._trace(
                tmp,
                [
                    ("tool_start", "1", "ping_host", "{'host_name': 'host_a'}", ""),
                    ("tool_end", "1", "ping_host", "", "host_a reachable"),
                ],
            )
            report = asyncio.run(
                run_active_exploration(
                    session_id="s1",
                    session_dir=tmp,
                    task_description="Inspect network health.",
                    tools=tools,
                    store=store,
                    llm=FakeModel(),
                    model="test-model",
                )
            )

        self.assertEqual(report["scheduled_tools"], ["ping_host", "inspect_state"])
        self.assertEqual(report["active_explorations"], 2)
        self.assertEqual(calls, ["ping:host_a", "inspect"])


    def test_self_driven_explorer_enforces_query_diversity(self) -> None:
        calls: list[str] = []

        async def ping_host(host_name: str) -> str:
            calls.append(host_name)
            return "reachable"

        class FakeModel:
            def with_structured_output(self, _schema):
                return self

            async def ainvoke(self, _prompt):
                return DraftExplorerDraft(
                    user_query="Check reachability for host_a.",
                    parameters={"host_name": "host_a"},
                )

        tool = StructuredTool.from_function(
            coroutine=ping_host,
            name="ping_host",
            description="Check whether one host is reachable.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=Path(tmp) / "library")
            ToolRefinementRuntime(
                session=SimpleNamespace(session_id="s2", session_dir=tmp),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            prior_trial = ToolTrial(
                trial_id="prior-trial",
                session_id="s1",
                tool_name="ping_host",
                task_description="Check reachability for host_a.",
                arguments={"host_name": "host_a"},
                status="success",
                output_summary="reachable",
            )
            state = store.load()
            state.trials.append(prior_trial)
            state.explorations.append(
                DraftExploration(
                    exploration_id="prior-exploration",
                    session_id="s1",
                    trial_id="prior-trial",
                    tool_name="ping_host",
                    user_query="Check reachability for host_a.",
                    parameters={"host_name": "host_a"},
                    observation="reachable",
                    status="success",
                )
            )
            store.save(state)
            self._trace(
                tmp,
                [
                    ("tool_start", "1", "ping_host", "{'host_name': 'host_a'}", ""),
                    ("tool_end", "1", "ping_host", "", "host_a reachable"),
                ],
            )
            report = asyncio.run(
                run_active_exploration(
                    session_id="s2",
                    session_dir=tmp,
                    task_description="Inspect network health.",
                    tools=[tool],
                    store=store,
                    llm=FakeModel(),
                    model="test-model",
                )
            )

        self.assertEqual(calls, [])
        self.assertEqual(report["active_explorations"], 0)
        self.assertIn("similarity", report["skipped"]["ping_host"])


    def test_structured_tool_output_distinguishes_execution_failure_from_evidence(
        self,
    ) -> None:
        timeout = ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": "[TIMEOUT] Command 'ping' exceeded 10s.",
                }
            ],
            tool_call_id="timeout-call",
            status="success",
        )
        unreachable = ToolMessage(
            content="Destination Host Unreachable; 100% packet loss",
            tool_call_id="evidence-call",
            status="success",
        )
        unknown = ToolMessage(
            content={"status": "unknown", "result": []},
            tool_call_id="unknown-call",
            status="success",
        )
        partial_unknown = ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "hosts": {"host_a": None, "host_b": "192.0.2.2"},
                            "results": [
                                {"src": "host_a", "status": "unknown"},
                                {"src": "host_b", "status": "ok"},
                            ],
                        }
                    ),
                }
            ],
            tool_call_id="partial-call",
            status="success",
        )

        self.assertEqual(classify_tool_outcome(timeout), "error")
        self.assertEqual(classify_tool_outcome(unreachable), "success")
        self.assertEqual(classify_tool_outcome(unknown), "unknown")
        self.assertEqual(classify_tool_outcome(partial_unknown), "success")
        self.assertEqual(
            classify_tool_outcome({"content": {"status": "unknown"}}),
            "unknown",
        )
        self.assertEqual(
            classify_tool_outcome(
                "content=[{'type': 'text', 'text': 'Cannot get IP address of host pc1.'}]"
            ),
            "error",
        )


    def test_callback_logger_preserves_structured_tool_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AgentCallbackLogger(agent="diagnosis", session_dir=tmp)
            logger.on_tool_end(
                ToolMessage(
                    content=[{"type": "text", "text": "eth0 up"}],
                    artifact={"structured_content": {"status": "ok"}},
                    tool_call_id="call-1",
                    name="show_iface",
                    status="success",
                )
            )
            event = json.loads(
                (Path(tmp) / "messages.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(event["event"], "tool_end")
        self.assertEqual(event["outcome"], "success")
        self.assertIsInstance(event["output"], dict)
        self.assertEqual(event["output"]["content"][0]["text"], "eth0 up")


    def test_documentation_replaces_runtime_identifiers_with_placeholders(
        self,
    ) -> None:
        doc = ToolDocumentation(
            name="show_iface",
            tool_usage_description="Inspect eth9 on router_dist_2_1.",
            parameters={
                "router_name": ToolParameterDoc(
                    name="router_name",
                    examples=["router_dist_2_1"],
                ),
                "interface": ToolParameterDoc(name="interface", examples=["eth9"]),
            },
            positive_examples=[
                {
                    "arguments": {
                        "router_name": "router_dist_2_1",
                        "interface": "eth9",
                    }
                }
            ],
        )

        changed = generalize_tool_documentation(doc)

        self.assertTrue(changed)
        self.assertNotIn("router_dist_2_1", doc.model_dump_json())
        self.assertNotIn("eth9", doc.model_dump_json())
        self.assertIn("<router_name>", doc.tool_usage_description)
        self.assertIn("<interface>", doc.tool_usage_description)


    def test_primitive_contract_change_reopens_frozen_documentation(self) -> None:
        def ping(host: str) -> str:
            return host

        def ping_with_count(host: str, count: int = 1) -> str:
            return host * count

        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
            first_tool = StructuredTool.from_function(
                ping,
                name="ping_host",
                description="Ping one host.",
            )
            ToolRefinementRuntime(
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
            ToolRefinementRuntime(
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
            store = ToolRefinementStore("draft", root=tmp)
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
            store = ToolRefinementStore("draft", root=tmp)
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
                task_description="Determine whether a service restart caused route failure.",
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
        self.assertTrue(
            all(item.observation == "route missing" for item in state.explorations)
        )
        self.assertTrue(all(item.read_only for item in state.explorations))
        self.assertTrue(
            all("restart" not in item.user_query for item in state.explorations)
        )
        self.assertEqual(state.explorations[0].diversity_score, 1.0)
        self.assertEqual(state.explorations[1].diversity_score, 0.0)
        self.assertEqual(state.explorations[1].reflection_count, 1)


    def test_documentation_mastery_is_independent_of_rca_score(self) -> None:
        def evolve(root: str, rca_f1: float) -> float:
            store = ToolRefinementStore("draft", root=root)
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


    def test_diagnostic_utility_updates_independently_of_frozen_docs(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "NIKA_TRAINING_LLM_BACKEND": "",
                    "NIKA_TRAINING_LLM_MODEL": "",
                },
            ),
        ):
            store = ToolRefinementStore("draft", root=tmp)
            first = ToolTrial(
                trial_id="first",
                session_id="s1",
                tool_name="show_route",
                status="success",
            )
            rewrite_documentation(
                store,
                trials=[first],
                tool_descriptions={"show_route": "Show routes."},
                metrics={
                    "detection_score": 1.0,
                    "localization_f1": 1.0,
                    "rca_f1": 1.0,
                },
            )
            state = store.load()
            state.documents["show_route"].frozen = True
            store.save(state)

            second = ToolTrial(
                trial_id="second",
                session_id="s2",
                tool_name="show_route",
                status="success",
            )
            rewrite_documentation(
                store,
                trials=[second],
                tool_descriptions={"show_route": "Show routes."},
                metrics={
                    "detection_score": 1.0,
                    "localization_f1": 0.0,
                    "rca_f1": 0.0,
                },
            )
            doc = store.get_document("show_route")
            assert doc is not None

        self.assertEqual(doc.contract_mastery_score, doc.mastery_score)
        self.assertEqual(doc.diagnostic_utility_count, 2)
        self.assertAlmostEqual(doc.diagnostic_utility_score, 0.55)
        self.assertFalse(doc.frozen)
        self.assertEqual(doc.frozen_reason, "")


    def test_diagnostic_utility_requires_confident_support(self) -> None:
        weak = ToolDocumentation(
            name="weak",
            diagnostic_utility_score=0.55,
            diagnostic_utility_count=2,
        )
        strong = ToolDocumentation(
            name="strong",
            diagnostic_utility_score=1.0,
            diagnostic_utility_count=2,
        )

        self.assertLess(_diagnostic_utility_lcb(weak), 0.3)
        self.assertGreater(_diagnostic_utility_lcb(strong), 0.3)


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


    def test_tool_end_with_explicit_failure_is_not_positive_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = self._trace(
                tmp,
                [
                    ("tool_start", "1", "show_route", "{'router': 'r1'}", ""),
                    ("tool_end", "1", "show_route", "", "Error: command failed"),
                    ("tool_start", "2", "show_route", "{'router': 'r2'}", ""),
                    ("tool_end", "2", "show_route", "", ""),
                    ("tool_start", "3", "show_route", "{'router': 'r3'}", ""),
                    ("tool_end", "3", "show_route", "", "route not found"),
                ],
            )
            trials, _ = extract_tool_trials(trace, session_id="semantic-outcome")

        self.assertEqual(
            [trial.status for trial in trials],
            ["error", "unknown", "success"],
        )
        self.assertFalse(trials[0].success)
        self.assertFalse(trials[1].success)
        self.assertTrue(trials[2].success)


    def test_atomic_store_failure_preserves_previous_tool_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("atomic", root=tmp)
            state = store.load()
            state.library_usage_description = "stable state"
            store.save(state)
            previous = store.state_path.read_text(encoding="utf-8")
            state.library_usage_description = "interrupted update"

            with patch(
                "agent.utils.atomic.os.replace",
                side_effect=OSError("simulated interruption"),
            ):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    store.save(state)

            self.assertEqual(store.state_path.read_text(encoding="utf-8"), previous)
            self.assertEqual(list(store.library_dir.glob(".*.tmp")), [])


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
                        "[Integrated training guidance - not evidence]\n"
                        "Active Skill-MDP option: seed."
                    ),
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            trials, _ = extract_tool_trials(trace, session_id="s1")

        self.assertIn("eth0 up", trials[0].output_summary)
        self.assertNotIn("Integrated training guidance", trials[0].output_summary)


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
            store = ToolRefinementStore("draft", root=tmp)
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
            store = ToolRefinementStore("draft", root=tmp)
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
        self.assertFalse(any(gap.gap_type == "diagnostic_semantic_gap" for gap in gaps))


    def test_draft_does_not_learn_from_documented_tool_without_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
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


    def test_runtime_appends_refined_docs_without_adding_tools(self) -> None:
        def ping(host: str) -> str:
            return host

        tool = StructuredTool.from_function(
            ping,
            name="ping_pair",
            description="Ping a host.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
            ToolRefinementRuntime(
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
            doc.published = True
            store.upsert_document(doc)
            runtime = ToolRefinementRuntime(
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
            store = ToolRefinementStore("draft", root=tmp)
            state = store.load()
            state.documents["ping_pair"] = ToolDocumentation(
                name="ping_pair",
                description="Ping a host.",
                usage_notes=["Use exact host names from the active topology."],
            )
            store.save(state)
            runtime = ToolRefinementRuntime(
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

