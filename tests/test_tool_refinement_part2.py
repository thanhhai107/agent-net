"""Tests for DRAFT-style tool documentation refinement."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.tools import StructuredTool
from unittest.mock import patch

from agent.tool_refinement.curator import (
    extract_tool_trials,
    finalize_tool_refinement_session,
    rewrite_documentation,
)
from agent.tool_refinement.models import (
    DraftAnalyzerDraft,
    DraftRewriteProposal,
    ToolDocumentation,
    ToolParameterDoc,
    ToolTrial,
)
from agent.tool_refinement.runtime import ToolRefinementRuntime
from agent.tool_refinement.store import ToolRefinementStore
from nika.evaluator.result_log import build_eval_result_from_session_dir
from nika.workflows.eval.session import run_eval_metrics




class DraftToolRefinementTestPart2(unittest.TestCase):
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
            store = ToolRefinementStore("draft", root=tmp)
            runtime = ToolRefinementRuntime(
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
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(
                        suggestion="Clarify the observed interface output."
                    )
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
            store = ToolRefinementStore("draft", root=tmp)
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
                "agent.tool_refinement.curator.load_model", return_value=FakeModel()
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
            store = ToolRefinementStore("draft", root=tmp)
            ToolRefinementRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            with patch(
                "agent.tool_refinement.curator.load_model", return_value=FakeModel()
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
            [
                "Tool arguments must be grounded in currently observed topology evidence."
            ],
        )
        self.assertEqual(doc.failure_modes, [])
        self.assertEqual(doc.usage_notes, [])
        self.assertNotIn("audit metadata", doc.refined_description(max_chars=4000))
        self.assertTrue(doc.frozen)
        self.assertIn("adaptive termination", doc.frozen_reason)


    def test_identifier_changes_do_not_satisfy_termination_diversity(self) -> None:
        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, _prompt):
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(suggestion="Clarify route output.")
                return DraftRewriteProposal(
                    tool_name="show_route",
                    tool_usage_description="Inspect routes on an observed router.",
                )

        def show_route(router_name: str) -> str:
            return router_name

        tool = StructuredTool.from_function(
            show_route,
            name="show_route",
            description="Inspect one routing table.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
            ToolRefinementRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            with patch(
                "agent.tool_refinement.curator.load_model", return_value=FakeModel()
            ):
                revisions = []
                for session_id, router_name in (("s1", "router_a"), ("s2", "router_b")):
                    revisions = rewrite_documentation(
                        store,
                        trials=[
                            ToolTrial(
                                trial_id=f"trial-{session_id}",
                                session_id=session_id,
                                tool_name="show_route",
                                arguments={"router_name": router_name},
                                status="success",
                                output_summary="route present",
                            )
                        ],
                        tool_descriptions={"show_route": "Inspect one routing table."},
                        metrics={},
                        llm_backend="custom",
                        model="test-model",
                    )
            doc = store.get_document("show_route")

        assert doc is not None
        self.assertFalse(doc.frozen)
        self.assertEqual(
            revisions[0].metrics["exploration_diversity_support"],
            0.5,
        )


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
            store = ToolRefinementStore("draft", root=tmp)
            ToolRefinementRuntime(
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
                "agent.tool_refinement.curator.load_model",
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


    def test_llm_rewrite_cannot_invent_numeric_contract_constraints(self) -> None:
        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, _prompt):
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(suggestion="Clarify the retry count.")
                return DraftRewriteProposal(
                    tool_name="ping_host",
                    tool_usage_description="Ping a host exactly 99 times.",
                    parameters={
                        "count": ToolParameterDoc(
                            name="count",
                            type_hint="int",
                            description="Use a count from 1 to 99.",
                        )
                    },
                )

        def ping_host(host_name: str, count: int = 4) -> str:
            return host_name * count

        tool = StructuredTool.from_function(
            ping_host,
            name="ping_host",
            description="Ping a host a configurable number of times.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
            ToolRefinementRuntime(
                session=object(),
                primitive_tools=[tool],
                library_id="draft",
                store=store,
            )
            trials, _ = extract_tool_trials(
                self._trace(
                    tmp,
                    [
                        (
                            "tool_start",
                            "1",
                            "ping_host",
                            "{'host_name': 'pc1', 'count': 4}",
                            "",
                        ),
                        ("tool_end", "1", "ping_host", "", "host reachable"),
                    ],
                ),
                session_id="s1",
            )
            with patch(
                "agent.tool_refinement.curator.load_model",
                return_value=FakeModel(),
            ):
                revisions = rewrite_documentation(
                    store,
                    trials=trials,
                    tool_descriptions={
                        "ping_host": "Ping a host a configurable number of times."
                    },
                    metrics={},
                    llm_backend="custom",
                    model="test-model",
                )
            doc = store.get_document("ping_host")

        assert doc is not None
        self.assertNotIn("99", doc.refined_description(max_chars=4000))
        self.assertEqual(revisions[0].metrics["llm_contract_rejected"], 1.0)
        self.assertIn("numeric constraints", revisions[0].llm_error)


    def test_llm_rewrite_failure_is_reported(self) -> None:
        class FailingModel:
            def __init__(self) -> None:
                self.invocations = 0

            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                self.invocations += 1
                raise TimeoutError("draft timeout")

        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
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
            failing_model = FailingModel()
            with (
                patch.dict(
                    os.environ,
                    {
                        "NIKA_TRAINING_LLM_BACKEND": "custom",
                        "NIKA_TRAINING_LLM_MODEL": "training-model",
                    },
                ),
                patch(
                    "agent.tool_refinement.curator.load_model",
                    return_value=failing_model,
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
        self.assertEqual(args[:2], ("custom", "training-model"))
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(failing_model.invocations, 1)
        self.assertEqual(revisions[0].metrics["llm_attempted"], 1.0)
        self.assertEqual(revisions[0].metrics["llm_failed"], 1.0)
        self.assertEqual(revisions[0].metrics["llm_rewrite"], 0.0)
        self.assertIn("TimeoutError", revisions[0].llm_error)


    def test_failed_rewriter_never_freezes_documentation(self) -> None:
        class FailingModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                raise TimeoutError("draft timeout")

        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
            with patch(
                "agent.tool_refinement.curator.load_model",
                return_value=FailingModel(),
            ):
                for session_id in ("s1", "s2"):
                    rewrite_documentation(
                        store,
                        trials=[
                            ToolTrial(
                                trial_id=f"trial-{session_id}",
                                session_id=session_id,
                                tool_name="inspect_state",
                                status="success",
                                output_summary="healthy",
                            )
                        ],
                        tool_descriptions={"inspect_state": "Inspect state."},
                        metrics={},
                        llm_backend="custom",
                        model="test-model",
                    )
            doc = store.get_document("inspect_state")

        assert doc is not None
        self.assertFalse(doc.frozen)
        self.assertEqual(doc.frozen_reason, "")


    def test_draft_rewrite_prompt_compacts_large_outputs(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            schema: type | None = None

            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
                if self.schema is DraftAnalyzerDraft:
                    return DraftAnalyzerDraft(
                        suggestion="Clarify the observed reachability output."
                    )
                return DraftRewriteProposal(
                    tool_name="get_reachability",
                    tool_usage_description="Summarize reachability matrix safely.",
                )

        huge_output = "reachable " + ("x" * 8000)
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolRefinementStore("draft", root=tmp)
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
                "agent.tool_refinement.curator.load_model", return_value=FakeModel()
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
            store = ToolRefinementStore("draft", root=tmp)
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


    def test_session_trials_are_collected_until_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_dirs = {
                "s1": root / "s1",
                "s2": root / "s2",
            }
            for session_id, session_dir in session_dirs.items():
                session_dir.mkdir()
                self._trace(
                    str(session_dir),
                    [
                        (
                            "tool_start",
                            session_id,
                            "show_route",
                            f"{{'router': '{session_id}'}}",
                            "",
                        ),
                        (
                            "tool_end",
                            session_id,
                            "show_route",
                            "",
                            "route present",
                        ),
                    ],
                )
            store = ToolRefinementStore("checkpoint", root=root / "library")

            class FakeSession:
                def load_closed_session(self, *, session_id):
                    self.session_id = session_id
                    self.session_dir = str(session_dirs[session_id])
                    self.tool_library_id = "checkpoint"
                    self.allow_training_updates = True
                    self.task_description = "Inspect route state."
                    self.llm_backend = ""
                    self.model = ""
                    return self

            with (
                patch("agent.tool_refinement.curator.Session", FakeSession),
                patch(
                    "agent.tool_refinement.curator.ToolRefinementStore",
                    return_value=store,
                ),
            ):
                collected = finalize_tool_refinement_session(
                    session_id="s1",
                    metrics={"rca_f1": 1.0},
                    rewrite=False,
                    min_new_trials=2,
                )
                updated = finalize_tool_refinement_session(
                    session_id="s2",
                    metrics={"rca_f1": 1.0},
                    rewrite=True,
                    min_new_trials=2,
                )

            state = store.load()

        self.assertEqual(collected["status"], "collected")
        self.assertEqual(collected["draft_pending_trials"], 1)
        self.assertEqual(updated["draft_selected_tools"], ["show_route"])
        self.assertEqual(updated["draft_pending_trials"], 0)
        self.assertEqual(len(state.processed_trial_ids), 2)


    def test_evaluation_finalizer_skips_without_mutating_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_dir = root / "evaluation"
            session_dir.mkdir()
            self._trace(
                str(session_dir),
                [
                    ("tool_start", "1", "show_route", "{'router': 'r1'}", ""),
                    ("tool_end", "1", "show_route", "", "route present"),
                ],
            )
            store = ToolRefinementStore("frozen", root=root / "library")
            state = store.load()
            state.documents["show_route"] = ToolDocumentation(
                name="show_route",
                description="Show routes.",
            )
            store.save(state)
            before = store.state_hash()

            class FakeSession:
                def load_closed_session(self, *, session_id):
                    self.session_id = session_id
                    self.session_dir = str(session_dir)
                    self.tool_library_id = "frozen"
                    self.allow_training_updates = False
                    return self

            with (
                patch("agent.tool_refinement.curator.Session", FakeSession),
                patch(
                    "agent.tool_refinement.curator.ToolRefinementStore",
                    return_value=store,
                ),
            ):
                report = finalize_tool_refinement_session(
                    session_id="evaluation",
                    metrics={"rca_f1": 1.0},
                )

            after = store.state_hash()

        self.assertEqual(report["status"], "skipped")
        self.assertEqual(before, after)


    def test_stable_published_document_skips_redundant_success_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_dir = root / "s1"
            session_dir.mkdir()
            self._trace(
                str(session_dir),
                [
                    (
                        "tool_start",
                        "1",
                        "show_route",
                        "{'router': 'r1'}",
                        "",
                    ),
                    ("tool_end", "1", "show_route", "", "route present"),
                ],
            )
            store = ToolRefinementStore("stable", root=root / "library")
            state = store.load()
            state.documents["show_route"] = ToolDocumentation(
                name="show_route",
                description="Show routes.",
                published=True,
                diagnostic_utility_score=1.0,
                diagnostic_utility_count=2,
                diagnostic_utility_sessions=["old-1", "old-2"],
            )
            store.save(state)

            class FakeSession:
                def load_closed_session(self, *, session_id):
                    self.session_id = session_id
                    self.session_dir = str(session_dir)
                    self.tool_library_id = "stable"
                    self.allow_training_updates = True
                    self.task_description = "Inspect route state."
                    self.llm_backend = ""
                    self.model = ""
                    return self

            with (
                patch("agent.tool_refinement.curator.Session", FakeSession),
                patch(
                    "agent.tool_refinement.curator.ToolRefinementStore",
                    return_value=store,
                ),
            ):
                report = finalize_tool_refinement_session(
                    session_id="s1",
                    metrics={
                        "detection_score": 1.0,
                        "localization_f1": 1.0,
                        "rca_f1": 1.0,
                    },
                    rewrite=True,
                    min_new_trials=1,
                )

            state = store.load()

        self.assertEqual(report["draft_selected_tools"], [])
        self.assertEqual(report["draft_pending_trials"], 0)
        self.assertEqual(len(state.processed_trial_ids), 1)
        self.assertEqual(state.revisions, [])

