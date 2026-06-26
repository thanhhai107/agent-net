"""Unit tests for persistent diagnostic tool evolution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool

from agent.langgraph.domain_agents.diagnosis_agent import DiagnosisAgent
from agent.tool_evolution.curator import (
    _composite_outcomes,
    _distill_trace,
    _minimal_successful_trace,
    _paired_primitive_calls,
    _sanitize_value,
    finalize_tool_evolution_session,
)
from agent.tool_evolution.models import (
    CompositeStep,
    CompositeTool,
    ToolEvolutionMode,
    ToolParameter,
    ToolUsageExample,
    ValidationEvidence,
)
from agent.tool_evolution.runtime import (
    COMPOSABLE_PRIMITIVE_TOOLS,
    NON_COMPOSABLE_PRIMITIVE_TOOLS,
    ToolEvolutionRuntime,
    _tool_output_is_error,
    _validate_composite_arguments,
    _validate_step_argument_policy,
)
from agent.tool_evolution.store import ToolEvolutionStore
from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers
from nika.workflows.eval.summary import run_eval_summary
from nika.workflows.benchmark.run import (
    _benchmark_row_cli_args,
    run_single_benchmark,
)
from nika.service.mcp_server import tool_evolution_mcp_server
from nika.utils.session import Session
from nika.utils.session_store import SessionStore


def _composite(name: str = "workflow_host_config") -> CompositeTool:
    return CompositeTool(
        name=name,
        description="Collect reusable host network configuration evidence.",
        parameters=[
            ToolParameter(
                name="host",
                description="Target network device.",
            )
        ],
        steps=[
            CompositeStep(
                tool="get_host_net_config",
                arguments={"host_name": "${host}"},
            )
        ],
    )


class ToolEvolutionModuleBoundaryTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _tool(name: str) -> StructuredTool:
        def invoke() -> str:
            return name

        return StructuredTool.from_function(
            invoke,
            name=name,
            description=f"Test tool {name}.",
        )

    async def test_read_only_primitive_surface_is_identical_when_disabled(self) -> None:
        safe = self._tool("get_reachability")
        unsafe = self._tool("exec_shell")
        agent = DiagnosisAgent.__new__(DiagnosisAgent)
        agent.client = MagicMock()
        agent.client.get_tools = AsyncMock(return_value=[safe, unsafe])
        agent.tool_evolution_enabled = False
        agent.tool_evolution_runtime = None

        await agent.load_tools()

        self.assertEqual([tool.name for tool in agent.tools], ["get_reachability"])

    async def test_enabled_module_wraps_the_same_read_only_primitives(self) -> None:
        safe = self._tool("get_reachability")
        unsafe = self._tool("exec_shell")
        manager = self._tool("search_diagnostic_tools")
        agent = DiagnosisAgent.__new__(DiagnosisAgent)
        agent.client = MagicMock()
        agent.client.get_tools = AsyncMock(return_value=[safe, unsafe])
        agent.tool_evolution_enabled = True
        agent.tool_evolution_runtime = None
        agent.session_id = "session"
        agent.tool_library_id = "library"
        agent.tool_evolution_mode = ToolEvolutionMode.DUAL
        agent.model = "model"
        runtime = MagicMock()
        runtime.build_tools.return_value = [safe, manager]

        with (
            patch(
                "agent.langgraph.domain_agents.diagnosis_agent.Session"
            ) as session_cls,
            patch(
                "agent.langgraph.domain_agents.diagnosis_agent.ToolEvolutionRuntime",
                return_value=runtime,
            ) as runtime_cls,
        ):
            session_cls.return_value.load_running_session.return_value = (
                SimpleNamespace(task_description="task")
            )
            await agent.load_tools()

        primitives = runtime_cls.call_args.kwargs["primitive_tools"]
        self.assertEqual([tool.name for tool in primitives], ["get_reachability"])
        self.assertEqual(
            [tool.name for tool in agent.tools],
            ["get_reachability", "search_diagnostic_tools"],
        )


class ToolEvolutionStoreTest(unittest.TestCase):
    def test_deduplicates_equivalent_composites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            first, created_first = store.register_composite(_composite())
            duplicate = CompositeTool(
                name="workflow_duplicate",
                description="Equivalent workflow with a renamed input parameter.",
                parameters=[
                    ToolParameter(
                        name="device",
                        description="Target network device.",
                    )
                ],
                steps=[
                    CompositeStep(
                        tool="get_host_net_config",
                        arguments={"host_name": "${device}"},
                    )
                ],
            )
            second, created_second = store.register_composite(duplicate)

        self.assertTrue(created_first)
        self.assertFalse(created_second)
        self.assertEqual(first.name, second.name)

    def test_promotes_only_after_distinct_successful_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            tool, _ = store.register_composite(_composite())
            first = store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="simple_bgp:fixed",
                    execution_success=True,
                    incident_success=True,
                    source="distillation",
                ),
            )
            second = store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="dc_clos_bgp:s",
                    execution_success=True,
                    incident_success=True,
                    source="runtime",
                ),
            )
            third = store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="dc_clos_bgp:m",
                    execution_success=True,
                    incident_success=True,
                    source="replay",
                ),
            )

        self.assertEqual(first.status, "candidate")
        self.assertEqual(second.status, "candidate")
        self.assertEqual(third.status, "promoted")

    def test_no_validation_ablation_promotes_after_first_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            tool, _ = store.register_composite(_composite())
            updated = store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="simple_bgp:fixed",
                    execution_success=True,
                    incident_success=True,
                ),
                validation_enabled=False,
            )

        self.assertEqual(updated.status, "promoted")

    def test_failed_execution_demotes_and_repeated_failure_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            tool, _ = store.register_composite(_composite())
            for context in ("simple_bgp:fixed", "dc_clos_bgp:s"):
                promoted = store.record_composite_evidence(
                    tool.name,
                    ValidationEvidence(
                        context_fingerprint=context,
                        execution_success=True,
                        incident_success=True,
                    ),
                )
            demoted = store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="dc_clos_bgp:m",
                    execution_success=False,
                    incident_success=False,
                ),
            )
            rejected = store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="dc_clos_bgp:l",
                    execution_success=False,
                    incident_success=False,
                ),
            )

        self.assertEqual(promoted.status, "promoted")
        self.assertEqual(demoted.status, "candidate")
        self.assertEqual(rejected.status, "rejected")

    def test_failure_then_success_in_same_context_are_distinct_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            tool, _ = store.register_composite(_composite())
            store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="simple_bgp:fixed",
                    execution_success=False,
                    incident_success=False,
                ),
            )
            recovered = store.record_composite_evidence(
                tool.name,
                ValidationEvidence(
                    context_fingerprint="simple_bgp:fixed",
                    execution_success=True,
                    incident_success=False,
                ),
            )

        self.assertEqual(len(recovered.evidence), 2)
        self.assertEqual(recovered.execution_count, 2)
        self.assertEqual(recovered.success_count, 1)

    def test_mastery_overlay_keeps_sanitized_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            mastery = store.upsert_mastery(
                "ping_pair",
                parameter_guidance=["Use valid source and destination device names."],
                usage_example=ToolUsageExample(
                    arguments={"host_a": "<device>", "host_b": "<device>"},
                    succeeded=True,
                ),
                calls=1,
                successes=1,
                source_model="model-a",
            )

        self.assertIn("Parameter guidance", mastery.agent_overlay())
        self.assertNotIn("pc1", mastery.model_dump_json())

    def test_tool_card_versions_only_when_semantics_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            first = store.upsert_mastery(
                "ping_pair",
                parameter_guidance=["Provide two reachable endpoint names."],
                rationale="Analyzer found one successful invocation.",
            )
            converged = store.upsert_mastery(
                "ping_pair",
                parameter_guidance=["Provide two reachable endpoint names."],
                rationale="Repeated evidence did not change the card.",
            )
            revised = store.upsert_mastery(
                "ping_pair",
                failure_semantics=["A timeout does not by itself localize the fault."],
                rationale="Analyzer observed a timeout.",
            )

        self.assertEqual(first.version, 1)
        self.assertEqual(converged.convergence_count, 1)
        self.assertEqual(revised.version, 2)
        self.assertEqual(len(revised.revisions), 2)

    def test_frozen_retrieval_does_not_mutate_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            tool, _ = store.register_composite(_composite())

            selected = store.search_composites(
                "host configuration",
                record_usage=False,
            )
            frozen = store.get_composite(tool.name)

        self.assertEqual([item.name for item in selected], [tool.name])
        self.assertEqual(frozen.retrieval_count, 0)
        self.assertIsNone(frozen.last_used_at)

    def test_capacity_prunes_low_utility_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp, capacity=10)
            names = []
            for index in range(11):
                composite = CompositeTool(
                    name=f"workflow_capacity_{index}",
                    description=f"Collect diagnostic evidence for capability {index}.",
                    parameters=[
                        ToolParameter(
                            name="host",
                            description="Target network device.",
                        )
                    ],
                    steps=[
                        CompositeStep(
                            tool="get_host_net_config",
                            arguments={
                                "host_name": "${host}",
                                "variant": index,
                            },
                        )
                    ],
                )
                registered, _ = store.register_composite(composite)
                names.append(registered.name)
            state = store.load()

        self.assertEqual(len(state.composites), 10)
        self.assertNotIn(names[0], state.composites)


class CompositeValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = ToolEvolutionRuntime.__new__(ToolEvolutionRuntime)
        self.runtime.session = SimpleNamespace(session_id="session-123")
        self.runtime._known_devices = {"pc1", "router1"}
        self.runtime.primitive_tools = {
            "get_host_net_config": object(),
            "exec_shell": object(),
        }

    def test_rejects_mutating_or_arbitrary_shell_primitives(self) -> None:
        unsafe = CompositeTool(
            name="workflow_unsafe",
            description="Execute an arbitrary command against a target device.",
            parameters=[
                ToolParameter(name="host", description="Target device."),
                ToolParameter(name="command", description="Command to execute."),
            ],
            steps=[
                CompositeStep(
                    tool="exec_shell",
                    arguments={
                        "host_name": "${host}",
                        "command": "${command}",
                    },
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.runtime.validate_composite(unsafe)

    def test_rejects_non_composable_observation_primitives(self) -> None:
        for tool_name in NON_COMPOSABLE_PRIMITIVE_TOOLS:
            composite = CompositeTool(
                name=f"workflow_{tool_name}",
                description="Try to persist a non-composable diagnostic action.",
                parameters=[
                    ToolParameter(name="host", description="Target device."),
                ],
                steps=[
                    CompositeStep(
                        tool=tool_name,
                        arguments={"host_name": "${host}"},
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "unsafe|unsupported"):
                self.runtime.validate_composite(composite)

    def test_composable_set_is_stricter_than_live_safe_surface(self) -> None:
        self.assertFalse(NON_COMPOSABLE_PRIMITIVE_TOOLS & COMPOSABLE_PRIMITIVE_TOOLS)
        self.assertIn("cat_file", NON_COMPOSABLE_PRIMITIVE_TOOLS)

    def test_rejects_hard_coded_incident_values(self) -> None:
        leaked = CompositeTool(
            name="workflow_leaked_host",
            description="Inspect pc1 network configuration for this incident.",
            steps=[
                CompositeStep(
                    tool="get_host_net_config",
                    arguments={"host_name": "pc1"},
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "hard-coded device"):
            self.runtime.validate_composite(leaked)

    def test_accepts_parameterized_read_only_workflow(self) -> None:
        self.runtime.validate_composite(_composite())

    def test_rejects_shell_control_literals(self) -> None:
        unsafe = CompositeTool(
            name="workflow_injected_args",
            description="Inspect one target with unsafe literal arguments.",
            parameters=[
                ToolParameter(name="host", description="Target device."),
            ],
            steps=[
                CompositeStep(
                    tool="get_host_net_config",
                    arguments={"host_name": "${host}; shutdown -h now"},
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "shell control"):
            self.runtime.validate_composite(unsafe)

    def test_rejects_root_cause_labels_in_persistent_lessons(self) -> None:
        with self.assertRaisesRegex(ValueError, "root-cause labels"):
            self.runtime._validate_persistent_text(
                "Use this specifically for link_down incidents."
            )

    def test_topology_endpoints_expose_device_names_for_sanitization(self) -> None:
        runtime = ToolEvolutionRuntime.__new__(ToolEvolutionRuntime)
        runtime.session = SimpleNamespace(
            topology=[
                ("router1:eth0", "router2:eth0"),
                ("router1:eth1", "pc1:eth0"),
            ]
        )

        devices = runtime._collect_devices()

        self.assertIn("router1", devices)
        self.assertIn("pc1", devices)
        self.assertIn("router1:eth0", devices)

    def test_capability_gap_rejects_bare_device_names(self) -> None:
        runtime = ToolEvolutionRuntime.__new__(ToolEvolutionRuntime)
        runtime._known_devices = {"router1", "router2"}

        with self.assertRaisesRegex(ValueError, "generalized role"):
            runtime._parse_gap_items(
                json.dumps(
                    [
                        {
                            "name": "router1",
                            "type": "string",
                            "description": "First router.",
                        }
                    ]
                ),
                "required_inputs_json",
            )

    def test_capability_gap_generalizes_devices_inside_descriptions(self) -> None:
        runtime = ToolEvolutionRuntime.__new__(ToolEvolutionRuntime)
        runtime._known_devices = {"router1", "router2"}

        values = runtime._parse_gap_items(
            json.dumps(["Collect router1 interface counters"]),
            "expected_observations_json",
        )

        self.assertEqual(values, ["Collect <device> interface counters"])

    def test_composite_arguments_enforce_required_unknown_and_types(self) -> None:
        composite = _composite()

        with self.assertRaisesRegex(ValueError, "missing required"):
            _validate_composite_arguments(composite, {})
        with self.assertRaisesRegex(ValueError, "unknown composite"):
            _validate_composite_arguments(
                composite,
                {"host": "pc9", "extra": "value"},
            )
        with self.assertRaisesRegex(ValueError, "must be of type str"):
            _validate_composite_arguments(composite, {"host": 9})
        self.assertEqual(
            _validate_composite_arguments(composite, {"host": "pc9"}),
            {"host": "pc9"},
        )

    def test_tool_parameter_accepts_json_schema_type_names(self) -> None:
        aliases = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
        }

        for raw, normalized in aliases.items():
            parameter = ToolParameter(
                name=f"value_{normalized}",
                type=raw,
                description="Generic typed value.",
            )
            self.assertEqual(parameter.type, normalized)

    def test_rejects_arguments_not_in_primitive_schema(self) -> None:
        async def ping_pair(host_a: str, host_b: str) -> dict:
            return {"reachable": True}

        self.runtime.primitive_tools["ping_pair"] = StructuredTool.from_function(
            coroutine=ping_pair,
            name="ping_pair",
            description="Ping between two hosts.",
        )
        invalid = CompositeTool(
            name="workflow_wrong_ping_schema",
            description="Collect pairwise reachability evidence.",
            parameters=[
                ToolParameter(name="source", description="Source host."),
                ToolParameter(name="destination", description="Destination host."),
            ],
            steps=[
                CompositeStep(
                    tool="ping_pair",
                    arguments={
                        "source": "${source}",
                        "destination": "${destination}",
                    },
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing argument"):
            self.runtime.validate_composite(invalid)

    def test_rejects_unbounded_composite_command_arguments(self) -> None:
        async def ping_pair(
            host_a: str,
            host_b: str,
            count: int = 4,
            args: str = "",
        ) -> dict:
            return {"reachable": True}

        self.runtime.primitive_tools["ping_pair"] = StructuredTool.from_function(
            coroutine=ping_pair,
            name="ping_pair",
            description="Ping between two hosts.",
        )

        unsafe_ping = CompositeTool(
            name="workflow_bad_ping_args",
            description="Collect pairwise reachability evidence.",
            parameters=[
                ToolParameter(name="source", description="Source host."),
                ToolParameter(name="destination", description="Destination host."),
            ],
            steps=[
                CompositeStep(
                    tool="ping_pair",
                    arguments={
                        "host_a": "${source}",
                        "host_b": "${destination}",
                        "count": 100,
                    },
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "ping_pair.count"):
            self.runtime.validate_composite(unsafe_ping)
        with self.assertRaisesRegex(ValueError, "ping_pair.args"):
            _validate_step_argument_policy(
                "ping_pair",
                {
                    "host_a": "${source}",
                    "host_b": "${destination}",
                    "args": "${flags}",
                },
                allow_placeholders=True,
            )

    def test_detects_mcp_error_string_and_nested_content(self) -> None:
        error = (
            "Error executing tool ping_pair: 2 validation errors for "
            "ping_pairArguments"
        )

        self.assertTrue(_tool_output_is_error(error))
        self.assertTrue(_tool_output_is_error({"content": [error]}))
        self.assertFalse(_tool_output_is_error("packet loss: 0%"))


class TraceDistillationTest(unittest.TestCase):
    def test_distillation_parameterizes_concrete_values(self) -> None:
        calls = [
            {
                "tool": "get_host_net_config",
                "arguments": {"host_name": "pc1"},
                "succeeded": True,
            },
            {
                "tool": "ping_pair",
                "arguments": {"host_a": "pc1", "host_b": "pc2", "count": 4},
                "succeeded": True,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            name, created = _distill_trace(
                store,
                calls,
                scenario_name="simple_bgp",
                deduplicate=True,
                context_fingerprint="simple_bgp:fixed",
            )
            state = store.load()

        self.assertTrue(created)
        self.assertIsNotNone(name)
        serialized = state.composites[name].model_dump_json()
        self.assertNotIn("pc1", serialized)
        self.assertNotIn("pc2", serialized)
        self.assertIn("${host_name}", serialized)
        self.assertEqual(state.composites[name].evidence, [])
        self.assertIn("${host_name}", state.composites[name].steps[1].arguments["host_a"])

    def test_minimal_trace_removes_failed_noncomposable_and_duplicates(self) -> None:
        calls = [
            {"tool": "cat_file", "arguments": {"host_name": "x", "file_path": "/etc/passwd"}, "succeeded": True},
            {"tool": "ping_pair", "arguments": {"a": "x"}, "succeeded": True},
            {"tool": "ping_pair", "arguments": {"a": "x"}, "succeeded": True},
            {"tool": "ping_pair", "arguments": {"host_a": "x", "host_b": "y", "count": 999}, "succeeded": True},
            {"tool": "netstat", "arguments": {"host": "x"}, "succeeded": False},
            {"tool": "netstat", "arguments": {"host": "y"}, "succeeded": True},
        ]

        selected = _minimal_successful_trace(calls)

        self.assertEqual([item["tool"] for item in selected], ["ping_pair", "netstat"])

    def test_parallel_calls_are_correlated_by_run_id(self) -> None:
        events = [
            {
                "agent": "diagnosis_agent",
                "event": "tool_start",
                "tool": {"name": "ping_pair"},
                "input": '{"host_a": "a", "host_b": "b"}',
                "run_id": "run-a",
            },
            {
                "agent": "diagnosis_agent",
                "event": "tool_start",
                "tool": {"name": "netstat"},
                "input": '{"host_name": "a"}',
                "run_id": "run-b",
            },
            {
                "agent": "diagnosis_agent",
                "event": "tool_error",
                "error": "timeout",
                "run_id": "run-b",
            },
            {
                "agent": "diagnosis_agent",
                "event": "tool_end",
                "output": "reachable",
                "run_id": "run-a",
            },
        ]

        calls = _paired_primitive_calls(events)

        self.assertEqual(calls[0]["tool"], "netstat")
        self.assertFalse(calls[0]["succeeded"])
        self.assertEqual(calls[1]["tool"], "ping_pair")
        self.assertTrue(calls[1]["succeeded"])

    def test_ephemeral_execution_is_counted_once_and_aliased_after_dedup(self) -> None:
        events = [
            {
                "event": "tool_evolution_composite_start",
                "name": "workflow_ephemeral",
                "status": "ephemeral",
            },
            {
                "event": "tool_evolution_composite_end",
                "name": "workflow_ephemeral",
            },
            {
                "event": "tool_evolution_candidate_verified",
                "source_name": "workflow_ephemeral",
                "name": "workflow_existing",
            },
        ]

        successes, errors, reuse_count = _composite_outcomes(events)

        self.assertEqual(successes, ["workflow_existing"])
        self.assertEqual(errors, [])
        self.assertEqual(reuse_count, 0)

    def test_mastery_examples_redact_incident_identifiers(self) -> None:
        sanitized = _sanitize_value(
            {
                "host": "pc1",
                "path": "/tmp/session-123/link_down.txt",
                "address": "10.0.0.1",
            },
            {"pc1"},
            {"session-123", "link_down"},
        )

        serialized = json.dumps(sanitized)
        self.assertNotIn("pc1", serialized)
        self.assertNotIn("session-123", serialized)
        self.assertNotIn("link_down", serialized)
        self.assertNotIn("10.0.0.1", serialized)


class TestTimeSynthesisTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _runtime_for_mode(mode: ToolEvolutionMode, tmp: str) -> ToolEvolutionRuntime:
        async def fake_get_host_net_config(host_name: str) -> dict:
            return {"host_name": host_name, "state": "up"}

        primitive = StructuredTool.from_function(
            coroutine=fake_get_host_net_config,
            name="get_host_net_config",
            description="Read host network configuration.",
        )
        session = SimpleNamespace(
            session_id="session-123",
            session_dir=tmp,
            scenario_name="simple_bgp",
            scenario_topo_size=None,
            topology=[],
            tool_evolution_update_enabled=True,
        )
        return ToolEvolutionRuntime(
            session=session,
            primitive_tools=[primitive],
            library_id=f"experiment-{mode.value}",
            mode=mode,
            model="model-a",
        )

    def test_manager_tools_match_evolution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mastery = self._runtime_for_mode(
                ToolEvolutionMode.MASTERY,
                tmp,
            )
            distill = self._runtime_for_mode(
                ToolEvolutionMode.DISTILL,
                tmp,
            )
            dual = self._runtime_for_mode(ToolEvolutionMode.DUAL, tmp)

        mastery_tools = {tool.name for tool in mastery._build_manager_tools()}
        distill_tools = {tool.name for tool in distill._build_manager_tools()}
        dual_tools = {tool.name for tool in dual._build_manager_tools()}

        self.assertIn("record_tool_lesson", mastery_tools)
        self.assertNotIn("propose_composite_tool", mastery_tools)
        self.assertIn("propose_composite_tool", distill_tools)
        self.assertNotIn("record_tool_lesson", distill_tools)
        self.assertIn("record_tool_lesson", dual_tools)
        self.assertIn("propose_composite_tool", dual_tools)

    async def test_ephemeral_tool_is_persisted_only_after_verification(self) -> None:
        async def fake_get_host_net_config(host_name: str) -> dict:
            return {"host_name": host_name, "state": "up"}

        primitive = StructuredTool.from_function(
            coroutine=fake_get_host_net_config,
            name="get_host_net_config",
            description="Read host network configuration.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            session = SimpleNamespace(
                session_id="session-123",
                session_dir=tmp,
                scenario_name="simple_bgp",
                scenario_topo_size=None,
                topology=[("pc1", "r1")],
                tool_evolution_update_enabled=True,
            )
            with patch(
                "agent.tool_evolution.runtime.ToolEvolutionStore",
                return_value=store,
            ):
                runtime = ToolEvolutionRuntime(
                    session=session,
                    primitive_tools=[primitive],
                    library_id="experiment",
                    mode=ToolEvolutionMode.DUAL,
                    model="model-a",
                    task_description="Investigate connectivity.",
                )
            tools = {tool.name: tool for tool in runtime._build_manager_tools()}
            gap_result = await tools["identify_capability_gap"].ainvoke(
                {
                    "description": "Collect reusable host configuration evidence.",
                    "required_inputs_json": '["target host"]',
                    "expected_observations_json": '["interface configuration"]',
                }
            )
            gap_id = gap_result.split("'")[1]
            proposed = await tools["propose_composite_tool"].ainvoke(
                {
                    "name": "workflow_ephemeral_host",
                    "description": "Collect reusable host configuration evidence.",
                    "gap_id": gap_id,
                    "parameters_json": json.dumps(
                        [{"name": "host", "description": "Target network device."}]
                    ),
                    "steps_json": json.dumps(
                        [
                            {
                                "tool": "get_host_net_config",
                                "arguments": {"host_name": "${host}"},
                                "label": "Read configuration.",
                            }
                        ]
                    ),
                }
            )

            self.assertIn("ephemeral", proposed)
            self.assertEqual(store.load().composites, {})

            output = await tools["execute_candidate_tool"].ainvoke(
                {
                    "name": "workflow_ephemeral_host",
                    "arguments_json": '{"host": "pc9"}',
                }
            )
            persisted = store.get_composite("workflow_ephemeral_host")
            gap = store.load().capability_gaps[gap_id]

        self.assertIn("persisted_as", output)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, "candidate")
        self.assertEqual(
            persisted.output_contract,
            ["interface configuration"],
        )
        self.assertEqual(gap.status, "resolved")
        self.assertTrue(
            any(
                report.stage == "semantic" and report.passed
                for report in persisted.verification_reports
            )
        )

    async def test_empty_runtime_output_does_not_persist_ephemeral_tool(self) -> None:
        async def empty_tool(host_name: str) -> dict:
            return {}

        primitive = StructuredTool.from_function(
            coroutine=empty_tool,
            name="get_host_net_config",
            description="Read host network configuration.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            session = SimpleNamespace(
                session_id="session-123",
                session_dir=tmp,
                scenario_name="simple_bgp",
                scenario_topo_size=None,
                topology=[],
                tool_evolution_update_enabled=True,
            )
            with patch(
                "agent.tool_evolution.runtime.ToolEvolutionStore",
                return_value=store,
            ):
                runtime = ToolEvolutionRuntime(
                    session=session,
                    primitive_tools=[primitive],
                    library_id="experiment",
                    mode=ToolEvolutionMode.DUAL,
                    model="model-a",
                )
            tools = {tool.name: tool for tool in runtime._build_manager_tools()}
            gap_result = await tools["identify_capability_gap"].ainvoke(
                {
                    "description": "Collect reusable host configuration evidence.",
                    "required_inputs_json": '["target host"]',
                    "expected_observations_json": '["interface configuration"]',
                }
            )
            gap_id = gap_result.split("'")[1]
            await tools["propose_composite_tool"].ainvoke(
                {
                    "name": "workflow_empty_output",
                    "description": "Collect reusable host configuration evidence.",
                    "gap_id": gap_id,
                    "parameters_json": json.dumps(
                        [{"name": "host", "description": "Target network device."}]
                    ),
                    "steps_json": json.dumps(
                        [
                            {
                                "tool": "get_host_net_config",
                                "arguments": {"host_name": "${host}"},
                            }
                        ]
                    ),
                }
            )
            output = await tools["execute_candidate_tool"].ainvoke(
                {
                    "name": "workflow_empty_output",
                    "arguments_json": '{"host": "pc9"}',
                }
            )

        self.assertIn("no informative primitive output", output)
        self.assertIsNone(store.get_composite("workflow_empty_output"))


class CurationIdempotencyTest(unittest.TestCase):
    def test_repeated_finalization_does_not_reapply_mastery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "session-1"
            session_dir.mkdir()
            events = [
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_start",
                    "tool": {"name": "get_host_net_config"},
                    "input": '{"host_name": "pc1"}',
                    "run_id": "run-1",
                },
                {
                    "agent": "diagnosis_agent",
                    "event": "tool_end",
                    "output": {"interfaces": ["eth0"]},
                    "run_id": "run-1",
                },
            ]
            (session_dir / "messages.jsonl").write_text(
                "\n".join(json.dumps(item) for item in events) + "\n",
                encoding="utf-8",
            )
            session = SimpleNamespace(
                session_id="session-1",
                session_dir=str(session_dir),
                agent_type="react",
                tool_evolution_enabled=True,
                tool_library_id="experiment",
                tool_evolution_mode="mastery",
                tool_evolution_update_enabled=True,
                scenario_name="simple_bgp",
                scenario_topo_size=None,
                topology=[("pc1", "r1")],
                problem_names=[],
                model="model-a",
            )
            loader = SimpleNamespace(
                load_closed_session=lambda session_id: session
            )
            store = ToolEvolutionStore("experiment", root=tmp)
            metrics = {
                "detection_score": 1.0,
                "localization_accuracy": 1.0,
                "rca_accuracy": 1.0,
            }
            with (
                patch(
                    "agent.tool_evolution.curator.Session",
                    return_value=loader,
                ),
                patch(
                    "agent.tool_evolution.curator.ToolEvolutionStore",
                    return_value=store,
                ),
            ):
                first = finalize_tool_evolution_session(
                    session_id="session-1",
                    metrics=metrics,
                )
                second = finalize_tool_evolution_session(
                    session_id="session-1",
                    metrics=metrics,
                )
            mastery = store.load().mastery["get_host_net_config"]

        self.assertEqual(first, second)
        self.assertEqual(mastery.calls, 1)


class NonOracleRoutingTest(unittest.TestCase):
    def test_problem_labels_are_ignored_without_oracle_flag(self) -> None:
        normal = select_diagnosis_servers(
            "generic_scenario",
            ["bgp_asn_misconfig"],
        )
        oracle = select_diagnosis_servers(
            "generic_scenario",
            ["bgp_asn_misconfig"],
            oracle=True,
        )

        self.assertNotIn("kathara_frr_mcp_server", normal)
        self.assertIn("kathara_frr_mcp_server", oracle)

    def test_p4_int_selects_bmv2_and_telemetry_from_scenario(self) -> None:
        servers = select_diagnosis_servers("p4_int", [])
        self.assertIn("kathara_bmv2_mcp_server", servers)
        self.assertIn("kathara_telemetry_mcp_server", servers)

    def test_toolbox_mcp_config_propagates_session_and_library(self) -> None:
        config = MCPServerConfig("session-123").load_toolbox_config("experiment")
        server = config["nika_diagnostic_toolbox"]
        self.assertEqual(server["env"]["NIKA_SESSION_ID"], "session-123")
        self.assertEqual(server["env"]["NIKA_TOOL_LIBRARY_ID"], "experiment")


class SessionTopologyPersistenceTest(unittest.TestCase):
    def test_running_and_closed_session_keep_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session()
            session.store = SessionStore(root / "sessions")
            with patch("nika.utils.session.RESULTS_DIR", root / "results"):
                session.init_session(
                    session_id="session-topology",
                    scenario_name="simple_bgp",
                    lab_name="test-lab",
                    scenario_topo_size=None,
                    topology=[("pc1", "router1"), ("router1", "router2")],
                )
                loaded = Session()
                loaded.store = session.store
                loaded.load_running_session("session-topology")
                loaded.clear_session()
                closed = Session()
                closed.store = session.store
                closed.load_closed_session("session-topology")

        expected = [["pc1", "router1"], ["router1", "router2"]]
        self.assertEqual(loaded.topology, expected)
        self.assertEqual(closed.topology, expected)


class EvolutionSummaryTest(unittest.TestCase):
    def test_summary_computes_sequential_efficiency_and_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (tokens, rca) in enumerate(((100, 0.0), (80, 1.0))):
                session_dir = root / f"session-{index}"
                session_dir.mkdir()
                (session_dir / "run.json").write_text(
                    json.dumps(
                        {
                            "status": "finished",
                            "session_id": f"session-{index}",
                            "agent_type": "react",
                            "tool_evolution_enabled": True,
                            "model": "model",
                            "scenario_name": "simple_bgp",
                            "scenario_topo_size": None,
                            "root_cause_name": "link_down",
                            "root_cause_category": "link_failure",
                            "tool_library_id": "experiment",
                            "evolution_stream": "link",
                            "evolution_sequence_index": index,
                        }
                    ),
                    encoding="utf-8",
                )
                (session_dir / "ground_truth.json").write_text(
                    json.dumps(
                        {
                            "is_anomaly": True,
                            "faulty_devices": ["pc1"],
                            "root_cause_name": ["link_down"],
                        }
                    ),
                    encoding="utf-8",
                )
                (session_dir / "eval_metrics.json").write_text(
                    json.dumps(
                        {
                            "in_tokens": tokens,
                            "out_tokens": 0,
                            "detection_score": 1.0,
                            "localization_accuracy": 1.0,
                            "rca_accuracy": rca,
                        }
                    ),
                    encoding="utf-8",
                )
            output = root / "summary.csv"
            run_eval_summary(results_dir=str(root), output_path=str(output))
            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertIn("efficiency_evolution_rate", rows[0])
        self.assertIn("tool_card_revisions", rows[0])
        self.assertIn("verified_composites", rows[0])
        self.assertIn("-0.2", rows[2])
        self.assertIn("0.3333", rows[2])


class ToolEvolutionBenchmarkArgsTest(unittest.TestCase):
    def test_stream_cli_args_preserve_library_and_skip_dash_tier(self) -> None:
        args = _benchmark_row_cli_args(
            {
                "problem": "link_down",
                "scenario": "simple_bgp",
                "topo_size": "-",
            },
            agent_type="react",
            llm_backend="openai",
            model="model",
            max_steps=10,
            max_attempts=2,
            run_judge=False,
            judge_llm_backend=None,
            judge_model=None,
            oracle_routing=False,
            tool_evolution_enabled=True,
            tool_library_id="experiment",
            tool_evolution_mode="dual",
        )

        self.assertIn("--tool-library", args)
        self.assertIn("--tool-evolution", args)
        self.assertIn("experiment", args)
        self.assertNotIn("-t", args)

    def test_failed_benchmark_closes_environment(self) -> None:
        session = SimpleNamespace(update_session=MagicMock())
        loader = MagicMock()
        loader.load_running_session.return_value = session
        with (
            patch(
                "nika.workflows.benchmark.run.start_net_env",
                return_value="session-failed",
            ),
            patch("nika.utils.session.Session", return_value=loader),
            patch("nika.workflows.benchmark.run.inject_failure"),
            patch(
                "nika.workflows.benchmark.run.start_agent",
                side_effect=RuntimeError("provider failed"),
            ),
            patch("nika.workflows.benchmark.run.close_session") as close,
        ):
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                run_single_benchmark(
                    problem="link_down",
                    scenario="simple_bgp",
                    topo_size="",
                    agent_type="react",
                    llm_backend="netmind",
                    model="model",
                    max_steps=10,
                )

        close.assert_called_once_with(
            session_id="session-failed",
            undeploy=True,
        )


class ToolEvolutionMcpAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_lists_and_executes_persistent_composite(self) -> None:
        async def fake_get_host_net_config(host_name: str) -> dict:
            return {"host_name": host_name, "state": "up"}

        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            promoted = _composite().model_copy(update={"status": "promoted"})
            tool, _ = store.register_composite(promoted)
            with (
                patch.object(
                    tool_evolution_mcp_server,
                    "ToolEvolutionStore",
                    return_value=store,
                ),
                patch.dict(
                    tool_evolution_mcp_server._PRIMITIVES,
                    {"get_host_net_config": fake_get_host_net_config},
                ),
                patch.dict(
                    "os.environ",
                    {"NIKA_TOOL_LIBRARY_ID": "experiment"},
                ),
            ):
                listed = tool_evolution_mcp_server.list_evolved_tools()
                output = await tool_evolution_mcp_server.execute_evolved_tool(
                    tool.name,
                    json.dumps({"host": "pc9"}),
                )

        self.assertEqual(listed[0]["name"], tool.name)
        self.assertEqual(json.loads(output)["observations"][0]["output"]["state"], "up")

    async def test_refuses_unpromoted_composite_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolEvolutionStore("experiment", root=tmp)
            candidate, _ = store.register_composite(_composite("workflow_candidate"))
            with (
                patch.object(
                    tool_evolution_mcp_server,
                    "ToolEvolutionStore",
                    return_value=store,
                ),
                patch.dict(
                    "os.environ",
                    {"NIKA_TOOL_LIBRARY_ID": "experiment"},
                ),
            ):
                listed = tool_evolution_mcp_server.list_evolved_tools()
                output = await tool_evolution_mcp_server.execute_evolved_tool(
                    candidate.name,
                    json.dumps({"host": "pc9"}),
                )

        self.assertEqual(listed, [])
        self.assertEqual(output["error"], "tool_execution_error")
        self.assertIn("unpromoted", output["details"])


if __name__ == "__main__":
    unittest.main()
