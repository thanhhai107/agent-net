"""Runtime tool mastery overlays, retrieval, creation, and safe composition."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from agent.tool_evolution.generated_tools import (
    run_generated_tool,
    validate_generated_tool_code,
)
from agent.tool_evolution.models import (
    CapabilityGap,
    CompositeStep,
    CompositeTool,
    GeneratedTool,
    ToolEvolutionMode,
    ToolParameter,
    ToolVerificationReport,
)
from agent.tool_evolution.store import ToolEvolutionStore
from agent.utils.loggers import MessageLogger
from nika.orchestrator.problems.prob_pool import list_avail_problem_names


COMPOSABLE_PRIMITIVE_TOOLS = frozenset(
    {
        "cat_file",
        "curl_web_test",
        "ethtool",
        "exec_shell",
        "exec_shell_dual",
        "frr_exec",
        "get_reachability",
        "ping_pair",
        "get_host_net_config",
        "get_tc_statistics",
        "iperf_test",
        "netstat",
        "ip_addr_statistics",
        "systemctl_ops",
        "frr_get_bgp_conf",
        "frr_show_bgp_summary",
        "frr_show_running_config",
        "frr_show_ip_route",
        "frr_get_ospf_conf",
        "bmv2_get_log",
        "bmv2_get_counter_arrays",
        "bmv2_read_p4_program",
        "bmv2_counter_read",
        "bmv2_show_tables",
        "bmv2_table_dump",
        "bmv2_get_register_arrays",
        "bmv2_register_read",
        "influx_list_buckets",
        "influx_get_measurements",
        "influx_count_measurements",
        "influx_query_measurement",
    }
)

# The upstream NIKA diagnosis surface exposes all of these primitives directly.
# Keep tool-evolution on the same surface instead of blocking stateful or
# open-ended tools at the composability layer.
NON_COMPOSABLE_PRIMITIVE_TOOLS = frozenset()

MANAGER_TOOL_NAMES = frozenset(
    {
        "search_diagnostic_tools",
        "identify_capability_gap",
        "propose_composite_tool",
        "propose_python_tool",
        "revise_composite_tool",
        "execute_candidate_tool",
        "record_tool_lesson",
    }
)

_PLACEHOLDER = re.compile(r"\$\{([a-z][a-z0-9_]*)\}")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SHELL_CONTROL = re.compile(r"[;&|`$><\n\r]")
_NETSTAT_ALLOWED_ARGS = frozenset({"", "-tuln", "-rn", "-s", "-i", "-ant", "-anu"})


class SearchToolsInput(BaseModel):
    query: str = Field(description="Capability or diagnostic evidence needed.")


class ProposeCompositeInput(BaseModel):
    name: str
    description: str
    gap_id: str = Field(
        description="Capability gap id returned by identify_capability_gap.",
    )
    parameters_json: str = Field(
        description="JSON list of {name,type,description,required,default}."
    )
    steps_json: str = Field(
        description=(
            "JSON list of read-only steps: "
            "{tool, arguments, label}. Use ${parameter_name} placeholders."
        )
    )


class ProposePythonToolInput(BaseModel):
    name: str
    description: str
    gap_id: str = Field(
        description="Capability gap id returned by identify_capability_gap.",
    )
    parameters_json: str = Field(
        description="JSON list of {name,type,description,required,default}."
    )
    code: str = Field(
        description=(
            "Complete Python source defining a function whose name and parameters "
            "match this generated tool."
        )
    )
    output_description: str = Field(
        default="",
        description="Short description of the generated function output.",
    )
    test_example_json: str = Field(
        default="{}",
        description="Optional JSON object with sample arguments for validation.",
    )


class ExecuteCandidateInput(BaseModel):
    name: str
    arguments_json: str = Field(description="JSON object of composite arguments.")


class ReviseCompositeInput(BaseModel):
    existing_name: str
    description: str
    parameters_json: str
    steps_json: str


class RecordLessonInput(BaseModel):
    tool_name: str
    precondition: str = ""
    parameter_guidance: str = ""
    output_interpretation: str = ""
    failure_semantics: str = ""


class CapabilityGapInput(BaseModel):
    description: str = Field(
        description="General diagnostic capability that is currently missing."
    )
    required_inputs_json: str = Field(
        default="[]",
        description=(
            "JSON list of generic input names, or objects containing a name. "
            "Do not use concrete device names."
        ),
    )
    expected_observations_json: str = Field(
        default="[]",
        description=(
            "JSON list of expected observation names, or objects containing a name."
        ),
    )


def _parameter_python_type(kind: str) -> type:
    return {"str": str, "int": int, "float": float, "bool": bool}[kind]


def _validate_composite_arguments(
    composite: CompositeTool,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _validate_parameter_arguments(
        composite.parameters,
        arguments,
        label="composite",
    )


def _validate_generated_arguments(
    tool: GeneratedTool,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _validate_parameter_arguments(
        tool.parameters,
        arguments,
        label="generated tool",
    )


def _validate_parameter_arguments(
    parameters: list[ToolParameter],
    arguments: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError(f"{label} arguments must be an object")
    declared = {parameter.name for parameter in parameters}
    unknown = set(arguments) - declared
    if unknown:
        raise ValueError(f"unknown {label} argument(s): {', '.join(sorted(unknown))}")

    normalized: dict[str, Any] = {}
    for parameter in parameters:
        if parameter.name in arguments:
            value = arguments[parameter.name]
        elif parameter.default is not None or not parameter.required:
            value = parameter.default
        else:
            raise ValueError(f"missing required {label} argument: {parameter.name}")
        if value is None and not parameter.required:
            normalized[parameter.name] = None
            continue
        expected = _parameter_python_type(parameter.type)
        valid = isinstance(value, expected)
        if parameter.type in {"int", "float"} and isinstance(value, bool):
            valid = False
        if (
            parameter.type == "float"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            value = float(value)
            valid = True
        if not valid:
            raise ValueError(
                f"argument '{parameter.name}' must be of type {parameter.type}"
            )
        normalized[parameter.name] = value
    return normalized


def _resolve_template(value: Any, arguments: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_template(item, arguments) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_template(item, arguments) for item in value]
    if not isinstance(value, str):
        return value
    exact = _PLACEHOLDER.fullmatch(value)
    if exact:
        return arguments[exact.group(1)]
    return _PLACEHOLDER.sub(lambda match: str(arguments[match.group(1)]), value)


def _validate_argument_safety(value: Any, *, allow_placeholders: bool) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_argument_safety(item, allow_placeholders=allow_placeholders)
        return
    if isinstance(value, list):
        for item in value:
            _validate_argument_safety(item, allow_placeholders=allow_placeholders)
        return
    if not isinstance(value, str):
        return
    inspected = _PLACEHOLDER.sub("", value) if allow_placeholders else value
    if len(inspected) > 512:
        raise ValueError("string arguments must be at most 512 characters")
    if _SHELL_CONTROL.search(inspected):
        raise ValueError(
            "shell control characters are not allowed in composite arguments"
        )


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLACEHOLDER.fullmatch(value))


def _validate_int_range(
    tool_name: str,
    arguments: dict[str, Any],
    field_name: str,
    *,
    minimum: int,
    maximum: int,
    allow_placeholders: bool,
) -> None:
    if field_name not in arguments:
        return
    value = arguments[field_name]
    if allow_placeholders and _is_placeholder(value):
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{tool_name}.{field_name} must be an integer between "
            f"{minimum} and {maximum}"
        )
    if value < minimum or value > maximum:
        raise ValueError(
            f"{tool_name}.{field_name} must be between {minimum} and {maximum}"
        )


def _validate_static_allowlist(
    tool_name: str,
    arguments: dict[str, Any],
    field_name: str,
    allowed_values: frozenset[str],
    *,
    allow_placeholders: bool,
) -> None:
    if field_name not in arguments:
        return
    value = arguments[field_name]
    if allow_placeholders and _is_placeholder(value):
        raise ValueError(
            f"{tool_name}.{field_name} cannot be parameterized in composite tools"
        )
    if not isinstance(value, str):
        raise ValueError(f"{tool_name}.{field_name} must be a string")
    normalized = " ".join(value.split()).strip()
    if normalized not in allowed_values:
        raise ValueError(
            f"{tool_name}.{field_name} must be one of "
            + ", ".join(repr(item) for item in sorted(allowed_values))
        )


def _validate_step_argument_policy(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    allow_placeholders: bool,
) -> None:
    """Enforce extra safety bounds for composable diagnostic primitives."""
    if tool_name not in COMPOSABLE_PRIMITIVE_TOOLS:
        raise ValueError(f"primitive tool is not composable: {tool_name}")
    if tool_name == "ping_pair":
        _validate_static_allowlist(
            tool_name,
            arguments,
            "args",
            frozenset({""}),
            allow_placeholders=allow_placeholders,
        )
        _validate_int_range(
            tool_name,
            arguments,
            "count",
            minimum=1,
            maximum=10,
            allow_placeholders=allow_placeholders,
        )
    elif tool_name == "netstat":
        _validate_static_allowlist(
            tool_name,
            arguments,
            "args",
            _NETSTAT_ALLOWED_ARGS,
            allow_placeholders=allow_placeholders,
        )
    elif tool_name == "bmv2_get_log":
        _validate_int_range(
            tool_name,
            arguments,
            "rows",
            minimum=1,
            maximum=200,
            allow_placeholders=allow_placeholders,
        )
    elif tool_name == "influx_query_measurement":
        _validate_int_range(
            tool_name,
            arguments,
            "limit",
            minimum=1,
            maximum=100,
            allow_placeholders=allow_placeholders,
        )
        _validate_int_range(
            tool_name,
            arguments,
            "offset",
            minimum=0,
            maximum=100_000,
            allow_placeholders=allow_placeholders,
        )


def _tool_output_is_error(output: Any) -> bool:
    if isinstance(output, str):
        normalized = output.strip().lower()
        return normalized.startswith(
            (
                "error executing tool",
                "tool execution failed",
                "composite execution failed",
            )
        )
    if isinstance(output, (list, tuple)):
        return any(_tool_output_is_error(item) for item in output)
    if isinstance(output, dict):
        if output.get("error"):
            return True
        return _tool_output_is_error(output.get("content"))
    if getattr(output, "status", None) == "error":
        return True
    content = getattr(output, "content", None)
    return _tool_output_is_error(content) if content is not None else False


def _compact_text(value: Any, *, limit: int = 96) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parameter_summary(parameters: list[ToolParameter], *, limit: int = 8) -> str:
    if not parameters:
        return "none"
    parts: list[str] = []
    for parameter in parameters[:limit]:
        requirement = (
            "required"
            if parameter.required and parameter.default is None
            else f"default={parameter.default!r}"
        )
        parts.append(
            f"{parameter.name} ({parameter.type}, {requirement}): "
            f"{_compact_text(parameter.description, limit=72)}"
        )
    if len(parameters) > limit:
        parts.append("...")
    return "; ".join(parts)


def _step_sequence_summary(steps: list[CompositeStep], *, limit: int = 6) -> str:
    pieces: list[str] = []
    for step in steps[:limit]:
        if step.arguments:
            args = ", ".join(
                f"{key}={_compact_text(value, limit=32)}"
                for key, value in step.arguments.items()
            )
            pieces.append(f"{step.tool}({args})")
        else:
            pieces.append(f"{step.tool}()")
    if len(steps) > limit:
        pieces.append("...")
    return " -> ".join(pieces)


def _output_contract_summary(output_contract: list[str], *, limit: int = 6) -> str:
    if not output_contract:
        return "one observation per declared step"
    values = [_compact_text(item, limit=80) for item in output_contract[:limit]]
    if len(output_contract) > limit:
        values.append("...")
    return "; ".join(values)


class ToolEvolutionRuntime:
    def __init__(
        self,
        *,
        session: Any,
        primitive_tools: list[BaseTool],
        library_id: str,
        mode: ToolEvolutionMode,
        model: str,
        task_description: str = "",
    ) -> None:
        self.session = session
        self.mode = mode
        self.model = model
        self.update_enabled = bool(
            getattr(session, "tool_evolution_update_enabled", True)
        )
        self.store = ToolEvolutionStore(library_id)
        self.logger = MessageLogger(
            agent="diagnosis_agent",
            session_dir=session.session_dir,
            extra_fields={"phase": "tool_evolution"},
        )
        self.primitive_tools = {tool.name: tool for tool in primitive_tools}
        self.retrieved_names: list[str] = []
        self.created_names: list[str] = []
        self.capability_gap_ids: list[str] = []
        self.mastery_used: list[str] = []
        self.cross_model_mastery: list[str] = []
        self._ephemeral_tools: dict[str, CompositeTool] = {}
        self._ephemeral_generated_tools: dict[str, GeneratedTool] = {}
        self._known_devices = self._collect_devices()
        self._apply_mastery_overlays()
        self.retrieved = self.store.search_composites(
            task_description,
            tags=[
                str(getattr(session, "scenario_name", "")),
                str(getattr(session, "scenario_topo_size", "")),
            ],
            top_k=5,
            include_candidates=True,
            record_usage=self.update_enabled,
        )
        self.retrieved_generated = self.store.search_generated_tools(
            task_description,
            tags=[
                str(getattr(session, "scenario_name", "")),
                str(getattr(session, "scenario_topo_size", "")),
            ],
            top_k=5,
            include_candidates=True,
            record_usage=self.update_enabled,
        )
        self.retrieved_names = [
            *[tool.name for tool in self.retrieved],
            *[tool.name for tool in self.retrieved_generated],
        ]
        if self.retrieved_names:
            self.logger.log(
                "tool_evolution_retrieved",
                {"tools": self.retrieved_names, "library_id": self.store.library_id},
            )

    def _collect_devices(self) -> set[str]:
        devices: set[str] = set()
        for edge in getattr(self.session, "topology", []) or []:
            if isinstance(edge, (list, tuple)):
                for item in edge:
                    if not item:
                        continue
                    endpoint = str(item)
                    devices.add(endpoint)
                    devices.add(endpoint.split(":", 1)[0])
        return devices

    def _apply_mastery_overlays(self) -> None:
        state = self.store.load()
        for tool_name, mastery in state.mastery.items():
            tool = self.primitive_tools.get(tool_name)
            if tool is None:
                continue
            overlay = mastery.agent_overlay()
            if not overlay:
                continue
            tool.description = (
                f"{tool.description}\n\nLearned from prior executions:\n{overlay}"
            )
            self.mastery_used.append(tool_name)
            if mastery.source_models and self.model not in mastery.source_models:
                self.cross_model_mastery.append(tool_name)

    def prompt_suffix(self) -> str:
        retrieved_guidance = self._retrieved_tool_guidance()
        return f"""\

Tool-evolution policy:
    - If a retrieved composite/generated tool below matches the evidence you need
      and you can supply its arguments, call it before manually repeating the same
      primitive sequence.
    - Search the diagnostic tool library before repeating any other multi-step
      investigation.
    - Candidate tools are reusable evidence collectors; verify observations before
      using them as the final conclusion.
    - When a reusable workflow or executable helper is missing, first identify the
      capability gap.
    - Synthesize either a parameterized composite or Python generated tool for that
      gap, then execute it.
    - A synthesized tool is persisted only after structural, runtime/sandbox, and
      semantic checks pass.
    - Revise a composite into a new version when execution reveals a general flaw.
    - Composite steps may use the same diagnosis primitives exposed by upstream
      NIKA; keep arguments parameterized and reusable across incidents.
    - Generated Python tools must be pure computational helpers with declared
      parameters.
    - Never encode incident labels, concrete device names, IP addresses, or session
      identifiers.
    - Record concise lessons when a primitive tool's documentation is incomplete or
      misleading.
{retrieved_guidance}"""

    def _retrieved_tool_guidance(self) -> str:
        if not self.retrieved and not self.retrieved_generated:
            return ""

        lines = ["", "Retrieved reusable tools for this task:"]
        for composite in self.retrieved:
            lines.extend(
                [
                    f"- {composite.name} [{composite.status}]: "
                    f"{_compact_text(composite.description, limit=180)}",
                    f"  Steps: {_step_sequence_summary(composite.steps)}",
                    f"  Args: {_parameter_summary(composite.parameters)}",
                    f"  Outputs: {_output_contract_summary(composite.output_contract)}",
                ]
            )
        for tool in self.retrieved_generated:
            lines.extend(
                [
                    f"- {tool.name} [{tool.status} generated_python]: "
                    f"{_compact_text(tool.description, limit=180)}",
                    f"  Args: {_parameter_summary(tool.parameters)}",
                    f"  Output: {_compact_text(tool.output_description or 'computed diagnostic value')}",
                ]
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _composite_payload(composite: CompositeTool) -> dict[str, Any]:
        return {
            "kind": "composite",
            "name": composite.name,
            "description": composite.description,
            "status": composite.status,
            "parameters": [
                parameter.model_dump() for parameter in composite.parameters
            ],
            "step_sequence": _step_sequence_summary(composite.steps),
            "steps": [
                {
                    "tool": step.tool,
                    "arguments": step.arguments,
                    "label": step.label,
                }
                for step in composite.steps
            ],
            "output_contract": composite.output_contract,
            "execution_hint": (
                "Call this tool directly when available, or use execute_candidate_tool "
                "with this name and arguments_json."
            ),
        }

    @staticmethod
    def _generated_payload(tool: GeneratedTool) -> dict[str, Any]:
        return {
            "kind": "generated_python",
            "name": tool.name,
            "description": tool.description,
            "status": tool.status,
            "parameters": [parameter.model_dump() for parameter in tool.parameters],
            "output_description": tool.output_description,
            "execution_hint": (
                "Call this tool directly when available, or use execute_candidate_tool "
                "with this name and arguments_json."
            ),
        }

    @staticmethod
    def _composite_tool_description(composite: CompositeTool) -> str:
        return "\n".join(
            [
                composite.description,
                f"Runs: {_step_sequence_summary(composite.steps)}",
                f"Arguments: {_parameter_summary(composite.parameters)}",
                f"Returns: {_output_contract_summary(composite.output_contract)}",
            ]
        )

    @staticmethod
    def _generated_tool_description(tool: GeneratedTool) -> str:
        return "\n".join(
            [
                tool.description,
                f"Arguments: {_parameter_summary(tool.parameters)}",
                f"Returns: {_compact_text(tool.output_description or 'computed diagnostic value')}",
            ]
        )

    def build_tools(self) -> list[BaseTool]:
        tools = list(self.primitive_tools.values())
        tools.extend(self._build_composite_tool(item) for item in self.retrieved)
        tools.extend(
            self._build_generated_tool(item) for item in self.retrieved_generated
        )
        tools.extend(self._build_manager_tools())
        return tools

    def _build_manager_tools(self) -> list[StructuredTool]:
        async def search_diagnostic_tools(query: str) -> str:
            composite_matches = self.store.search_composites(
                query,
                tags=[str(getattr(self.session, "scenario_name", ""))],
                top_k=8,
                include_candidates=True,
                record_usage=self.update_enabled,
            )
            generated_matches = self.store.search_generated_tools(
                query,
                tags=[str(getattr(self.session, "scenario_name", ""))],
                top_k=8,
                include_candidates=True,
                record_usage=self.update_enabled,
            )
            payload: list[dict[str, Any]] = [
                self._composite_payload(item) for item in composite_matches
            ]
            payload.extend(self._generated_payload(item) for item in generated_matches)
            self.logger.log(
                "tool_evolution_search",
                {"query": query, "result_names": [item["name"] for item in payload]},
            )
            return json.dumps(payload, ensure_ascii=False)

        async def identify_capability_gap(
            description: str,
            required_inputs_json: str = "[]",
            expected_observations_json: str = "[]",
        ) -> str:
            if not self.mode.distillation_enabled:
                return "Tool synthesis is disabled in mastery-only mode."
            if not self.update_enabled:
                return "Tool library updates are disabled for this run."
            try:
                self._validate_persistent_text(description)
                required_inputs = self._parse_gap_items(
                    required_inputs_json,
                    "required_inputs_json",
                )
                expected_observations = self._parse_gap_items(
                    expected_observations_json,
                    "expected_observations_json",
                )
                self._validate_persistent_text(
                    " ".join([*required_inputs, *expected_observations])
                )
                seed = json.dumps(
                    {
                        "description": " ".join(description.lower().split()),
                        "required_inputs": required_inputs,
                        "expected_observations": expected_observations,
                    },
                    sort_keys=True,
                )
                gap_id = "gap_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
                gap = self.store.record_capability_gap(
                    CapabilityGap(
                        gap_id=gap_id,
                        description=description,
                        required_inputs=required_inputs,
                        expected_observations=expected_observations,
                    )
                )
            except Exception as exc:
                return f"Rejected capability gap: {exc}"
            if gap.gap_id not in self.capability_gap_ids:
                self.capability_gap_ids.append(gap.gap_id)
            self.logger.log(
                "tool_evolution_gap_identified",
                {
                    "gap_id": gap.gap_id,
                    "required_inputs": gap.required_inputs,
                    "expected_observations": gap.expected_observations,
                },
            )
            return (
                f"Recorded capability gap '{gap.gap_id}'. "
                "Synthesize an ephemeral composite or Python tool for this gap."
            )

        async def propose_composite_tool(
            name: str,
            description: str,
            gap_id: str,
            parameters_json: str,
            steps_json: str,
        ) -> str:
            if not self.mode.distillation_enabled:
                return "Tool distillation is disabled in mastery-only mode."
            if not self.update_enabled:
                return "Tool library updates are disabled for this run."
            try:
                if not gap_id:
                    raise ValueError(
                        "gap_id is required; identify the capability gap first"
                    )
                gap = self.store.load().capability_gaps.get(gap_id)
                if gap is None:
                    raise ValueError(f"unknown capability gap: {gap_id}")
                parameters = [
                    ToolParameter.model_validate(item)
                    for item in json.loads(parameters_json)
                ]
                steps = [
                    CompositeStep.model_validate(item)
                    for item in json.loads(steps_json)
                ]
                composite = CompositeTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    steps=steps,
                    output_contract=list(gap.expected_observations),
                    tags=[str(getattr(self.session, "scenario_name", ""))],
                    status="ephemeral",
                )
                self.validate_composite(composite)
                composite.verification_reports.append(
                    ToolVerificationReport(
                        stage="structural",
                        passed=True,
                        checks=[
                            "schema",
                            "parameterization",
                            "read_only_allowlist",
                            "metadata_sanitization",
                        ],
                        context_fingerprint=self._context_fingerprint(),
                    )
                )
                self.store.resolve_capability_gap(
                    gap_id,
                    proposed_tool=composite.name,
                )
                self._ephemeral_tools[composite.name] = composite
            except Exception as exc:
                self.logger.log(
                    "tool_evolution_candidate_rejected",
                    {"name": name, "error": str(exc)},
                )
                return f"Rejected composite tool: {exc}"
            self.logger.log(
                "tool_evolution_ephemeral_created",
                {
                    "name": composite.name,
                    "signature": composite.ensure_signature().signature,
                    "gap_id": gap_id,
                },
            )
            return (
                f"Synthesized ephemeral tool '{composite.name}'. "
                "Execute it with execute_candidate_tool to verify and persist it."
            )

        async def propose_python_tool(
            name: str,
            description: str,
            gap_id: str,
            parameters_json: str,
            code: str,
            output_description: str = "",
            test_example_json: str = "{}",
        ) -> str:
            if not self.mode.distillation_enabled:
                return "Tool generation is disabled in mastery-only mode."
            if not self.update_enabled:
                return "Tool library updates are disabled for this run."
            try:
                if not gap_id:
                    raise ValueError(
                        "gap_id is required; identify the capability gap first"
                    )
                gap = self.store.load().capability_gaps.get(gap_id)
                if gap is None:
                    raise ValueError(f"unknown capability gap: {gap_id}")
                parameters = [
                    ToolParameter.model_validate(item)
                    for item in json.loads(parameters_json)
                ]
                test_example = json.loads(test_example_json or "{}")
                if not isinstance(test_example, dict):
                    raise ValueError("test_example_json must decode to an object")
                generated = GeneratedTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    code=code,
                    output_description=output_description
                    or ", ".join(gap.expected_observations),
                    tags=[str(getattr(self.session, "scenario_name", ""))],
                    status="ephemeral",
                    test_example=test_example,
                )
                self.validate_generated_tool(generated)
                generated.verification_reports.append(
                    ToolVerificationReport(
                        stage="structural",
                        passed=True,
                        checks=validate_generated_tool_code(generated),
                        context_fingerprint=self._context_fingerprint(),
                    )
                )
                self.store.resolve_capability_gap(
                    gap_id,
                    proposed_tool=generated.name,
                )
                self._ephemeral_generated_tools[generated.name] = generated
            except Exception as exc:
                self.logger.log(
                    "tool_evolution_candidate_rejected",
                    {"name": name, "kind": "generated_python", "error": str(exc)},
                )
                return f"Rejected generated Python tool: {exc}"
            self.logger.log(
                "tool_evolution_generated_ephemeral_created",
                {
                    "name": generated.name,
                    "signature": generated.ensure_signature().signature,
                    "gap_id": gap_id,
                },
            )
            return (
                f"Synthesized ephemeral Python tool '{generated.name}'. "
                "Execute it with execute_candidate_tool to sandbox-verify and persist it."
            )

        async def execute_candidate_tool(name: str, arguments_json: str) -> str:
            try:
                arguments = json.loads(arguments_json)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments_json must decode to an object")
                ephemeral_generated = self._ephemeral_generated_tools.get(name)
                generated = ephemeral_generated or self.store.get_generated_tool(name)
                if generated is not None:
                    result = await self.execute_generated_tool(generated, arguments)
                    checks = self._verify_generated_result(generated, result)
                    report = ToolVerificationReport(
                        stage="semantic",
                        passed=True,
                        checks=checks,
                        context_fingerprint=self._context_fingerprint(),
                    )
                    if ephemeral_generated is not None:
                        ephemeral_generated.verification_reports.extend(
                            [
                                ToolVerificationReport(
                                    stage="sandbox",
                                    passed=True,
                                    checks=list(result.get("checks") or []),
                                    context_fingerprint=self._context_fingerprint(),
                                ),
                                report,
                            ]
                        )
                        registered, created = self.store.register_generated_tool(
                            ephemeral_generated,
                            deduplicate=self.mode.dedup_enabled,
                        )
                        if not created:
                            self.store.record_generated_verification(
                                registered.name,
                                report,
                            )
                        self._ephemeral_generated_tools.pop(name, None)
                        if created:
                            self.created_names.append(registered.name)
                        for gap_id in self.capability_gap_ids:
                            gap = self.store.load().capability_gaps.get(gap_id)
                            if gap and gap.proposed_tool == name:
                                self.store.resolve_capability_gap(
                                    gap_id,
                                    proposed_tool=registered.name,
                                    resolved=True,
                                )
                        self.logger.log(
                            "tool_evolution_generated_verified",
                            {
                                "name": registered.name,
                                "source_name": name,
                                "created": created,
                                "checks": checks,
                            },
                        )
                        result["persisted_as"] = registered.name
                    else:
                        if self.update_enabled:
                            self.store.record_generated_verification(
                                generated.name,
                                report,
                            )
                    return json.dumps(result, ensure_ascii=False, default=str)

                ephemeral = self._ephemeral_tools.get(name)
                composite = ephemeral or self.store.get_composite(name)
                if composite is None:
                    raise ValueError(f"Unknown composite tool: {name}")
                result = await self.execute_composite(composite, arguments)
                semantic_checks = self._verify_result(composite, result)
                report = ToolVerificationReport(
                    stage="semantic",
                    passed=True,
                    checks=semantic_checks,
                    context_fingerprint=self._context_fingerprint(),
                )
                if ephemeral is not None:
                    ephemeral.verification_reports.extend(
                        [
                            ToolVerificationReport(
                                stage="runtime",
                                passed=True,
                                checks=["all primitive calls completed without error"],
                                context_fingerprint=self._context_fingerprint(),
                            ),
                            report,
                        ]
                    )
                    registered, created = self.store.register_composite(
                        ephemeral,
                        deduplicate=self.mode.dedup_enabled,
                    )
                    if not created:
                        self.store.record_verification(registered.name, report)
                    self._ephemeral_tools.pop(name, None)
                    if created:
                        self.created_names.append(registered.name)
                    for gap_id in self.capability_gap_ids:
                        gap = self.store.load().capability_gaps.get(gap_id)
                        if gap and gap.proposed_tool == name:
                            self.store.resolve_capability_gap(
                                gap_id,
                                proposed_tool=registered.name,
                                resolved=True,
                            )
                    self.logger.log(
                        "tool_evolution_candidate_verified",
                        {
                            "name": registered.name,
                            "source_name": name,
                            "created": created,
                            "checks": semantic_checks,
                        },
                    )
                    result["persisted_as"] = registered.name
                else:
                    if self.update_enabled:
                        self.store.record_verification(composite.name, report)
                return json.dumps(result, ensure_ascii=False, default=str)
            except Exception as exc:
                composite = self._ephemeral_tools.get(name)
                if composite is not None:
                    composite.verification_reports.append(
                        ToolVerificationReport(
                            stage="runtime",
                            passed=False,
                            error=str(exc),
                            context_fingerprint=self._context_fingerprint(),
                        )
                    )
                generated = self._ephemeral_generated_tools.get(name)
                if generated is not None:
                    generated.verification_reports.append(
                        ToolVerificationReport(
                            stage="sandbox",
                            passed=False,
                            error=str(exc),
                            context_fingerprint=self._context_fingerprint(),
                        )
                    )
                return f"Candidate execution failed: {exc}"

        async def revise_composite_tool(
            existing_name: str,
            description: str,
            parameters_json: str,
            steps_json: str,
        ) -> str:
            if not self.mode.distillation_enabled:
                return "Tool distillation is disabled in mastery-only mode."
            if not self.update_enabled:
                return "Tool library updates are disabled for this run."
            existing = self._ephemeral_tools.get(existing_name)
            existing = existing or self.store.get_composite(existing_name)
            if existing is None:
                return f"Unknown composite tool: {existing_name}"
            try:
                version = existing.version + 1
                suffix = f"_v{version}"
                revised = CompositeTool(
                    name=f"{existing.name[: 64 - len(suffix)]}{suffix}",
                    description=description,
                    parameters=[
                        ToolParameter.model_validate(item)
                        for item in json.loads(parameters_json)
                    ],
                    steps=[
                        CompositeStep.model_validate(item)
                        for item in json.loads(steps_json)
                    ],
                    tags=existing.tags,
                    version=version,
                    parent_name=existing.name,
                    status="ephemeral",
                )
                self.validate_composite(revised)
                revised.verification_reports.append(
                    ToolVerificationReport(
                        stage="structural",
                        passed=True,
                        checks=[
                            "schema",
                            "parameterization",
                            "read_only_allowlist",
                            "metadata_sanitization",
                        ],
                        context_fingerprint=self._context_fingerprint(),
                    )
                )
                self._ephemeral_tools[revised.name] = revised
            except Exception as exc:
                return f"Rejected composite revision: {exc}"
            self.logger.log(
                "tool_evolution_ephemeral_created",
                {
                    "name": revised.name,
                    "parent_name": existing.name,
                    "version": revised.version,
                    "signature": revised.ensure_signature().signature,
                },
            )
            return (
                f"Synthesized ephemeral revision '{revised.name}' from "
                f"'{existing.name}'. Execute it before persistence."
            )

        async def record_tool_lesson(
            tool_name: str,
            precondition: str = "",
            parameter_guidance: str = "",
            output_interpretation: str = "",
            failure_semantics: str = "",
        ) -> str:
            if not self.mode.mastery_enabled:
                return "Tool mastery is disabled in distillation-only mode."
            if not self.update_enabled:
                return "Tool library updates are disabled for this run."
            if tool_name not in self.primitive_tools:
                return f"Unknown primitive tool: {tool_name}"
            if not any(
                (
                    precondition.strip(),
                    parameter_guidance.strip(),
                    output_interpretation.strip(),
                    failure_semantics.strip(),
                )
            ):
                return "Rejected tool lesson: provide at least one evidence-grounded field."
            try:
                self._validate_persistent_text(
                    " ".join(
                        [
                            precondition,
                            parameter_guidance,
                            output_interpretation,
                            failure_semantics,
                        ]
                    )
                )
            except ValueError as exc:
                return f"Rejected tool lesson: {exc}"
            mastery = self.store.upsert_mastery(
                tool_name,
                preconditions=[precondition] if precondition else [],
                parameter_guidance=[parameter_guidance] if parameter_guidance else [],
                output_interpretation=(
                    [output_interpretation] if output_interpretation else []
                ),
                failure_semantics=[failure_semantics] if failure_semantics else [],
                source_model=self.model,
                revision_source="runtime",
                rationale="Recorded during a live diagnostic trajectory.",
            )
            self.logger.log(
                "tool_evolution_mastery_recorded",
                {"tool_name": tool_name},
            )
            return f"Recorded tool lesson for {mastery.tool_name}."

        tools = [
            StructuredTool.from_function(
                coroutine=search_diagnostic_tools,
                name="search_diagnostic_tools",
                description="Retrieve reusable diagnostic composite tools by capability.",
                args_schema=SearchToolsInput,
            ),
        ]
        if self.mode.distillation_enabled:
            tools.extend(
                [
                    StructuredTool.from_function(
                        coroutine=identify_capability_gap,
                        name="identify_capability_gap",
                        description=(
                            "Record a general missing diagnostic capability before "
                            "synthesizing a test-time tool."
                        ),
                        args_schema=CapabilityGapInput,
                    ),
                    StructuredTool.from_function(
                        coroutine=propose_composite_tool,
                        name="propose_composite_tool",
                        description=(
                            "Create a parameterized read-only diagnostic workflow. "
                            "Use only composable primitive tools and ${parameter} placeholders."
                        ),
                        args_schema=ProposeCompositeInput,
                    ),
                    StructuredTool.from_function(
                        coroutine=propose_python_tool,
                        name="propose_python_tool",
                        description=(
                            "Create an executable Python helper for a missing "
                            "computational capability. The code must define the "
                            "named function and pass sandbox validation."
                        ),
                        args_schema=ProposePythonToolInput,
                    ),
                    StructuredTool.from_function(
                        coroutine=execute_candidate_tool,
                        name="execute_candidate_tool",
                        description=(
                            "Execute a newly proposed or probationary composite/generated tool."
                        ),
                        args_schema=ExecuteCandidateInput,
                    ),
                    StructuredTool.from_function(
                        coroutine=revise_composite_tool,
                        name="revise_composite_tool",
                        description=(
                            "Create a new probationary version of an existing composite tool."
                        ),
                        args_schema=ReviseCompositeInput,
                    ),
                ]
            )
        if self.mode.mastery_enabled:
            tools.append(
                StructuredTool.from_function(
                    coroutine=record_tool_lesson,
                    name="record_tool_lesson",
                    description=(
                        "Persist an evidence-grounded lesson about how to use a primitive tool."
                    ),
                    args_schema=RecordLessonInput,
                )
            )
        return tools

    def _build_composite_tool(self, composite: CompositeTool) -> StructuredTool:
        fields: Any = {}
        for parameter in composite.parameters:
            py_type = _parameter_python_type(parameter.type)
            default = (
                ...
                if parameter.required and parameter.default is None
                else parameter.default
            )
            fields[parameter.name] = (
                py_type,
                Field(default=default, description=parameter.description),
            )
        args_model = create_model(
            f"{composite.name.title().replace('_', '')}Input",
            **fields,
        )

        async def execute(**kwargs: Any) -> str:
            result = await self.execute_composite(composite, kwargs)
            checks = self._verify_result(composite, result)
            if self.update_enabled:
                self.store.record_verification(
                    composite.name,
                    ToolVerificationReport(
                        stage="semantic",
                        passed=True,
                        checks=checks,
                        context_fingerprint=self._context_fingerprint(),
                    ),
                )
            return json.dumps(result, ensure_ascii=False, default=str)

        status_note = (
            "Promoted and validated."
            if composite.status == "promoted"
            else (
                "Candidate reusable workflow: use it for matching evidence "
                "collection, then cross-check observations before the final conclusion."
            )
        )
        return StructuredTool.from_function(
            coroutine=execute,
            name=composite.name,
            description=f"{self._composite_tool_description(composite)}\n{status_note}",
            args_schema=args_model,
        )

    def _build_generated_tool(self, tool: GeneratedTool) -> StructuredTool:
        fields: Any = {}
        for parameter in tool.parameters:
            py_type = _parameter_python_type(parameter.type)
            default = (
                ...
                if parameter.required and parameter.default is None
                else parameter.default
            )
            fields[parameter.name] = (
                py_type,
                Field(default=default, description=parameter.description),
            )
        args_model = create_model(
            f"{tool.name.title().replace('_', '')}Input",
            **fields,
        )

        async def execute(**kwargs: Any) -> str:
            result = await self.execute_generated_tool(tool, kwargs)
            checks = self._verify_generated_result(tool, result)
            if self.update_enabled:
                self.store.record_generated_verification(
                    tool.name,
                    ToolVerificationReport(
                        stage="semantic",
                        passed=True,
                        checks=checks,
                        context_fingerprint=self._context_fingerprint(),
                    ),
                )
            return json.dumps(result, ensure_ascii=False, default=str)

        status_note = (
            "Promoted and validated generated Python tool."
            if tool.status == "promoted"
            else (
                "Candidate generated Python tool: use it for matching computations, "
                "then cross-check before the final conclusion."
            )
        )
        return StructuredTool.from_function(
            coroutine=execute,
            name=tool.name,
            description=f"{self._generated_tool_description(tool)}\n{status_note}",
            args_schema=args_model,
        )

    def validate_composite(self, composite: CompositeTool) -> None:
        declared = {parameter.name for parameter in composite.parameters}
        serialized = composite.model_dump_json().lower()
        self._validate_persistent_text(serialized)
        for step in composite.steps:
            if step.tool not in COMPOSABLE_PRIMITIVE_TOOLS:
                raise ValueError(f"unsafe or unsupported primitive tool: {step.tool}")
            if step.tool not in self.primitive_tools:
                raise ValueError(
                    f"primitive tool is unavailable in this scenario: {step.tool}"
                )
            self._validate_primitive_step_schema(
                self.primitive_tools[step.tool],
                step,
            )
            refs = set(_PLACEHOLDER.findall(json.dumps(step.arguments, default=str)))
            unknown = refs - declared
            if unknown:
                raise ValueError(
                    f"undeclared parameter reference(s): {', '.join(sorted(unknown))}"
                )
            _validate_argument_safety(
                step.arguments,
                allow_placeholders=True,
            )
            _validate_step_argument_policy(
                step.tool,
                step.arguments,
                allow_placeholders=True,
            )
        used = set()
        for step in composite.steps:
            used.update(_PLACEHOLDER.findall(json.dumps(step.arguments, default=str)))
        unused = declared - used
        if unused:
            raise ValueError(f"unused parameter(s): {', '.join(sorted(unused))}")
        defaults = {
            parameter.name: parameter.default
            for parameter in composite.parameters
            if parameter.default is not None or not parameter.required
        }
        if defaults:
            partial = composite.model_copy(
                update={
                    "parameters": [
                        parameter.model_copy(update={"required": False})
                        for parameter in composite.parameters
                        if parameter.name in defaults
                    ]
                }
            )
            _validate_composite_arguments(partial, defaults)

    def validate_generated_tool(self, tool: GeneratedTool) -> None:
        serialized = tool.model_dump_json().lower()
        self._validate_persistent_text(serialized)
        validate_generated_tool_code(tool)
        defaults = {
            parameter.name: parameter.default
            for parameter in tool.parameters
            if parameter.default is not None or not parameter.required
        }
        if defaults:
            partial = tool.model_copy(
                update={
                    "parameters": [
                        parameter.model_copy(update={"required": False})
                        for parameter in tool.parameters
                        if parameter.name in defaults
                    ]
                }
            )
            _validate_generated_arguments(partial, defaults)

    @staticmethod
    def _validate_primitive_step_schema(
        primitive: BaseTool,
        step: CompositeStep,
    ) -> None:
        args_schema = getattr(primitive, "args_schema", None)
        if args_schema is None or not hasattr(args_schema, "model_json_schema"):
            return
        schema = args_schema.model_json_schema()
        properties = set((schema.get("properties") or {}).keys())
        required = set(schema.get("required") or [])
        provided = set(step.arguments)
        missing = required - provided
        unknown = provided - properties
        if missing:
            raise ValueError(
                f"primitive '{step.tool}' is missing argument(s): "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                f"primitive '{step.tool}' has unknown argument(s): "
                + ", ".join(sorted(unknown))
            )

    def _validate_persistent_text(self, text: str) -> None:
        serialized = text.lower()
        if _IPV4.search(serialized):
            raise ValueError("hard-coded IP addresses are not allowed")
        session_id = str(getattr(self.session, "session_id", "")).lower()
        if len(session_id) >= 6 and session_id in serialized:
            raise ValueError("session identifiers are not allowed")
        for device in self._known_devices:
            if device and re.search(rf"\b{re.escape(device.lower())}\b", serialized):
                raise ValueError(f"hard-coded device name is not allowed: {device}")
        normalized = re.sub(r"[^a-z0-9]+", "_", serialized)
        for problem_name in list_avail_problem_names():
            if problem_name.lower() in normalized:
                raise ValueError("root-cause labels are not allowed in tool metadata")

    def _parse_gap_items(
        self,
        raw: str,
        field_name: str,
    ) -> list[str]:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(
                f"{field_name} must be a JSON list of strings or named objects"
            )
        values: list[str] = []
        known_devices = {item.lower() for item in self._known_devices}
        for index, item in enumerate(parsed[:12]):
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = str(
                    item.get("name")
                    or item.get("description")
                    or item.get("label")
                    or ""
                )
            else:
                value = ""
            value = " ".join(value.split()).strip()[:160]
            if not value:
                raise ValueError(
                    f"{field_name} items must be non-empty strings or named objects"
                )
            if value.lower() in known_devices:
                raise ValueError(
                    f"{field_name} item must describe a generalized role, "
                    "not a concrete device name"
                )
            else:
                for device in sorted(
                    self._known_devices,
                    key=len,
                    reverse=True,
                ):
                    value = re.sub(
                        rf"\b{re.escape(device)}\b",
                        "<device>",
                        value,
                        flags=re.IGNORECASE,
                    )
            values.append(value)
        return values

    def _context_fingerprint(self) -> str:
        scenario = str(getattr(self.session, "scenario_name", "unknown"))
        tier = str(getattr(self.session, "scenario_topo_size", "") or "fixed")
        return f"{scenario}:{tier}"

    @staticmethod
    def _verify_result(
        composite: CompositeTool,
        result: dict[str, Any],
    ) -> list[str]:
        if result.get("status") != "success":
            raise ValueError("composite did not return success status")
        observations = result.get("observations")
        if not isinstance(observations, list):
            raise ValueError("composite observations must be a list")
        if len(observations) != len(composite.steps):
            raise ValueError("composite did not produce one observation per step")
        for index, (observation, step) in enumerate(
            zip(observations, composite.steps, strict=True)
        ):
            if not isinstance(observation, dict):
                raise ValueError(f"observation {index} must be an object")
            if observation.get("step") != index or observation.get("tool") != step.tool:
                raise ValueError(
                    f"observation {index} does not match its declared step"
                )
            if not ToolEvolutionRuntime._informative_output(observation.get("output")):
                raise ValueError(
                    f"observation {index} returned no informative primitive output"
                )
        checks = [
            "success status",
            "one observation per declared step",
            "no missing primitive outputs",
        ]
        if composite.output_contract:
            if not all(item.strip() for item in composite.output_contract):
                raise ValueError(
                    "composite output contract contains an empty observation"
                )
            checks.append(
                "declared observation contract: " + ", ".join(composite.output_contract)
            )
        return checks

    @staticmethod
    def _verify_generated_result(
        tool: GeneratedTool,
        result: dict[str, Any],
    ) -> list[str]:
        if result.get("status") != "success" or not result.get("success"):
            raise ValueError("generated tool did not return success status")
        if not ToolEvolutionRuntime._informative_output(result.get("result")):
            raise ValueError("generated tool returned no informative output")
        checks = [
            "success status",
            "informative generated output",
        ]
        if tool.output_description:
            checks.append(f"declared output: {tool.output_description}")
        return checks

    @staticmethod
    def _informative_output(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (dict, list, tuple, set)):
            return bool(value)
        content = getattr(value, "content", None)
        if content is not None:
            return ToolEvolutionRuntime._informative_output(content)
        return True

    async def execute_composite(
        self,
        composite: CompositeTool,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        self.validate_composite(composite)
        arguments = _validate_composite_arguments(composite, arguments)
        self.logger.log(
            "tool_evolution_composite_start",
            {"name": composite.name, "status": composite.status},
        )
        outputs: list[dict[str, Any]] = []
        try:
            for index, step in enumerate(composite.steps):
                primitive = self.primitive_tools[step.tool]
                resolved = _resolve_template(step.arguments, arguments)
                _validate_argument_safety(resolved, allow_placeholders=False)
                _validate_step_argument_policy(
                    step.tool,
                    resolved,
                    allow_placeholders=False,
                )
                self.logger.log(
                    "tool_evolution_primitive_start",
                    {
                        "composite": composite.name,
                        "step": index,
                        "tool": {"name": step.tool},
                        "input": resolved,
                    },
                )
                output = await primitive.ainvoke(
                    resolved,
                    config={"callbacks": []},
                )
                if _tool_output_is_error(output):
                    raise RuntimeError(
                        f"Primitive tool '{step.tool}' returned an error: {output}"
                    )
                outputs.append(
                    {
                        "step": index,
                        "tool": step.tool,
                        "label": step.label,
                        "output": output,
                    }
                )
                self.logger.log(
                    "tool_evolution_primitive_end",
                    {
                        "composite": composite.name,
                        "step": index,
                        "tool": {"name": step.tool},
                    },
                )
        except Exception as exc:
            self.logger.log(
                "tool_evolution_composite_error",
                {"name": composite.name, "error": str(exc)},
            )
            raise
        self.logger.log(
            "tool_evolution_composite_end",
            {"name": composite.name, "steps": len(outputs)},
        )
        return {"tool": composite.name, "status": "success", "observations": outputs}

    async def execute_generated_tool(
        self,
        tool: GeneratedTool,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        self.validate_generated_tool(tool)
        arguments = _validate_generated_arguments(tool, arguments)
        self.logger.log(
            "tool_evolution_generated_start",
            {"name": tool.name, "status": tool.status},
        )
        try:
            output = await asyncio.to_thread(run_generated_tool, tool, arguments)
        except Exception as exc:
            self.logger.log(
                "tool_evolution_generated_error",
                {"name": tool.name, "error": str(exc)},
            )
            raise
        if not output.get("success"):
            self.logger.log(
                "tool_evolution_generated_error",
                {"name": tool.name, "error": output.get("stderr", "")},
            )
            raise RuntimeError(output.get("stderr") or "generated tool failed")
        self.logger.log(
            "tool_evolution_generated_end",
            {"name": tool.name},
        )
        return {
            "tool": tool.name,
            "status": "success",
            "success": True,
            "result": output.get("result"),
            "checks": output.get("checks", []),
        }
