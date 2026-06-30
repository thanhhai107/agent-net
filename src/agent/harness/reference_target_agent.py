"""Reference executable target agent for NIKA SIA-H harness evolution.

This file is copied into each evolution run as ``target_agent.py``. Feedback
agents are allowed to rewrite this harness across generations. Keep this module
self-contained and executable through the CLI contract documented in ``main``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.llm.model_factory import load_model
from agent.utils.loggers import AgentCallbackLogger, MESSAGES_FILENAME
from agent.utils.mcp_servers import MCPServerConfig


DIAGNOSIS_SYSTEM_PROMPT = """\
You are a network troubleshooting expert.
Use the available MCP tools to diagnose the current network incident.
Focus on anomaly detection, exact faulty-device localization, and root-cause
identification. Do not propose mitigation.

Investigation requirements:
- Gather concrete evidence with MCP tools before concluding.
- Prefer direct checks against the affected endpoint, service, interface, or
  routing protocol over broad speculation.
- Keep an evidence ledger with symptom, observations, suspected faulty device,
  and root-cause hypothesis.
- Stop exploration only when the evidence can support detection,
  localization, and RCA.
"""

SUBMISSION_SYSTEM_PROMPT = """\
You are an expert network engineer.
Use the task MCP tools to submit the final structured solution.
First call list_avail_problems() and choose root_cause_name only from that list.
Then call submit(is_anomaly, faulty_devices, root_cause_name).
Do not submit if there is no diagnosis report.
"""


def _load_case_context(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "case_context.json"
    if not path.exists():
        raise FileNotFoundError(f"case context not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("case_context.json must contain a JSON object")
    return value


def _message_events(working_dir: Path) -> list[dict[str, Any]]:
    path = working_dir / MESSAGES_FILENAME
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _write_execution(
    *,
    working_dir: Path,
    benchmark_index: int | str | None,
    context: dict[str, Any],
    diagnosis_report: str,
    submission_result: str,
    error: str | None = None,
) -> None:
    execution_dir = working_dir / "agent_execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(benchmark_index) if benchmark_index is not None else "case"
    payload = {
        "case": {
            "benchmark_index": benchmark_index,
            "scenario_name": context.get("scenario_name"),
            "topology_class": context.get("scenario_topo_size"),
        },
        "diagnosis_report": diagnosis_report,
        "submission_result": submission_result,
        "error": error,
        "messages": _message_events(working_dir),
    }
    (execution_dir / f"execution_{suffix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (working_dir / "agent_execution.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _diagnosis_config(session_id: str, context: dict[str, Any]) -> dict[str, Any]:
    mcp = MCPServerConfig(session_id=session_id)
    server_names = [
        str(item)
        for item in context.get("diagnosis_mcp_servers", [])
        if str(item).strip()
    ]
    config = (
        mcp.load_filtered_config(server_names)
        if server_names
        else mcp.load_config(if_submit=False)
    )
    tool_library_id = str(context.get("tool_library_id") or "").strip()
    if tool_library_id:
        config.update(mcp.load_toolbox_config(tool_library_id))
    return config


def _build_task_prompt(context: dict[str, Any]) -> str:
    parts = [
        "# Network diagnosis task",
        "",
        str(context.get("task_description") or ""),
        "",
        "# Public topology summary",
        json.dumps(
            {
                "scenario_name": context.get("scenario_name"),
                "scenario_topo_size": context.get("scenario_topo_size"),
                "topology": context.get("topology", []),
                "diagnosis_mcp_servers": context.get("diagnosis_mcp_servers", []),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
    ]
    memory_context = str(context.get("memory_context") or "").strip()
    if memory_context:
        parts.extend(["", "# Retrieved procedural memory", memory_context])
    tool_note = str(context.get("tool_evolution_note") or "").strip()
    if tool_note:
        parts.extend(["", "# Evolved toolbox note", tool_note])
    return "\n".join(parts)


async def _run_diagnosis(
    *,
    session_id: str,
    working_dir: Path,
    context: dict[str, Any],
    backend: str,
    model: str,
    max_steps: int,
) -> str:
    client = MultiServerMCPClient(connections=_diagnosis_config(session_id, context))
    tools = await client.get_tools()
    for tool in tools:
        tool.handle_tool_error = True
        tool.handle_validation_error = True
    llm = load_model(backend, model)
    agent = create_agent(
        model=llm,
        system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
        tools=tools,
        name="HarnessDiagnosisAgent",
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=_build_task_prompt(context))]},
        config={
            "callbacks": [
                AgentCallbackLogger(
                    agent="diagnosis_agent",
                    session_dir=str(working_dir),
                )
            ],
            "recursion_limit": max_steps,
        },
    )
    messages = result.get("messages") or []
    return str(messages[-1].content) if messages else ""


async def _run_submission(
    *,
    session_id: str,
    working_dir: Path,
    diagnosis_report: str,
    backend: str,
    model: str,
    max_steps: int,
) -> str:
    client = MultiServerMCPClient(
        connections=MCPServerConfig(session_id=session_id).load_config(if_submit=True)
    )
    tools = await client.get_tools()
    for tool in tools:
        tool.handle_tool_error = True
        tool.handle_validation_error = True
    llm = load_model(backend, model)
    agent = create_agent(
        model=llm,
        system_prompt=SUBMISSION_SYSTEM_PROMPT,
        tools=tools,
        name="HarnessSubmissionAgent",
    )
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Based on this diagnosis report, call the task tools and "
                        f"submit the final answer:\n\n{diagnosis_report}"
                    )
                )
            ]
        },
        config={
            "callbacks": [
                AgentCallbackLogger(
                    agent="submission_agent",
                    session_dir=str(working_dir),
                )
            ],
            "recursion_limit": max(8, min(24, max_steps)),
        },
    )
    messages = result.get("messages") or []
    return str(messages[-1].content) if messages else ""


async def run_target(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    context = _load_case_context(dataset_dir)
    diagnosis_report = ""
    submission_result = ""
    error: str | None = None
    try:
        diagnosis_report = await _run_diagnosis(
            session_id=args.session_id,
            working_dir=working_dir,
            context=context,
            backend=args.backend,
            model=args.model,
            max_steps=args.max_steps,
        )
        submission_result = await _run_submission(
            session_id=args.session_id,
            working_dir=working_dir,
            diagnosis_report=diagnosis_report,
            backend=args.backend,
            model=args.model,
            max_steps=args.max_steps,
        )
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (working_dir / "harness_error.json").write_text(
            json.dumps(
                {"error": error, "traceback": traceback.format_exc()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(error, flush=True)
        return 1
    finally:
        _write_execution(
            working_dir=working_dir,
            benchmark_index=context.get("benchmark_index"),
            context=context,
            diagnosis_report=diagnosis_report,
            submission_result=submission_result,
            error=error,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a NIKA SIA-H target agent.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run_target(build_parser().parse_args())))


if __name__ == "__main__":
    main()
