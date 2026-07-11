"""Mock LLM agent that simulates BasicReActAgent behaviour without a real LLM.

The agent mirrors the two-phase architecture of BasicReActAgent:
  1. diagnosis phase  – calls lab MCP tools and emits a deterministic report
  2. submission phase – calls list_avail_problems + submit via task MCP server

Test-only. See ``tests/README.md``.
"""

import json
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.utils.mcp_client import begin_submission_mcp_phase, load_session_mcp_config
from agent.utils.mcp_servers import select_diagnosis_servers
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.runtime.factory import resolve_backend
from nika.utils.session import Session
from nika.utils.session_store import SessionStore

_KATHARA_DIAGNOSIS_REPORT = (
    "Anomaly detected: high packet loss between pc1 and pc2.  "
    "BGP routes on r1 show an unreachable prefix (10.0.1.0/24).  "
    "Suspected root cause: link failure on the path between r1 and pc2."
)

_CLAB_DIAGNOSIS_REPORT = (
    "Anomaly detected: high packet loss between client1 and client2.  "
    "BGP routes on leaf1 show an unreachable prefix (10.0.0.24/31).  "
    "Suspected root cause: link failure on the path between leaf1 and spine."
)


def _tool_text_list(result: object) -> list[str]:
    """Normalize MCP tool output into plain strings."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return [result]
    if not isinstance(result, list):
        return [str(result)]

    texts: list[str] = []
    for item in result:
        if isinstance(item, dict) and "text" in item:
            texts.append(str(item["text"]))
        else:
            texts.append(str(item))
    return texts


def _mock_diagnosis_tool_calls(
    backend: str, server_names: list[str]
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = [("get_reachability", {})]
    if "pingmesh_mcp_server" in server_names:
        calls.append(("run_pingmesh_snapshot", {}))
    if backend == "containerlab":
        calls.append(("ping_pair", {"host_a": "client1", "host_b": "client2"}))
        if "containerlab_srl_mcp_server" in server_names:
            calls.append(("srl_show_ip_route", {"device_name": "leaf1"}))
        else:
            calls.append(("exec_shell", {"host_name": "leaf1", "command": "hostname"}))
    else:
        calls.append(("ping_pair", {"host_a": "pc1", "host_b": "pc2"}))
        if "kathara_frr_mcp_server" in server_names:
            calls.append(("frr_show_ip_route", {"router_name": "r1"}))
        else:
            calls.append(("exec_shell", {"host_name": "pc1", "command": "hostname"}))
    return calls


def _mock_faulty_device(backend: str) -> str:
    return "leaf1" if backend == "containerlab" else "pc1"


def _mock_diagnosis_report(backend: str) -> str:
    return (
        _CLAB_DIAGNOSIS_REPORT
        if backend == "containerlab"
        else _KATHARA_DIAGNOSIS_REPORT
    )


class MockAgent:
    """Deterministic mock agent that mirrors the BasicReActAgent interface."""

    def __init__(
        self,
        session_id: str,
        model: str = "mock-v1",
        max_steps: int = 20,
    ) -> None:
        self.session_id = session_id
        self.model = model
        self.max_steps = max_steps
        self.session = Session()
        self.session.load_running_session(session_id=session_id)

    def load_session(self) -> None:
        self.session.load_running_session(session_id=self.session_id)

    async def run(self, task_description: str) -> dict[str, Any]:
        self.load_session()
        diagnosis_report = await self._run_diagnosis(task_description)
        await self._run_submission(diagnosis_report)
        return {"diagnosis_report": diagnosis_report}

    async def _run_diagnosis(self, task_description: str) -> str:
        logger = self._make_logger(DIAGNOSIS)
        logger.log(
            "llm_start",
            {
                "messages": {"role": "user", "content": task_description},
                "model": {"name": self.model},
            },
        )

        session_row = SessionStore().get_session(self.session_id)
        backend = resolve_backend(session_row)
        scenario = str(session_row.get("scenario_name") or "")
        server_names = select_diagnosis_servers(scenario, backend=backend)

        config = load_session_mcp_config(self.session_id, scenario, backend=backend)
        client = MultiServerMCPClient(connections=config)
        tools = {tool.name: tool for tool in await client.get_tools()}

        tool_calls = _mock_diagnosis_tool_calls(backend, server_names)
        diagnosis_report = _mock_diagnosis_report(backend)

        for tool_name, tool_input in tool_calls:
            if tool_name not in tools:
                continue
            logger.log(
                "tool_start",
                {"tool": {"name": tool_name}, "input": json.dumps(tool_input)},
            )
            tool_output = await tools[tool_name].ainvoke(tool_input)
            logger.log(
                "tool_end",
                {
                    "output": str(tool_output),
                    "output_type": type(tool_output).__name__,
                },
            )

        logger.log("llm_end", {"text": diagnosis_report})
        return diagnosis_report

    async def _run_submission(self, diagnosis_report: str) -> None:
        logger = self._make_logger(SUBMISSION)

        session_row = SessionStore().get_session(self.session_id)
        backend = resolve_backend(session_row)
        faulty_device = _mock_faulty_device(backend)
        scenario = str(session_row.get("scenario_name") or "")

        logger.log(
            "llm_start",
            {
                "messages": {
                    "role": "user",
                    "content": (
                        f"Based on diagnosis: {diagnosis_report}. Please call list_avail_problems and then submit."
                    ),
                },
                "model": {"name": self.model},
            },
        )

        begin_submission_mcp_phase(self.session_id)
        config = load_session_mcp_config(
            self.session_id, scenario, backend=backend
        )
        client = MultiServerMCPClient(connections=config)
        tools = {tool.name: tool for tool in await client.get_tools()}

        logger.log(
            "tool_start", {"tool": {"name": "list_avail_problems"}, "input": "{}"}
        )
        avail_raw = await tools["list_avail_problems"].ainvoke({})
        avail = _tool_text_list(avail_raw)
        session_root_cause = getattr(self.session, "root_cause_name", None)
        if session_root_cause in avail:
            mock_root_cause = session_root_cause
        else:
            mock_root_cause = avail[0] if avail else "link_down"
        logger.log(
            "tool_end",
            {"output": json.dumps(avail[:5]) + " ...", "output_type": "list"},
        )

        submission: dict[str, Any] = {
            "is_anomaly": True,
            "faulty_devices": [faulty_device],
            "root_cause_name": [mock_root_cause],
        }
        logger.log(
            "tool_start",
            {"tool": {"name": "submit"}, "input": json.dumps(submission)},
        )
        submit_result = await tools["submit"].ainvoke(submission)
        logger.log(
            "tool_end",
            {"output": str(submit_result), "output_type": type(submit_result).__name__},
        )

        logger.log(
            "llm_end",
            {
                "text": (
                    f"Submitted: root cause = {mock_root_cause}, "
                    f"faulty device = {faulty_device}"
                )
            },
        )

    def _make_logger(self, agent_name: str):
        """Return a MessageLogger for *agent_name*."""
        from agent.utils.loggers import MessageLogger  # noqa: PLC0415

        return MessageLogger(agent=agent_name, session_dir=self.session.session_dir)
