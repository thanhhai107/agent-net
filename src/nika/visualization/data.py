"""Read NIKA session artifacts without requiring a running Kathará lab."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nika.config import RESULTS_DIR, SESSIONS_DIR


@dataclass(frozen=True)
class SessionBundle:
    """All artifacts used by the visualization dashboard."""

    session_id: str
    meta: dict[str, Any]
    ground_truth: dict[str, Any]
    submission: dict[str, Any]
    metrics: dict[str, Any]
    judge: dict[str, Any]
    events: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    failure_injections: list[dict[str, Any]]
    session_dir: Path


@dataclass(frozen=True)
class ReplayStep:
    """One human-readable step in an agent execution trace."""

    index: int
    timestamp: str
    agent: str
    kind: str
    title: str
    input: str
    output: str
    devices: tuple[str, ...]
    raw: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        pass
    return rows


def discover_sessions(
    *,
    results_dir: Path = RESULTS_DIR,
    sessions_dir: Path = SESSIONS_DIR,
) -> list[dict[str, Any]]:
    """Return running and finished sessions, newest first."""
    discovered: dict[str, dict[str, Any]] = {}

    if results_dir.exists():
        for run_path in results_dir.rglob("run.json"):
            if "0_summary" in run_path.relative_to(results_dir).parts:
                continue
            run = _read_json(run_path)
            session_id = str(run.get("session_id") or run_path.parent.name)
            run["session_id"] = session_id
            run["session_dir"] = str(run_path.parent)
            run["_source"] = "results"
            discovered[session_id] = run

    if sessions_dir.exists():
        for runtime_path in sessions_dir.glob("*.json"):
            runtime = _read_json(runtime_path)
            session_id = str(runtime.get("session_id") or runtime_path.stem)
            base = discovered.get(session_id, {})
            merged = {**base, **runtime}
            merged["session_id"] = session_id
            merged["_source"] = "runtime"
            discovered[session_id] = merged

    valid_sessions = []
    for item in discovered.values():
        s_dir = item.get("session_dir") or str(results_dir / item["session_id"])
        if Path(s_dir).exists():
            valid_sessions.append(item)

    return sorted(
        valid_sessions,
        key=lambda item: str(item.get("created_at") or item.get("session_id") or ""),
        reverse=True,
    )


def load_session_bundle(
    session_id: str,
    *,
    results_dir: Path = RESULTS_DIR,
    sessions_dir: Path = SESSIONS_DIR,
) -> SessionBundle:
    """Load one session and its optional result artifacts."""
    sessions = {
        str(item["session_id"]): item
        for item in discover_sessions(
            results_dir=results_dir, sessions_dir=sessions_dir
        )
    }
    if session_id not in sessions:
        raise FileNotFoundError(f"Session '{session_id}' not found.")

    meta = sessions[session_id]
    session_dir = Path(meta.get("session_dir") or results_dir / session_id)
    run_meta = _read_json(session_dir / "run.json")
    meta = {**run_meta, **meta}
    failure_injections = meta.get("failure_injections", [])
    if not isinstance(failure_injections, list):
        failure_injections = []

    return SessionBundle(
        session_id=session_id,
        meta=meta,
        ground_truth=_read_json(session_dir / "ground_truth.json"),
        submission=_read_json(session_dir / "submission.json"),
        metrics=_read_json(session_dir / "eval_metrics.json"),
        judge=_read_json(session_dir / "llm_judge.json"),
        events=_read_jsonl(session_dir / "events.jsonl"),
        messages=_read_jsonl(session_dir / "messages.jsonl"),
        failure_injections=failure_injections,
        session_dir=session_dir,
    )


_TOPOLOGY_PAIR = re.compile(r"\(([^,()]+),\s*([^()]+)\)")


def parse_topology(meta: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract endpoint pairs from the topology snapshot or task description."""
    raw_topology = meta.get("topology")
    pairs: list[tuple[str, str]] = []
    if isinstance(raw_topology, list):
        for raw_pair in raw_topology:
            if isinstance(raw_pair, (list, tuple)) and len(raw_pair) == 2:
                pairs.append((str(raw_pair[0]).strip(), str(raw_pair[1]).strip()))
    if pairs:
        return pairs

    description = str(meta.get("task_description") or "")
    topology_text = (
        description.split("Topology:", 1)[1]
        if "Topology:" in description
        else description
    )
    return [
        (left.strip(), right.strip())
        for left, right in _TOPOLOGY_PAIR.findall(topology_text)
    ]


def endpoint_parts(endpoint: str) -> tuple[str, str]:
    """Split ``device:interface`` while tolerating missing interfaces."""
    device, separator, interface = endpoint.partition(":")
    return device.strip(), interface.strip() if separator else ""


def faulty_devices(payload: dict[str, Any]) -> set[str]:
    values = payload.get("faulty_devices", [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return set()
    return {endpoint_parts(str(value))[0] for value in values}


def fault_endpoints(bundle: SessionBundle) -> set[tuple[str, str]]:
    """Find interfaces implicated by injection records and verification events."""
    found: set[tuple[str, str]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            host = value.get("host") or value.get("host_name")
            interface = (
                value.get("intf") or value.get("intf_name") or value.get("interface")
            )
            if host and interface:
                found.add((str(host), str(interface)))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(bundle.failure_injections)
    for event in bundle.events:
        visit(event.get("data"))
        message = str(event.get("message") or "")
        match = re.search(r"\bon\s+([\w.-]+):(eth\d+|[\w.-]+)", message)
        if match:
            found.add((match.group(1), match.group(2)))
    return found


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _mentioned_devices(value: Any, device_names: set[str]) -> tuple[str, ...]:
    """Return topology device names mentioned in a trace payload."""
    searchable = _display_value(value).lower()
    found = [
        device
        for device in sorted(device_names, key=lambda item: (-len(item), item))
        if re.search(rf"(?<![\w.-]){re.escape(device.lower())}(?![\w.-])", searchable)
    ]
    return tuple(sorted(found))


def replay_steps(
    bundle: SessionBundle,
    endpoint_pairs: list[tuple[str, str]] | None = None,
) -> list[ReplayStep]:
    """Normalize agent JSONL records into replayable steps."""
    pairs = (
        endpoint_pairs if endpoint_pairs is not None else parse_topology(bundle.meta)
    )
    device_names = {
        endpoint_parts(endpoint)[0]
        for pair in pairs
        for endpoint in pair
        if endpoint_parts(endpoint)[0]
    }
    steps: list[ReplayStep] = []
    pending_tools: dict[str, list[int]] = {}

    for message in bundle.messages:
        event = str(message.get("event") or "")
        agent = str(message.get("agent") or "agent")
        timestamp = str(message.get("timestamp") or "")

        if event == "tool_start":
            tool = message.get("tool") or {}
            name = (
                str(tool.get("name") or "tool") if isinstance(tool, dict) else str(tool)
            )
            input_text = _display_value(message.get("input"))
            step = ReplayStep(
                index=len(steps),
                timestamp=timestamp,
                agent=agent,
                kind="tool",
                title=name,
                input=input_text,
                output="",
                devices=_mentioned_devices(input_text, device_names),
                raw=message,
            )
            steps.append(step)
            pending_tools.setdefault(agent, []).append(step.index)
            continue

        if event in {"tool_end", "tool_error"}:
            pending = pending_tools.get(agent, [])
            if pending:
                step_index = pending.pop()
                current = steps[step_index]
                output = message.get("output") or message.get("error")
                output_text = _display_value(output)
                merged_devices = tuple(
                    sorted(
                        set(current.devices)
                        | set(_mentioned_devices(output_text, device_names))
                    )
                )
                steps[step_index] = ReplayStep(
                    index=current.index,
                    timestamp=current.timestamp,
                    agent=current.agent,
                    kind="error" if event == "tool_error" else current.kind,
                    title=current.title,
                    input=current.input,
                    output=output_text,
                    devices=merged_devices,
                    raw={**current.raw, "result": message},
                )
            continue

        if event in {"llm_start", "llm_end", "llm_end_error"}:
            if event == "llm_start":
                title = "Prompt received"
                input_text = _display_value(message.get("messages"))
                output_text = ""
                kind = "prompt"
            else:
                title = "Agent response" if event == "llm_end" else "Agent error"
                input_text = ""
                output_text = _display_value(
                    message.get("text") or message.get("error")
                )
                kind = "response" if event == "llm_end" else "error"
            payload = f"{input_text}\n{output_text}"
            steps.append(
                ReplayStep(
                    index=len(steps),
                    timestamp=timestamp,
                    agent=agent,
                    kind=kind,
                    title=title,
                    input=input_text,
                    output=output_text,
                    devices=_mentioned_devices(payload, device_names),
                    raw=message,
                )
            )
            continue

    return steps
