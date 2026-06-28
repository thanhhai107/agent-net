"""Validation and sandbox execution for generated Python tools."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent.tool_evolution.models import GeneratedTool

SAFE_IMPORT_MODULES = frozenset(
    {
        "cmath",
        "collections",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
    }
)
FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
        "vars",
    }
)


def validate_generated_tool_code(tool: GeneratedTool) -> list[str]:
    """Statically validate generated Python source before sandbox execution."""
    try:
        module = ast.parse(tool.code)
    except SyntaxError as exc:
        raise ValueError(f"generated tool code has invalid syntax: {exc}") from exc

    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not any(node.name == tool.name for node in functions):
        raise ValueError(f"generated code must define function '{tool.name}'")

    for node in module.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)
        ):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        raise ValueError(
            "generated code may contain only imports and function definitions"
        )

    declared = {parameter.name for parameter in tool.parameters}
    target = next(node for node in functions if node.name == tool.name)
    actual = {argument.arg for argument in target.args.args}
    vararg = target.args.vararg or target.args.kwarg
    if vararg:
        raise ValueError("generated tool functions may not use *args or **kwargs")
    if actual != declared:
        missing = declared - actual
        extra = actual - declared
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("extra " + ", ".join(sorted(extra)))
        raise ValueError(
            "function signature does not match parameters: " + "; ".join(details)
        )

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in SAFE_IMPORT_MODULES:
                    raise ValueError(
                        f"import is not allowed in generated tools: {root}"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in SAFE_IMPORT_MODULES:
                raise ValueError(f"import is not allowed in generated tools: {root}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FORBIDDEN_CALLS:
                raise ValueError(f"call is not allowed in generated tools: {name}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal state is not allowed in generated tools")

    return ["syntax", "function_signature", "import_allowlist", "no_forbidden_calls"]


def run_generated_tool(
    tool: GeneratedTool,
    arguments: dict[str, Any],
    *,
    timeout: int | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Execute a generated Python tool in Docker when available, else a subprocess."""
    checks = validate_generated_tool_code(tool)
    timeout = timeout or int(os.environ.get("NIKA_GENERATED_TOOL_TIMEOUT", "30"))
    mode = (mode or os.environ.get("NIKA_GENERATED_TOOL_RUNNER", "auto")).lower()
    if mode not in {"auto", "docker", "local"}:
        raise ValueError(
            "NIKA_GENERATED_TOOL_RUNNER must be one of: auto, docker, local"
        )
    if mode in {"auto", "docker"} and shutil.which("docker"):
        try:
            result = _run_in_docker(tool, arguments, timeout=timeout)
            result["checks"] = checks + ["docker_sandbox"]
            return result
        except Exception:
            if mode == "docker":
                raise
    result = _run_in_subprocess(tool, arguments, timeout=timeout)
    result["checks"] = checks + ["subprocess_runner"]
    return result


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _runner_script(tool: GeneratedTool, arguments: dict[str, Any]) -> str:
    return f"""
import asyncio
import json

namespace = {{}}
exec({tool.code!r}, namespace)
function = namespace[{tool.name!r}]
arguments = json.loads({json.dumps(arguments, default=str)!r})
value = function(**arguments)
if hasattr(value, "__await__"):
    value = asyncio.run(value)
print(json.dumps({{"success": True, "result": value}}, ensure_ascii=False, default=str))
"""


def _parse_runner_output(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if completed.returncode != 0:
        return {
            "success": False,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "result": None,
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "success": False,
            "stdout": "",
            "stderr": "missing runner output",
            "result": None,
        }
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "stdout": completed.stdout,
            "stderr": f"invalid runner output: {exc}",
            "result": None,
        }
    payload.setdefault("stdout", completed.stdout)
    payload.setdefault("stderr", completed.stderr)
    return payload


def _run_in_subprocess(
    tool: GeneratedTool,
    arguments: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", encoding="utf-8", delete=False
    ) as handle:
        handle.write(_runner_script(tool, arguments))
        script_path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return _parse_runner_output(completed)
    finally:
        Path(script_path).unlink(missing_ok=True)


def _run_in_docker(
    tool: GeneratedTool,
    arguments: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    image = os.environ.get("NIKA_GENERATED_TOOL_DOCKER_IMAGE", "python:3.11-slim")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "run_generated_tool.py"
        script_path.write_text(_runner_script(tool, arguments), encoding="utf-8")
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{Path(tmp).resolve()}:/app:ro",
                "-w",
                "/app",
                image,
                "python",
                "run_generated_tool.py",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    return _parse_runner_output(completed)
