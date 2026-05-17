"""Typed parameter schema helpers for failure injections."""

from dataclasses import dataclass
import ast
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal


ParamType = Literal["str", "int", "float", "bool"]


@dataclass(frozen=True)
class FailureParamField:
    name: str
    param_type: ParamType
    description: str
    required: bool = False
    default: Any = None
    choices: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class FailureParamSchema:
    problem_name: str
    summary: str
    fields: tuple[FailureParamField, ...]
    example: str


_SCHEMAS: dict[str, FailureParamSchema] = {}


def list_schema_problem_names() -> list[str]:
    names = set(_SCHEMAS.keys())
    names.update(_load_schemas_from_source().keys())
    return sorted(names)


def _get_problem_defined_schema(problem_name: str) -> FailureParamSchema | None:
    return _load_schemas_from_source().get(problem_name)


def _const_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def _parse_field_call(node: ast.AST) -> FailureParamField | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "FailureParamField":
        return None
    args = list(node.args)
    if len(args) < 3:
        return None
    name = _const_value(args[0])
    param_type = _const_value(args[1])
    description = _const_value(args[2])
    if not isinstance(name, str) or not isinstance(param_type, str) or not isinstance(description, str):
        return None

    kwargs: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        if isinstance(kw.value, ast.Tuple):
            vals = tuple(_const_value(v) for v in kw.value.elts)
            kwargs[kw.arg] = vals
        else:
            kwargs[kw.arg] = _const_value(kw.value)
    return FailureParamField(
        name=name,
        param_type=param_type,  # type: ignore[arg-type]
        description=description,
        required=bool(kwargs.get("required", False)),
        default=kwargs.get("default", None),
        choices=kwargs.get("choices", None),
    )


def _parse_schema_call(node: ast.AST) -> FailureParamSchema | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "FailureParamSchema":
        return None

    data: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        if kw.arg == "fields" and isinstance(kw.value, ast.Tuple):
            fields: list[FailureParamField] = []
            for elt in kw.value.elts:
                parsed = _parse_field_call(elt)
                if parsed is not None:
                    fields.append(parsed)
            data["fields"] = tuple(fields)
            continue
        data[kw.arg] = _const_value(kw.value)

    if not isinstance(data.get("problem_name"), str):
        return None
    if not isinstance(data.get("summary"), str):
        return None
    if not isinstance(data.get("example"), str):
        return None
    if "fields" not in data:
        data["fields"] = tuple()

    return FailureParamSchema(
        problem_name=data["problem_name"],
        summary=data["summary"],
        fields=data["fields"],
        example=data["example"],
    )


@lru_cache(maxsize=1)
def _load_schemas_from_source() -> dict[str, FailureParamSchema]:
    base = Path(__file__).resolve().parents[1] / "orchestrator" / "problems"
    schemas: dict[str, FailureParamSchema] = {}
    for path in base.rglob("*.py"):
        if path.name in {"prob_pool.py", "problem_base.py", "multi_problems.py"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "FAILURE_PARAM_SCHEMA":
                        schema = _parse_schema_call(stmt.value)
                        if schema is not None:
                            schemas[schema.problem_name] = schema
    return schemas


def get_failure_param_schema(problem_name: str) -> FailureParamSchema | None:
    schema = _get_problem_defined_schema(problem_name)
    if schema is not None:
        return schema
    return _SCHEMAS.get(problem_name)


def _coerce_value(raw: str, target_type: ParamType) -> Any:
    if target_type == "str":
        return raw
    if target_type == "int":
        return int(raw)
    if target_type == "float":
        return float(raw)
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def resolve_failure_params(
    problem_name: str,
    overrides: dict[str, str] | None,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = get_failure_param_schema(problem_name)
    if schema is None:
        if overrides:
            raise ValueError(
                f"Problem '{problem_name}' does not define FAILURE_PARAM_SCHEMA; "
                "cannot apply --set overrides."
            )
        return {}

    ov = dict(overrides or {})
    allowed = {field.name for field in schema.fields}
    unknown = sorted(k for k in ov if k not in allowed)
    if unknown:
        raise ValueError(
            f"Unsupported parameter(s) for '{problem_name}': {', '.join(unknown)}. "
            f"Supported keys: {', '.join(sorted(allowed))}."
        )

    ctx = dict(context or {})
    resolved: dict[str, Any] = {}
    for field in schema.fields:
        if field.name in ov:
            try:
                value = _coerce_value(ov[field.name], field.param_type)
            except ValueError as exc:
                raise ValueError(f"Invalid value for '{field.name}': {exc}") from exc
        elif field.name in ctx and ctx[field.name] is not None:
            value = ctx[field.name]
        elif field.default is not None:
            value = field.default
        elif field.required:
            raise ValueError(f"Missing required parameter '{field.name}' for '{problem_name}'.")
        else:
            value = None

        if value is not None and field.choices is not None and value not in field.choices:
            choices = ", ".join(str(v) for v in field.choices)
            raise ValueError(f"Invalid value for '{field.name}': {value!r}. Allowed: {choices}.")
        if value is not None:
            resolved[field.name] = value
    return resolved

