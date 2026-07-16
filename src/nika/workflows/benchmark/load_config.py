"""Load benchmark case definitions from YAML."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def is_no_fault_problem(problem: str) -> bool:
    """Return whether a benchmark problem is the explicit clean control."""

    return problem.strip().lower() == "no_fault"


@dataclass(frozen=True)
class BenchmarkManifest:
    """Validated benchmark manifest plus a stable resume fingerprint."""

    path: Path
    role: str
    seed: int | None
    counts: dict[str, int]
    cases: list[dict[str, Any]]
    fingerprint: str


def benchmark_case_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return the scenario/topology/problem identity used for leakage checks."""

    return (
        str(row["scenario"]),
        str(row.get("topo_size") or ""),
        str(row["problem"]),
    )


def _manifest_fingerprint(
    *,
    role: str,
    seed: int | None,
    cases: list[dict[str, Any]],
) -> str:
    payload = {
        "benchmark_role": role,
        "seed": seed,
        "cases": cases,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_benchmark_yaml(path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark cases from a YAML file.

    Expected shape::

        cases:
          - scenario: simple_bgp
            topo_size: null
            problem: link_down
            inject:
              host_name: pc1
              intf_name: eth0
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"Invalid benchmark YAML (missing top-level 'cases'): {path}")
    cases = data["cases"]
    if not isinstance(cases, list):
        raise ValueError(f"Invalid benchmark YAML ('cases' must be a list): {path}")
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError(f"Benchmark case {idx} must be a mapping")
        required = ("scenario", "problem")
        for key in required:
            if key not in row:
                raise ValueError(f"Benchmark case {idx} missing required field {key!r}")
        topo = row.get("topo_size")
        if topo is None:
            topo = ""
        inject = row.get("inject") or {}
        if not isinstance(inject, dict):
            raise ValueError(f"Benchmark case {idx} 'inject' must be a mapping")
        problem = str(row["problem"])
        if not inject and not is_no_fault_problem(problem):
            raise ValueError(
                f"Benchmark case {idx} ({row.get('scenario')}/{row.get('problem')}) "
                f"missing non-empty inject map in {path}"
            )
        normalized.append(
            {
                "scenario": str(row["scenario"]),
                "problem": problem,
                "topo_size": "" if topo in ("-", "", None) else str(topo),
                "inject": {str(k): str(v) for k, v in inject.items()},
            }
        )
    return normalized


def load_benchmark_manifest(
    path: str | Path,
    *,
    expected_role: str | None = None,
) -> BenchmarkManifest:
    """Load a benchmark with role/count metadata for experiment pipelines.

    Custom manifests may omit ``benchmark_role`` and ``counts``. When a caller
    supplies ``expected_role``, the missing role is filled from that execution
    context; an explicitly conflicting role is always rejected.
    """

    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark YAML (expected a mapping): {source}")
    cases = load_benchmark_yaml(source)
    if not cases:
        raise ValueError(f"Benchmark YAML has no cases: {source}")

    declared_role = str(data.get("benchmark_role") or "").strip().lower()
    if declared_role and declared_role not in {"learning", "evaluation"}:
        raise ValueError(
            "Invalid benchmark YAML ('benchmark_role' must be learning or "
            f"evaluation): {source}"
        )
    if expected_role not in {None, "learning", "evaluation"}:
        raise ValueError(f"Invalid expected benchmark role: {expected_role!r}")
    if declared_role and expected_role and declared_role != expected_role:
        raise ValueError(
            f"Benchmark role mismatch for {source}: expected {expected_role}, "
            f"found {declared_role}"
        )
    role = declared_role or expected_role or "evaluation"

    seed_value = data.get("seed")
    if seed_value is not None and (
        isinstance(seed_value, bool) or not isinstance(seed_value, int)
    ):
        raise ValueError(
            f"Invalid benchmark YAML ('seed' must be an integer): {source}"
        )
    seed = seed_value if isinstance(seed_value, int) else None

    no_fault = sum(is_no_fault_problem(row["problem"]) for row in cases)
    actual_counts = {
        "total": len(cases),
        "fault": len(cases) - no_fault,
        "no_fault": no_fault,
    }
    raw_counts = data.get("counts")
    if raw_counts is not None:
        if not isinstance(raw_counts, dict):
            raise ValueError(
                f"Invalid benchmark YAML ('counts' must be a mapping): {source}"
            )
        declared_counts: dict[str, int] = {}
        for name in ("total", "fault", "no_fault"):
            value = raw_counts.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Invalid benchmark YAML ('counts.{name}' must be an integer): "
                    f"{source}"
                )
            declared_counts[name] = value
        if declared_counts != actual_counts:
            raise ValueError(
                f"Benchmark counts mismatch for {source}: declared "
                f"{declared_counts}, actual {actual_counts}"
            )

    identities = [benchmark_case_identity(row) for row in cases]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Benchmark contains duplicate case identities: {source}")

    return BenchmarkManifest(
        path=source.resolve(),
        role=role,
        seed=seed,
        counts=actual_counts,
        cases=cases,
        fingerprint=_manifest_fingerprint(role=role, seed=seed, cases=cases),
    )
