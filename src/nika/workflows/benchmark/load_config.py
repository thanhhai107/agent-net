"""Load benchmark case definitions from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def is_no_fault_problem(problem: str) -> bool:
    """Return whether a benchmark problem is the explicit clean control."""

    return problem.strip().lower() == "no_fault"


def load_benchmark_evolve_first_cases(path: str | Path) -> int | None:
    """Return the optional evolve/read curriculum boundary from a benchmark."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"Invalid benchmark YAML (missing top-level 'cases'): {path}")
    cases = data["cases"]
    if not isinstance(cases, list):
        raise ValueError(f"Invalid benchmark YAML ('cases' must be a list): {path}")
    cutoff = data.get("evolve_first_cases")
    if cutoff is None:
        return None
    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        raise ValueError(
            f"Invalid benchmark YAML ('evolve_first_cases' must be an integer): {path}"
        )
    if not 0 <= cutoff <= len(cases):
        raise ValueError(
            "Invalid benchmark YAML ('evolve_first_cases' must be between 0 "
            f"and {len(cases)}): {path}"
        )
    return cutoff


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
