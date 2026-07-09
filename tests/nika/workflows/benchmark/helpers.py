"""Shared helpers for benchmark-related tests."""

from __future__ import annotations

from nika.config import BENCHMARK_DIR
from nika.workflows.benchmark.load_config import load_benchmark_yaml


def inject_params_from_benchmark_yaml(
    scenario: str,
    problem: str,
    topo_size: str = "",
) -> dict[str, str]:
    """Load inject parameters for a benchmark row from bundled YAML configs."""
    normalized_topo = topo_size or ""
    for yaml_name in ("benchmark_full.yaml", "benchmark_selected.yaml"):
        path = BENCHMARK_DIR / yaml_name
        if not path.is_file():
            continue
        for row in load_benchmark_yaml(path):
            if (
                row["scenario"] == scenario
                and row["problem"] == problem
                and (row.get("topo_size") or "") == normalized_topo
            ):
                return dict(row["inject"])
    raise ValueError(
        f"No benchmark inject entry for scenario={scenario!r}, problem={problem!r}, "
        f"topo_size={topo_size!r}; pass explicit inject parameters."
    )
