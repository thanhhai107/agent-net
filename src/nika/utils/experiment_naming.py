"""Shared sequential names for experiments and learning libraries."""

from __future__ import annotations

import re
from pathlib import Path

from agent.extensions.config import PROCEDURAL_MEMORY_DIR, TOOL_REFINEMENT_DIR
from nika.config import RESULTS_DIR, RUNTIME_DIR

STREAMLIT_RUNS_DIR = RUNTIME_DIR / "streamlit_runs"
SEQUENCE_WIDTH = 2


def slugify_experiment_name(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    return slug or "experiment"


def experiment_stem(benchmark: str | Path) -> str:
    del benchmark
    return "experiment"


# Compatibility for callers that imported the old helper. New IDs are always
# generated through the experiment-prefixed stem above.
benchmark_stem = experiment_stem


def experiment_id(name: str, index: int) -> str:
    return f"{slugify_experiment_name(name)}-{index:0{SEQUENCE_WIDTH}d}"


def _existing_indices(prefix: str, roots: list[Path]) -> set[int]:
    pattern = re.compile(
        rf"^{re.escape(slugify_experiment_name(prefix))}-(\d{{{SEQUENCE_WIDTH},}})$"
    )
    indices: set[int] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                indices.add(int(match.group(1)))
    return indices


def next_experiment_id(
    benchmark: str | Path,
    *,
    roots: list[Path] | None = None,
) -> str:
    stem = experiment_stem(benchmark)
    roots = roots or [
        Path(RESULTS_DIR),
        STREAMLIT_RUNS_DIR,
        Path(PROCEDURAL_MEMORY_DIR),
        Path(TOOL_REFINEMENT_DIR),
    ]
    used = _existing_indices(stem, roots)
    index = 1
    while index in used:
        index += 1
    return experiment_id(stem, index)
