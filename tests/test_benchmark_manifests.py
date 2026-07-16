"""Integrity and leakage checks for the canonical benchmark manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from benchmark.generate_benchmark import (
    EVALUATION_ONLY_PROBLEMS,
    NO_FAULT_CONTROLS,
    case_identity,
    select_learning_cases,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"


def _load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((BENCHMARK_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _assert_manifest_counts(
    manifest: dict[str, Any],
    *,
    role: str,
    total: int,
    fault: int,
    no_fault: int,
) -> None:
    cases = manifest["cases"]
    actual_no_fault = sum(row["problem"] == "no_fault" for row in cases)
    assert manifest["benchmark_role"] == role
    assert manifest["seed"] == 42
    assert manifest["counts"] == {
        "total": total,
        "fault": fault,
        "no_fault": no_fault,
    }
    assert len(cases) == total
    assert actual_no_fault == no_fault
    assert total - actual_no_fault == fault


def test_canonical_manifest_names_and_counts() -> None:
    learning = _load("benchmark_learning.yaml")
    selected = _load("benchmark_selected.yaml")
    full = _load("benchmark_full.yaml")

    _assert_manifest_counts(learning, role="learning", total=100, fault=90, no_fault=10)
    _assert_manifest_counts(selected, role="evaluation", total=56, fault=56, no_fault=0)
    _assert_manifest_counts(full, role="evaluation", total=702, fault=702, no_fault=0)

    legacy_names = (
        "benchmark_" + "evolve.yaml",
        "benchmark_" + "evaluate.yaml",
    )
    assert all(not (BENCHMARK_DIR / name).exists() for name in legacy_names)


def test_learning_controls_are_interleaved_and_empty() -> None:
    cases = _load("benchmark_learning.yaml")["cases"]
    controls = [row for row in cases if row["problem"] == "no_fault"]

    assert [
        index
        for index, row in enumerate(cases, start=1)
        if row["problem"] == "no_fault"
    ] == list(range(10, 101, 10))
    assert [(row["scenario"], row["topo_size"]) for row in controls] == list(
        NO_FAULT_CONTROLS
    )
    assert all(row["inject"] == {} for row in controls)


def test_learning_faults_are_unique_and_evaluation_disjoint() -> None:
    learning_cases = _load("benchmark_learning.yaml")["cases"]
    selected_cases = _load("benchmark_selected.yaml")["cases"]
    full_cases = _load("benchmark_full.yaml")["cases"]

    learning_identities = [case_identity(row) for row in learning_cases]
    selected_identities = {case_identity(row) for row in selected_cases}
    full_identities = {case_identity(row) for row in full_cases}
    learning_fault_identities = {
        case_identity(row) for row in learning_cases if row["problem"] != "no_fault"
    }
    assert len(learning_identities) == len(set(learning_identities))
    assert set(learning_identities).isdisjoint(selected_identities)
    assert learning_fault_identities.issubset(full_identities)

    learning_fault_problems = {
        row["problem"] for row in learning_cases if row["problem"] != "no_fault"
    }
    selected_problems = {row["problem"] for row in selected_cases}
    assert learning_fault_problems == selected_problems - EVALUATION_ONLY_PROBLEMS
    assert len(learning_fault_problems) == 54

    for problem in EVALUATION_ONLY_PROBLEMS:
        full_variants = [row for row in full_cases if row["problem"] == problem]
        assert len(full_variants) == 1
        assert case_identity(full_variants[0]) in selected_identities
        assert problem not in learning_fault_problems


def test_learning_selection_is_pure_and_deterministic() -> None:
    learning_cases = _load("benchmark_learning.yaml")["cases"]
    selected_cases = _load("benchmark_selected.yaml")["cases"]
    full_cases = _load("benchmark_full.yaml")["cases"]

    assert select_learning_cases(full_cases, selected_cases, seed=42) == learning_cases
    assert (
        select_learning_cases(
            list(reversed(full_cases)),
            list(reversed(selected_cases)),
            seed=42,
        )
        == learning_cases
    )
