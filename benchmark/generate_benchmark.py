"""Generate the learning, selected, and full benchmark manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from nika.net_env.net_env_pool import list_all_net_envs, scenario_requires_topo_size
from nika.problems.prob_pool import list_avail_problem_instances

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path)
from inject_resolve import (  # noqa: E402
    DEFAULT_SEED,
    resolve_inject_params,
    validate_benchmark_case,
)

# One best-matching traditional Kathara scenario per failure (k8s/llmd appear in full only).
SELECTED_SCENARIO_FOR_PROBLEM: dict[str, str] = {
    "arp_acl_block": "ospf_enterprise_dhcp",
    "arp_cache_poisoning": "ospf_enterprise_dhcp",
    "bgp_acl_block": "dc_clos_bgp",
    "bgp_asn_misconfig": "dc_clos_bgp",
    "bgp_blackhole_route_leak": "dc_clos_bgp",
    "bgp_hijacking": "dc_clos_service",
    "bgp_missing_route_advertisement": "dc_clos_bgp",
    "bmv2_switch_down": "p4_bloom_filter",
    "dhcp_missing_subnet": "ospf_enterprise_dhcp",
    "dhcp_service_down": "ospf_enterprise_dhcp",
    "dhcp_spoofed_dns": "ospf_enterprise_dhcp",
    "dhcp_spoofed_gateway": "ospf_enterprise_dhcp",
    "dhcp_spoofed_subnet": "ospf_enterprise_dhcp",
    "dns_lookup_latency": "ospf_enterprise_dhcp",
    "dns_port_blocked": "ospf_enterprise_dhcp",
    "dns_record_error": "ospf_enterprise_dhcp",
    "dns_service_down": "ospf_enterprise_dhcp",
    "flow_rule_loop": "sdn_clos",
    "flow_rule_shadowing": "sdn_clos",
    "frr_service_down": "ospf_enterprise_dhcp",
    "host_crash": "dc_clos_bgp",
    "host_incorrect_dns": "ospf_enterprise_dhcp",
    "host_incorrect_gateway": "ospf_enterprise_dhcp",
    "host_incorrect_ip": "ospf_enterprise_static",
    "host_incorrect_netmask": "ospf_enterprise_static",
    "host_ip_conflict": "dc_clos_bgp",
    "host_missing_ip": "ospf_enterprise_static",
    "host_static_blackhole": "dc_clos_bgp",
    "host_vpn_membership_missing": "rip_small_internet_vpn",
    "http_acl_block": "ospf_enterprise_dhcp",
    "icmp_acl_block": "ospf_enterprise_dhcp",
    "incast_traffic_network_limitation": "ospf_enterprise_dhcp",
    "link_bandwidth_throttling": "dc_clos_bgp",
    "link_detach": "dc_clos_bgp",
    "link_down": "dc_clos_bgp",
    "link_flap": "dc_clos_bgp",
    "link_fragmentation_disabled": "dc_clos_bgp",
    "link_high_packet_corruption": "dc_clos_bgp",
    "load_balancer_overload": "ospf_enterprise_dhcp",
    "mac_address_conflict": "ospf_enterprise_dhcp",
    "mpls_label_limit_exceeded": "p4_mpls",
    "ospf_acl_block": "ospf_enterprise_dhcp",
    "ospf_area_misconfiguration": "ospf_enterprise_dhcp",
    "ospf_neighbor_missing": "ospf_enterprise_dhcp",
    "p4_aggressive_detection_thresholds": "p4_bloom_filter",
    "p4_compilation_error_parser_state": "p4_bloom_filter",
    "p4_header_definition_error": "p4_bloom_filter",
    "p4_table_entry_misconfig": "p4_bloom_filter",
    "p4_table_entry_missing": "p4_bloom_filter",
    "receiver_resource_contention": "ospf_enterprise_dhcp",
    "sdn_controller_crash": "sdn_clos",
    "sender_application_delay": "ospf_enterprise_dhcp",
    "sender_resource_contention": "ospf_enterprise_dhcp",
    "southbound_port_block": "sdn_clos",
    "southbound_port_mismatch": "sdn_clos",
    "web_dos_attack": "ospf_enterprise_dhcp",
}

LEARNING_FAULT_CASES = 90
LEARNING_NO_FAULT_CASES = 10
EVALUATION_ONLY_PROBLEMS = frozenset(
    {
        "mpls_label_limit_exceeded",
        "p4_aggressive_detection_thresholds",
    }
)
NO_FAULT_CONTROLS: tuple[tuple[str, str | None], ...] = (
    ("p4_mpls", None),
    ("sdn_clos", "s"),
    ("ospf_enterprise_dhcp", "s"),
    ("dc_clos_service", "m"),
    ("simple_bgp", None),
    ("rip_small_internet_vpn", "s"),
    ("dc_clos_service", "l"),
    ("ospf_enterprise_static", "l"),
    ("sdn_clos", "l"),
    ("dc_clos_bgp", "l"),
)


def _topo_sizes_for_scenario(scenario: str) -> list[str]:
    if scenario_requires_topo_size(scenario):
        return ["s", "m", "l"]
    return [""]


def _make_row(scenario: str, problem: str, topo_size: str, *, seed: int) -> dict:
    inject = resolve_inject_params(problem, scenario, topo_size, seed=seed)
    validate_benchmark_case(scenario, problem, inject, topo_size)
    return {
        "scenario": scenario,
        "topo_size": topo_size or None,
        "problem": problem,
        "inject": inject,
    }


def iter_full_cases(*, seed: int) -> list[dict]:
    net_envs = list_all_net_envs()
    problem_instances = list_avail_problem_instances()
    rows: list[dict] = []

    for prob_name, problem_class in problem_instances.items():
        problem_instance = problem_class
        for net_env_name, net_env_cls in net_envs.items():
            if not set(problem_instance.TAGS).issubset(set(net_env_cls.TAGS)):
                continue
            for topo_size in _topo_sizes_for_scenario(net_env_name):
                rows.append(_make_row(net_env_name, prob_name, topo_size, seed=seed))
    return rows


def iter_selected_cases(*, seed: int) -> list[dict]:
    net_envs = list_all_net_envs()
    problem_instances = list_avail_problem_instances()
    rows: list[dict] = []

    for prob_name in sorted(problem_instances.keys()):
        scenario = SELECTED_SCENARIO_FOR_PROBLEM.get(prob_name)
        if scenario is None:
            raise ValueError(f"No selected scenario mapping for problem {prob_name!r}")
        net_env_cls = net_envs[scenario]
        problem_instance = problem_instances[prob_name]
        if not set(problem_instance.TAGS).issubset(set(net_env_cls.TAGS)):
            raise ValueError(
                f"Selected scenario {scenario} not tag-compatible with {prob_name} "
                f"(problem={problem_instance.TAGS}, scenario={net_env_cls.TAGS})"
            )
        topo_size = "s" if scenario_requires_topo_size(scenario) else ""
        rows.append(_make_row(scenario, prob_name, topo_size, seed=seed))
    return rows


def normalize_topo_size(value: object) -> str:
    """Return the canonical topology-size component used in case identities."""

    if value in (None, "", "-"):
        return ""
    return str(value)


def case_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the leakage identity shared by generated benchmark manifests."""

    return (
        str(row["scenario"]),
        normalize_topo_size(row.get("topo_size")),
        str(row["problem"]),
    )


def _stable_rank(seed: int, namespace: str, value: object) -> str:
    payload = json.dumps(
        [seed, namespace, value],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fault_learning_cases(
    full_cases: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Select 90 deterministic, evaluation-disjoint fault cases.

    The first pass covers every transferable root cause. The remaining cases
    favor under-represented scenarios and topology sizes. Hash ranks make every
    tie-break stable across Python processes and input dictionary ordering.
    """

    selected_identities = {case_identity(row) for row in selected_cases}
    selected_problems = {str(row["problem"]) for row in selected_cases}
    transferable_problems = selected_problems - EVALUATION_ONLY_PROBLEMS
    if len(transferable_problems) != 54:
        raise ValueError(
            "Expected 54 transferable problems after reserving the two "
            f"evaluation-only problems, found {len(transferable_problems)}"
        )

    unique_full: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in full_cases:
        identity = case_identity(row)
        if identity in unique_full:
            raise ValueError(f"Duplicate full benchmark identity: {identity!r}")
        unique_full[identity] = row

    nonoverlap = {
        identity: row
        for identity, row in unique_full.items()
        if identity not in selected_identities
    }
    unexpected_singleton_variants = sorted(
        problem
        for problem in EVALUATION_ONLY_PROBLEMS
        if any(str(row["problem"]) == problem for row in nonoverlap.values())
    )
    if unexpected_singleton_variants:
        raise ValueError(
            "Evaluation-only problems unexpectedly have transferable variants: "
            + ", ".join(unexpected_singleton_variants)
        )

    candidates = [
        row
        for row in nonoverlap.values()
        if str(row["problem"]) in transferable_problems
    ]
    by_problem: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_problem[str(row["problem"])].append(row)
    missing = sorted(transferable_problems - by_problem.keys())
    if missing:
        raise ValueError("No learning candidate for: " + ", ".join(missing))

    scenario_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    chosen_identities: set[tuple[str, str, str]] = set()

    def candidate_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
        scenario = str(row["scenario"])
        topology = normalize_topo_size(row.get("topo_size"))
        canonical_row = {
            "scenario": scenario,
            "topo_size": topology,
            "problem": str(row["problem"]),
            "inject": row.get("inject", {}),
        }
        return (
            scenario_counts[scenario],
            topology_counts[topology],
            _stable_rank(seed, "learning-case", canonical_row),
        )

    def choose(row: Mapping[str, Any]) -> None:
        identity = case_identity(row)
        if identity in chosen_identities:
            raise ValueError(f"Learning identity selected twice: {identity!r}")
        chosen.append(deepcopy(dict(row)))
        chosen_identities.add(identity)
        scenario_counts[str(row["scenario"])] += 1
        topology_counts[normalize_topo_size(row.get("topo_size"))] += 1

    problem_order = sorted(
        transferable_problems,
        key=lambda problem: (_stable_rank(seed, "learning-problem", problem), problem),
    )
    for problem in problem_order:
        choose(min(by_problem[problem], key=candidate_key))

    remaining = [
        row for row in candidates if case_identity(row) not in chosen_identities
    ]
    while len(chosen) < LEARNING_FAULT_CASES:
        if not remaining:
            raise ValueError(
                f"Only {len(chosen)} disjoint learning fault cases are available"
            )
        next_row = min(remaining, key=candidate_key)
        choose(next_row)
        next_identity = case_identity(next_row)
        remaining = [row for row in remaining if case_identity(row) != next_identity]

    return chosen


def select_learning_cases(
    full_cases: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Build the deterministic 90-fault/10-control learning curriculum."""

    fault_cases = _fault_learning_cases(full_cases, selected_cases, seed=seed)
    rows: list[dict[str, Any]] = []
    for control_index, control in enumerate(NO_FAULT_CONTROLS):
        start = control_index * 9
        rows.extend(fault_cases[start : start + 9])
        scenario, topo_size = control
        rows.append(
            {
                "scenario": scenario,
                "topo_size": topo_size,
                "problem": "no_fault",
                "inject": {},
            }
        )
    if len(rows) != LEARNING_FAULT_CASES + LEARNING_NO_FAULT_CASES:
        raise AssertionError(f"Unexpected learning case count: {len(rows)}")
    return rows


def benchmark_manifest(
    role: str, rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    """Wrap cases in the common manifest metadata."""

    no_fault = sum(str(row["problem"]) == "no_fault" for row in rows)
    return {
        "benchmark_role": role,
        "seed": seed,
        "counts": {
            "total": len(rows),
            "fault": len(rows) - no_fault,
            "no_fault": no_fault,
        },
        "cases": list(rows),
    }


def _print_stats(label: str, rows: list[dict]) -> None:
    by_scenario = Counter(r["scenario"] for r in rows)
    by_problem = Counter(r["problem"] for r in rows)
    print(
        f"\n{label}: {len(rows)} cases, {len(by_problem)} problems, {len(by_scenario)} scenarios"
    )
    for scenario, count in sorted(by_scenario.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {scenario}: {count}")


def generate_benchmark(
    *, seed: int = DEFAULT_SEED
) -> tuple[list[dict], list[dict], list[dict]]:
    full_rows = iter_full_cases(seed=seed)
    selected_rows = iter_selected_cases(seed=seed)
    learning_rows = select_learning_cases(full_rows, selected_rows, seed=seed)

    _print_stats("benchmark_learning.yaml", learning_rows)
    _print_stats("benchmark_full.yaml", full_rows)
    _print_stats("benchmark_selected.yaml", selected_rows)

    benchmark_dir = Path(cur_path)
    for name, role, rows in (
        ("benchmark_learning.yaml", "learning", learning_rows),
        ("benchmark_selected.yaml", "evaluation", selected_rows),
        ("benchmark_full.yaml", "evaluation", full_rows),
    ):
        out_path = benchmark_dir / name
        out_path.write_text(
            yaml.dump(
                benchmark_manifest(role, rows, seed=seed),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} cases to {out_path} (seed={seed})")

    return full_rows, selected_rows, learning_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark YAML configs.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Global random seed for inject param selection (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_benchmark(seed=args.seed)
