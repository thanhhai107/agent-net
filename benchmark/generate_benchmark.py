"""Generate benchmark YAML files from prob_pool and net_env_pool."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import yaml

from nika.net_env.net_env_pool import list_all_net_envs, scenario_requires_topo_tier
from nika.orchestrator.problems.prob_pool import list_avail_problem_instances
from nika.workflows.benchmark.inject_defaults import resolve_inject_params, validate_benchmark_case

cur_path = os.path.dirname(os.path.abspath(__file__))

SELECTED_EVOLVING_CASES: list[tuple[str, str, str]] = [
    # Services and address management.
    ("ospf_enterprise_dhcp", "dns_record_error", "s"),
    ("ospf_enterprise_dhcp", "dns_service_down", "s"),
    ("ospf_enterprise_dhcp", "dns_port_blocked", "s"),
    ("ospf_enterprise_dhcp", "dhcp_missing_subnet", "s"),
    ("ospf_enterprise_dhcp", "dhcp_spoofed_gateway", "s"),
    # Host and link failures.
    ("ospf_enterprise_static", "host_missing_ip", "s"),
    ("ospf_enterprise_static", "host_incorrect_ip", "s"),
    ("ospf_enterprise_dhcp", "host_incorrect_gateway", "s"),
    ("ospf_enterprise_dhcp", "host_incorrect_dns", "s"),
    ("dc_clos_bgp", "host_ip_conflict", "s"),
    ("dc_clos_bgp", "link_down", "s"),
    ("dc_clos_bgp", "link_flap", "s"),
    ("dc_clos_bgp", "link_bandwidth_throttling", "s"),
    ("dc_clos_bgp", "link_high_packet_corruption", "s"),
    # Routing failures.
    ("dc_clos_bgp", "bgp_asn_misconfig", "s"),
    ("dc_clos_bgp", "bgp_missing_route_advertisement", "s"),
    ("dc_clos_bgp", "bgp_blackhole_route_leak", "s"),
    ("dc_clos_bgp", "host_static_blackhole", "s"),
    ("ospf_enterprise_dhcp", "ospf_neighbor_missing", "s"),
    ("ospf_enterprise_dhcp", "frr_service_down", "s"),
    # Security and attack-like faults.
    ("ospf_enterprise_dhcp", "arp_cache_poisoning", "s"),
    ("ospf_enterprise_dhcp", "http_acl_block", "s"),
    ("ospf_enterprise_dhcp", "web_dos_attack", "s"),
    ("dc_clos_service", "bgp_hijacking", "s"),
    # Resource contention.
    ("ospf_enterprise_dhcp", "incast_traffic_network_limitation", "s"),
    ("ospf_enterprise_dhcp", "receiver_resource_contention", "s"),
    # SDN/P4 control-plane and data-plane cases.
    ("sdn_clos", "flow_rule_loop", "s"),
    ("sdn_clos", "sdn_controller_crash", "s"),
    ("p4_bloom_filter", "p4_table_entry_missing", ""),
    ("p4_bloom_filter", "p4_table_entry_misconfig", ""),
]

EVALUATE_EVOLUTION_PROBLEMS: list[str] = [
    # Service and address-management motifs.
    "dns_lookup_latency",
    "dns_port_blocked",
    "dns_record_error",
    "dns_service_down",
    "dhcp_missing_subnet",
    "dhcp_service_down",
    "dhcp_spoofed_dns",
    "dhcp_spoofed_gateway",
    "dhcp_spoofed_subnet",
    # Host/IP configuration motifs.
    "host_incorrect_dns",
    "host_incorrect_gateway",
    "host_incorrect_ip",
    "host_incorrect_netmask",
    "host_ip_conflict",
    "host_missing_ip",
    "host_static_blackhole",
    # Link and traffic-quality motifs.
    "link_bandwidth_throttling",
    "link_down",
    "link_flap",
    "link_high_packet_corruption",
    # Routing and control-plane motifs.
    "bgp_acl_block",
    "bgp_asn_misconfig",
    "bgp_blackhole_route_leak",
    "bgp_hijacking",
    "bgp_missing_route_advertisement",
    "frr_service_down",
    "ospf_acl_block",
    "ospf_area_misconfiguration",
    "ospf_neighbor_missing",
    # Security and policy motifs.
    "arp_acl_block",
    "arp_cache_poisoning",
    "http_acl_block",
    "icmp_acl_block",
    # Application/resource-pressure motifs.
    "incast_traffic_network_limitation",
    "receiver_resource_contention",
    "web_dos_attack",
    # SDN motifs.
    "flow_rule_loop",
    "flow_rule_shadowing",
    "sdn_controller_crash",
    "southbound_port_block",
    # P4 motifs.
    "bmv2_switch_down",
    "p4_header_definition_error",
    "p4_table_entry_misconfig",
    "p4_table_entry_missing",
]

EVALUATE_HEADER = (
    "# 100-case curriculum-evaluation subset of benchmark_full.yaml.\n"
    "# Cases 1-44 are evolution variants for broad motif learning.\n"
    "# Cases 45-100 are benchmark_selected.yaml: one case per root cause.\n"
)

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


def _topo_sizes_for_scenario(net_env_cls) -> list[str]:
    if net_env_cls.TOPO_SIZE is None:
        return [""]
    return ["s", "m", "l"]


def _make_row(scenario: str, problem: str, topo_size: str) -> dict:
    inject = resolve_inject_params(problem, scenario, topo_size)
    validate_benchmark_case(scenario, problem, inject, topo_size)
    return {
        "scenario": scenario,
        "topo_size": topo_size or None,
        "problem": problem,
        "inject": inject,
    }


def iter_full_cases() -> list[dict]:
    net_envs = list_all_net_envs()
    problem_instances = list_avail_problem_instances()
    rows: list[dict] = []

    for prob_name, prob_task_levels in problem_instances.items():
        problem_instance = prob_task_levels["detection"]
        for net_env_name, net_env_cls in net_envs.items():
            if not set(problem_instance.TAGS).issubset(set(net_env_cls.TAGS)):
                continue
            for topo_size in _topo_sizes_for_scenario(net_env_cls):
                rows.append(_make_row(net_env_name, prob_name, topo_size))
    return rows


def iter_selected_cases() -> list[dict]:
    net_envs = list_all_net_envs()
    problem_instances = list_avail_problem_instances()
    rows: list[dict] = []

    for prob_name in sorted(problem_instances.keys()):
        scenario = SELECTED_SCENARIO_FOR_PROBLEM.get(prob_name)
        if scenario is None:
            raise ValueError(f"No selected scenario mapping for problem {prob_name!r}")
        net_env_cls = net_envs[scenario]
        problem_instance = problem_instances[prob_name]["detection"]
        if not set(problem_instance.TAGS).issubset(set(net_env_cls.TAGS)):
            raise ValueError(
                f"Selected scenario {scenario} not tag-compatible with {prob_name} "
                f"(problem={problem_instance.TAGS}, scenario={net_env_cls.TAGS})"
            )
        topo_size = "s" if scenario_requires_topo_tier(scenario) else ""
        rows.append(_make_row(scenario, prob_name, topo_size))
    return rows


def iter_test_cases() -> list[dict]:
    net_envs = list_all_net_envs()
    problem_instances = list_avail_problem_instances()
    rows: list[dict] = []

    seen: set[tuple[str, str, str]] = set()
    for scenario, prob_name, topo_size in SELECTED_EVOLVING_CASES:
        key = (scenario, prob_name, topo_size)
        if key in seen:
            raise ValueError(f"Duplicate test benchmark case: {key!r}")
        seen.add(key)
        if prob_name not in problem_instances:
            raise ValueError(f"Unknown test problem {prob_name!r}")
        if scenario not in net_envs:
            raise ValueError(f"Unknown test scenario {scenario!r}")
        net_env_cls = net_envs[scenario]
        problem_instance = problem_instances[prob_name]["detection"]
        if not set(problem_instance.TAGS).issubset(set(net_env_cls.TAGS)):
            raise ValueError(
                f"Test scenario {scenario} not tag-compatible with {prob_name} "
                f"(problem={problem_instance.TAGS}, scenario={net_env_cls.TAGS})"
            )
        requires_tier = scenario_requires_topo_tier(scenario)
        if requires_tier and topo_size not in {"s", "m", "l"}:
            raise ValueError(f"Test scenario {scenario} requires topo_size s/m/l")
        if not requires_tier and topo_size:
            raise ValueError(f"Test scenario {scenario} does not accept topo_size")
        rows.append(_make_row(scenario, prob_name, topo_size))
    return rows


def _row_key(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _evaluate_variant_score(
    row: dict,
    *,
    selected_row: dict,
) -> tuple[int, int, int, str]:
    same_scenario = row["scenario"] == selected_row["scenario"]
    scenario_rank = 0 if same_scenario else 1
    size_rank = {"m": 0, "l": 1, "s": 2, None: 3}.get(row.get("topo_size"), 4)
    p4_rank = {
        "p4_counter": 0,
        "p4_int": 1,
        "p4_mpls": 2,
        "p4_bloom_filter": 3,
    }.get(row["scenario"], 4)
    return (scenario_rank, size_rank, p4_rank, row["scenario"])


def iter_evaluate_cases(
    *,
    full_rows: list[dict] | None = None,
    selected_rows: list[dict] | None = None,
) -> list[dict]:
    full_rows = full_rows or iter_full_cases()
    selected_rows = selected_rows or iter_selected_cases()
    selected_by_problem = {row["problem"]: row for row in selected_rows}
    selected_keys = {_row_key(row) for row in selected_rows}
    full_by_problem: dict[str, list[dict]] = {}
    for row in full_rows:
        full_by_problem.setdefault(row["problem"], []).append(row)

    if len(EVALUATE_EVOLUTION_PROBLEMS) != 44:
        raise ValueError("Evaluate evolution phase must contain exactly 44 problems")
    if len(set(EVALUATE_EVOLUTION_PROBLEMS)) != len(EVALUATE_EVOLUTION_PROBLEMS):
        raise ValueError("Evaluate evolution phase contains duplicate problems")

    evolution_rows: list[dict] = []
    for problem in EVALUATE_EVOLUTION_PROBLEMS:
        selected_row = selected_by_problem.get(problem)
        if selected_row is None:
            raise ValueError(f"Evaluate problem missing selected case: {problem}")
        candidates = [
            row for row in full_by_problem.get(problem, [])
            if _row_key(row) not in selected_keys
        ]
        if not candidates:
            raise ValueError(f"No non-selected evaluate variant for {problem}")
        evolution_rows.append(
            sorted(
                candidates,
                key=lambda row: _evaluate_variant_score(
                    row,
                    selected_row=selected_row,
                ),
            )[0]
        )

    rows = evolution_rows + selected_rows
    row_keys = {_row_key(row) for row in rows}
    full_keys = {_row_key(row) for row in full_rows}
    if len(rows) != 100 or len(row_keys) != 100:
        raise ValueError("benchmark_evaluate.yaml must contain 100 unique cases")
    if not row_keys.issubset(full_keys):
        raise ValueError("benchmark_evaluate.yaml must be a subset of full")
    if len({row["problem"] for row in rows[44:]}) != len(selected_rows):
        raise ValueError("Evaluate coverage phase must cover each selected problem")
    return rows


def _print_stats(label: str, rows: list[dict]) -> None:
    by_scenario = Counter(r["scenario"] for r in rows)
    by_problem = Counter(r["problem"] for r in rows)
    print(f"\n{label}: {len(rows)} cases, {len(by_problem)} problems, {len(by_scenario)} scenarios")
    for scenario, count in sorted(by_scenario.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {scenario}: {count}")


def generate_benchmark() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    full_rows = iter_full_cases()
    selected_rows = iter_selected_cases()
    test_rows = iter_test_cases()
    evaluate_rows = iter_evaluate_cases(
        full_rows=full_rows,
        selected_rows=selected_rows,
    )

    _print_stats("benchmark_full.yaml", full_rows)
    _print_stats("benchmark_evaluate.yaml", evaluate_rows)
    _print_stats("benchmark_selected.yaml", selected_rows)
    _print_stats("benchmark_test.yaml", test_rows)

    benchmark_dir = Path(cur_path)
    for name, rows in (
        ("benchmark_full.yaml", full_rows),
        ("benchmark_evaluate.yaml", evaluate_rows),
        ("benchmark_selected.yaml", selected_rows),
        ("benchmark_test.yaml", test_rows),
    ):
        out_path = benchmark_dir / name
        content = yaml.dump({"cases": rows}, sort_keys=False, allow_unicode=True)
        if name == "benchmark_evaluate.yaml":
            content = EVALUATE_HEADER + content
        out_path.write_text(
            content,
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} cases to {out_path}")

    return full_rows, evaluate_rows, selected_rows, test_rows


if __name__ == "__main__":
    generate_benchmark()
