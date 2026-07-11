"""Unit tests for benchmark inject param resolution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from nika.net_env.net_env_pool import list_all_net_envs, scenario_requires_topo_size
from nika.problems.prob_pool import list_avail_problem_instances

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from inject_resolve import (  # noqa: E402
    DEFAULT_SEED,
    _device_interfaces,
    _get_net_env_for_benchmark,
    _load_inventory,
    resolve_inject_params,
    validate_benchmark_case,
)


def _topo_sizes_for_scenario(scenario: str) -> list[str]:
    if scenario_requires_topo_size(scenario):
        return ["s", "m", "l"]
    return [""]


def _iter_all_cases() -> list[tuple[str, str, str]]:
    net_envs = list_all_net_envs()
    problems = list_avail_problem_instances()
    cases: list[tuple[str, str, str]] = []
    for prob_name, problem_class in problems.items():
        for scenario, net_env_cls in net_envs.items():
            if not set(problem_class.TAGS).issubset(set(net_env_cls.TAGS)):
                continue
            for topo_size in _topo_sizes_for_scenario(scenario):
                cases.append((scenario, prob_name, topo_size))
    return cases


class TestInjectResolveReproducibility(unittest.TestCase):
    def test_same_seed_produces_identical_params(self) -> None:
        first = resolve_inject_params("link_down", "dc_clos_bgp", "s", seed=42)
        second = resolve_inject_params("link_down", "dc_clos_bgp", "s", seed=42)
        self.assertEqual(first, second)

    def test_different_seed_produces_different_params(self) -> None:
        a = resolve_inject_params("link_down", "dc_clos_bgp", "s", seed=1)
        b = resolve_inject_params("link_down", "dc_clos_bgp", "s", seed=2)
        self.assertNotEqual(a, b)

    def test_different_topo_size_produces_different_params(self) -> None:
        small = resolve_inject_params("link_down", "dc_clos_bgp", "s", seed=42)
        medium = resolve_inject_params("link_down", "dc_clos_bgp", "m", seed=42)
        self.assertNotEqual(small, medium)


class TestInjectResolveValidation(unittest.TestCase):
    def test_all_cases_validate(self) -> None:
        for scenario, problem, topo_size in _iter_all_cases():
            with self.subTest(scenario=scenario, problem=problem, topo_size=topo_size):
                inject = resolve_inject_params(
                    problem, scenario, topo_size, seed=DEFAULT_SEED
                )
                validate_benchmark_case(scenario, problem, inject, topo_size)

    def test_interface_belongs_to_device(self) -> None:
        inject = resolve_inject_params("link_down", "dc_clos_bgp", "s", seed=42)
        net_env = _get_net_env_for_benchmark("dc_clos_bgp", "s")
        _load_inventory(net_env)
        ifaces = _device_interfaces(net_env).get(inject["host_name"], [])
        if ifaces:
            self.assertIn(inject["intf_name"], ifaces)

    def test_dual_host_problems_use_distinct_hosts_when_possible(self) -> None:
        inject = resolve_inject_params("host_ip_conflict", "dc_clos_bgp", "s", seed=42)
        self.assertNotEqual(inject["host_name"], inject["host_name_2"])

    def test_mac_conflict_pair_are_distinct_on_multi_host_scenario(self) -> None:
        inject = resolve_inject_params(
            "mac_address_conflict", "ospf_enterprise_dhcp", "s", seed=42
        )
        self.assertNotEqual(inject["host_name"], inject["host_name_2"])

    def test_min3clos_link_fault_uses_router_interface(self) -> None:
        inject = resolve_inject_params("link_down", "min3clos", "", seed=42)
        net_env = _get_net_env_for_benchmark("min3clos", "")
        _load_inventory(net_env)
        self.assertIn(inject["host_name"], net_env.routers)
        ifaces = _device_interfaces(net_env).get(inject["host_name"], [])
        if ifaces:
            self.assertIn(inject["intf_name"], ifaces)


if __name__ == "__main__":
    unittest.main()
