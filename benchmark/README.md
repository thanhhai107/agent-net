# Benchmark configs

Benchmark cases are defined in YAML files with explicit inject parameters:

- `benchmark_selected.yaml` — **56** curated cases (one per failure, default `nika benchmark run` config)
- `benchmark_full.yaml` — **685** cases (all scenario × failure × size combinations)

Each case includes an `inject` map (device names, etc.) that is passed to `nika failure inject` as `--set` flags. Device names must match the target scenario topology (see lab definitions under `src/nika/net_env/`). IP and netmask values are derived from the live lab at inject time.

## Statistics

| Metric | Count |
|--------|------:|
| Failure types (root causes) | 56 |
| Full benchmark cases | 685 |
| Selected benchmark cases | 56 |
| Scenarios in full matrix | 14 |

### Full matrix by scenario

| Scenario | Cases |
|----------|------:|
| `ospf_enterprise_dhcp` | 111 |
| `dc_clos_service` | 102 |
| `ospf_enterprise_static` | 78 |
| `rip_small_internet_vpn` | 72 |
| `dc_clos_bgp` | 66 |
| `sdn_clos` | 57 |
| `sdn_star` | 57 |
| `k8s_lab` | 22 |
| `simple_bgp` | 22 |
| `llmd_lab` | 20 |
| `p4_bloom_filter` | 20 |
| `p4_mpls` | 20 |
| `p4_counter` | 19 |
| `p4_int` | 19 |

### Selected matrix by scenario

| Scenario | Cases |
|----------|------:|
| `ospf_enterprise_dhcp` | 26 |
| `dc_clos_bgp` | 13 |
| `p4_bloom_filter` | 6 |
| `sdn_clos` | 5 |
| `ospf_enterprise_static` | 3 |
| `dc_clos_service` | 1 |
| `p4_mpls` | 1 |
| `rip_small_internet_vpn` | 1 |

Kubernetes scenarios (`k8s_lab`, `llmd_lab`) appear in the full matrix only; selected cases use traditional Kathara labs as the best-matching scenario per failure.

## Regeneration

Regenerate both YAML files:

```shell
uv run python benchmark/generate_benchmark.py
```

`nika benchmark run` reads YAML via `--config` (default: `benchmark/benchmark_selected.yaml`).
