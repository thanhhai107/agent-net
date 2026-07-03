# Benchmark configs

Benchmark cases are defined in YAML files with explicit inject parameters:

- `benchmark_test.yaml` — **30** evolution-focused curated cases (default `nika benchmark run` config)
- `benchmark_evaluate.yaml` — **125** sequential curriculum-evaluation cases (44 evolution variants + 56 full root-cause coverage cases + 25 clean controls)
- `benchmark_selected.yaml` — **56** curated cases (one per failure)
- `benchmark_full.yaml` — **685** cases (all scenario × failure × size combinations)

Fault cases include an `inject` map (device names, etc.) that is passed to `nika failure inject` as `--set` flags. Device names must match the target scenario topology (see lab definitions under `src/nika/net_env/`). IP and netmask values are derived from the live lab at inject time. Clean controls use `problem: no_fault` and `inject: {}`; the benchmark runner deploys the lab without injecting failures.

## Statistics

| Metric | Count |
|--------|------:|
| Failure types (root causes) | 56 |
| Full benchmark cases | 685 |
| Evaluate benchmark cases | 125 |
| Test benchmark cases | 30 |
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

### Test matrix by scenario

| Scenario | Cases |
|----------|------:|
| `ospf_enterprise_dhcp` | 14 |
| `dc_clos_bgp` | 9 |
| `ospf_enterprise_static` | 2 |
| `p4_bloom_filter` | 2 |
| `sdn_clos` | 2 |
| `dc_clos_service` | 1 |

Kubernetes scenarios (`k8s_lab`, `llmd_lab`) appear in the full matrix only. The selected slice uses traditional Kathara labs and keeps related diagnostic clusters for short Agent Evolution runs.

### Evaluate matrix

`benchmark_evaluate.yaml` has 100 fault cases plus 25 clean controls. The fault
cases are a subset of `benchmark_full.yaml`; if clean rows are ignored, the
remaining fault stream preserves the curriculum order: 44 non-duplicate
evolution variants followed by exactly `benchmark_selected.yaml`, so every root
cause is covered once. The no-fault controls are deterministically interleaved
throughout the sequence to measure false-positive behavior without making
"clean at the end" a learnable artifact. This makes the file suitable for
sequential memory and tool-evolution experiments where earlier cases update the
skill bank and DRAFT tool documentation before broad root-cause coverage and
clean controls. It is not a neutral held-out split.

| Scenario | Cases |
|----------|------:|
| `ospf_enterprise_dhcp` | 51 |
| `dc_clos_bgp` | 26 |
| `sdn_clos` | 12 |
| `ospf_enterprise_static` | 9 |
| `p4_bloom_filter` | 7 |
| `dc_clos_service` | 5 |
| `p4_counter` | 5 |
| `rip_small_internet_vpn` | 4 |
| `p4_mpls` | 2 |
| `sdn_star` | 2 |
| `p4_int` | 1 |
| `simple_bgp` | 1 |

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

Regenerate benchmark YAML files:

```shell
uv run python benchmark/generate_benchmark.py
```

`nika benchmark run` reads YAML via `--file` (default: `benchmark/benchmark_test.yaml`).
