# Benchmark configs

Benchmark cases are defined in YAML files with explicit inject parameters:

- `benchmark_test.yaml` — **30** evolution-focused curated cases (default `nika benchmark run` config)
- `benchmark_evaluate.yaml` — **100** sequential curriculum-evaluation cases (44 evolution variants + 56 full root-cause coverage cases)
- `benchmark_selected.yaml` — **56** curated cases (one per failure)
- `benchmark_full.yaml` — **685** cases (all scenario × failure × size combinations)

Each case includes an `inject` map (device names, etc.) that is passed to `nika failure inject` as `--set` flags. Device names must match the target scenario topology (see lab definitions under `src/nika/net_env/`). IP and netmask values are derived from the live lab at inject time.

## Statistics

| Metric | Count |
|--------|------:|
| Failure types (root causes) | 56 |
| Full benchmark cases | 685 |
| Evaluate benchmark cases | 100 |
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

`benchmark_evaluate.yaml` is a 100-case subset of `benchmark_full.yaml`. Cases
1-44 are non-duplicate evolution variants for broad motif learning. Cases
45-100 are exactly `benchmark_selected.yaml`, so the second phase covers every
root cause once. This makes the file suitable for sequential memory and
tool-evolution experiments where earlier cases update the skill bank and DRAFT
tool documentation before a full root-cause coverage phase. It is not a neutral
held-out split.

| Scenario | Cases |
|----------|------:|
| `ospf_enterprise_dhcp` | 48 |
| `dc_clos_bgp` | 23 |
| `sdn_clos` | 9 |
| `ospf_enterprise_static` | 6 |
| `p4_bloom_filter` | 6 |
| `p4_counter` | 4 |
| `dc_clos_service` | 2 |
| `rip_small_internet_vpn` | 1 |
| `p4_mpls` | 1 |

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
