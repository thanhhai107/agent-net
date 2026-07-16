# Benchmark configs

Benchmark cases are defined in YAML files with explicit inject parameters:

- `benchmark_learning.yaml` — **100** learning cases (**90 fault + 10 no-fault**)
- `benchmark_selected.yaml` — **56** curated evaluation cases (one per failure; default evaluation config)
- `benchmark_full.yaml` — **702** evaluation cases (all compatible scenario × failure × size combinations)

Each case includes an `inject` map (device names, etc.) that is passed to `nika failure inject` as `--set` flags. Device names must match the target scenario topology (see lab definitions under `src/nika/net_env/`). IP and netmask values are derived from the live lab at inject time.

Every manifest declares `benchmark_role`, `seed`, and exact fault/no-fault
`counts`. The learning set is generated deterministically with seed 42. Its 90
fault identities do not overlap `benchmark_selected.yaml`; it covers 54
transferable root causes while reserving `mpls_label_limit_exceeded` and
`p4_aggressive_detection_thresholds` for evaluation. One no-fault control is
placed after every nine fault cases.

## Statistics

| Metric | Count |
|--------|------:|
| Failure types (root causes) | 56 |
| Transferable learning root causes | 54 |
| Learning benchmark cases | 100 |
| Learning fault cases | 90 |
| Learning no-fault cases | 10 |
| Full benchmark cases | 702 |
| Selected benchmark cases | 56 |
| Scenarios in full matrix | 15 |

### Full matrix by scenario

| Scenario | Cases |
|----------|------:|
| `ospf_enterprise_dhcp` | 111 |
| `dc_clos_service` | 102 |
| `ospf_enterprise_static` | 78 |
| `rip_small_internet_vpn` | 72 |
| `dc_clos_bgp` | 69 |
| `sdn_clos` | 57 |
| `sdn_star` | 57 |
| `k8s_lab` | 23 |
| `simple_bgp` | 23 |
| `llmd_lab` | 20 |
| `p4_bloom_filter` | 20 |
| `p4_mpls` | 20 |
| `p4_counter` | 19 |
| `p4_int` | 19 |
| `min3clos` | 12 |

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

Kubernetes scenarios (`k8s_lab`, `llmd_lab`) do not appear in the selected
evaluation set; selected cases use traditional Kathara labs as the best-matching
scenario per failure. They remain eligible as disjoint variants in the learning
and full manifests.

## Regeneration

Regenerate all three YAML files:

```shell
uv run python benchmark/generate_benchmark.py
```

The generator builds full and selected cases from the problem/environment pools,
then derives learning cases with the pure `select_learning_cases()` selector.
`nika benchmark run` reads an evaluation YAML via `--config` (default:
`benchmark/benchmark_selected.yaml`). Learning pipelines use
`benchmark_learning.yaml` first and consume the selected or full evaluation
manifest only after the learned state is frozen.
