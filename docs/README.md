# NIKA Documentation

This directory keeps project notes that are useful for running experiments and
understanding the current system boundary.

## Current Docs

| Path | Purpose |
|---|---|
| `learning_modules.md` | Current boundary and usage for memory, Tool Evolution, and Harness Evolution |
| `memory/README.md` | Detailed procedural-memory design notes and implementation rationale |
| `report/` | Thesis/report source moved from the old top-level `report/` directory |

## Experiment Studio

Launch the Streamlit runner UI:

```bash
uv run nika studio
```

The studio writes run specs and logs under `runtime/streamlit_runs/`. It runs
the same CLI workflows as the terminal commands, so benchmark artifacts still
land under `results/` and can be inspected with `nika visualize`.

## Benchmark Contract

Benchmark CSV files should use the minimal online-timeline schema:

```csv
problem,scenario,topo_size
```

The row order is the evaluation order. Stateful modules such as memory evolution
and Tool Evolution update after each row, so evolving runs must stay sequential
unless the experiment isolates state per worker.

`benchmark/benchmark_test.csv` is the lightweight 30-case suite for quick
comparisons. `benchmark/benchmark_selected.csv` and `benchmark/benchmark_full.csv`
remain larger sources for expanded runs.

## Report Notes

The report source lives under `docs/report/`. Keep source files such as `.tex`,
`.bib`, images, and final PDFs there. LaTeX build products such as `.aux`,
`.log`, `.toc`, `.fls`, `.fdb_latexmk`, `.bbl`, and `.blg` are ignored by the
root `.gitignore` and should not be committed.
