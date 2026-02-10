# Reporting Pipeline

This document describes how to generate PB96-like tables and plots from VRPTW GA runs.

## What It Generates
- Paper-like tables matching Potvin & Bengio (1996) columns:
  - K (Number of Routes)
  - Distance (travel time)
  - Waiting Time
  - Route Time (travel + waiting + service)
  - Computation Time (min:sec)
- OUR-GA rows at generation marks (default 0/20/50) and OUR-GA-BEST at final generation
- Paper reference rows (I1, GENEROUS-00/20/50) from PB96
- Plots:
  - Line plots with mean + std band for K and Route Time across generations
  - Bar charts: OUR vs PAPER at gen 50 for K and Route Time
  - Boxplots: final K and Route Time across seeds
- A `reports/summary.json` file with metadata, configuration signatures, file list, and warnings

## Run Experiments
Single run example:

```bash
python -m experiments.run \
  --instance data/solomon/C101.txt \
  --seed 42 \
  --pop 150 \
  --gens 50 \
  --time_limit 60
```

Reports are generated automatically after each run. To disable:

```bash
python -m experiments.run \
  --instance data/solomon/C101.txt \
  --seed 42 \
  --no-report
```

Batch run example:

```bash
python -m experiments.batch_run \
  --instances data/solomon/C101.txt,data/solomon/C102.txt \
  --seeds 1,2,3 \
  --pop 150 \
  --gens 50 \
  --time_limit 60
```

Disable report generation:

```bash
python -m experiments.batch_run \
  --instances data/solomon/C101.txt,data/solomon/C102.txt \
  --seeds 1,2,3 \
  --no-report
```

Results are saved under `results/` as:
- `run_*.json` (final summary)
- `run_*.csv` (flat summary)
- `progress_*.csv` (per-generation history)

## Generate Reports

```bash
python -m experiments.report \
  --results_dir results \
  --out_dir reports \
  --paper_ref docs/pb96_reference_tables.csv \
  --gen_marks 0 20 50
```

Key options:
- `--only_best`: skip generation marks and gen-based plots
- `--include_paper / --no-include_paper`: toggle PB96 reference rows
- `--group_by solomon_type|instance_name`
- `--format md,csv` (tables)

Outputs:
- `reports/tables/Table_1a_like_R.*`
- `reports/tables/Table_1b_like_C.*`
- `reports/tables/Table_1c_like_RC.*`
- `reports/plots/*.png`
- `reports/summary.json`

## Generate PB96 Comparison Package (canva)

Creates a `canva/` package organized by the same comparison blocks used in Potvin & Bengio (1996):
- Table 1 replication block (I1 vs GENEROUS checkpoints vs OUR-GA)
- Table 2 operator ablation reference block (RC1)
- Table 3 literature comparison block (+ OUR-GA overlay)
- Table 4 selected-problem comparison block (+ OUR-GA overlay)

```bash
python -m experiments.report_pb96_comparisons \
  --results_root results \
  --out_dir canva
```

Outputs:
- `canva/01_table1_replication/`
- `canva/02_table2_operators_rc1/`
- `canva/03_table3_literature/`
- `canva/04_table4_optimum/`
- `canva/README.md`

## Interpretation Notes
- Route Time is computed as `distance + waiting + service`. The report warns if a row violates this.
- If a progress log does not include the exact generation marks, the report uses the closest generation <= target (or nearest overall if none) and logs a warning in `summary.json`.
- If progress logs are missing, OUR-GA-BEST tables still work; gen-based plots are skipped.

## PB96 Reference Data
The paper reference averages are stored in `docs/pb96_reference_tables.csv`.
They come from PB96 Tables 1a/1b/1c (averages across problems).

## Reproducibility
- `reports/summary.json` includes:
  - list of input files used
  - git commit hash (if available)
  - full CLI arguments
  - distinct config signatures
  - warnings
