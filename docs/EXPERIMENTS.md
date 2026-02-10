# Experiments

This project defines three GA-only experiments with consistent metrics and PB96-style reporting.

## Global Protocol
- GA-only: no local search, tabu, SA, VNS, relocate, 2-opt.
- Solomon VRPTW benchmark (100 customers).
- Reproducibility via explicit seeds.
- Metrics: K, distance, waiting, route time (distance + waiting + service), computation time.
- Per-generation snapshots are logged.

**Deviation from PB96**
- PB96 GENEROUS uses a third mutation (LSM). We do not include LSM to keep GA-only local-search-free.
- Computation times in PB96 were measured on SPARC10; our wall-clock times are not directly comparable.

## Experiment 1 — Paper Replication (PB96)
**Goal:** replicate PB96 GENEROUS protocol as closely as possible with GA-only constraints.

**Config (default):**
- Population: 150
- Generations: 50
- Crossover rate: 0.6
- Mutation rate: 0.6
- Objective: lexicographic (feasible → K → route_time)
- Init: I1 constructive

**Run (direct):**
```bash
python -m experiments.exp1_run \
  --groups C1,C2 \
  --seeds 1,2,3,4,5,6,7,8,9,10
```

**Report:**
```bash
python -m experiments.report_exp1
```

## Experiment 2 — Operator Sensitivity
**Goal:** evaluate crossover/mutation probabilities.

**Grid:**
- Crossover rate ∈ {0.4, 0.6, 0.8}
- Mutation rate ∈ {0.2, 0.4, 0.6}

**Run:**
```bash
python -m experiments.exp2_run \
  --groups C1,C2 \
  --seeds 1,2,3,4,5,6,7,8,9,10
```

**Report:**
```bash
python -m experiments.report_exp2
```

## Experiment 3 — Population/Time Sensitivity
**Goal:** evaluate population size and time budget tradeoffs.

**Grid:**
- Population ∈ {50, 100, 150, 300}
- Time limit ∈ {30, 60} seconds

**Run:**
```bash
python -m experiments.exp3_run \
  --groups C1,C2 \
  --seeds 1,2,3,4,5,6,7,8,9,10
```

**Report:**
```bash
python -m experiments.report_exp3
```

## Pipeline (Small vs Full)
Use the orchestrator to run one representative instance per group (small) or all 56 instances (full).

```bash
python -m experiments.pipeline --mode small --exp all --seeds 3 --time_limit 10
python -m experiments.pipeline --mode full --exp 1 --seeds 10 --time_limit 60
python -m experiments.pipeline --mode full --exp 1 --archive_old
```

Cleanup behavior:
- The pipeline deletes the target `results/<exp>/<mode>/` folder before running.
- With `--archive_old`, the folder is moved to `results/_archive/<timestamp>_<exp>_<mode>/`.

Small mode instance list:
- C1: C101
- C2: C201
- R1: R101
- R2: R201
- RC1: RC101
- RC2: RC201

Full mode instance ranges:
- C1: C101–C109
- C2: C201–C208
- R1: R101–R112
- R2: R201–R211
- RC1: RC101–RC108
- RC2: RC201–RC208

## Directory Structure
```
results/
  exp1_replication/
    small/
      raw/
      aggregated/
      tables/
      plots/
    full/
      raw/
      aggregated/
      tables/
      plots/
  exp2_operator_sensitivity/
    small/...
    full/...
  exp3_population_sensitivity/
    small/...
    full/...
```
