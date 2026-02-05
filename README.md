# VRPTW-GA (GA-only, PB96-aligned)

Baseline Genetic Algorithm for the Vehicle Routing Problem with Time Windows (VRPTW), aligned to **Potvin & Bengio (1996)** “The Vehicle Routing Problem with Time Windows – Part II: Genetic Search”.

**Hard rule:** GA-only. No local search / hybrid metaheuristics.

## Features
- Solomon instance parser
- Route-based GA with **PB96-inspired crossover** (SBX/RBX + repair)
- Feasible greedy constructor for initial population
- Objective: **lexicographic** (K then distance) on feasible solutions
- Deterministic runs via RNG seed
- CLI runner saving CSV/JSON results and optional plot

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional plotting support:
```bash
pip install -e .[plot]
```

## Data
Place Solomon `.txt` instances in:
```
data/solomon/
```

## Run
```bash
python -m experiments.run --instance data/solomon/C101.txt --seed 42 --pop 150 --gens 50 --time_limit 60
```

Outputs are saved under:
```
results/
```
- `run_<instance>_<seed>_<timestamp>.csv`
- `run_<instance>_<seed>_<timestamp>.json`
- `progress_<instance>_<seed>_<timestamp>.csv`
- optional `run_<instance>_<seed>_<timestamp>.png`

## Key Flags
- `--crossover pb96|ox|pmx` (default `pb96`)
- `--objective lexicographic|penalized` (default `lexicographic`)
- `--init i1|random_perm|feasible_greedy|mixed` (default `i1`)
- `--decoder sequential|split` (used for permutation decoding)
- Legacy/ignored flags (deprecated): `--adaptive_penalty`, `--diversity_lambda`, `--diversity_metric`, `--variant`, `--report_k`

## Metrics
- **K**: number of routes/vehicles
- **total_distance**: sum of travel distances
- **total_timewarp**: total lateness beyond due dates
- **best_penalized**: `distance + penalty_tw * timewarp` (used for infeasible ranking)

## Alignment with Potvin & Bengio (1996)
Checklist of GA-only components aligned to the paper:
- Route-based crossover (SBX/RBX with repair) implemented as `pb96` crossover
- Linear ranking selection + SUS (stochastic universal sampling)
- Lexicographic preference: feasible > infeasible; then minimize `K`, then **route_time** (distance + waiting + service)
- Feasible constructive initialization (Solomon I1-style insertion)
- Population size 150, generations 50, crossover rate 0.6, mutation rate 0.6 (defaults)

**Simplifications vs PB96 (GA-only):**
- No local search (paper uses 1M/2M/LSM, including Or-opt).
 - Our mutation is swap/inversion on permutation + re-decode.
 - Penalized fitness uses travel distance; feasible tie-break uses route_time.

## Paper-aligned GA-only example
```bash
python -m experiments.run \
  --instance data/solomon/C101.txt \
  --seed 42 \
  --pop 150 \
  --gens 50 \
  --time_limit 60 \
  --crossover pb96 \
  --objective lexicographic \
  --init mixed
```

## Tests
```bash
pip install -e ".[dev]"
pytest -q
```
