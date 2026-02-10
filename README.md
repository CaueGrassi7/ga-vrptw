# VRPTW-GA

GA-only implementation for the Vehicle Routing Problem with Time Windows (VRPTW), aligned with Potvin & Bengio (1996).

## Scope
- Genetic Algorithm only (no local search / hybrid metaheuristics).
- Solomon benchmark instances (`100` customers).
- PB96-inspired route-based crossover (SBX/RBX + repair).
- Lexicographic preference for feasible solutions: minimize `K`, then `route_time`.

## Repository Structure
- `src/vrptw_ga/`: core VRPTW + GA implementation.
- `src/experiments/`: experiment runners, pipeline, and reporting.
- `data/solomon/`: Solomon instances (`*.txt`).
- `docs/`: experiment and reporting documentation.
- `tests/`: smoke tests.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional plotting/report dependencies:
```bash
pip install -e ".[plot]"
```

Dev dependencies:
```bash
pip install -e ".[dev]"
```

## Single Run
```bash
python -m experiments.run \
  --instance data/solomon/C101.txt \
  --seed 42 \
  --pop 150 \
  --gens 50 \
  --time_limit 60
```

## Key Flags
- `--crossover pb96|ox|pmx`
- `--objective lexicographic|penalized`
- `--init i1|random_perm|feasible_greedy|mixed`
- `--decoder sequential|split`

## Experiments
Main experiment scripts:
- `python -m experiments.exp1_run`
- `python -m experiments.exp2_run`
- `python -m experiments.exp3_run`

Pipeline orchestrator:
```bash
python -m experiments.pipeline --mode small --exp all --seeds 3 --time_limit 10
python -m experiments.pipeline --mode full --exp all --seeds 10 --time_limit 60
```

## Reporting
- General PB96-like report:
```bash
python -m experiments.report \
  --results_dir results \
  --out_dir reports \
  --paper_ref docs/pb96_reference_tables.csv \
  --gen_marks 0 20 50
```

- Experiment-specific reports:
```bash
python -m experiments.report_exp1
python -m experiments.report_exp2
python -m experiments.report_exp3
```

See:
- `docs/EXPERIMENTS.md`
- `docs/REPORTING.md`
- `docs/RESULTS_GUIDE.md`

## Metrics
- `K`: number of routes.
- `distance`: travel distance.
- `waiting`: waiting time.
- `service`: service time.
- `route_time`: `distance + waiting + service`.
- `timewarp`: time-window violation.

## Reproducibility
- Explicit RNG seed support.
- Run artifacts include config and per-generation history.
- Report summaries include input files and metadata.

## Tests
```bash
pytest -q
```

## Reference
Potvin, J.-Y. and Bengio, S. (1996), *The Vehicle Routing Problem with Time Windows Part II: Genetic Search*.
