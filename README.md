# VRPTW-GA (Baseline)

Baseline Genetic Algorithm for the Vehicle Routing Problem with Time Windows (VRPTW) using Solomon benchmark instances. This is a **first working version** meant for academic, reproducible experiments.

## Features
- Solomon instance parser
- Permutation chromosome decoding into routes (capacity-respecting)
- Time-window evaluation with time-warp penalties
- Baseline GA: tournament selection, OX crossover, swap + inversion mutation, elitism
- Deterministic runs via single RNG seed
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
Example filenames: `C101.txt`, `R101.txt`, `RC101.txt`.

## Run
```bash
python -m experiments.run --instance data/solomon/C101.txt --seed 42 --pop 100 --gens 300 --time_limit 60 --penalty_tw 1000
```

Optional flags:
- `--log_every 10` to reduce logging noise
- `--log_file results/progress.csv` to save best-per-generation metrics
- `--repair_tw` to split routes on time-window violations (simple repair)
- `--plot` to save a fitness curve image (requires matplotlib)

Outputs are saved under:
```
results/
```
- `run_YYYYmmdd_HHMMSS.csv`
- `run_YYYYmmdd_HHMMSS.json`
- optional `run_YYYYmmdd_HHMMSS.png`

## Metrics
- **total_distance**: sum of Euclidean distances over all routes
- **total_timewarp**: total lateness beyond due dates (time-window violations)
- **objective**: `total_distance + penalty_tw * total_timewarp`
- **feasible_timewindows**: `total_timewarp == 0`
- **capacity_violation**: sum of load above capacity (should be zero in this baseline)

## Sanity check
On small instances (e.g., `C101`), best fitness should generally decrease over generations with a non-zero penalty.

## Notes
- Capacity is enforced in decoding (hard constraint), so capacity violations are zero by design.
- Time windows are treated softly via penalties (time-warp).

## License
See `LICENSE`.
