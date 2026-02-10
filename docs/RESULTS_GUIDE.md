# Results Guide

This guide explains how to read the tables and plots produced by the experiments.

## Paper-Like Metrics
The tables follow PB96 conventions:
- **K**: number of routes
- **Distance**: total travel distance
- **Waiting Time**: total waiting across all routes
- **Route Time**: distance + waiting + service
- **Computation Time**: wall-clock time (not directly comparable to PB96 SPARC10)

## Experiment 1 Tables
Tables are written under `results/exp1_replication/<mode>/tables/`:
- `Table_1a_like_R.*`
- `Table_1b_like_C.*`
- `Table_1c_like_RC.*`

Rows include:
- PB96 reference rows (I1, GENEROUS-00/20/50)
- OUR-GA rows at gen 0/20/50 and OUR-GA-BEST

## Experiment 1 Plots
`results/exp1_replication/<mode>/plots/` contains:
- `line_best_k_<group>.png`
- `line_best_route_time_<group>.png`

These show mean and standard deviation across seeds.

`results/exp1_replication/<mode>/aggregated/` also contains:
- `runs.csv`: one row per instance/seed/run
- `per_instance.*`: aggregated by instance and checkpoint row

## Experiment 2 Outputs
`results/exp2_operator_sensitivity/<mode>/tables/` contains:
- `operator_sensitivity.*` with mean K, mean route time, and mean convergence generation per configuration.

Plots:
- `route_time_vs_gen_<group>.png`: one curve per configuration.
- `box_route_time_<group>.png`: distribution of final route time across seeds per configuration.

`results/exp2_operator_sensitivity/<mode>/aggregated/` also contains:
- `runs.csv`
- `per_instance.*`

## Experiment 3 Outputs
`results/exp3_population_sensitivity/<mode>/tables/` contains:
- `population_sensitivity.*` with mean K, route time, runtime per (pop, time_limit) pair.

Plots:
- `tradeoff_route_time_<group>.png`: route time vs runtime tradeoff.
- `tradeoff_k_<group>.png`: K vs runtime tradeoff.

`results/exp3_population_sensitivity/<mode>/aggregated/` also contains:
- `runs.csv`
- `per_instance.*`

## Notes on Comparability
- PB96 computation times are not directly comparable to modern machines.
- Any deviations from the paper’s operators or protocol are explicitly noted in `docs/EXPERIMENTS.md`.
