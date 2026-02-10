from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from vrptw_ga.ga import GAConfig
from vrptw_ga.utils import ensure_dir, now_timestamp

from .instance_utils import list_solomon_instances, parse_csv_list
from .runner_utils import run_and_save


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment 3: Sensitivity to population size and time budget")
    p.add_argument("--groups", type=str, default="C1,C2", help="Comma-separated groups")
    p.add_argument("--instances", type=str, default="", help="Comma-separated instance paths (overrides groups)")
    p.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds")
    p.add_argument("--results_dir", type=str, default="results/exp3_population_sensitivity/raw")
    p.add_argument("--pops", type=str, default="50,100,150,300")
    p.add_argument("--time_limits", type=str, default="30,60")
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--crossover_rate", type=float, default=0.6)
    p.add_argument("--mutation_rate", type=float, default=0.6)
    p.add_argument("--objective", type=str, default="lexicographic")
    p.add_argument("--init", type=str, default="i1")
    p.add_argument("--crossover", type=str, default="pb96")
    p.add_argument("--decoder", type=str, default="sequential")
    p.add_argument("--log_every", type=int, default=10)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    pops = [int(x) for x in parse_csv_list(args.pops)]
    time_limits = [float(x) for x in parse_csv_list(args.time_limits)]

    if args.instances:
        instances = parse_csv_list(args.instances)
    else:
        data_dir = Path("data/solomon")
        instances = list_solomon_instances(data_dir, parse_csv_list(args.groups))

    if not instances:
        raise SystemExit("No instances found. Provide --instances or valid --groups.")

    base_dir = ensure_dir(Path(args.results_dir))
    batch_ts = now_timestamp()

    for pop in pops:
        for time_limit in time_limits:
            tag = f"pop{pop}_t{int(time_limit)}"
            out_dir = ensure_dir(base_dir / tag)
            for inst in instances:
                for seed in seeds:
                    config = GAConfig(
                        pop_size=pop,
                        generations=args.gens,
                        time_limit=time_limit,
                        penalty_tw=1000.0,
                        crossover_rate=args.crossover_rate,
                        mutation_rate=args.mutation_rate,
                        elite=1,
                        p_swap=0.2,
                        p_inversion=0.1,
                        log_every=args.log_every,
                        repair_tw=False,
                        decoder=args.decoder,
                        crossover=args.crossover,
                        objective=args.objective,
                        init=args.init,
                    )
                    run_and_save(inst, seed, config, out_dir, batch_ts=batch_ts)


if __name__ == "__main__":
    main()
