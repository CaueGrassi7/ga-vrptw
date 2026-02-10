from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from vrptw_ga.ga import GAConfig
from vrptw_ga.utils import ensure_dir

from .pipeline_config import DEFAULT_GRIDS
from .pipeline_utils import build_seed_list, ensure_within_results, resolve_instances, safe_clean_dir
from .runner_utils import run_and_save


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment pipeline orchestrator")
    p.add_argument("--mode", choices=["small", "full"], required=True)
    p.add_argument("--exp", choices=["1", "2", "3", "all"], required=True)
    p.add_argument("--seeds", type=str, default="10", help="Seed count or comma list")
    p.add_argument("--time_limit", type=float, default=60.0)
    p.add_argument("--out_root", type=str, default="results")
    p.add_argument("--archive_old", action="store_true")
    return p


def _run_report(exp: str, results_dir: Path, out_dir: Path) -> None:
    if exp == "1":
        from experiments import report_exp1 as report_mod

        argv = [
            "report_exp1",
            "--results_dir",
            str(results_dir),
            "--out_dir",
            str(out_dir),
        ]
    elif exp == "2":
        from experiments import report_exp2 as report_mod

        argv = [
            "report_exp2",
            "--results_dir",
            str(results_dir),
            "--out_dir",
            str(out_dir),
        ]
    else:
        from experiments import report_exp3 as report_mod

        argv = [
            "report_exp3",
            "--results_dir",
            str(results_dir),
            "--out_dir",
            str(out_dir),
        ]

    import sys

    old_argv = sys.argv
    try:
        sys.argv = argv
        report_mod.main()
    finally:
        sys.argv = old_argv


def _exp1_run(instances: List[str], seeds: List[int], raw_dir: Path, time_limit: float) -> None:
    for inst in instances:
        for seed in seeds:
            config = GAConfig(
                pop_size=150,
                generations=50,
                time_limit=time_limit,
                penalty_tw=1000.0,
                crossover_rate=0.6,
                mutation_rate=0.6,
                elite=1,
                p_swap=0.2,
                p_inversion=0.1,
                log_every=10,
                repair_tw=False,
                decoder="sequential",
                crossover="pb96",
                objective="lexicographic",
                init="i1",
            )
            run_and_save(inst, seed, config, raw_dir)


def _exp2_run(instances: List[str], seeds: List[int], raw_dir: Path, time_limit: float) -> None:
    for cr in DEFAULT_GRIDS.exp2_crossover_rates:
        for mr in DEFAULT_GRIDS.exp2_mutation_rates:
            tag = f"cr{cr}_mr{mr}".replace(".", "p")
            out_dir = ensure_dir(raw_dir / tag)
            for inst in instances:
                for seed in seeds:
                    config = GAConfig(
                        pop_size=150,
                        generations=50,
                        time_limit=time_limit,
                        penalty_tw=1000.0,
                        crossover_rate=cr,
                        mutation_rate=mr,
                        elite=1,
                        p_swap=0.2,
                        p_inversion=0.1,
                        log_every=10,
                        repair_tw=False,
                        decoder="sequential",
                        crossover="pb96",
                        objective="lexicographic",
                        init="i1",
                    )
                    run_and_save(inst, seed, config, out_dir)


def _exp3_run(instances: List[str], seeds: List[int], raw_dir: Path) -> None:
    for pop in DEFAULT_GRIDS.exp3_pop_sizes:
        for time_limit in DEFAULT_GRIDS.exp3_time_limits:
            tag = f"pop{pop}_t{int(time_limit)}"
            out_dir = ensure_dir(raw_dir / tag)
            for inst in instances:
                for seed in seeds:
                    config = GAConfig(
                        pop_size=pop,
                        generations=50,
                        time_limit=time_limit,
                        penalty_tw=1000.0,
                        crossover_rate=0.6,
                        mutation_rate=0.6,
                        elite=1,
                        p_swap=0.2,
                        p_inversion=0.1,
                        log_every=10,
                        repair_tw=False,
                        decoder="sequential",
                        crossover="pb96",
                        objective="lexicographic",
                        init="i1",
                    )
                    run_and_save(inst, seed, config, out_dir)


def main() -> None:
    args = _build_parser().parse_args()
    seeds = build_seed_list(args.seeds)

    results_root = Path(args.out_root)
    data_dir = Path("data/solomon")
    instances, missing = resolve_instances(args.mode, data_dir)

    if missing:
        print("Missing instance files:")
        for path in missing:
            print(f"- {path}")

    if not instances:
        raise SystemExit("No instances found for selected mode.")

    exp_map = {
        "1": "exp1_replication",
        "2": "exp2_operator_sensitivity",
        "3": "exp3_population_sensitivity",
    }
    exp_list = ["1", "2", "3"] if args.exp == "all" else [args.exp]

    for exp in exp_list:
        exp_dir = results_root / exp_map[exp] / args.mode
        ensure_within_results(exp_dir, results_root)
        print(f"Cleaning output directory: {exp_dir}")
        safe_clean_dir(
            exp_dir,
            args.archive_old,
            results_root / "_archive",
            archive_label=f"{exp_map[exp]}_{args.mode}",
        )

        raw_dir = ensure_dir(exp_dir / "raw")
        ensure_dir(exp_dir / "aggregated")
        ensure_dir(exp_dir / "tables")
        ensure_dir(exp_dir / "plots")

        if exp == "1":
            _exp1_run(instances, seeds, raw_dir, args.time_limit)
        elif exp == "2":
            _exp2_run(instances, seeds, raw_dir, args.time_limit)
        else:
            _exp3_run(instances, seeds, raw_dir)

        _run_report(exp, raw_dir, exp_dir)


if __name__ == "__main__":
    main()
