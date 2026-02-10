from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from vrptw_ga.ga import GAConfig, run_ga
from vrptw_ga.metrics import penalized_fitness
from vrptw_ga.parser import parse_solomon
from vrptw_ga.utils import append_csv, ensure_dir, now_timestamp, save_json, set_seed


def _parse_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch runner for VRPTW GA experiments")
    p.add_argument("--instances", required=True, help="Comma-separated instance paths")
    p.add_argument("--seeds", required=True, help="Comma-separated seeds (e.g., 0,1,2)")
    p.add_argument("--crossover", default="pb96", help="Crossover: pb96, ox, pmx")
    p.add_argument("--objective", default="lexicographic", help="Objective: lexicographic or penalized")
    p.add_argument("--init", default="i1", help="Init: i1, random_perm, feasible_greedy, mixed")
    p.add_argument("--pop", type=int, default=150)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--time_limit", type=float, default=60.0)
    p.add_argument("--penalty_tw", type=float, default=1000.0)
    p.add_argument("--elite", type=int, default=1)
    p.add_argument("--p_swap", type=float, default=0.2)
    p.add_argument("--p_inversion", type=float, default=0.1)
    p.add_argument("--crossover_rate", type=float, default=0.6)
    p.add_argument("--mutation_rate", type=float, default=0.6)
    p.add_argument("--decoder", type=str, default="sequential")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--repair_tw", action="store_true")
    p.add_argument("--report", action=argparse.BooleanOptionalAction, default=True, help="Generate reports after batch")
    p.add_argument("--report_results_dir", type=str, default="results", help="Report input directory")
    p.add_argument("--report_out_dir", type=str, default="reports", help="Report output directory")
    p.add_argument(
        "--report_paper_ref",
        type=str,
        default="docs/pb96_reference_tables.csv",
        help="PB96 reference CSV",
    )
    p.add_argument("--report_gen_marks", nargs="*", type=int, default=[0, 20, 50], help="Report gen marks")
    p.add_argument("--report_only_best", action="store_true", help="Report only final best")
    p.add_argument(
        "--report_include_paper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include PB96 reference rows",
    )
    return p


def _maybe_run_report(args: argparse.Namespace) -> None:
    if not args.report:
        return
    import sys
    from experiments import report as report_mod

    report_args = [
        "--results_dir",
        args.report_results_dir,
        "--out_dir",
        args.report_out_dir,
        "--paper_ref",
        args.report_paper_ref,
        "--gen_marks",
        *[str(g) for g in args.report_gen_marks],
    ]
    if args.report_only_best:
        report_args.append("--only_best")
    if not args.report_include_paper:
        report_args.append("--no-include_paper")

    old_argv = sys.argv
    try:
        sys.argv = ["experiments.report", *report_args]
        report_mod.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = _build_parser().parse_args()
    instances = _parse_list(args.instances)
    seeds = [int(s) for s in _parse_list(args.seeds)]

    results_dir = ensure_dir(Path("results"))
    ts = now_timestamp()
    run_rows: List[Dict[str, Any]] = []

    for inst_path in instances:
        instance = parse_solomon(inst_path)
        for seed in seeds:
            rng = set_seed(seed)
            config = GAConfig(
                pop_size=args.pop,
                generations=args.gens,
                time_limit=args.time_limit,
                penalty_tw=args.penalty_tw,
                elite=args.elite,
                p_swap=args.p_swap,
                p_inversion=args.p_inversion,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                crossover=args.crossover,
                objective=args.objective,
                init=args.init,
                decoder=args.decoder,
                log_every=args.log_every,
                repair_tw=args.repair_tw,
            )

            start = time.time()
            result = run_ga(instance, rng, config)
            elapsed = time.time() - start
            best = result["best_solution"]
            history = result["history"]

            run_id = f"{instance.name}_{seed}_{ts}"
            config_json: Dict[str, Any] = {
                "run_id": run_id,
                "timestamp": ts,
                "instance_path": str(inst_path),
                "instance_name": instance.name,
                "seed": seed,
                "config": config,
                "elapsed_seconds": elapsed,
                "generations_run": len(history),
                "best_perm": [c for r in best.routes for c in r.customers],
                "best_routes": [r.customers for r in best.routes],
                "best_metrics": {
                    "best_k": len(best.routes),
                    "best_distance": best.total_distance,
                    "best_total_waiting": best.total_waiting,
                    "best_total_service": best.total_service,
                    "best_total_route_time": best.total_route_time,
                    "best_timewarp": best.total_timewarp,
                    "best_penalized": penalized_fitness(best, args.penalty_tw),
                    "capacity_violation": best.capacity_violation,
                    "feasible_timewindows": best.feasible_timewindows,
                    "feasible_capacity": best.feasible_capacity,
                    "feasible": best.feasible_timewindows and best.feasible_capacity,
                },
            }
            save_json(results_dir / f"run_{run_id}.json", config_json)

            progress_path = results_dir / f"progress_{run_id}.csv"
            for row_h in history:
                append_csv(progress_path, row_h)

            row = {
                "run_id": run_id,
                "instance": instance.name,
                "instance_path": inst_path,
                "crossover": args.crossover,
                "objective": args.objective,
                "init": args.init,
                "decoder": args.decoder,
                "pop": args.pop,
                "gens": args.gens,
                "time_limit": args.time_limit,
                "penalty_tw": args.penalty_tw,
                "elite": args.elite,
                "p_swap": args.p_swap,
                "p_inversion": args.p_inversion,
                "crossover_rate": args.crossover_rate,
                "mutation_rate": args.mutation_rate,
                "repair_tw": args.repair_tw,
                "seed": seed,
                "best_k": len(best.routes),
                "best_distance": best.total_distance,
                "best_total_waiting": best.total_waiting,
                "best_total_service": best.total_service,
                "best_total_route_time": best.total_route_time,
                "best_timewarp": best.total_timewarp,
                "best_penalized": penalized_fitness(best, args.penalty_tw),
                "feasible": best.feasible_timewindows and best.feasible_capacity,
                "runtime_s": round(elapsed, 3),
            }
            run_rows.append(row)
            append_csv(results_dir / f"batch_runs_{ts}.csv", row)

    # Aggregate stats
    stats_rows: List[Dict[str, Any]] = []
    by_key: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in run_rows:
        key = (
            row["instance"],
            row["crossover"],
            row["objective"],
            row["init"],
            row["decoder"],
            row["pop"],
            row["gens"],
            row["time_limit"],
            row["penalty_tw"],
            row["elite"],
            row["p_swap"],
            row["p_inversion"],
            row["crossover_rate"],
            row["mutation_rate"],
            row["repair_tw"],
        )
        by_key.setdefault(key, []).append(row)

    for (
        inst,
        crossover,
        objective,
        init,
        decoder,
        pop,
        gens,
        time_limit,
        penalty_tw,
        elite,
        p_swap,
        p_inversion,
        crossover_rate,
        mutation_rate,
        repair_tw,
    ), rows in by_key.items():
        best_distances = [r["best_distance"] for r in rows]
        best_penalized = [r["best_penalized"] for r in rows]
        best_k = [r["best_k"] for r in rows]
        feasibles = [r["feasible"] for r in rows]
        stats_rows.append(
            {
                "instance": inst,
                "crossover": crossover,
                "objective": objective,
                "init": init,
                "decoder": decoder,
                "pop": pop,
                "gens": gens,
                "time_limit": time_limit,
                "penalty_tw": penalty_tw,
                "elite": elite,
                "p_swap": p_swap,
                "p_inversion": p_inversion,
                "crossover_rate": crossover_rate,
                "mutation_rate": mutation_rate,
                "repair_tw": repair_tw,
                "n_runs": len(rows),
                "feasible_rate": sum(1 for f in feasibles if f) / max(1, len(feasibles)),
                "mean_best_k": statistics.mean(best_k),
                "std_best_k": statistics.pstdev(best_k) if len(best_k) > 1 else 0.0,
                "mean_best_distance": statistics.mean(best_distances),
                "std_best_distance": statistics.pstdev(best_distances) if len(best_distances) > 1 else 0.0,
                "min_best_distance": min(best_distances),
                "mean_best_penalized": statistics.mean(best_penalized),
                "std_best_penalized": statistics.pstdev(best_penalized) if len(best_penalized) > 1 else 0.0,
                "min_best_penalized": min(best_penalized),
            }
        )

    for row in stats_rows:
        append_csv(results_dir / f"batch_stats_{ts}.csv", row)

    save_json(
        results_dir / f"batch_meta_{ts}.json",
        {
            "instances": instances,
            "seeds": seeds,
            "crossover": args.crossover,
            "objective": args.objective,
            "init": args.init,
            "decoder": args.decoder,
            "pop": args.pop,
            "gens": args.gens,
            "time_limit": args.time_limit,
            "penalty_tw": args.penalty_tw,
            "elite": args.elite,
            "p_swap": args.p_swap,
            "p_inversion": args.p_inversion,
            "crossover_rate": args.crossover_rate,
            "mutation_rate": args.mutation_rate,
            "repair_tw": args.repair_tw,
        },
    )
    _maybe_run_report(args)


if __name__ == "__main__":
    main()
