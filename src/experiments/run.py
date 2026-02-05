from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

from vrptw_ga.ga import GAConfig, run_ga
from vrptw_ga.metrics import penalized_fitness
from vrptw_ga.parser import parse_solomon
from vrptw_ga.utils import append_csv, ensure_dir, now_timestamp, save_json, set_seed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GA-only VRPTW (Potvin & Bengio 1996 aligned)")
    parser.add_argument("--instance", required=True, help="Path to Solomon .txt file")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--pop", type=int, default=150, help="Population size")
    parser.add_argument("--gens", type=int, default=50, help="Max generations")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Wall-clock time limit (seconds)")
    parser.add_argument("--penalty_tw", type=float, default=1000.0, help="Time window penalty weight")
    parser.add_argument("--elite", type=int, default=1, help="Elitism count")
    parser.add_argument("--p_swap", type=float, default=0.2, help="Swap mutation probability")
    parser.add_argument("--p_inversion", type=float, default=0.1, help="Inversion mutation probability")
    parser.add_argument("--crossover", type=str, default="pb96", help="Crossover: pb96, ox, pmx")
    parser.add_argument("--crossover_rate", type=float, default=0.6, help="Crossover rate")
    parser.add_argument("--mutation_rate", type=float, default=0.6, help="Mutation rate")
    parser.add_argument("--objective", type=str, default="lexicographic", help="Objective: lexicographic or penalized")
    parser.add_argument("--init", type=str, default="i1", help="Init: i1, random_perm, feasible_greedy, mixed")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N generations (0 disables)")
    parser.add_argument("--log_file", type=str, default="", help="CSV path to log best per generation")
    parser.add_argument("--repair_tw", action="store_true", help="Repair routes by splitting on time-window violations")
    parser.add_argument("--decoder", type=str, default="sequential", help="Decoder: sequential or split")
    parser.add_argument("--report_k", action="store_true", help="DEPRECATED (no-op). Outputs always include K.")
    parser.add_argument("--adaptive_penalty", action="store_true", help="DEPRECATED (ignored).")
    parser.add_argument("--diversity_lambda", type=float, default=None, help="DEPRECATED (ignored).")
    parser.add_argument("--diversity_metric", type=str, default=None, help="DEPRECATED (ignored).")
    parser.add_argument("--variant", type=str, default=None, help="DEPRECATED (ignored).")
    parser.add_argument("--plot", action="store_true", help="Save fitness plot if matplotlib is available")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.adaptive_penalty or args.diversity_lambda is not None or args.diversity_metric or args.variant:
        print(
            "Warning: legacy flags (--adaptive_penalty/--diversity_*/--variant) are deprecated and ignored.",
            file=sys.stderr,
        )

    instance = parse_solomon(args.instance)
    rng = set_seed(args.seed)

    if args.log_file:
        ensure_dir(Path(args.log_file).parent)

    config = GAConfig(
        pop_size=args.pop,
        generations=args.gens,
        time_limit=args.time_limit,
        penalty_tw=args.penalty_tw,
        elite=args.elite,
        p_swap=args.p_swap,
        p_inversion=args.p_inversion,
        crossover=args.crossover,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        objective=args.objective,
        init=args.init,
        log_every=args.log_every,
        repair_tw=args.repair_tw,
        decoder=args.decoder,
    )

    print("Config:")
    print(config)

    start = time.time()
    result = run_ga(instance, rng, config)
    elapsed = time.time() - start

    best = result["best_solution"]
    history = result["history"]

    results_dir = ensure_dir(Path("results"))
    ts = now_timestamp()
    run_id = f"{instance.name}_{args.seed}_{ts}"

    config_json: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": ts,
        "instance_path": str(args.instance),
        "instance_name": instance.name,
        "seed": args.seed,
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

    row = {
        "run_id": run_id,
        "timestamp": ts,
        "instance": instance.name,
        "instance_path": str(args.instance),
        "seed": args.seed,
        "crossover": args.crossover,
        "objective": args.objective,
        "init": args.init,
        "decoder": args.decoder,
        "pop": args.pop,
        "gens": args.gens,
        "gens_run": len(history),
        "time_limit": args.time_limit,
        "elapsed_sec": round(elapsed, 3),
        "penalty_tw": args.penalty_tw,
        "elite": args.elite,
        "p_swap": args.p_swap,
        "p_inversion": args.p_inversion,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "best_k": len(best.routes),
        "best_distance": best.total_distance,
        "best_total_waiting": best.total_waiting,
        "best_total_service": best.total_service,
        "best_total_route_time": best.total_route_time,
        "best_timewarp": best.total_timewarp,
        "best_penalized": penalized_fitness(best, args.penalty_tw),
        "capacity_violation": best.capacity_violation,
        "routes": len(best.routes),
        "feasible_timewindows": best.feasible_timewindows,
        "feasible_capacity": best.feasible_capacity,
        "feasible": best.feasible_timewindows and best.feasible_capacity,
        "repair_tw": args.repair_tw,
    }
    append_csv(results_dir / f"run_{run_id}.csv", row)

    progress_path = results_dir / f"progress_{run_id}.csv"
    for row_h in history:
        append_csv(progress_path, row_h)
    if args.log_file:
        for row_h in history:
            append_csv(args.log_file, row_h)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.plot([h["best_distance"] for h in history], label="Best Distance")
            plt.plot([h["best_k"] for h in history], label="Best K")
            plt.title("Best Distance and K per Generation")
            plt.xlabel("Generation")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / f"run_{run_id}.png", dpi=150)
            plt.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
