from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from vrptw_ga.ga import GAConfig, run_ga
from vrptw_ga.parser import parse_solomon
from vrptw_ga.utils import append_csv, ensure_dir, now_timestamp, save_json, set_seed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline GA for VRPTW (Solomon instances)")
    parser.add_argument("--instance", required=True, help="Path to Solomon .txt file")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gens", type=int, default=300, help="Max generations")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Wall-clock time limit (seconds)")
    parser.add_argument("--penalty_tw", type=float, default=1000.0, help="Time window penalty weight")
    parser.add_argument("--elite", type=int, default=2, help="Elitism count")
    parser.add_argument("--tournament_k", type=int, default=3, help="Tournament size")
    parser.add_argument("--p_swap", type=float, default=0.2, help="Swap mutation probability")
    parser.add_argument("--p_inversion", type=float, default=0.1, help="Inversion mutation probability")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N generations (0 disables)")
    parser.add_argument("--log_file", type=str, default="", help="CSV path to log best per generation")
    parser.add_argument("--repair_tw", action="store_true", help="Repair routes by splitting on time-window violations")
    parser.add_argument("--plot", action="store_true", help="Save fitness plot if matplotlib is available")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

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
        tournament_k=args.tournament_k,
        p_swap=args.p_swap,
        p_inversion=args.p_inversion,
        log_every=args.log_every,
        repair_tw=args.repair_tw,
        log_file=args.log_file if args.log_file else None,
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

    config_json: Dict[str, Any] = {
        "timestamp": ts,
        "instance_path": str(args.instance),
        "instance_name": instance.name,
        "seed": args.seed,
        "config": config,
        "elapsed_seconds": elapsed,
        "generations_run": len(history),
    }
    save_json(results_dir / f"run_{ts}.json", config_json)

    row = {
        "timestamp": ts,
        "instance": instance.name,
        "instance_path": str(args.instance),
        "seed": args.seed,
        "pop": args.pop,
        "gens": args.gens,
        "gens_run": len(history),
        "time_limit": args.time_limit,
        "elapsed_sec": round(elapsed, 3),
        "penalty_tw": args.penalty_tw,
        "elite": args.elite,
        "tournament_k": args.tournament_k,
        "p_swap": args.p_swap,
        "p_inversion": args.p_inversion,
        "best_fitness": best.objective,
        "best_distance": best.total_distance,
        "best_timewarp": best.total_timewarp,
        "capacity_violation": best.capacity_violation,
        "routes": len(best.routes),
        "feasible_timewindows": best.feasible_timewindows,
        "feasible_capacity": best.feasible_capacity,
    }
    append_csv(results_dir / f"run_{ts}.csv", row)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.plot(history)
            plt.title("Best Fitness per Generation")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.tight_layout()
            plt.savefig(results_dir / f"run_{ts}.png", dpi=150)
            plt.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
