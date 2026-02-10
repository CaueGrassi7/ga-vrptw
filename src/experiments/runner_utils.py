from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from vrptw_ga.ga import GAConfig, run_ga
from vrptw_ga.metrics import penalized_fitness
from vrptw_ga.parser import parse_solomon
from vrptw_ga.utils import append_csv, ensure_dir, now_timestamp, save_json, set_seed


def run_and_save(
    instance_path: str,
    seed: int,
    config: GAConfig,
    out_dir: Path,
    run_tag: str | None = None,
    batch_ts: str | None = None,
) -> str:
    instance = parse_solomon(instance_path)
    rng = set_seed(seed)

    start = time.time()
    result = run_ga(instance, rng, config)
    elapsed = time.time() - start

    best = result["best_solution"]
    history = result["history"]

    ensure_dir(out_dir)
    ts = batch_ts or now_timestamp()
    if run_tag:
        run_id = f"{instance.name}_{seed}_{run_tag}_{ts}"
    else:
        run_id = f"{instance.name}_{seed}_{ts}"

    config_json: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": ts,
        "instance_path": str(instance_path),
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
            "best_penalized": penalized_fitness(best, config.penalty_tw),
            "capacity_violation": best.capacity_violation,
            "feasible_timewindows": best.feasible_timewindows,
            "feasible_capacity": best.feasible_capacity,
            "feasible": best.feasible_timewindows and best.feasible_capacity,
        },
    }
    save_json(out_dir / f"run_{run_id}.json", config_json)

    row = {
        "run_id": run_id,
        "timestamp": ts,
        "instance": instance.name,
        "instance_path": str(instance_path),
        "seed": seed,
        "crossover": config.crossover,
        "objective": config.objective,
        "init": config.init,
        "decoder": config.decoder,
        "pop": config.pop_size,
        "gens": config.generations,
        "gens_run": len(history),
        "time_limit": config.time_limit,
        "elapsed_sec": round(elapsed, 3),
        "penalty_tw": config.penalty_tw,
        "elite": config.elite,
        "p_swap": config.p_swap,
        "p_inversion": config.p_inversion,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "best_k": len(best.routes),
        "best_distance": best.total_distance,
        "best_total_waiting": best.total_waiting,
        "best_total_service": best.total_service,
        "best_total_route_time": best.total_route_time,
        "best_timewarp": best.total_timewarp,
        "best_penalized": penalized_fitness(best, config.penalty_tw),
        "capacity_violation": best.capacity_violation,
        "feasible_timewindows": best.feasible_timewindows,
        "feasible_capacity": best.feasible_capacity,
        "feasible": best.feasible_timewindows and best.feasible_capacity,
        "repair_tw": config.repair_tw,
    }
    append_csv(out_dir / f"run_{run_id}.csv", row)

    progress_path = out_dir / f"progress_{run_id}.csv"
    for row_h in history:
        append_csv(progress_path, row_h)

    return run_id
