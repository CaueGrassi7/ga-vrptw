from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vrptw_ga.utils import ensure_dir, now_timestamp

from .reporting import aggregate, io, plots


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report for Experiment 3 (population/time sensitivity)")
    p.add_argument("--results_dir", type=str, default="results/exp3_population_sensitivity/raw")
    p.add_argument("--out_dir", type=str, default="results/exp3_population_sensitivity")
    p.add_argument("--format", type=str, default="md,csv")
    return p


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_config(pop: int, time_limit: float) -> str:
    return f"pop={pop},t={int(time_limit)}s"


def main() -> None:
    args = _build_parser().parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]

    load_result = io.load_runs_recursive(results_dir)
    warnings = list(load_result.warnings)
    if not load_result.runs:
        raise SystemExit(f"No runs found in {results_dir}")

    tables_dir = ensure_dir(out_dir / "tables")
    plots_dir = ensure_dir(out_dir / "plots")
    agg_dir = ensure_dir(out_dir / "aggregated")

    # Per-run rows
    run_header = [
        "run_id",
        "instance",
        "seed",
        "pop_size",
        "time_limit",
        "best_k",
        "best_distance",
        "best_total_waiting",
        "best_total_service",
        "best_total_route_time",
        "elapsed_seconds",
    ]
    run_rows = []
    for run in load_result.runs:
        metrics = run.best_metrics
        run_rows.append(
            {
                "run_id": run.run_id,
                "instance": run.instance_name,
                "seed": run.seed,
                "pop_size": run.config.get("pop_size"),
                "time_limit": run.config.get("time_limit"),
                "best_k": metrics.get("best_k"),
                "best_distance": metrics.get("best_distance"),
                "best_total_waiting": metrics.get("best_total_waiting"),
                "best_total_service": metrics.get("best_total_service"),
                "best_total_route_time": metrics.get("best_total_route_time"),
                "elapsed_seconds": run.elapsed_seconds,
            }
        )
    if "csv" in formats:
        with (agg_dir / "runs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_header)
            writer.writeheader()
            writer.writerows(run_rows)

    rows: List[Dict[str, Any]] = []

    for run in load_result.runs:
        pop = int(run.config.get("pop_size", 0))
        time_limit = float(run.config.get("time_limit", 0.0))
        group = aggregate.solomon_group(run.instance_name)
        best_k = run.best_metrics.get("best_k")
        best_rt = run.best_metrics.get("best_total_route_time")
        if best_k is None or best_rt is None:
            warnings.append(f"Missing metrics for run {run.run_id}")
            continue
        rows.append(
            {
                "group": group,
                "pop_size": pop,
                "time_limit": time_limit,
                "best_k": float(best_k),
                "best_route_time": float(best_rt),
                "runtime_sec": float(run.elapsed_seconds),
            }
        )

    table_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["group"], row["pop_size"], row["time_limit"])
        grouped[key].append(row)

    for (group, pop, time_limit), items in sorted(grouped.items()):
        table_rows.append(
            {
                "group": group,
                "pop_size": pop,
                "time_limit": time_limit,
                "mean_best_k": _mean([r["best_k"] for r in items]),
                "mean_best_route_time": _mean([r["best_route_time"] for r in items]),
                "mean_runtime_sec": _mean([r["runtime_sec"] for r in items]),
                "n": len(items),
            }
        )

    header = [
        "group",
        "pop_size",
        "time_limit",
        "mean_best_k",
        "mean_best_route_time",
        "mean_runtime_sec",
        "n",
    ]

    if "csv" in formats:
        with (tables_dir / "population_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(table_rows)

    if "md" in formats:
        with (tables_dir / "population_sensitivity.md").open("w") as f:
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "|".join(["---"] * len(header)) + "|\n")
            for row in table_rows:
                f.write(
                    "| "
                    + " | ".join(str(row.get(h, "")) if row.get(h) is not None else "" for h in header)
                    + " |\n"
                )

    # Aggregate per instance + config
    inst_rows: List[Dict[str, Any]] = []
    grouped_inst: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = defaultdict(list)
    for run in load_result.runs:
        pop = int(run.config.get("pop_size", 0))
        time_limit = float(run.config.get("time_limit", 0.0))
        inst = run.instance_name
        best_k = run.best_metrics.get("best_k")
        best_rt = run.best_metrics.get("best_total_route_time")
        if best_k is None or best_rt is None:
            continue
        grouped_inst[(inst, pop, time_limit)].append(
            {
                "best_k": float(best_k),
                "best_route_time": float(best_rt),
            }
        )

    inst_header = ["instance", "pop_size", "time_limit", "mean_best_k", "mean_best_route_time", "n"]
    for (inst, pop, time_limit), items in sorted(grouped_inst.items()):
        inst_rows.append(
            {
                "instance": inst,
                "pop_size": pop,
                "time_limit": time_limit,
                "mean_best_k": _mean([r["best_k"] for r in items]),
                "mean_best_route_time": _mean([r["best_route_time"] for r in items]),
                "n": len(items),
            }
        )

    if "csv" in formats:
        with (agg_dir / "per_instance.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=inst_header)
            writer.writeheader()
            writer.writerows(inst_rows)
    if "md" in formats:
        with (agg_dir / "per_instance.md").open("w") as f:
            f.write("| " + " | ".join(inst_header) + " |\n")
            f.write("|" + "|".join(["---"] * len(inst_header)) + "|\n")
            for row in inst_rows:
                f.write(
                    "| "
                    + " | ".join(str(row.get(h, "")) if row.get(h) is not None else "" for h in inst_header)
                    + " |\n"
                )

    # Tradeoff plots by group
    grouped_points: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    grouped_points_k: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    for row in table_rows:
        group = row["group"]
        label = _format_config(int(row["pop_size"]), float(row["time_limit"]))
        rt = row["mean_best_route_time"]
        runtime = row["mean_runtime_sec"]
        k = row["mean_best_k"]
        if rt is not None and runtime is not None:
            grouped_points[group][label] = (runtime, rt)
        if k is not None and runtime is not None:
            grouped_points_k[group][label] = (runtime, k)

    for group, points in grouped_points.items():
        plots.plot_scatter_labels(
            points,
            plots_dir / f"tradeoff_route_time_{group}.png",
            title=f"Route Time vs Runtime ({group})",
            xlabel="Runtime (s)",
            ylabel="Route Time",
        )

    for group, points in grouped_points_k.items():
        plots.plot_scatter_labels(
            points,
            plots_dir / f"tradeoff_k_{group}.png",
            title=f"K vs Runtime ({group})",
            xlabel="Runtime (s)",
            ylabel="K",
        )

    summary = {
        "generated_at": now_timestamp(),
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "run_count": len(load_result.runs),
        "used_files": sorted(load_result.used_files),
        "warnings": warnings,
    }
    ensure_dir(out_dir)
    (agg_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
