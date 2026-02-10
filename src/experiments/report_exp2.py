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
    p = argparse.ArgumentParser(description="Report for Experiment 2 (operator sensitivity)")
    p.add_argument("--results_dir", type=str, default="results/exp2_operator_sensitivity/raw")
    p.add_argument("--out_dir", type=str, default="results/exp2_operator_sensitivity")
    p.add_argument("--format", type=str, default="md,csv")
    return p


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _convergence_gen(history: List[Dict[str, Any]], target_k: float) -> float | None:
    if not history:
        return None
    sorted_hist = sorted(history, key=lambda h: float(h.get("gen", 0.0)))
    for entry in sorted_hist:
        if entry.get("best_k") is not None and float(entry.get("best_k")) <= target_k:
            return float(entry.get("gen", 0.0))
    return None


def _format_config(cr: float, mr: float) -> str:
    return f"cr={cr:.1f},mr={mr:.1f}"


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
        "crossover_rate",
        "mutation_rate",
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
                "crossover_rate": run.config.get("crossover_rate"),
                "mutation_rate": run.config.get("mutation_rate"),
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
    series_by_group: Dict[str, Dict[str, Dict[float, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    box_by_group: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for run in load_result.runs:
        cr = float(run.config.get("crossover_rate", 0.0))
        mr = float(run.config.get("mutation_rate", 0.0))
        group = aggregate.solomon_group(run.instance_name)
        config_label = _format_config(cr, mr)

        best_k = run.best_metrics.get("best_k")
        best_rt = run.best_metrics.get("best_total_route_time")
        if best_k is None or best_rt is None:
            warnings.append(f"Missing metrics for run {run.run_id}")
            continue

        conv_gen = _convergence_gen(run.history, float(best_k))
        rows.append(
            {
                "group": group,
                "crossover_rate": cr,
                "mutation_rate": mr,
                "best_k": float(best_k),
                "best_route_time": float(best_rt),
                "convergence_gen": conv_gen,
            }
        )

        box_by_group[group][config_label].append(float(best_rt))

        for entry in run.history:
            gen = float(entry.get("gen", 0.0))
            rt = entry.get("best_total_route_time")
            if rt is None:
                continue
            series_by_group[group][config_label][gen].append(float(rt))

    # Aggregate table
    table_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["group"], row["crossover_rate"], row["mutation_rate"])
        grouped[key].append(row)

    for (group, cr, mr), items in sorted(grouped.items()):
        table_rows.append(
            {
                "group": group,
                "crossover_rate": cr,
                "mutation_rate": mr,
                "mean_best_k": _mean([r["best_k"] for r in items]),
                "mean_best_route_time": _mean([r["best_route_time"] for r in items]),
                "mean_convergence_gen": _mean([r["convergence_gen"] for r in items if r["convergence_gen"] is not None]),
                "n": len(items),
            }
        )

    header = [
        "group",
        "crossover_rate",
        "mutation_rate",
        "mean_best_k",
        "mean_best_route_time",
        "mean_convergence_gen",
        "n",
    ]

    if "csv" in formats:
        with (tables_dir / "operator_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(table_rows)

    if "md" in formats:
        with (tables_dir / "operator_sensitivity.md").open("w") as f:
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
    grouped_inst: Dict[Tuple[str, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for run in load_result.runs:
        cr = float(run.config.get("crossover_rate", 0.0))
        mr = float(run.config.get("mutation_rate", 0.0))
        inst = run.instance_name
        best_k = run.best_metrics.get("best_k")
        best_rt = run.best_metrics.get("best_total_route_time")
        if best_k is None or best_rt is None:
            continue
        grouped_inst[(inst, cr, mr)].append(
            {
                "best_k": float(best_k),
                "best_route_time": float(best_rt),
            }
        )

    inst_header = ["instance", "crossover_rate", "mutation_rate", "mean_best_k", "mean_best_route_time", "n"]
    for (inst, cr, mr), items in sorted(grouped_inst.items()):
        inst_rows.append(
            {
                "instance": inst,
                "crossover_rate": cr,
                "mutation_rate": mr,
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

    # Plots
    for group, configs in series_by_group.items():
        series_for_plot: Dict[str, Dict[float, float]] = {}
        for label, series in configs.items():
            mean_series = {gen: _mean(vals) for gen, vals in series.items() if vals}
            series_for_plot[label] = mean_series
        plots.plot_multi_line(
            series_for_plot,
            plots_dir / f"route_time_vs_gen_{group}.png",
            title=f"Route Time vs Generation ({group})",
            ylabel="Route Time",
        )

    for group, configs in box_by_group.items():
        plots.plot_boxplots(
            configs,
            plots_dir / f"box_route_time_{group}.png",
            title=f"Final Route Time by Config ({group})",
            ylabel="Route Time",
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
