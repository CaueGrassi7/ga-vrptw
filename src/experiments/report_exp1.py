from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from vrptw_ga.utils import ensure_dir, now_timestamp

from .reporting import aggregate, io, pb96_like, plots


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report for Experiment 1 (PB96 replication)")
    p.add_argument("--results_dir", type=str, default="results/exp1_replication/raw")
    p.add_argument("--out_dir", type=str, default="results/exp1_replication")
    p.add_argument("--paper_ref", type=str, default="docs/pb96_reference_tables.csv")
    p.add_argument("--gen_marks", nargs="*", type=int, default=[0, 20, 50])
    p.add_argument("--format", type=str, default="md,csv", help="Output formats: md,csv,json")
    p.add_argument("--include_paper", action=argparse.BooleanOptionalAction, default=True)
    return p


def _git_commit() -> str | None:
    try:
        import subprocess

        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def main() -> None:
    args = _build_parser().parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]

    load_result = io.load_runs(results_dir)
    warnings = list(load_result.warnings)
    if not load_result.runs:
        raise SystemExit(f"No runs found in {results_dir}")

    paper_rows: List[Dict[str, Any]] = []
    if args.include_paper:
        paper_path = Path(args.paper_ref)
        if paper_path.exists():
            paper_rows = io.load_paper_reference(paper_path)
        else:
            warnings.append(f"Paper reference not found: {paper_path}")

    agg = aggregate.build_aggregate(
        runs=load_result.runs,
        group_by="solomon_type",
        gen_marks=sorted(set(args.gen_marks)),
        only_best=False,
    )
    warnings.extend(agg.warnings)

    tables_dir = ensure_dir(out_dir / "tables")
    plots_dir = ensure_dir(out_dir / "plots")
    aggregated_dir = ensure_dir(out_dir / "aggregated")

    # Per-run rows
    run_header = [
        "run_id",
        "instance",
        "seed",
        "pop_size",
        "generations",
        "time_limit",
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
                "pop_size": run.config.get("pop_size"),
                "generations": run.config.get("generations"),
                "time_limit": run.config.get("time_limit"),
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
        import csv

        with (aggregated_dir / "runs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_header)
            writer.writeheader()
            writer.writerows(run_rows)

    pb96_like.write_pb96_like_tables(agg.group_rows, paper_rows, tables_dir, formats)

    # Per-instance aggregated table (OUR rows only)
    inst_agg = aggregate.build_aggregate(
        runs=load_result.runs,
        group_by="instance_name",
        gen_marks=sorted(set(args.gen_marks)),
        only_best=False,
    )
    inst_header = ["instance", "row", "K", "distance", "waiting", "route_time", "n"]
    inst_rows = [
        {
            "instance": row["group"],
            "row": row["row"],
            "K": row["K"],
            "distance": row["distance"],
            "waiting": row["waiting"],
            "route_time": row["route_time"],
            "n": row.get("n"),
        }
        for row in inst_agg.group_rows
    ]
    if "csv" in formats:
        import csv

        with (aggregated_dir / "per_instance.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=inst_header)
            writer.writeheader()
            writer.writerows(inst_rows)
    if "md" in formats:
        with (aggregated_dir / "per_instance.md").open("w") as f:
            f.write("| " + " | ".join(inst_header) + " |\\n")
            f.write("|" + "|".join(["---"] * len(inst_header)) + "|\\n")
            for row in inst_rows:
                f.write(
                    "| "
                    + " | ".join(str(row.get(h, "")) if row.get(h) is not None else "" for h in inst_header)
                    + " |\\n"
                )

    for group, series in agg.series_by_group.items():
        if series:
            plots.plot_line_with_band(
                series,
                plots_dir / f"line_best_k_{group}.png",
                title=f"Best K over Generations ({group})",
                ylabel="Best K",
                mean_key="K_mean",
                std_key="K_std",
            )
            plots.plot_line_with_band(
                series,
                plots_dir / f"line_best_route_time_{group}.png",
                title=f"Best Route Time over Generations ({group})",
                ylabel="Route Time",
                mean_key="route_time_mean",
                std_key="route_time_std",
            )

    summary = {
        "generated_at": now_timestamp(),
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "args": vars(args),
        "run_count": len(load_result.runs),
        "groups": sorted({aggregate.group_key(r, "solomon_type") for r in load_result.runs}),
        "used_files": sorted(load_result.used_files),
        "git_commit": _git_commit(),
        "warnings": warnings,
    }
    ensure_dir(out_dir)
    (aggregated_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
