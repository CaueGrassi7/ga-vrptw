from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from vrptw_ga.utils import ensure_dir, now_timestamp

from .reporting import aggregate, io, pb96_like, plots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PB96-like reports for VRPTW GA runs")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with run_*.json and progress_*.csv")
    parser.add_argument("--out_dir", type=str, default="reports", help="Output directory for tables/plots")
    parser.add_argument("--paper_ref", type=str, default="docs/pb96_reference_tables.csv", help="PB96 reference CSV")
    parser.add_argument("--gen_marks", nargs="*", type=int, default=[0, 20, 50], help="Generation marks")
    parser.add_argument("--metric", type=str, default="route_time", help="Primary metric for plots")
    parser.add_argument("--include_paper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only_best", action="store_true", help="Skip generation marks and plots")
    parser.add_argument("--format", type=str, default="md,csv", help="Output formats: md,csv,json")
    parser.add_argument("--fail_on_missing", action="store_true", help="Fail if required inputs are missing")
    parser.add_argument("--group_by", type=str, choices=["solomon_type", "instance_name"], default="solomon_type")
    return parser


def _git_commit() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def _config_signature(config: Dict[str, Any]) -> str:
    keys = [
        "pop_size",
        "generations",
        "time_limit",
        "penalty_tw",
        "crossover_rate",
        "mutation_rate",
        "elite",
        "tournament_k",
        "p_swap",
        "p_inversion",
        "decoder",
        "crossover",
        "objective",
        "init",
    ]
    parts = []
    for key in keys:
        if key in config:
            parts.append(f"{key}={config.get(key)}")
    return ";".join(parts)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]

    load_result = io.load_runs(results_dir)
    warnings = list(load_result.warnings)

    if not load_result.runs:
        msg = f"No runs found in {results_dir}"
        if args.fail_on_missing:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg)
        return

    paper_rows: List[Dict[str, Any]] = []
    if args.include_paper:
        paper_path = Path(args.paper_ref)
        if paper_path.exists():
            paper_rows = io.load_paper_reference(paper_path)
        else:
            warnings.append(f"Paper reference not found: {paper_path}")
            if args.fail_on_missing:
                print(f"Missing paper reference: {paper_path}", file=sys.stderr)
                sys.exit(1)

    agg = aggregate.build_aggregate(
        runs=load_result.runs,
        group_by=args.group_by,
        gen_marks=sorted(set(args.gen_marks)),
        only_best=args.only_best,
    )
    warnings.extend(agg.warnings)

    tables_dir = ensure_dir(out_dir / "tables")
    plots_dir = ensure_dir(out_dir / "plots")

    pb96_like.write_pb96_like_tables(agg.group_rows, paper_rows, tables_dir, formats)

    if not args.only_best:
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

    if args.include_paper and not args.only_best:
        our_by_group = {row["group"]: row for row in agg.group_rows if row.get("row") == "OUR-GA-50"}
        paper_by_group = {row["group"]: row for row in paper_rows if row.get("row") == "GENEROUS-50"}
        groups = [g for g in ["R1", "R2", "C1", "C2", "RC1", "RC2"] if g in our_by_group and g in paper_by_group]
        if groups:
            paper_k = [float(paper_by_group[g]["K"]) for g in groups]
            our_k = [float(our_by_group[g]["K"]) for g in groups]
            plots.plot_bar_compare(
                groups,
                paper_k,
                our_k,
                plots_dir / "bar_k_paper_vs_our_gen50.png",
                title="K at Generation 50: Paper vs Our GA",
                ylabel="K",
                paper_label="PB96 GENEROUS-50",
                our_label="OUR-GA-50",
            )

            paper_rt = [float(paper_by_group[g]["route_time"]) for g in groups]
            our_rt = [float(our_by_group[g]["route_time"]) for g in groups]
            plots.plot_bar_compare(
                groups,
                paper_rt,
                our_rt,
                plots_dir / "bar_route_time_paper_vs_our_gen50.png",
                title="Route Time at Generation 50: Paper vs Our GA",
                ylabel="Route Time",
                paper_label="PB96 GENEROUS-50",
                our_label="OUR-GA-50",
            )
        else:
            warnings.append("Missing OUR-GA-50 or GENEROUS-50 data for bar comparisons")

    for metric, title, ylabel in (
        ("K", "Final Best K Distribution", "K"),
        ("route_time", "Final Best Route Time Distribution", "Route Time"),
    ):
        values_by_group = {g: vals.get(metric, []) for g, vals in agg.final_values.items()}
        plots.plot_boxplots(
            values_by_group,
            plots_dir / f"box_{metric}_final.png",
            title=title,
            ylabel=ylabel,
        )

    config_signatures = sorted({_config_signature(run.config) for run in load_result.runs})
    run_summaries = [
        {
            "run_id": run.run_id,
            "instance_name": run.instance_name,
            "seed": run.seed,
            "time_limit": run.config.get("time_limit"),
            "pop_size": run.config.get("pop_size"),
            "generations": run.config.get("generations"),
            "crossover": run.config.get("crossover"),
            "init": run.config.get("init"),
            "objective": run.config.get("objective"),
        }
        for run in load_result.runs
    ]

    summary = {
        "generated_at": now_timestamp(),
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "args": vars(args),
        "run_count": len(load_result.runs),
        "groups": sorted({aggregate.group_key(r, args.group_by) for r in load_result.runs}),
        "used_files": sorted(load_result.used_files),
        "config_signatures": config_signatures,
        "runs": run_summaries,
        "git_commit": _git_commit(),
        "warnings": warnings,
    }
    ensure_dir(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
