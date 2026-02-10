from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

from vrptw_ga.utils import ensure_dir, now_timestamp


GROUP_ORDER = ["R1", "R2", "C1", "C2", "RC1", "RC2"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross comparison between exp1/exp2/exp3")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_dir", type=str, default="canva/05_experiments_comparison")
    return p


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.3f}"
    return str(v)


def _solomon_group(instance: str) -> str:
    n = instance.upper()
    if n.startswith("RC"):
        num = int(n[2:])
        if 101 <= num <= 108:
            return "RC1"
        if 201 <= num <= 208:
            return "RC2"
    elif n.startswith("R"):
        num = int(n[1:])
        if 101 <= num <= 112:
            return "R1"
        if 201 <= num <= 211:
            return "R2"
    elif n.startswith("C"):
        num = int(n[1:])
        if 101 <= num <= 109:
            return "C1"
        if 201 <= num <= 208:
            return "C2"
    return "UNKNOWN"


def _inst_sort(inst: str) -> Tuple[str, int]:
    u = inst.upper()
    if u.startswith("RC"):
        return ("RC", int(u[2:]))
    if u.startswith("R"):
        return ("R", int(u[1:]))
    if u.startswith("C"):
        return ("C", int(u[1:]))
    return ("Z", 0)


def _render_table_png(path: Path, title: str, cols: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    cell_text = [[_fmt(r.get(c)) for c in cols] for r in rows]
    fig_w = max(11.0, min(24.0, 1.5 * len(cols)))
    fig_h = max(4.5, min(28.0, 0.42 * len(rows) + 2.3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=list(cols), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f7fbff" if r % 2 else "#e8f1fa")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def _write_csv(path: Path, cols: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pick_best(rows: List[Dict[str, str]], k_col: str, rt_col: str) -> Dict[str, str]:
    return min(rows, key=lambda r: (float(r[k_col]), float(r[rt_col])))


def _mean(vals: List[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _plot_bar(
    path: Path,
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    zoom: bool,
) -> None:
    plt.figure(figsize=(7, 4))
    bars = plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if zoom:
        vmin = min(values)
        vmax = max(values)
        span = max(vmax - vmin, 1e-9)
        pad = span * 0.35
        plt.ylim(vmin - pad, vmax + pad)
    else:
        plt.ylim(0, max(values) * 1.08)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.results_root)
    out = Path(args.out_dir)
    ensure_dir(out)

    exp1_runs = _read_csv(root / "exp1_replication" / "full" / "aggregated" / "runs.csv")
    exp2_pi = _read_csv(root / "exp2_operator_sensitivity" / "full" / "aggregated" / "per_instance.csv")
    exp3_pi = _read_csv(root / "exp3_population_sensitivity" / "full" / "aggregated" / "per_instance.csv")

    # Exp1 best per instance using available rows (dedupe by instance+seed keeping best lexicographic)
    exp1_by_inst_seed: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in exp1_runs:
        exp1_by_inst_seed[(r["instance"], r["seed"])].append(r)
    exp1_seed_best: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (inst, _seed), rows in exp1_by_inst_seed.items():
        b = min(rows, key=lambda x: (float(x["best_k"]), float(x["best_total_route_time"])))
        exp1_seed_best[inst].append((float(b["best_k"]), float(b["best_total_route_time"])))
    exp1_best: Dict[str, Tuple[float, float]] = {}
    for inst, vals in exp1_seed_best.items():
        exp1_best[inst] = (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))

    exp2_group: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in exp2_pi:
        exp2_group[r["instance"]].append(r)
    exp2_best: Dict[str, Dict[str, Any]] = {}
    for inst, rows in exp2_group.items():
        b = _pick_best(rows, "mean_best_k", "mean_best_route_time")
        exp2_best[inst] = {
            "k": float(b["mean_best_k"]),
            "rt": float(b["mean_best_route_time"]),
            "cr": float(b["crossover_rate"]),
            "mr": float(b["mutation_rate"]),
        }

    exp3_group: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in exp3_pi:
        exp3_group[r["instance"]].append(r)
    exp3_best: Dict[str, Dict[str, Any]] = {}
    for inst, rows in exp3_group.items():
        b = _pick_best(rows, "mean_best_k", "mean_best_route_time")
        exp3_best[inst] = {
            "k": float(b["mean_best_k"]),
            "rt": float(b["mean_best_route_time"]),
            "pop": int(float(b["pop_size"])),
            "time_limit": float(b["time_limit"]),
        }

    instances = sorted(set(exp1_best.keys()) & set(exp2_best.keys()) & set(exp3_best.keys()), key=_inst_sort)
    per_inst_rows: List[Dict[str, Any]] = []
    wins = {"exp1": 0, "exp2": 0, "exp3": 0}
    wins_by_group = defaultdict(lambda: {"exp1": 0, "exp2": 0, "exp3": 0})

    for inst in instances:
        grp = _solomon_group(inst)
        e1k, e1rt = exp1_best[inst]
        e2 = exp2_best[inst]
        e3 = exp3_best[inst]
        cand = {
            "exp1": (e1k, e1rt),
            "exp2": (e2["k"], e2["rt"]),
            "exp3": (e3["k"], e3["rt"]),
        }
        best_src = min(cand.keys(), key=lambda s: (cand[s][0], cand[s][1], s))
        wins[best_src] += 1
        wins_by_group[grp][best_src] += 1
        per_inst_rows.append(
            {
                "instance": inst,
                "group": grp,
                "exp1_k": e1k,
                "exp1_route_time": e1rt,
                "exp2_k": e2["k"],
                "exp2_route_time": e2["rt"],
                "exp2_best_cr": e2["cr"],
                "exp2_best_mr": e2["mr"],
                "exp3_k": e3["k"],
                "exp3_route_time": e3["rt"],
                "exp3_best_pop": e3["pop"],
                "exp3_best_time_limit": e3["time_limit"],
                "best_source": best_src,
                "best_k": cand[best_src][0],
                "best_route_time": cand[best_src][1],
                "delta_exp2_vs_exp1_route_time": e2["rt"] - e1rt,
                "delta_exp3_vs_exp1_route_time": e3["rt"] - e1rt,
            }
        )

    cols_inst = [
        "instance",
        "group",
        "exp1_k",
        "exp1_route_time",
        "exp2_k",
        "exp2_route_time",
        "exp2_best_cr",
        "exp2_best_mr",
        "exp3_k",
        "exp3_route_time",
        "exp3_best_pop",
        "exp3_best_time_limit",
        "best_source",
        "best_k",
        "best_route_time",
        "delta_exp2_vs_exp1_route_time",
        "delta_exp3_vs_exp1_route_time",
    ]
    _write_csv(out / "cross_per_instance.csv", cols_inst, per_inst_rows)

    # Summaries
    global_rows = []
    for src, kcol, rtcol in [
        ("exp1", "exp1_k", "exp1_route_time"),
        ("exp2", "exp2_k", "exp2_route_time"),
        ("exp3", "exp3_k", "exp3_route_time"),
        ("best", "best_k", "best_route_time"),
    ]:
        global_rows.append(
            {
                "source": src,
                "mean_k": _mean([float(r[kcol]) for r in per_inst_rows]),
                "mean_route_time": _mean([float(r[rtcol]) for r in per_inst_rows]),
                "n_instances": len(per_inst_rows),
            }
        )
    _write_csv(out / "cross_global_summary.csv", ["source", "mean_k", "mean_route_time", "n_instances"], global_rows)

    group_rows = []
    for g in GROUP_ORDER:
        g_rows = [r for r in per_inst_rows if r["group"] == g]
        if not g_rows:
            continue
        for src, kcol, rtcol in [
            ("exp1", "exp1_k", "exp1_route_time"),
            ("exp2", "exp2_k", "exp2_route_time"),
            ("exp3", "exp3_k", "exp3_route_time"),
            ("best", "best_k", "best_route_time"),
        ]:
            group_rows.append(
                {
                    "group": g,
                    "source": src,
                    "mean_k": _mean([float(r[kcol]) for r in g_rows]),
                    "mean_route_time": _mean([float(r[rtcol]) for r in g_rows]),
                    "n_instances": len(g_rows),
                }
            )
    _write_csv(out / "cross_group_summary.csv", ["group", "source", "mean_k", "mean_route_time", "n_instances"], group_rows)

    wins_rows = [{"source": k, "wins": v} for k, v in wins.items()]
    _write_csv(out / "cross_wins.csv", ["source", "wins"], wins_rows)
    wins_group_rows = []
    for g in GROUP_ORDER:
        if g not in wins_by_group:
            continue
        wins_group_rows.append(
            {
                "group": g,
                "wins_exp1": wins_by_group[g]["exp1"],
                "wins_exp2": wins_by_group[g]["exp2"],
                "wins_exp3": wins_by_group[g]["exp3"],
            }
        )
    _write_csv(out / "cross_wins_by_group.csv", ["group", "wins_exp1", "wins_exp2", "wins_exp3"], wins_group_rows)

    # Presentation PNG tables
    _render_table_png(
        out / "Table_Cross_Global_Summary_presentation.png",
        "Cross-Experiment Global Summary (Exp1 vs Exp2 vs Exp3 vs Best)",
        ["source", "mean_k", "mean_route_time", "n_instances"],
        global_rows,
    )
    _render_table_png(
        out / "Table_Cross_Wins_By_Group_presentation.png",
        "Cross-Experiment Wins by Group",
        ["group", "wins_exp1", "wins_exp2", "wins_exp3"],
        wins_group_rows,
    )

    # Charts
    x = list(range(len(global_rows)))
    labels = [r["source"] for r in global_rows]
    rt = [float(r["mean_route_time"]) for r in global_rows]
    k = [float(r["mean_k"]) for r in global_rows]

    _plot_bar(
        out / "plot_cross_mean_route_time.png",
        labels,
        rt,
        "Cross-Experiment Mean Route Time (absolute scale)",
        "Mean Route Time",
        zoom=False,
    )
    _plot_bar(
        out / "plot_cross_mean_route_time_zoom.png",
        labels,
        rt,
        "Cross-Experiment Mean Route Time (zoomed scale)",
        "Mean Route Time",
        zoom=True,
    )
    _plot_bar(
        out / "plot_cross_mean_k.png",
        labels,
        k,
        "Cross-Experiment Mean K (absolute scale)",
        "Mean K",
        zoom=False,
    )
    _plot_bar(
        out / "plot_cross_mean_k_zoom.png",
        labels,
        k,
        "Cross-Experiment Mean K (zoomed scale)",
        "Mean K",
        zoom=True,
    )

    g_labels = [r["group"] for r in wins_group_rows]
    y1 = [r["wins_exp1"] for r in wins_group_rows]
    y2 = [r["wins_exp2"] for r in wins_group_rows]
    y3 = [r["wins_exp3"] for r in wins_group_rows]
    plt.figure(figsize=(8, 4))
    plt.bar(g_labels, y1, label="exp1")
    plt.bar(g_labels, y2, bottom=y1, label="exp2")
    plt.bar(g_labels, y3, bottom=[a + b for a, b in zip(y1, y2)], label="exp3")
    plt.ylabel("Wins")
    plt.title("Cross-Experiment Wins by Group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "plot_cross_wins_by_group_stacked.png", dpi=160)
    plt.close()

    readme = out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Cross-Experiment Comparison",
                "",
                f"- Generated at: `{now_timestamp()}`",
                "- Compared experiments: `exp1` baseline vs best-per-instance from `exp2` and `exp3`.",
                "- Ranking rule: lexicographic `(K, route_time)`.",
                "",
                "## Files",
                "- `cross_per_instance.csv`",
                "- `cross_global_summary.csv`",
                "- `cross_group_summary.csv`",
                "- `cross_wins.csv`",
                "- `cross_wins_by_group.csv`",
                "- `Table_Cross_Global_Summary_presentation.png`",
                "- `Table_Cross_Wins_By_Group_presentation.png`",
                "- `plot_cross_mean_route_time.png`",
                "- `plot_cross_mean_k.png`",
                "- `plot_cross_mean_route_time_zoom.png`",
                "- `plot_cross_mean_k_zoom.png`",
                "- `plot_cross_wins_by_group_stacked.png`",
            ]
        )
        + "\n"
    )

    print(f"cross-experiment package generated at: {out}")


if __name__ == "__main__":
    main()
