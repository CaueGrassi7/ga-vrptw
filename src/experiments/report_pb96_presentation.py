from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

from vrptw_ga.utils import ensure_dir, now_timestamp


TABLE2_REFERENCE = [
    {"Algorithm": "I1", "Routes": 13.5, "Route Time": 2775.0, "Comp. Time": "---"},
    {"Algorithm": "GENEROUS", "Routes": 12.1, "Route Time": 2509.9, "Comp. Time": "10:58"},
    {"Algorithm": "GEN-SBX-1M", "Routes": 12.9, "Route Time": 2731.5, "Comp. Time": "1:59"},
    {"Algorithm": "GEN-SBX-2M", "Routes": 12.9, "Route Time": 2729.1, "Comp. Time": "2:45"},
    {"Algorithm": "GEN-SBX-LSM", "Routes": 12.6, "Route Time": 2521.7, "Comp. Time": "22:32"},
    {"Algorithm": "GEN-RBX-1M", "Routes": 12.9, "Route Time": 2722.2, "Comp. Time": "3:38"},
    {"Algorithm": "GEN-RBX-2M", "Routes": 12.8, "Route Time": 2732.2, "Comp. Time": "5:11"},
    {"Algorithm": "GEN-RBX-LSM", "Routes": 12.5, "Route Time": 2515.2, "Comp. Time": "28:35"},
]

TABLE3_REFERENCE = [
    {"Set": "R1", "I1": 13.6, "PARIS": 13.3, "GRASP": 13.1, "CTA": 13.0, "GIDEON": 12.8, "TABU": 12.5, "GENEROUS": 12.6},
    {"Set": "R2", "I1": 3.3, "PARIS": 3.1, "GRASP": 3.1, "CTA": 3.1, "GIDEON": 3.2, "TABU": 3.1, "GENEROUS": 3.0},
    {"Set": "C1", "I1": 10.0, "PARIS": 10.7, "GRASP": 10.6, "CTA": 10.0, "GIDEON": 10.0, "TABU": 10.0, "GENEROUS": 10.0},
    {"Set": "C2", "I1": 3.1, "PARIS": 3.4, "GRASP": 3.4, "CTA": 3.0, "GIDEON": 3.0, "TABU": 3.0, "GENEROUS": 3.0},
    {"Set": "RC1", "I1": 13.5, "PARIS": 13.4, "GRASP": 12.8, "CTA": 13.0, "GIDEON": 12.5, "TABU": 12.6, "GENEROUS": 12.1},
    {"Set": "RC2", "I1": 3.9, "PARIS": 3.6, "GRASP": 3.6, "CTA": 3.7, "GIDEON": 3.4, "TABU": 3.4, "GENEROUS": 3.4},
]

TABLE4_REFERENCE = [
    {"Problem": "R101", "I1 (k,d)": "21, 1867.1", "TABU (k,d)": "19, 1650.7", "GENEROUS (k,d)": "19, 1669.4", "OPTIMUM (k,d)": "18, 1607.7"},
    {"Problem": "R102", "I1 (k,d)": "19, 1699.5", "TABU (k,d)": "18, 1471.8", "GENEROUS (k,d)": "17, 1532.1", "OPTIMUM (k,d)": "17, 1434.0"},
    {"Problem": "C101", "I1 (k,d)": "10, 851.4", "TABU (k,d)": "10, 827.3", "GENEROUS (k,d)": "10, 827.3", "OPTIMUM (k,d)": "10, 827.3"},
    {"Problem": "C102", "I1 (k,d)": "10, 966.7", "TABU (k,d)": "10, 827.3", "GENEROUS (k,d)": "10, 827.3", "OPTIMUM (k,d)": "10, 827.3"},
    {"Problem": "C106", "I1 (k,d)": "10, 916.0", "TABU (k,d)": "10, 827.3", "GENEROUS (k,d)": "10, 827.3", "OPTIMUM (k,d)": "10, 827.3"},
    {"Problem": "C107", "I1 (k,d)": "10, 902.4", "TABU (k,d)": "10, 827.3", "GENEROUS (k,d)": "10, 827.3", "OPTIMUM (k,d)": "10, 827.3"},
    {"Problem": "C108", "I1 (k,d)": "10, 853.1", "TABU (k,d)": "10, 827.3", "GENEROUS (k,d)": "10, 827.3", "OPTIMUM (k,d)": "10, 827.3"},
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate PB96 presentation-ready tables as PNG")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_dir", type=str, default="canva")
    return p


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}"
    return str(value)


def _render_table_png(
    path: Path,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    row_highlight: Callable[[Dict[str, Any]], str | None] | None = None,
) -> None:
    ensure_dir(path.parent)
    cell_text = [[_fmt(row.get(col)) for col in columns] for row in rows]
    fig_w = max(11.0, min(24.0, 1.55 * len(columns)))
    fig_h = max(4.8, min(24.0, 0.42 * len(rows) + 2.4))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=list(columns), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)

    header_bg = "#1f4e79"
    header_fg = "white"
    zebra1 = "#f7fbff"
    zebra2 = "#e8f1fa"
    high_best = "#d9f2d9"
    high_ref = "#fdecc8"

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color(header_fg)
            cell.get_text().set_weight("bold")
            continue
        row = rows[r - 1]
        base = zebra1 if (r % 2 == 1) else zebra2
        if row_highlight:
            tag = row_highlight(row)
            if tag == "best":
                base = high_best
            elif tag == "ref":
                base = high_ref
        cell.set_facecolor(base)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def _load_exp1_best_by_instance(exp1_raw_dir: Path) -> Dict[str, Tuple[float, float]]:
    best_map: Dict[Tuple[str, int, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
    for run_path in sorted(exp1_raw_dir.glob("run_*.json")):
        data = json.loads(run_path.read_text())
        best = data.get("best_metrics", {})
        k = _to_float(best.get("best_k"))
        d = _to_float(best.get("best_distance"))
        if k is None or d is None:
            continue
        cfg = tuple(sorted((str(k0), str(v0)) for k0, v0 in dict(data.get("config", {})).items()))
        key = (str(data.get("instance_name", "")), int(data.get("seed", 0)), cfg)
        old = best_map.get(key)
        candidate = {
            "run_id": str(data.get("run_id", "")),
            "timestamp": str(data.get("timestamp", "")),
            "instance": str(data.get("instance_name", "")),
            "k": k,
            "d": d,
        }
        if old is None or (candidate["timestamp"], candidate["run_id"]) > (old["timestamp"], old["run_id"]):
            best_map[key] = candidate

    per_instance: Dict[str, List[Tuple[float, float]]] = {}
    for row in best_map.values():
        per_instance.setdefault(row["instance"], []).append((row["k"], row["d"]))

    out: Dict[str, Tuple[float, float]] = {}
    for inst, vals in per_instance.items():
        k, d = min(vals, key=lambda t: (t[0], t[1]))
        out[inst] = (k, int(d * 10) / 10.0)
    return out


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.results_root)
    out = Path(args.out_dir)

    # Clean current canva content.
    if out.exists():
        for f in sorted(out.glob("**/*"), reverse=True):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                try:
                    f.rmdir()
                except OSError:
                    pass
    ensure_dir(out)

    exp1_tables = root / "exp1_replication" / "full" / "tables"
    exp1_raw = root / "exp1_replication" / "full" / "raw"
    if not exp1_tables.exists() or not exp1_raw.exists():
        raise SystemExit("Missing exp1 full results.")

    d1 = ensure_dir(out / "01_table1_replication")
    d2 = ensure_dir(out / "02_table2_operators_rc1")
    d3 = ensure_dir(out / "03_table3_literature")
    d4 = ensure_dir(out / "04_table4_optimum")

    # Table 1 (three panels) + gap presentation table
    t1_files = [
        ("Table_1a_like_R.csv", "Table 1a (R) - PB96 vs OUR"),
        ("Table_1b_like_C.csv", "Table 1b (C) - PB96 vs OUR"),
        ("Table_1c_like_RC.csv", "Table 1c (RC) - PB96 vs OUR"),
    ]
    for file_name, title in t1_files:
        rows = _read_csv(exp1_tables / file_name)
        cols = ["Group", "Row", "K", "Distance", "Waiting Time", "Route Time", "Computation Time"]
        _render_table_png(
            d1 / f"{Path(file_name).stem}_presentation.png",
            title,
            cols,
            rows,
            row_highlight=lambda r: "best" if r["Row"] == "OUR-GA-BEST" else ("ref" if r["Row"] == "GENEROUS-50" else None),
        )

    gap_rows = _read_csv(d1.parent / "01_table1_replication" / "table1_group_gap_our50_vs_generous50.csv") if (d1 / "table1_group_gap_our50_vs_generous50.csv").exists() else []
    if not gap_rows:
        # rebuild from existing table files
        group_rows: List[Dict[str, str]] = []
        for file_name, _ in t1_files:
            group_rows.extend(_read_csv(exp1_tables / file_name))
        by_key = {(r["Group"], r["Row"]): r for r in group_rows}
        for grp in ["R1", "R2", "C1", "C2", "RC1", "RC2"]:
            o = by_key.get((grp, "OUR-GA-50"))
            p = by_key.get((grp, "GENEROUS-50"))
            if not o or not p:
                continue
            gap_rows.append(
                {
                    "Group": grp,
                    "Delta K": _to_float(o["K"]) - _to_float(p["K"]),
                    "Delta Distance": _to_float(o["Distance"]) - _to_float(p["Distance"]),
                    "Delta Route Time": _to_float(o["Route Time"]) - _to_float(p["Route Time"]),
                }
            )
    _render_table_png(
        d1 / "Table_1_gap_summary_presentation.png",
        "Table 1 - Gap Summary (OUR-GA-50 minus GENEROUS-50)",
        ["Group", "Delta K", "Delta Distance", "Delta Route Time"],
        gap_rows,
    )

    # Table 2
    _render_table_png(
        d2 / "Table_2_reference_presentation.png",
        "Table 2 - RC1 Operator Comparison (Paper Reference)",
        ["Algorithm", "Routes", "Route Time", "Comp. Time"],
        TABLE2_REFERENCE,
        row_highlight=lambda r: "ref" if r["Algorithm"] == "GENEROUS" else None,
    )

    # Table 3 with OUR
    exp1_group_rows: List[Dict[str, str]] = []
    for file_name, _ in t1_files:
        exp1_group_rows.extend(_read_csv(exp1_tables / file_name))
    our_by_group = {r["Group"]: _to_float(r["K"]) for r in exp1_group_rows if r["Row"] == "OUR-GA-BEST"}
    table3_rows: List[Dict[str, Any]] = []
    for row in TABLE3_REFERENCE:
        our = our_by_group.get(row["Set"])
        table3_rows.append({**row, "OUR_EXP1_BEST": our, "Delta OUR-GENEROUS": (our - row["GENEROUS"]) if our is not None else None})
    _render_table_png(
        d3 / "Table_3_with_our_presentation.png",
        "Table 3 - Literature vs OUR (Average Number of Routes)",
        ["Set", "I1", "PARIS", "GRASP", "CTA", "GIDEON", "TABU", "GENEROUS", "OUR_EXP1_BEST", "Delta OUR-GENEROUS"],
        table3_rows,
    )

    # Table 4 with OUR
    our_best = _load_exp1_best_by_instance(exp1_raw)
    table4_rows: List[Dict[str, Any]] = []
    for row in TABLE4_REFERENCE:
        inst = row["Problem"]
        pair = our_best.get(inst)
        if pair is None:
            our_str = ""
            d_vs_opt = None
        else:
            our_str = f"{int(pair[0])}, {pair[1]:.1f}"
            opt_d = float(str(row["OPTIMUM (k,d)"]).split(",")[1].strip())
            d_vs_opt = pair[1] - opt_d
        table4_rows.append({**row, "OUR_EXP1_BEST (k,d)": our_str, "Delta OUR-OPT (d)": d_vs_opt})
    _render_table_png(
        d4 / "Table_4_with_our_presentation.png",
        "Table 4 - Selected Problems (Paper vs OUR)",
        ["Problem", "I1 (k,d)", "TABU (k,d)", "GENEROUS (k,d)", "OPTIMUM (k,d)", "OUR_EXP1_BEST (k,d)", "Delta OUR-OPT (d)"],
        table4_rows,
    )

    readme = out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# PB96 Presentation Tables",
                "",
                f"- Generated at: `{now_timestamp()}`",
                "- All tables were rebuilt in presentation PNG format.",
                "",
                "## Files",
                "- `01_table1_replication/*_presentation.png`",
                "- `02_table2_operators_rc1/Table_2_reference_presentation.png`",
                "- `03_table3_literature/Table_3_with_our_presentation.png`",
                "- `04_table4_optimum/Table_4_with_our_presentation.png`",
            ]
        )
        + "\n"
    )

    print(f"Presentation tables generated at: {out}")


if __name__ == "__main__":
    main()
