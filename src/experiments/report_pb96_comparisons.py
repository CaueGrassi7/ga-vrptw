from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

from vrptw_ga.utils import ensure_dir, now_timestamp


TABLE2_REFERENCE = [
    {"algorithm": "I1", "routes": 13.5, "route_time": 2775.0, "comp_time_minsec": "---"},
    {"algorithm": "GENEROUS", "routes": 12.1, "route_time": 2509.9, "comp_time_minsec": "10:58"},
    {"algorithm": "GEN-SBX-1M", "routes": 12.9, "route_time": 2731.5, "comp_time_minsec": "1:59"},
    {"algorithm": "GEN-SBX-2M", "routes": 12.9, "route_time": 2729.1, "comp_time_minsec": "2:45"},
    {"algorithm": "GEN-SBX-LSM", "routes": 12.6, "route_time": 2521.7, "comp_time_minsec": "22:32"},
    {"algorithm": "GEN-RBX-1M", "routes": 12.9, "route_time": 2722.2, "comp_time_minsec": "3:38"},
    {"algorithm": "GEN-RBX-2M", "routes": 12.8, "route_time": 2732.2, "comp_time_minsec": "5:11"},
    {"algorithm": "GEN-RBX-LSM", "routes": 12.5, "route_time": 2515.2, "comp_time_minsec": "28:35"},
]

TABLE3_REFERENCE = [
    {"set": "R1", "I1": 13.6, "PARIS": 13.3, "GRASP": 13.1, "CTA": 13.0, "GIDEON": 12.8, "TABU": 12.5, "GENEROUS": 12.6},
    {"set": "R2", "I1": 3.3, "PARIS": 3.1, "GRASP": 3.1, "CTA": 3.1, "GIDEON": 3.2, "TABU": 3.1, "GENEROUS": 3.0},
    {"set": "C1", "I1": 10.0, "PARIS": 10.7, "GRASP": 10.6, "CTA": 10.0, "GIDEON": 10.0, "TABU": 10.0, "GENEROUS": 10.0},
    {"set": "C2", "I1": 3.1, "PARIS": 3.4, "GRASP": 3.4, "CTA": 3.0, "GIDEON": 3.0, "TABU": 3.0, "GENEROUS": 3.0},
    {"set": "RC1", "I1": 13.5, "PARIS": 13.4, "GRASP": 12.8, "CTA": 13.0, "GIDEON": 12.5, "TABU": 12.6, "GENEROUS": 12.1},
    {"set": "RC2", "I1": 3.9, "PARIS": 3.6, "GRASP": 3.6, "CTA": 3.7, "GIDEON": 3.4, "TABU": 3.4, "GENEROUS": 3.4},
]

TABLE4_REFERENCE = [
    {"problem": "R101", "I1_k": 21, "I1_distance": 1867.1, "TABU_k": 19, "TABU_distance": 1650.7, "GENEROUS_k": 19, "GENEROUS_distance": 1669.4, "OPTIMUM_k": 18, "OPTIMUM_distance": 1607.7},
    {"problem": "R102", "I1_k": 19, "I1_distance": 1699.5, "TABU_k": 18, "TABU_distance": 1471.8, "GENEROUS_k": 17, "GENEROUS_distance": 1532.1, "OPTIMUM_k": 17, "OPTIMUM_distance": 1434.0},
    {"problem": "C101", "I1_k": 10, "I1_distance": 851.4, "TABU_k": 10, "TABU_distance": 827.3, "GENEROUS_k": 10, "GENEROUS_distance": 827.3, "OPTIMUM_k": 10, "OPTIMUM_distance": 827.3},
    {"problem": "C102", "I1_k": 10, "I1_distance": 966.7, "TABU_k": 10, "TABU_distance": 827.3, "GENEROUS_k": 10, "GENEROUS_distance": 827.3, "OPTIMUM_k": 10, "OPTIMUM_distance": 827.3},
    {"problem": "C106", "I1_k": 10, "I1_distance": 916.0, "TABU_k": 10, "TABU_distance": 827.3, "GENEROUS_k": 10, "GENEROUS_distance": 827.3, "OPTIMUM_k": 10, "OPTIMUM_distance": 827.3},
    {"problem": "C107", "I1_k": 10, "I1_distance": 902.4, "TABU_k": 10, "TABU_distance": 827.3, "GENEROUS_k": 10, "GENEROUS_distance": 827.3, "OPTIMUM_k": 10, "OPTIMUM_distance": 827.3},
    {"problem": "C108", "I1_k": 10, "I1_distance": 853.1, "TABU_k": 10, "TABU_distance": 827.3, "GENEROUS_k": 10, "GENEROUS_distance": 827.3, "OPTIMUM_k": 10, "OPTIMUM_distance": 827.3},
]

GROUP_ORDER = ["R1", "R2", "C1", "C2", "RC1", "RC2"]


@dataclass
class RunBest:
    run_id: str
    timestamp: str
    instance: str
    seed: int
    config_sig: Tuple[Tuple[str, str], ...]
    best_k: float
    best_distance: float


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate PB96-style comparison package in canva/")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--out_dir", type=str, default="canva")
    return p


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}"
    return str(value)


def _write_table_png(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]], title: str) -> None:
    ensure_dir(path.parent)
    col_labels = list(columns)
    cell_text = [[_format_cell(row.get(col)) for col in col_labels] for row in rows]

    fig_w = max(8.0, min(22.0, 1.6 * len(col_labels)))
    fig_h = max(2.8, min(24.0, 0.38 * (len(cell_text) + 2) + 1.8))
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.12)
    ax.set_title(title, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _load_exp1_best_runs(exp1_raw_dir: Path) -> List[RunBest]:
    dedup: Dict[Tuple[str, int, Tuple[Tuple[str, str], ...]], RunBest] = {}
    for json_path in sorted(exp1_raw_dir.glob("run_*.json")):
        data = json.loads(json_path.read_text())
        config = data.get("config", {})
        best = data.get("best_metrics", {})
        best_k = _to_float(best.get("best_k"))
        best_distance = _to_float(best.get("best_distance"))
        if best_k is None or best_distance is None:
            continue
        cfg_sig = tuple(sorted((str(k), str(v)) for k, v in config.items()))
        row = RunBest(
            run_id=str(data.get("run_id", "")),
            timestamp=str(data.get("timestamp", "")),
            instance=str(data.get("instance_name", "")),
            seed=int(data.get("seed", 0)),
            config_sig=cfg_sig,
            best_k=best_k,
            best_distance=best_distance,
        )
        key = (row.instance, row.seed, row.config_sig)
        old = dedup.get(key)
        if old is None or (row.timestamp, row.run_id) > (old.timestamp, old.run_id):
            dedup[key] = row
    return list(dedup.values())


def _exp1_group_rows_from_tables(exp1_tables_dir: Path) -> List[Dict[str, Any]]:
    files = [
        exp1_tables_dir / "Table_1a_like_R.csv",
        exp1_tables_dir / "Table_1b_like_C.csv",
        exp1_tables_dir / "Table_1c_like_RC.csv",
    ]
    rows: List[Dict[str, Any]] = []
    for file in files:
        rows.extend(_read_csv(file))
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "group": row["Group"],
                "row": row["Row"],
                "K": _to_float(row["K"]),
                "distance": _to_float(row["Distance"]),
                "waiting": _to_float(row["Waiting Time"]),
                "route_time": _to_float(row["Route Time"]),
                "computation_time": row["Computation Time"],
            }
        )
    return out


def _table1_block(exp1_tables_dir: Path, out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    src_files = [
        "Table_1a_like_R.csv",
        "Table_1b_like_C.csv",
        "Table_1c_like_RC.csv",
        "Table_1a_like_R.md",
        "Table_1b_like_C.md",
        "Table_1c_like_RC.md",
    ]
    for name in src_files:
        src = exp1_tables_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            if name.endswith(".csv"):
                csv_rows = _read_csv(src)
                cols = list(csv_rows[0].keys()) if csv_rows else []
                if cols:
                    _write_table_png(
                        out_dir / f"{Path(name).stem}.png",
                        cols,
                        csv_rows,
                        title=Path(name).stem.replace("_", " "),
                    )

    rows = _exp1_group_rows_from_tables(exp1_tables_dir)
    by_key = {(row["group"], row["row"]): row for row in rows}

    gap_50_rows: List[Dict[str, Any]] = []
    gap_best_rows: List[Dict[str, Any]] = []
    for group in GROUP_ORDER:
        p50 = by_key.get((group, "GENEROUS-50"))
        o50 = by_key.get((group, "OUR-GA-50"))
        ob = by_key.get((group, "OUR-GA-BEST"))
        if p50 and o50:
            gap_50_rows.append(
                {
                    "group": group,
                    "our_row": "OUR-GA-50",
                    "paper_row": "GENEROUS-50",
                    "delta_K": o50["K"] - p50["K"],
                    "delta_distance": o50["distance"] - p50["distance"],
                    "delta_route_time": o50["route_time"] - p50["route_time"],
                }
            )
        if p50 and ob:
            gap_best_rows.append(
                {
                    "group": group,
                    "our_row": "OUR-GA-BEST",
                    "paper_row": "GENEROUS-50",
                    "delta_K": ob["K"] - p50["K"],
                    "delta_distance": ob["distance"] - p50["distance"],
                    "delta_route_time": ob["route_time"] - p50["route_time"],
                }
            )

    _write_csv(
        out_dir / "table1_group_gap_our50_vs_generous50.csv",
        ["group", "our_row", "paper_row", "delta_K", "delta_distance", "delta_route_time"],
        gap_50_rows,
    )
    _write_table_png(
        out_dir / "table1_group_gap_our50_vs_generous50.png",
        ["group", "our_row", "paper_row", "delta_K", "delta_distance", "delta_route_time"],
        gap_50_rows,
        "Table1 gap OUR-GA-50 vs GENEROUS-50",
    )
    _write_csv(
        out_dir / "table1_group_gap_ourbest_vs_generous50.csv",
        ["group", "our_row", "paper_row", "delta_K", "delta_distance", "delta_route_time"],
        gap_best_rows,
    )
    _write_table_png(
        out_dir / "table1_group_gap_ourbest_vs_generous50.png",
        ["group", "our_row", "paper_row", "delta_K", "delta_distance", "delta_route_time"],
        gap_best_rows,
        "Table1 gap OUR-GA-BEST vs GENEROUS-50",
    )

    plt.figure(figsize=(8, 4))
    x = list(range(len(gap_50_rows)))
    labels = [row["group"] for row in gap_50_rows]
    y1 = [row["delta_route_time"] for row in gap_50_rows]
    y2 = [row["delta_route_time"] for row in gap_best_rows]
    width = 0.35
    plt.bar([i - width / 2 for i in x], y1, width=width, label="OUR-GA-50 - GENEROUS-50")
    plt.bar([i + width / 2 for i in x], y2, width=width, label="OUR-GA-BEST - GENEROUS-50")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel("Delta Route Time")
    plt.title("Table 1: Gap por Grupo (negativo = melhor que PB96)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_table1_gap_route_time.png", dpi=150)
    plt.close()

    return {"gap_50_rows": gap_50_rows, "gap_best_rows": gap_best_rows}


def _table2_block(out_dir: Path) -> None:
    ensure_dir(out_dir)
    _write_csv(
        out_dir / "table2_reference_rc1.csv",
        ["algorithm", "routes", "route_time", "comp_time_minsec"],
        TABLE2_REFERENCE,
    )
    _write_table_png(
        out_dir / "table2_reference_rc1.png",
        ["algorithm", "routes", "route_time", "comp_time_minsec"],
        TABLE2_REFERENCE,
        "Table 2 reference (RC1)",
    )

    plt.figure(figsize=(8, 4))
    labels = [row["algorithm"] for row in TABLE2_REFERENCE]
    values = [row["route_time"] for row in TABLE2_REFERENCE]
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Route Time")
    plt.title("Table 2 (Paper): RC1 - Implementacoes Geneticas")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_table2_reference_route_time.png", dpi=150)
    plt.close()

    _write_md(
        out_dir / "notes.md",
        "\n".join(
            [
                "# Tabela 2 (paper) vs experimento atual",
                "",
                "- A Tabela 2 do paper compara operadores `SBX/RBX` x `1M/2M/LSM` em `RC1`.",
                "- O experimento atual do projeto (`exp2`) varia taxas (`crossover_rate`, `mutation_rate`) e nao reproduz essa ablation diretamente.",
                "- Este bloco traz os numeros de referencia do paper para guiar uma reproducao futura fiel.",
            ]
        )
        + "\n",
    )


def _table3_block(exp1_group_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    ensure_dir(out_dir)
    _write_csv(
        out_dir / "table3_reference.csv",
        ["set", "I1", "PARIS", "GRASP", "CTA", "GIDEON", "TABU", "GENEROUS"],
        TABLE3_REFERENCE,
    )
    _write_table_png(
        out_dir / "table3_reference.png",
        ["set", "I1", "PARIS", "GRASP", "CTA", "GIDEON", "TABU", "GENEROUS"],
        TABLE3_REFERENCE,
        "Table 3 reference",
    )

    our_best_map: Dict[str, float] = {}
    for row in exp1_group_rows:
        if row["row"] == "OUR-GA-BEST":
            our_best_map[row["group"]] = float(row["K"])

    with_our_rows: List[Dict[str, Any]] = []
    for row in TABLE3_REFERENCE:
        group = row["set"]
        our = our_best_map.get(group)
        generous = float(row["GENEROUS"])
        with_our_rows.append(
            {
                **row,
                "OUR_EXP1_BEST": our,
                "delta_our_minus_generous": (our - generous) if our is not None else None,
            }
        )

    _write_csv(
        out_dir / "table3_with_our.csv",
        ["set", "I1", "PARIS", "GRASP", "CTA", "GIDEON", "TABU", "GENEROUS", "OUR_EXP1_BEST", "delta_our_minus_generous"],
        with_our_rows,
    )
    _write_table_png(
        out_dir / "table3_with_our.png",
        ["set", "I1", "PARIS", "GRASP", "CTA", "GIDEON", "TABU", "GENEROUS", "OUR_EXP1_BEST", "delta_our_minus_generous"],
        with_our_rows,
        "Table 3 with OUR_EXP1_BEST",
    )

    methods = ["I1", "PARIS", "GRASP", "CTA", "GIDEON", "TABU", "GENEROUS", "OUR_EXP1_BEST"]
    groups = [row["set"] for row in with_our_rows]
    x = list(range(len(groups)))
    width = 0.1
    plt.figure(figsize=(12, 4.5))
    for idx, method in enumerate(methods):
        vals = [row[method] for row in with_our_rows]
        offset = (idx - (len(methods) - 1) / 2) * width
        plt.bar([i + offset for i in x], vals, width=width, label=method)
    plt.xticks(x, groups)
    plt.ylabel("Average Number of Routes")
    plt.title("Table 3: Comparacao de Heuristicas (com OUR_EXP1_BEST)")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_table3_routes_comparison.png", dpi=150)
    plt.close()


def _table4_block(exp1_runs: List[RunBest], out_dir: Path) -> None:
    ensure_dir(out_dir)
    _write_csv(
        out_dir / "table4_reference.csv",
        [
            "problem",
            "I1_k",
            "I1_distance",
            "TABU_k",
            "TABU_distance",
            "GENEROUS_k",
            "GENEROUS_distance",
            "OPTIMUM_k",
            "OPTIMUM_distance",
        ],
        TABLE4_REFERENCE,
    )
    _write_table_png(
        out_dir / "table4_reference.png",
        [
            "problem",
            "I1_k",
            "I1_distance",
            "TABU_k",
            "TABU_distance",
            "GENEROUS_k",
            "GENEROUS_distance",
            "OPTIMUM_k",
            "OPTIMUM_distance",
        ],
        TABLE4_REFERENCE,
        "Table 4 reference",
    )

    targets = {row["problem"] for row in TABLE4_REFERENCE}
    by_problem: Dict[str, List[RunBest]] = {}
    for run in exp1_runs:
        if run.instance in targets:
            by_problem.setdefault(run.instance, []).append(run)

    our_best: Dict[str, Tuple[float, float]] = {}
    for problem, runs in by_problem.items():
        best = min(runs, key=lambda r: (r.best_k, r.best_distance, r.seed, r.run_id))
        # Paper Table 4 truncates distance to first decimal.
        trunc_dist = int(best.best_distance * 10) / 10.0
        our_best[problem] = (best.best_k, trunc_dist)

    with_our_rows: List[Dict[str, Any]] = []
    for ref in TABLE4_REFERENCE:
        problem = ref["problem"]
        our_k, our_d = our_best.get(problem, (None, None))
        with_our_rows.append(
            {
                **ref,
                "OUR_EXP1_BEST_k": our_k,
                "OUR_EXP1_BEST_distance": our_d,
                "delta_our_vs_generous_k": (our_k - ref["GENEROUS_k"]) if our_k is not None else None,
                "delta_our_vs_generous_distance": (our_d - ref["GENEROUS_distance"]) if our_d is not None else None,
                "delta_our_vs_optimum_k": (our_k - ref["OPTIMUM_k"]) if our_k is not None else None,
                "delta_our_vs_optimum_distance": (our_d - ref["OPTIMUM_distance"]) if our_d is not None else None,
            }
        )

    _write_csv(
        out_dir / "table4_with_our_best.csv",
        [
            "problem",
            "I1_k",
            "I1_distance",
            "TABU_k",
            "TABU_distance",
            "GENEROUS_k",
            "GENEROUS_distance",
            "OPTIMUM_k",
            "OPTIMUM_distance",
            "OUR_EXP1_BEST_k",
            "OUR_EXP1_BEST_distance",
            "delta_our_vs_generous_k",
            "delta_our_vs_generous_distance",
            "delta_our_vs_optimum_k",
            "delta_our_vs_optimum_distance",
        ],
        with_our_rows,
    )
    _write_table_png(
        out_dir / "table4_with_our_best.png",
        [
            "problem",
            "I1_k",
            "I1_distance",
            "TABU_k",
            "TABU_distance",
            "GENEROUS_k",
            "GENEROUS_distance",
            "OPTIMUM_k",
            "OPTIMUM_distance",
            "OUR_EXP1_BEST_k",
            "OUR_EXP1_BEST_distance",
            "delta_our_vs_generous_k",
            "delta_our_vs_generous_distance",
            "delta_our_vs_optimum_k",
            "delta_our_vs_optimum_distance",
        ],
        with_our_rows,
        "Table 4 with OUR_EXP1_BEST",
    )

    problems = [row["problem"] for row in with_our_rows]
    x = list(range(len(problems)))
    width = 0.18
    y_tabu = [row["TABU_distance"] for row in with_our_rows]
    y_gen = [row["GENEROUS_distance"] for row in with_our_rows]
    y_opt = [row["OPTIMUM_distance"] for row in with_our_rows]
    y_our = [row["OUR_EXP1_BEST_distance"] if row["OUR_EXP1_BEST_distance"] is not None else 0.0 for row in with_our_rows]

    plt.figure(figsize=(10, 4.2))
    plt.bar([i - 1.5 * width for i in x], y_tabu, width=width, label="TABU")
    plt.bar([i - 0.5 * width for i in x], y_gen, width=width, label="GENEROUS")
    plt.bar([i + 0.5 * width for i in x], y_our, width=width, label="OUR_EXP1_BEST")
    plt.bar([i + 1.5 * width for i in x], y_opt, width=width, label="OPTIMUM")
    plt.xticks(x, problems)
    plt.ylabel("Distance (1-dec trunc)")
    plt.title("Table 4: Distancia por Problema (Paper vs OUR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_table4_distance_comparison.png", dpi=150)
    plt.close()


def _write_root_readme(out_dir: Path) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# PB96 Comparison Package",
                "",
                f"- Generated at: `{now_timestamp()}`",
                "- Source paper: Potvin & Bengio (1996), Table 1/2/3/4 comparison blocks.",
                "",
                "## Structure",
                "- `01_table1_replication/`: Tabela 1a/1b/1c + gaps OUR vs GENEROUS.",
                "- `02_table2_operators_rc1/`: referencia da Tabela 2 (SBX/RBX x 1M/2M/LSM).",
                "- `03_table3_literature/`: referencia da Tabela 3 + `OUR_EXP1_BEST` por grupo.",
                "- `04_table4_optimum/`: referencia da Tabela 4 + comparativo com `OUR_EXP1_BEST` por instancia.",
                "",
                "## Notes",
                "- Tabelas 2, 3 e 4 de referencia foram transcritas do artigo.",
                "- Para Tabela 4, `OUR_EXP1_BEST_distance` foi truncada para 1 casa decimal para manter comparabilidade com o paper.",
                "- Todas as tabelas possuem versao `.csv` e `.png`.",
                "",
                "## Paths",
                f"- `{_safe_rel(out_dir / '01_table1_replication')}`",
                f"- `{_safe_rel(out_dir / '02_table2_operators_rc1')}`",
                f"- `{_safe_rel(out_dir / '03_table3_literature')}`",
                f"- `{_safe_rel(out_dir / '04_table4_optimum')}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    args = _build_parser().parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)

    exp1_tables_dir = results_root / "exp1_replication" / "full" / "tables"
    exp1_raw_dir = results_root / "exp1_replication" / "full" / "raw"

    if not exp1_tables_dir.exists():
        raise SystemExit(f"Missing directory: {exp1_tables_dir}")
    if not exp1_raw_dir.exists():
        raise SystemExit(f"Missing directory: {exp1_raw_dir}")

    t1_dir = ensure_dir(out_dir / "01_table1_replication")
    t2_dir = ensure_dir(out_dir / "02_table2_operators_rc1")
    t3_dir = ensure_dir(out_dir / "03_table3_literature")
    t4_dir = ensure_dir(out_dir / "04_table4_optimum")

    _table1_block(exp1_tables_dir=exp1_tables_dir, out_dir=t1_dir)
    exp1_group_rows = _exp1_group_rows_from_tables(exp1_tables_dir)
    exp1_runs = _load_exp1_best_runs(exp1_raw_dir)

    _table2_block(out_dir=t2_dir)
    _table3_block(exp1_group_rows=exp1_group_rows, out_dir=t3_dir)
    _table4_block(exp1_runs=exp1_runs, out_dir=t4_dir)
    _write_root_readme(out_dir=out_dir)

    print(f"PB96 comparison package generated at: {out_dir}")


if __name__ == "__main__":
    main()
