from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.3f}"


def _format_comp_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes = int(seconds // 60)
    secs = int(round(seconds - minutes * 60))
    return f"{minutes}:{secs:02d}"


def _rows_for_groups(rows: List[Dict[str, Any]], groups: List[str]) -> List[Dict[str, Any]]:
    group_set = set(groups)
    return [row for row in rows if row.get("group") in group_set]


def _table_rows(rows: List[Dict[str, Any]]) -> List[List[str]]:
    table = []
    for row in rows:
        table.append(
            [
                row.get("group", ""),
                row.get("row", ""),
                _format_number(row.get("K")),
                _format_number(row.get("distance")),
                _format_number(row.get("waiting")),
                _format_number(row.get("route_time")),
                row.get("comp_time", ""),
            ]
        )
    return table


def write_pb96_like_tables(
    our_rows: List[Dict[str, Any]],
    paper_rows: List[Dict[str, Any]],
    out_dir: Path,
    formats: List[str],
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_our: List[Dict[str, Any]] = []
    for row in our_rows:
        enriched = dict(row)
        comp_time = row.get("comp_time_sec")
        enriched["comp_time"] = _format_comp_time(comp_time)
        enriched_our.append(enriched)

    combined = paper_rows + enriched_our

    tables = {
        "R": ["R1", "R2"],
        "C": ["C1", "C2"],
        "RC": ["RC1", "RC2"],
        "UNKNOWN": ["UNKNOWN"],
    }
    row_order = {
        "I1": 0,
        "GENEROUS-00": 1,
        "GENEROUS-20": 2,
        "GENEROUS-50": 3,
        "OUR-GA-00": 4,
        "OUR-GA-20": 5,
        "OUR-GA-50": 6,
        "OUR-GA-BEST": 7,
    }

    written: List[str] = []
    for key, groups in tables.items():
        rows = _rows_for_groups(combined, groups)
        if not rows:
            continue
        rows = sorted(
            rows,
            key=lambda r: (r.get("group", ""), row_order.get(r.get("row", ""), 99), r.get("row", "")),
        )
        header = ["Group", "Row", "K", "Distance", "Waiting Time", "Route Time", "Computation Time"]
        table = _table_rows(rows)

        base = f"Table_1{('a' if key == 'R' else 'b' if key == 'C' else 'c')}_like_{key}"
        if "csv" in formats:
            csv_path = out_dir / f"{base}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(table)
            written.append(str(csv_path))
        if "md" in formats:
            md_path = out_dir / f"{base}.md"
            with md_path.open("w") as f:
                f.write("| " + " | ".join(header) + " |\n")
                f.write("|" + "|".join(["---"] * len(header)) + "|\n")
                for row in table:
                    f.write("| " + " | ".join(row) + " |\n")
            written.append(str(md_path))

    return written
