from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    instance_name: str
    instance_path: str
    seed: int
    config: Dict[str, Any]
    elapsed_seconds: float
    generations_run: int
    best_metrics: Dict[str, Any]
    best_perm: List[int]
    best_routes: List[List[int]]
    history: List[Dict[str, Any]]
    source_json: Path
    source_progress: Path | None


@dataclass
class LoadResult:
    runs: List[RunRecord]
    warnings: List[str]
    used_files: List[str]


def _to_number(value: str | None) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return float(text)
    except ValueError:
        return text


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_progress_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            parsed = {k: _to_number(v) for k, v in row.items()}
            rows.append(parsed)
        return rows


def load_runs(results_dir: Path) -> LoadResult:
    warnings: List[str] = []
    used_files: List[str] = []
    runs: List[RunRecord] = []

    for json_path in sorted(results_dir.glob("run_*.json")):
        data = _load_json(json_path)
        run_id = data.get("run_id")
        if not run_id:
            warnings.append(f"Missing run_id in {json_path}")
            continue
        progress_path = results_dir / f"progress_{run_id}.csv"
        history: List[Dict[str, Any]] = []
        if progress_path.exists():
            history = _load_progress_csv(progress_path)
            used_files.append(str(progress_path))
        else:
            warnings.append(f"Missing progress CSV for run_id={run_id}")

        used_files.append(str(json_path))
        runs.append(
            RunRecord(
                run_id=run_id,
                timestamp=str(data.get("timestamp", "")),
                instance_name=str(data.get("instance_name", "")),
                instance_path=str(data.get("instance_path", "")),
                seed=int(data.get("seed", 0)),
                config=dict(data.get("config", {})),
                elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
                generations_run=int(data.get("generations_run", 0)),
                best_metrics=dict(data.get("best_metrics", {})),
                best_perm=list(data.get("best_perm", [])),
                best_routes=list(data.get("best_routes", [])),
                history=history,
                source_json=json_path,
                source_progress=progress_path if progress_path.exists() else None,
            )
        )

    if not runs:
        warnings.append(f"No run_*.json files found in {results_dir}")

    return LoadResult(runs=runs, warnings=warnings, used_files=used_files)


def load_runs_recursive(results_dir: Path) -> LoadResult:
    warnings: List[str] = []
    used_files: List[str] = []
    runs: List[RunRecord] = []

    for json_path in sorted(results_dir.glob("**/run_*.json")):
        data = _load_json(json_path)
        run_id = data.get("run_id")
        if not run_id:
            warnings.append(f"Missing run_id in {json_path}")
            continue
        progress_path = json_path.parent / f"progress_{run_id}.csv"
        history: List[Dict[str, Any]] = []
        if progress_path.exists():
            history = _load_progress_csv(progress_path)
            used_files.append(str(progress_path))
        else:
            warnings.append(f"Missing progress CSV for run_id={run_id}")

        used_files.append(str(json_path))
        runs.append(
            RunRecord(
                run_id=run_id,
                timestamp=str(data.get("timestamp", "")),
                instance_name=str(data.get("instance_name", "")),
                instance_path=str(data.get("instance_path", "")),
                seed=int(data.get("seed", 0)),
                config=dict(data.get("config", {})),
                elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
                generations_run=int(data.get("generations_run", 0)),
                best_metrics=dict(data.get("best_metrics", {})),
                best_perm=list(data.get("best_perm", [])),
                best_routes=list(data.get("best_routes", [])),
                history=history,
                source_json=json_path,
                source_progress=progress_path if progress_path.exists() else None,
            )
        )

    if not runs:
        warnings.append(f"No run_*.json files found in {results_dir} (recursive)")

    return LoadResult(runs=runs, warnings=warnings, used_files=used_files)


def load_paper_reference(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {
                "group": row.get("group", "").strip(),
                "row": row.get("row", "").strip(),
                "K": _to_number(row.get("K")),
                "distance": _to_number(row.get("distance")),
                "waiting": _to_number(row.get("waiting")),
                "route_time": _to_number(row.get("route_time")),
                "comp_time": row.get("comp_time", "").strip(),
            }
            rows.append(parsed)
    return rows
