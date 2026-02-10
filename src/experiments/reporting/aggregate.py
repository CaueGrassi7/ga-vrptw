from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .io import RunRecord


@dataclass
class MarkResult:
    gen: float
    metrics: Dict[str, Any]


@dataclass
class AggregateResult:
    group_rows: List[Dict[str, Any]]
    warnings: List[str]
    series_by_group: Dict[str, Dict[float, Dict[str, float]]]
    final_values: Dict[str, Dict[str, List[float]]]


GROUPS = ["R1", "R2", "C1", "C2", "RC1", "RC2", "UNKNOWN"]


def solomon_group(instance_name: str) -> str:
    name = instance_name.strip().upper()
    if name.startswith("R") and not name.startswith("RC"):
        try:
            num = int(name[1:])
        except ValueError:
            return "UNKNOWN"
        if 101 <= num <= 112:
            return "R1"
        if 201 <= num <= 211:
            return "R2"
        return "UNKNOWN"
    if name.startswith("C") and not name.startswith("RC"):
        try:
            num = int(name[1:])
        except ValueError:
            return "UNKNOWN"
        if 101 <= num <= 109:
            return "C1"
        if 201 <= num <= 208:
            return "C2"
        return "UNKNOWN"
    if name.startswith("RC"):
        try:
            num = int(name[2:])
        except ValueError:
            return "UNKNOWN"
        if 101 <= num <= 108:
            return "RC1"
        if 201 <= num <= 208:
            return "RC2"
        return "UNKNOWN"
    return "UNKNOWN"


def group_key(run: RunRecord, mode: str) -> str:
    if mode == "instance_name":
        return run.instance_name or "UNKNOWN"
    return solomon_group(run.instance_name)


def _metric_from_entry(entry: Dict[str, Any], warnings: List[str], context: str) -> Dict[str, Any]:
    distance = entry.get("best_distance")
    waiting = entry.get("best_total_waiting")
    service = entry.get("best_total_service")
    route_time = entry.get("best_total_route_time")
    if route_time is None and distance is not None and waiting is not None and service is not None:
        route_time = distance + waiting + service
    if route_time is not None and distance is not None and waiting is not None and service is not None:
        expected = distance + waiting + service
        if abs(route_time - expected) > 1e-6:
            warnings.append(
                f"Route time mismatch in {context}: route_time={route_time} expected={expected}"
            )
    return {
        "K": entry.get("best_k"),
        "distance": distance,
        "waiting": waiting,
        "service": service,
        "route_time": route_time,
        "timewarp": entry.get("best_timewarp"),
    }


def _select_mark(history: List[Dict[str, Any]], target: float, warnings: List[str], run_id: str) -> MarkResult | None:
    if not history:
        return None
    sorted_hist = sorted(history, key=lambda r: float(r.get("gen", 0.0)))
    if target == 0:
        entry = sorted_hist[0]
        return MarkResult(gen=float(entry.get("gen", 0.0)), metrics=entry)

    candidates = [h for h in sorted_hist if float(h.get("gen", 0.0)) <= target]
    if candidates:
        entry = max(candidates, key=lambda r: float(r.get("gen", 0.0)))
        if float(entry.get("gen", 0.0)) != target:
            warnings.append(
                f"Run {run_id}: using gen {entry.get('gen')} for target {target}"
            )
        return MarkResult(gen=float(entry.get("gen", 0.0)), metrics=entry)

    entry = min(sorted_hist, key=lambda r: abs(float(r.get("gen", 0.0)) - target))
    warnings.append(
        f"Run {run_id}: no gen <= {target}; using gen {entry.get('gen')}"
    )
    return MarkResult(gen=float(entry.get("gen", 0.0)), metrics=entry)


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: List[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    if mean is None:
        return None
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def build_aggregate(
    runs: List[RunRecord],
    group_by: str,
    gen_marks: List[int],
    only_best: bool,
) -> AggregateResult:
    warnings: List[str] = []
    group_rows: List[Dict[str, Any]] = []
    series_by_group: Dict[str, Dict[float, Dict[str, float]]] = defaultdict(dict)
    final_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    grouped: Dict[str, List[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[group_key(run, group_by)].append(run)

    for group, group_runs in grouped.items():
        if not only_best:
            for mark in gen_marks:
                values: Dict[str, List[float]] = defaultdict(list)
                for run in group_runs:
                    mark_res = _select_mark(run.history, mark, warnings, run.run_id)
                    if mark_res is None:
                        continue
                    metric = _metric_from_entry(mark_res.metrics, warnings, f"run_id={run.run_id} gen={mark_res.gen}")
                    for key, value in metric.items():
                        if value is not None:
                            values[key].append(float(value))
                row = {
                    "group": group,
                    "row": f"OUR-GA-{mark:02d}",
                    "K": _mean(values["K"]),
                    "distance": _mean(values["distance"]),
                    "waiting": _mean(values["waiting"]),
                    "route_time": _mean(values["route_time"]),
                    "comp_time_sec": None,
                    "n": len(values["K"]),
                }
                group_rows.append(row)

        values_best: Dict[str, List[float]] = defaultdict(list)
        elapsed: List[float] = []
        for run in group_runs:
            metric = _metric_from_entry(run.best_metrics, warnings, f"run_id={run.run_id} best")
            for key, value in metric.items():
                if value is not None:
                    values_best[key].append(float(value))
            elapsed.append(float(run.elapsed_seconds))

            if metric.get("K") is not None:
                final_values[group]["K"].append(float(metric["K"]))
            if metric.get("route_time") is not None:
                final_values[group]["route_time"].append(float(metric["route_time"]))

        group_rows.append(
            {
                "group": group,
                "row": "OUR-GA-BEST",
                "K": _mean(values_best["K"]),
                "distance": _mean(values_best["distance"]),
                "waiting": _mean(values_best["waiting"]),
                "route_time": _mean(values_best["route_time"]),
                "comp_time_sec": _mean(elapsed),
                "n": len(values_best["K"]),
            }
        )

        for run in group_runs:
            if not run.history:
                continue
            for entry in run.history:
                gen = float(entry.get("gen", 0.0))
                metric = _metric_from_entry(entry, warnings, f"run_id={run.run_id} gen={gen}")
                if metric.get("K") is not None:
                    series_by_group[group].setdefault(gen, {}).setdefault("K", []).append(float(metric["K"]))
                if metric.get("route_time") is not None:
                    series_by_group[group].setdefault(gen, {}).setdefault("route_time", []).append(
                        float(metric["route_time"])
                    )

    series_stats: Dict[str, Dict[float, Dict[str, float]]] = defaultdict(dict)
    for group, gen_map in series_by_group.items():
        for gen, metrics in gen_map.items():
            series_stats[group][gen] = {
                "K_mean": _mean(metrics.get("K", [])),
                "K_std": _std(metrics.get("K", [])),
                "route_time_mean": _mean(metrics.get("route_time", [])),
                "route_time_std": _std(metrics.get("route_time", [])),
                "n": len(metrics.get("K", [])),
            }

    return AggregateResult(
        group_rows=group_rows,
        warnings=warnings,
        series_by_group=series_stats,
        final_values=final_values,
    )
