from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ExperimentGrids:
    exp2_crossover_rates: List[float]
    exp2_mutation_rates: List[float]
    exp3_pop_sizes: List[int]
    exp3_time_limits: List[float]


SMALL_INSTANCES: Dict[str, str] = {
    "C1": "C101",
    "C2": "C201",
    "R1": "R101",
    "R2": "R201",
    "RC1": "RC101",
    "RC2": "RC201",
}

FULL_INSTANCES: Dict[str, List[str]] = {
    "C1": [f"C10{i}" for i in range(1, 10)],
    "C2": [f"C20{i}" for i in range(1, 9)],
    "R1": [f"R10{i}" for i in range(1, 10)] + ["R110", "R111", "R112"],
    "R2": [f"R20{i}" for i in range(1, 10)] + ["R210", "R211"],
    "RC1": [f"RC10{i}" for i in range(1, 9)],
    "RC2": [f"RC20{i}" for i in range(1, 9)],
}

DEFAULT_GRIDS = ExperimentGrids(
    exp2_crossover_rates=[0.4, 0.6, 0.8],
    exp2_mutation_rates=[0.2, 0.4, 0.6],
    exp3_pop_sizes=[50, 100, 150, 300],
    exp3_time_limits=[30.0, 60.0],
)
