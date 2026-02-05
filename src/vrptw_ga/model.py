from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Customer:
    """Single customer node in a Solomon instance."""

    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass(frozen=True)
class Instance:
    """VRPTW instance with customers and distance matrix."""

    name: str
    capacity: float
    depot_id: int
    customers: Dict[int, Customer]
    distance_matrix: np.ndarray
    id_order: List[int]
    id_to_index: Dict[int, int]

    @property
    def customer_ids(self) -> List[int]:
        return sorted(cid for cid in self.customers.keys() if cid != self.depot_id)


@dataclass
class Route:
    """Route represented by an ordered list of customer ids (excluding depot)."""

    customers: List[int]
    distance: float = 0.0
    timewarp: float = 0.0
    load: float = 0.0


@dataclass
class Solution:
    """Solution as list of routes with cached metrics."""

    routes: List[Route] = field(default_factory=list)
    total_distance: float = 0.0
    total_timewarp: float = 0.0
    total_load: float = 0.0
    capacity_violation: float = 0.0
    objective: float = 0.0
    feasible_timewindows: bool = False
    feasible_capacity: bool = True

    def update_objective(self, penalty_tw: float) -> None:
        self.objective = self.total_distance + penalty_tw * self.total_timewarp
        self.feasible_timewindows = self.total_timewarp == 0.0
        self.feasible_capacity = self.capacity_violation == 0.0
