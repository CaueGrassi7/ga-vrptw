from __future__ import annotations

from typing import List

from .distance import distance_between
from .evaluate import evaluate_solution
from .model import Instance, Route, Solution
from .split import split_dp


def _first_timewindow_violation(instance: Instance, route: Route) -> int | None:
    if not route.customers:
        return None

    current_time = 0.0
    prev = instance.depot_id

    for idx, cid in enumerate(route.customers):
        cust = instance.customers[cid]
        travel = distance_between(instance.distance_matrix, instance.id_to_index, prev, cid)
        arrival = current_time + travel
        if arrival > cust.due_date:
            return idx
        if arrival < cust.ready_time:
            current_time = cust.ready_time + cust.service_time
        else:
            current_time = arrival + cust.service_time
        prev = cid
    return None


def _repair_timewindows(instance: Instance, routes: List[Route]) -> List[Route]:
    repaired: List[Route] = []
    for route in routes:
        queue = [route]
        while queue:
            current = queue.pop(0)
            if len(current.customers) <= 1:
                repaired.append(current)
                continue
            idx = _first_timewindow_violation(instance, current)
            if idx is None or idx == 0:
                repaired.append(current)
                continue
            before = Route(customers=current.customers[:idx])
            after = Route(customers=current.customers[idx:])
            queue.append(before)
            queue.append(after)
    return repaired


def decode_chromosome(
    instance: Instance,
    chromosome: List[int],
    penalty_tw: float,
    repair_tw: bool = False,
    decoder: str = "sequential",
) -> Solution:
    """Decode a permutation into routes respecting capacity.

    Time windows are evaluated with time-warp penalties (soft constraint).
    """

    if decoder == "split":
        routes = split_dp(instance, chromosome)
    else:
        routes = []
        current_route: List[int] = []
        current_load = 0.0

        for cid in chromosome:
            demand = instance.customers[cid].demand
            if current_route and current_load + demand > instance.capacity:
                routes.append(Route(customers=current_route))
                current_route = []
                current_load = 0.0

            current_route.append(cid)
            current_load += demand

        if current_route:
            routes.append(Route(customers=current_route))

    if repair_tw:
        routes = _repair_timewindows(instance, routes)

    return evaluate_solution(instance, routes, penalty_tw)
