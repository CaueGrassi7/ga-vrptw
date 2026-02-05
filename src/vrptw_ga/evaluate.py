from __future__ import annotations

from typing import Iterable, Tuple

from .distance import distance_between
from .model import Instance, Route, Solution


def evaluate_route(instance: Instance, route: Route) -> Tuple[float, float, float, float, float]:
    """Compute distance, timewarp, and load for a single route.

    Time window handling:
    - travel time equals Euclidean distance
    - if arrival is early, vehicle waits until ready_time
    - if arrival is late (arrival > due_date), timewarp is accumulated
    """

    if not route.customers:
        return 0.0, 0.0, 0.0

    distance = 0.0
    timewarp = 0.0
    load = 0.0
    waiting = 0.0
    service = 0.0

    current_time = 0.0
    prev = instance.depot_id

    for cid in route.customers:
        cust = instance.customers[cid]
        travel = distance_between(instance.distance_matrix, instance.id_to_index, prev, cid)
        distance += travel
        arrival = current_time + travel

        if arrival < cust.ready_time:
            waiting += cust.ready_time - arrival
            service_start = cust.ready_time
        else:
            service_start = arrival

        if arrival > cust.due_date:
            timewarp += arrival - cust.due_date

        current_time = service_start + cust.service_time
        service += cust.service_time
        load += cust.demand
        prev = cid

    # Return to depot (distance only for baseline)
    distance += distance_between(instance.distance_matrix, instance.id_to_index, prev, instance.depot_id)

    return distance, timewarp, load, waiting, service


def evaluate_solution(instance: Instance, routes: Iterable[Route], penalty_tw: float) -> Solution:
    solution = Solution(routes=list(routes))

    for route in solution.routes:
        distance, timewarp, load, waiting, service = evaluate_route(instance, route)
        route.distance = distance
        route.timewarp = timewarp
        route.load = load
        route.waiting = waiting
        route.service = service

        solution.total_distance += distance
        solution.total_timewarp += timewarp
        solution.total_load += load
        solution.total_waiting += waiting
        solution.total_service += service
        if load > instance.capacity:
            solution.capacity_violation += load - instance.capacity

    solution.update_objective(penalty_tw)
    return solution
