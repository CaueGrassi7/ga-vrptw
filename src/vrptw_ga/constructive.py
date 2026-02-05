from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .distance import distance_between
from .model import Instance, Route, Solution
from .evaluate import evaluate_solution, evaluate_route


def _route_schedule(instance: Instance, route: List[int]) -> Tuple[bool, float]:
    current_time = 0.0
    prev = instance.depot_id
    for cid in route:
        cust = instance.customers[cid]
        travel = distance_between(instance.distance_matrix, instance.id_to_index, prev, cid)
        arrival = current_time + travel
        if arrival > cust.due_date:
            return False, arrival - cust.due_date
        if arrival < cust.ready_time:
            current_time = cust.ready_time + cust.service_time
        else:
            current_time = arrival + cust.service_time
        prev = cid
    return True, 0.0


def _route_distance(instance: Instance, route: List[int]) -> float:
    if not route:
        return 0.0
    dist = 0.0
    prev = instance.depot_id
    for cid in route:
        dist += distance_between(instance.distance_matrix, instance.id_to_index, prev, cid)
        prev = cid
    dist += distance_between(instance.distance_matrix, instance.id_to_index, prev, instance.depot_id)
    return dist


def _route_time(instance: Instance, route: List[int]) -> float:
    distance, timewarp, _, waiting, service = evaluate_route(instance, Route(customers=route))
    if timewarp > 0.0:
        return float("inf")
    return distance + waiting + service


def _route_start_times(instance: Instance, route: List[int]) -> Tuple[List[float], float] | None:
    current_time = 0.0
    prev = instance.depot_id
    starts: List[float] = []

    for cid in route:
        cust = instance.customers[cid]
        travel = distance_between(instance.distance_matrix, instance.id_to_index, prev, cid)
        arrival = current_time + travel
        if arrival > cust.due_date:
            return None
        start = cust.ready_time if arrival < cust.ready_time else arrival
        current_time = start + cust.service_time
        starts.append(start)
        prev = cid

    arrival_depot = current_time + distance_between(instance.distance_matrix, instance.id_to_index, prev, instance.depot_id)
    return starts, arrival_depot


def _feasible_insertion_positions(instance: Instance, route: List[int], cid: int) -> List[Tuple[int, float]]:
    positions: List[Tuple[int, float]] = []
    base_time = _route_time(instance, route)
    for pos in range(len(route) + 1):
        new_route = route[:pos] + [cid] + route[pos:]
        load = sum(instance.customers[c].demand for c in new_route)
        if load > instance.capacity:
            continue
        feasible, _ = _route_schedule(instance, new_route)
        if feasible:
            new_time = _route_time(instance, new_route)
            positions.append((pos, new_time - base_time))
    return positions


def greedy_feasible_construction(
    instance: Instance, rng: np.random.Generator, top_k: int = 3
) -> Solution:
    remaining = set(instance.customer_ids)
    routes: List[Route] = []

    while remaining:
        # Start a new route with a seed (earliest due date with randomness)
        candidates = list(remaining)
        dues = [instance.customers[c].due_date for c in candidates]
        idxs = np.argsort(dues)[: max(1, min(top_k, len(candidates)))]
        seed = candidates[int(rng.choice(idxs))]
        route = [seed]
        remaining.remove(seed)

        improved = True
        while improved and remaining:
            improved = False
            best_c = None
            best_pos = None
            best_cost = float("inf")
            for cid in list(remaining):
                positions = _feasible_insertion_positions(instance, route, cid)
                if not positions:
                    continue
                pos, cost = min(positions, key=lambda x: x[1])
                if cost < best_cost:
                    best_cost = cost
                    best_c = cid
                    best_pos = pos
            if best_c is not None and best_pos is not None:
                route = route[:best_pos] + [best_c] + route[best_pos:]
                remaining.remove(best_c)
                improved = True

        routes.append(Route(customers=route))

    return evaluate_solution(instance, routes, penalty_tw=1.0)


def solomon_i1_construction(instance: Instance, rng: np.random.Generator) -> Solution:
    remaining = set(instance.customer_ids)
    routes: List[Route] = []

    while remaining:
        mu = rng.random()
        lam = rng.random()
        alpha1 = rng.random()
        alpha2 = 1.0 - alpha1

        seed = int(rng.choice(list(remaining)))
        route: List[int] = [seed]
        remaining.remove(seed)

        while True:
            best_cid = None
            best_pos = None
            best_c2 = -float("inf")

            base_times = _route_start_times(instance, route)
            if base_times is None:
                break
            base_starts, base_depot_time = base_times

            for cid in list(remaining):
                best_c1 = float("inf")
                best_insert = None

                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cid] + route[pos:]
                    load = sum(instance.customers[c].demand for c in new_route)
                    if load > instance.capacity:
                        continue
                    new_times = _route_start_times(instance, new_route)
                    if new_times is None:
                        continue
                    new_starts, new_depot_time = new_times

                    if pos == 0:
                        i = instance.depot_id
                        j = route[0]
                        b_j = base_starts[0]
                        b_ju = new_starts[1]
                    elif pos == len(route):
                        i = route[-1]
                        j = instance.depot_id
                        b_j = base_depot_time
                        b_ju = new_depot_time
                    else:
                        i = route[pos - 1]
                        j = route[pos]
                        b_j = base_starts[pos]
                        b_ju = new_starts[pos + 1]

                    d_iu = distance_between(instance.distance_matrix, instance.id_to_index, i, cid)
                    d_uj = distance_between(instance.distance_matrix, instance.id_to_index, cid, j)
                    d_ij = distance_between(instance.distance_matrix, instance.id_to_index, i, j)
                    c11 = d_iu + d_uj - mu * d_ij
                    c12 = b_ju - b_j
                    c1 = alpha1 * c11 + alpha2 * c12

                    if c1 < best_c1:
                        best_c1 = c1
                        best_insert = pos

                if best_insert is None:
                    continue

                depot_dist = distance_between(
                    instance.distance_matrix, instance.id_to_index, instance.depot_id, cid
                )
                c2 = lam * depot_dist - best_c1
                if c2 > best_c2:
                    best_c2 = c2
                    best_cid = cid
                    best_pos = best_insert

            if best_cid is None or best_pos is None:
                break

            route = route[:best_pos] + [best_cid] + route[best_pos:]
            remaining.remove(best_cid)

        routes.append(Route(customers=route))

    return evaluate_solution(instance, routes, penalty_tw=1.0)


def repair_and_insert_unrouted(
    instance: Instance, routes: List[Route], rng: np.random.Generator
) -> tuple[List[Route], bool]:
    seen: set[int] = set()
    cleaned: List[Route] = []
    for route in routes:
        new_customers = []
        for cid in route.customers:
            if cid not in seen:
                new_customers.append(cid)
                seen.add(cid)
        if new_customers:
            cleaned.append(Route(customers=new_customers))

    unrouted = [cid for cid in instance.customer_ids if cid not in seen]

    for cid in unrouted:
        best_route_idx = None
        best_pos = None
        best_cost = float("inf")
        for r_idx, r in enumerate(cleaned):
            positions = _feasible_insertion_positions(instance, r.customers, cid)
            if not positions:
                continue
            pos, cost = min(positions, key=lambda x: x[1])
            if cost < best_cost:
                best_cost = cost
                best_route_idx = r_idx
                best_pos = pos
        if best_route_idx is not None and best_pos is not None:
            r = cleaned[best_route_idx]
            r.customers = r.customers[:best_pos] + [cid] + r.customers[best_pos:]
        else:
            return cleaned, False

    return cleaned, True
