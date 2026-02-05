from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .distance import distance_between
from .evaluate import evaluate_route
from .model import Instance, Route


def _clone_routes(routes: List[Route]) -> List[Route]:
    return [Route(customers=r.customers[:]) for r in routes]


def _route_time(instance: Instance, customers: List[int]) -> float:
    if not customers:
        return 0.0
    distance, timewarp, _, waiting, service = evaluate_route(instance, Route(customers=customers))
    if timewarp > 0.0:
        return float("inf")
    return distance + waiting + service


def _best_feasible_insertion(instance: Instance, route: List[int], cid: int) -> Tuple[int | None, float]:
    base_time = _route_time(instance, route)
    best_pos = None
    best_delta = float("inf")
    for pos in range(len(route) + 1):
        new_route = route[:pos] + [cid] + route[pos:]
        load = sum(instance.customers[c].demand for c in new_route)
        if load > instance.capacity:
            continue
        new_time = _route_time(instance, new_route)
        if new_time == float("inf"):
            continue
        delta = new_time - base_time
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
    return best_pos, best_delta


def _weighted_route_index(rng: np.random.Generator, routes: List[Route]) -> int:
    weights = np.array([1.0 / max(1, len(r.customers)) for r in routes], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(len(routes), p=weights))


def one_level_exchange(instance: Instance, rng: np.random.Generator, routes: List[Route]) -> List[Route]:
    if len(routes) <= 1:
        return _clone_routes(routes)

    new_routes = _clone_routes(routes)
    r_idx = _weighted_route_index(rng, new_routes)
    source = new_routes[r_idx].customers[:]
    if not source:
        return new_routes

    for cid in source:
        best = None
        for j, r in enumerate(new_routes):
            if j == r_idx:
                continue
            pos, delta = _best_feasible_insertion(instance, r.customers, cid)
            if pos is None:
                continue
            if best is None or delta < best[2]:
                best = (j, pos, delta)
        if best is None:
            return _clone_routes(routes)
        j, pos, _ = best
        target = new_routes[j]
        target.customers = target.customers[:pos] + [cid] + target.customers[pos:]
        new_routes[r_idx].customers.remove(cid)

    if not new_routes[r_idx].customers:
        del new_routes[r_idx]
    return new_routes


def _can_replace_customer(
    instance: Instance, route: List[int], replace_idx: int, cid: int
) -> bool:
    new_route = route[:replace_idx] + [cid] + route[replace_idx + 1 :]
    load = sum(instance.customers[c].demand for c in new_route)
    if load > instance.capacity:
        return False
    _, timewarp, _, _, _ = evaluate_route(instance, Route(customers=new_route))
    return timewarp == 0.0


def two_level_exchange(instance: Instance, rng: np.random.Generator, routes: List[Route]) -> List[Route]:
    if len(routes) <= 1:
        return _clone_routes(routes)

    new_routes = _clone_routes(routes)
    r_idx = _weighted_route_index(rng, new_routes)
    source = new_routes[r_idx].customers[:]
    if not source:
        return new_routes

    for cid in source:
        for j, r in enumerate(new_routes):
            if j == r_idx:
                continue
            for k, other_c in enumerate(r.customers):
                if not _can_replace_customer(instance, r.customers, k, cid):
                    continue
                # find a feasible insertion for other_c in any route except source
                best = None
                for t_idx, t_route in enumerate(new_routes):
                    if t_idx == r_idx:
                        continue
                    pos, delta = _best_feasible_insertion(instance, t_route.customers, other_c)
                    if pos is None:
                        continue
                    if best is None or delta < best[2]:
                        best = (t_idx, pos, delta)
                if best is None:
                    continue

                # apply exchange
                r.customers = r.customers[:k] + [cid] + r.customers[k + 1 :]
                t_idx, pos, _ = best
                target = new_routes[t_idx]
                target.customers = target.customers[:pos] + [other_c] + target.customers[pos:]
                new_routes[r_idx].customers.remove(cid)
                if not new_routes[r_idx].customers:
                    del new_routes[r_idx]
                return new_routes

    return _clone_routes(routes)
