from __future__ import annotations

"""Route-based crossover operators inspired by Potvin & Bengio (1996)."""

from typing import List, Optional

import numpy as np

from ..constructive import repair_and_insert_unrouted
from ..model import Instance, Route


def _flatten(routes: List[Route]) -> List[int]:
    return [cid for r in routes for cid in r.customers]


def sbx(parent_a: List[Route], parent_b: List[Route], rng: np.random.Generator) -> List[Route]:
    if not parent_a or not parent_b:
        return [Route(customers=_flatten(parent_a or parent_b))]

    ra = rng.integers(0, len(parent_a))
    rb = rng.integers(0, len(parent_b))
    route_a = parent_a[int(ra)].customers
    route_b = parent_b[int(rb)].customers

    if len(route_a) < 2 or len(route_b) < 2:
        return [Route(customers=_flatten(parent_a))]

    cut_a = rng.integers(1, len(route_a))
    cut_b = rng.integers(0, len(route_b) - 1)

    new_route = route_a[: int(cut_a)] + route_b[int(cut_b) :]

    offspring = [Route(customers=r.customers[:]) for r in parent_a]
    offspring[int(ra)] = Route(customers=new_route)
    return offspring


def rbx(parent_a: List[Route], parent_b: List[Route], rng: np.random.Generator) -> List[Route]:
    if not parent_a:
        return [Route(customers=_flatten(parent_b))]

    ra = rng.integers(0, len(parent_a))
    route_a = parent_a[int(ra)].customers

    # Replace a random route in parent_b
    if parent_b:
        rb = rng.integers(0, len(parent_b))
        offspring = [Route(customers=r.customers[:]) for r in parent_b]
        offspring[int(rb)] = Route(customers=route_a[:])
        return offspring

    return [Route(customers=route_a[:])]


def pb96_crossover(
    rng: np.random.Generator, instance: Instance, parent_a: List[Route], parent_b: List[Route]
) -> Optional[List[Route]]:
    # Randomly choose SBX or RBX
    if rng.random() < 0.5:
        raw = sbx(parent_a, parent_b, rng)
    else:
        raw = rbx(parent_a, parent_b, rng)

    repaired, valid = repair_and_insert_unrouted(instance, raw, rng)
    if not valid:
        return None
    return repaired
