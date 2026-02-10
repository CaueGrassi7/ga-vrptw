from __future__ import annotations

from dataclasses import dataclass

from .model import Solution


@dataclass(frozen=True)
class RankKey:
    feasible: bool
    k: int
    route_time: float
    distance: float
    penalized: float


def penalized_fitness(solution: Solution, penalty_tw: float) -> float:
    return solution.total_distance + penalty_tw * solution.total_timewarp


def rank_key(solution: Solution, penalty_tw: float) -> RankKey:
    feasible = solution.feasible_timewindows and solution.feasible_capacity
    k = len(solution.routes)
    route_time = solution.total_route_time
    distance = solution.total_distance
    penalized = penalized_fitness(solution, penalty_tw)
    return RankKey(feasible=feasible, k=k, route_time=route_time, distance=distance, penalized=penalized)


def dominates(a: Solution, b: Solution, penalty_tw: float) -> bool:
    ka = rank_key(a, penalty_tw)
    kb = rank_key(b, penalty_tw)
    if ka.feasible and not kb.feasible:
        return True
    if not ka.feasible and kb.feasible:
        return False
    if ka.feasible and kb.feasible:
        return (ka.k, ka.route_time) < (kb.k, kb.route_time)
    return ka.penalized < kb.penalized
