from __future__ import annotations

from typing import List

import numpy as np

from .metrics import rank_key
from .model import Solution


def rank_population(population: List[Solution], penalty_tw: float, objective: str = "lexicographic") -> List[int]:
    keys = [(i, rank_key(sol, penalty_tw)) for i, sol in enumerate(population)]
    if objective == "penalized":
        keys.sort(key=lambda x: x[1].penalized)
    else:
        keys.sort(
            key=lambda x: (
                0 if x[1].feasible else 1,
                x[1].k,
                x[1].route_time,
            )
        )
    return [i for i, _ in keys]


def linear_ranking_fitness(pop_size: int, s_max: float = 1.6, s_min: float = 0.4) -> List[float]:
    # Sum of expected values = 1 for each individual on average (s_max + s_min = 2)
    if pop_size <= 1:
        return [1.0]
    return [
        s_max - (s_max - s_min) * (rank - 1) / (pop_size - 1)
        for rank in range(1, pop_size + 1)
    ]


def stochastic_universal_sampling(
    rng: np.random.Generator,
    ranked_indices: List[int],
    fitness_values: List[float],
    n_select: int,
) -> List[int]:
    total = sum(fitness_values)
    if total <= 0:
        return [ranked_indices[0]] * n_select
    step = total / n_select
    start = rng.random() * step
    pointers = [start + i * step for i in range(n_select)]

    selected: List[int] = []
    cum = 0.0
    i = 0
    for p in pointers:
        while cum < p and i < len(fitness_values):
            cum += fitness_values[i]
            i += 1
        idx = ranked_indices[max(0, i - 1)]
        selected.append(idx)
    return selected
