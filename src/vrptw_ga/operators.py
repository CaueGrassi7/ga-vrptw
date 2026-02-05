from __future__ import annotations

from typing import List

import numpy as np


def tournament_selection(rng: np.random.Generator, fitness: List[float], k: int = 3) -> int:
    """Return index of selected individual (lower fitness is better)."""

    n = len(fitness)
    candidates = rng.integers(0, n, size=k)
    best = min(candidates, key=lambda i: fitness[i])
    return int(best)


def ordered_crossover(rng: np.random.Generator, parent1: List[int], parent2: List[int]) -> List[int]:
    """Ordered crossover (OX) for permutations."""

    n = len(parent1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = [-1] * n

    child[a:b] = parent1[a:b]
    fill = [g for g in parent2 if g not in child]

    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1

    return child


def swap_mutation(rng: np.random.Generator, chromosome: List[int], p: float) -> List[int]:
    if rng.random() >= p:
        return chromosome

    n = len(chromosome)
    i, j = rng.choice(n, size=2, replace=False)
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


def inversion_mutation(rng: np.random.Generator, chromosome: List[int], p: float) -> List[int]:
    if rng.random() >= p:
        return chromosome

    n = len(chromosome)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    chromosome[i:j] = list(reversed(chromosome[i:j]))
    return chromosome
