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


def pmx_crossover(rng: np.random.Generator, parent1: List[int], parent2: List[int]) -> List[int]:
    """Partially Mapped Crossover (PMX) for permutations."""
    n = len(parent1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = [-1] * n

    # Copy segment from parent1
    child[a:b] = parent1[a:b]

    # Mapping from parent2 segment to parent1 segment
    mapping = {parent2[i]: parent1[i] for i in range(a, b)}

    for i in range(a, b):
        val = parent2[i]
        if val in child:
            continue
        pos = i
        while True:
            mapped = mapping.get(parent2[pos], None)
            if mapped is None:
                break
            if mapped not in child:
                pos = parent2.index(mapped)
            else:
                break
        if child[pos] == -1:
            child[pos] = val

    for i in range(n):
        if child[i] == -1:
            child[i] = parent2[i]

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
