from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .decode import decode_chromosome
from .model import Instance, Solution
from .operators import (
    inversion_mutation,
    ordered_crossover,
    swap_mutation,
    tournament_selection,
)
from .utils import append_csv


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 100
    generations: int = 300
    time_limit: float | None = 60.0
    penalty_tw: float = 1000.0
    elite: int = 2
    tournament_k: int = 3
    p_swap: float = 0.2
    p_inversion: float = 0.1
    heuristic_frac: float = 0.2
    log_every: int = 10
    repair_tw: bool = False
    log_file: str | None = None


def _nearest_neighbor_perm(instance: Instance, rng: np.random.Generator) -> List[int]:
    unvisited = set(instance.customer_ids)
    order: List[int] = []
    current = instance.depot_id

    while unvisited:
        candidates = list(unvisited)
        distances = [
            instance.distance_matrix[instance.id_to_index[current], instance.id_to_index[c]]
            for c in candidates
        ]
        min_dist = min(distances)
        nearest = [c for c, d in zip(candidates, distances) if d == min_dist]
        next_c = rng.choice(nearest)
        order.append(int(next_c))
        unvisited.remove(next_c)
        current = next_c

    return order


def _init_population(instance: Instance, rng: np.random.Generator, config: GAConfig) -> List[List[int]]:
    pop: List[List[int]] = []
    heuristic_count = max(1, int(config.pop_size * config.heuristic_frac))

    for _ in range(heuristic_count):
        pop.append(_nearest_neighbor_perm(instance, rng))

    for _ in range(config.pop_size - heuristic_count):
        pop.append(rng.permutation(instance.customer_ids).tolist())
    return pop


def _evaluate_population(
    instance: Instance, population: List[List[int]], penalty_tw: float, repair_tw: bool
) -> Tuple[List[Solution], List[float]]:
    solutions: List[Solution] = []
    fitness: List[float] = []
    for chrom in population:
        sol = decode_chromosome(instance, chrom, penalty_tw, repair_tw=repair_tw)
        solutions.append(sol)
        fitness.append(sol.objective)
    return solutions, fitness


def run_ga(instance: Instance, rng: np.random.Generator, config: GAConfig) -> Dict[str, object]:
    population = _init_population(instance, rng, config)

    best_solution: Solution | None = None
    best_fitness = float("inf")
    history: List[float] = []

    start_time = time.time()
    log_path: Path | None = Path(config.log_file) if config.log_file else None

    for gen in range(config.generations):
        solutions, fitness = _evaluate_population(instance, population, config.penalty_tw, config.repair_tw)

        gen_best_idx = int(min(range(len(fitness)), key=lambda i: fitness[i]))
        gen_best_fit = fitness[gen_best_idx]
        gen_best_sol = solutions[gen_best_idx]

        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_solution = gen_best_sol

        history.append(gen_best_fit)
        if config.log_every > 0 and (gen % config.log_every == 0 or gen == config.generations - 1):
            print(
                f"gen={gen} best={gen_best_fit:.3f} dist={gen_best_sol.total_distance:.3f} tw={gen_best_sol.total_timewarp:.3f}",
                flush=True,
            )
        if log_path is not None:
            append_csv(
                log_path,
                {
                    "gen": gen,
                    "best_fitness": gen_best_fit,
                    "best_distance": gen_best_sol.total_distance,
                    "best_timewarp": gen_best_sol.total_timewarp,
                    "routes": len(gen_best_sol.routes),
                    "feasible_timewindows": gen_best_sol.feasible_timewindows,
                },
            )

        elapsed = time.time() - start_time
        if config.time_limit is not None and elapsed >= config.time_limit:
            break

        # Elitism
        elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[: config.elite]
        new_population: List[List[int]] = [population[i][:] for i in elite_indices]

        while len(new_population) < config.pop_size:
            i1 = tournament_selection(rng, fitness, config.tournament_k)
            i2 = tournament_selection(rng, fitness, config.tournament_k)
            p1 = population[i1]
            p2 = population[i2]

            child = ordered_crossover(rng, p1, p2)
            child = swap_mutation(rng, child, config.p_swap)
            child = inversion_mutation(rng, child, config.p_inversion)
            new_population.append(child)

        population = new_population

    if best_solution is None:
        raise RuntimeError("GA did not produce any solution.")

    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "history": history,
    }
