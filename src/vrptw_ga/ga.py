from __future__ import annotations

"""GA-only implementation aligned with Potvin & Bengio (1996) VRPTW genetic search.

Key elements:
- Route-based crossover (SBX/RBX) with repair
- Linear ranking selection + SUS
- Lexicographic preference: feasible > infeasible; then minimize K, then distance
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .constructive import greedy_feasible_construction, solomon_i1_construction
from .decode import decode_chromosome
from .metrics import dominates, penalized_fitness
from .model import Instance, Route, Solution
from .evaluate import evaluate_solution
from .selection import linear_ranking_fitness, rank_population, stochastic_universal_sampling
from .crossover.route_based_pb96 import pb96_crossover
from .operators import ordered_crossover, pmx_crossover
from .mutation_pb96 import one_level_exchange, two_level_exchange


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 150
    generations: int = 50
    time_limit: float | None = 60.0
    penalty_tw: float = 1000.0
    crossover_rate: float = 0.6
    mutation_rate: float = 0.6
    elite: int = 1
    tournament_k: int = 3
    p_swap: float = 0.2
    p_inversion: float = 0.1
    log_every: int = 10
    repair_tw: bool = False
    decoder: str = "sequential"
    crossover: str = "pb96"  # pb96, ox, pmx
    objective: str = "lexicographic"  # lexicographic or penalized
    init: str = "i1"  # i1, random_perm, feasible_greedy, mixed


def _flatten_routes(routes: List[Route]) -> List[int]:
    return [cid for r in routes for cid in r.customers]


def _build_solution_from_perm(instance: Instance, perm: List[int], config: GAConfig) -> Solution:
    return decode_chromosome(
        instance, perm, penalty_tw=config.penalty_tw, repair_tw=config.repair_tw, decoder=config.decoder
    )


def _init_population(instance: Instance, rng: np.random.Generator, config: GAConfig) -> List[Solution]:
    pop: List[Solution] = []
    if config.init == "i1":
        for _ in range(config.pop_size):
            pop.append(solomon_i1_construction(instance, rng))
        return pop
    if config.init == "feasible_greedy":
        for _ in range(config.pop_size):
            pop.append(greedy_feasible_construction(instance, rng))
        return pop

    if config.init == "random_perm":
        for _ in range(config.pop_size):
            perm = rng.permutation(instance.customer_ids).tolist()
            pop.append(_build_solution_from_perm(instance, perm, config))
        return pop

    # mixed
    n_feasible = max(1, config.pop_size // 2)
    for _ in range(n_feasible):
        pop.append(solomon_i1_construction(instance, rng))
    for _ in range(config.pop_size - n_feasible):
        perm = rng.permutation(instance.customer_ids).tolist()
        pop.append(_build_solution_from_perm(instance, perm, config))
    return pop


def _evaluate_population(instance: Instance, population: List[Solution], config: GAConfig) -> None:
    # metrics are computed inside decode/evaluate; nothing to do here
    return None


def _select_parents(
    rng: np.random.Generator, population: List[Solution], config: GAConfig
) -> List[Solution]:
    ranked = rank_population(population, config.penalty_tw, config.objective)
    fitness_values = linear_ranking_fitness(len(ranked))
    selected_idx = stochastic_universal_sampling(rng, ranked, fitness_values, len(ranked))
    return [population[i] for i in selected_idx]


def _crossover(
    rng: np.random.Generator,
    instance: Instance,
    parent_a: Solution,
    parent_b: Solution,
    config: GAConfig,
) -> Solution | None:
    if config.crossover == "pb96":
        routes = pb96_crossover(rng, instance, parent_a.routes, parent_b.routes)
        if routes is None:
            return None
        return evaluate_solution(instance, routes, penalty_tw=config.penalty_tw)
    if config.crossover == "pmx":
        perm = pmx_crossover(rng, _flatten_routes(parent_a.routes), _flatten_routes(parent_b.routes))
    else:
        perm = ordered_crossover(rng, _flatten_routes(parent_a.routes), _flatten_routes(parent_b.routes))
    return _build_solution_from_perm(instance, perm, config)


def _mutate(rng: np.random.Generator, instance: Instance, sol: Solution, config: GAConfig) -> Solution:
    if rng.random() < 0.5:
        routes = one_level_exchange(instance, rng, sol.routes)
    else:
        routes = two_level_exchange(instance, rng, sol.routes)
    return evaluate_solution(instance, routes, penalty_tw=config.penalty_tw)


def _best_solution(population: List[Solution], config: GAConfig) -> Solution:
    ranked = rank_population(population, config.penalty_tw, config.objective)
    return population[ranked[0]]


def run_ga(instance: Instance, rng: np.random.Generator, config: GAConfig) -> Dict[str, object]:
    population = _init_population(instance, rng, config)
    start_time = time.time()

    history: List[Dict[str, float]] = []
    best = _best_solution(population, config)
    max_tries = 50

    for gen in range(config.generations):
        _evaluate_population(instance, population, config)
        best = _best_solution(population, config)

        feasible_rate = sum(1 for s in population if s.feasible_timewindows and s.feasible_capacity) / max(1, len(population))
        history.append(
            {
                "gen": float(gen),
                "best_k": float(len(best.routes)),
                "best_distance": float(best.total_distance),
                "best_total_route_time": float(best.total_route_time),
                "best_timewarp": float(best.total_timewarp),
                "best_penalized": float(penalized_fitness(best, config.penalty_tw)),
                "feasible_rate": float(feasible_rate),
            }
        )

        if config.log_every > 0 and (gen % config.log_every == 0 or gen == config.generations - 1):
            print(
                f"gen={gen} K={len(best.routes)} dist={best.total_distance:.3f} tw={best.total_timewarp:.3f}",
                flush=True,
            )

        if config.time_limit is not None and (time.time() - start_time) >= config.time_limit:
            break

        # Selection
        parents = _select_parents(rng, population, config)

        # Recombination
        offspring: List[Solution] = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]
            tries = 0
            while True:
                if rng.random() < config.crossover_rate:
                    child = _crossover(rng, instance, p1, p2, config)
                    if child is None:
                        tries += 1
                        if tries >= max_tries:
                            print(
                                "Warning: PB96 crossover failed after max tries; using best parent.",
                                flush=True,
                            )
                            child = p1 if dominates(p1, p2, config.penalty_tw) else p2
                            break
                        retry = _select_parents(rng, population, config)
                        if len(retry) >= 2:
                            p1, p2 = retry[0], retry[1]
                        continue
                    break
                else:
                    child = p1
                    break
            if rng.random() < config.mutation_rate:
                child = _mutate(rng, instance, child, config)
            offspring.append(child)

        # Elitism: keep best
        combined = offspring
        if config.elite > 0:
            combined.append(best)

        # Survivor selection by ranking
        ranked = rank_population(combined, config.penalty_tw, config.objective)
        population = [combined[i] for i in ranked[: config.pop_size]]

    return {
        "best_solution": best,
        "history": history,
    }
