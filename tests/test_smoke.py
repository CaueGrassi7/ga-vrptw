from __future__ import annotations

import numpy as np

from vrptw_ga.constructive import greedy_feasible_construction
from vrptw_ga.constructive import repair_and_insert_unrouted
from vrptw_ga.crossover.route_based_pb96 import pb96_crossover
from vrptw_ga.distance import build_distance_matrix
from vrptw_ga.metrics import dominates
from vrptw_ga.model import Customer, Instance, Route, Solution
from vrptw_ga.smoke_instance import build_smoke_instance
from vrptw_ga.evaluate import evaluate_solution


def test_feasible_builder() -> None:
    instance = build_smoke_instance()
    rng = np.random.default_rng(123)
    sol = greedy_feasible_construction(instance, rng)
    assert sol.total_timewarp == 0.0
    assert sol.feasible_timewindows


def test_pb96_crossover_valid() -> None:
    instance = build_smoke_instance()
    rng = np.random.default_rng(7)
    parent_a = greedy_feasible_construction(instance, rng)
    parent_b = greedy_feasible_construction(instance, rng)
    child_routes = None
    for _ in range(20):
        child_routes = pb96_crossover(rng, instance, parent_a.routes, parent_b.routes)
        if child_routes is not None:
            break
    assert child_routes is not None
    child = evaluate_solution(instance, child_routes, penalty_tw=1000.0)
    all_customers = sorted([c for r in child.routes for c in r.customers])
    assert all_customers == sorted(instance.customer_ids)


def test_ranking_prefers_smaller_k() -> None:
    instance = build_smoke_instance()
    # Two feasible solutions with different K; capacity is 6 (demand=2 each).
    r1 = [
        Route(customers=instance.customer_ids[:3]),
        Route(customers=instance.customer_ids[3:]),
    ]
    r2 = [
        Route(customers=instance.customer_ids[:2]),
        Route(customers=instance.customer_ids[2:4]),
        Route(customers=instance.customer_ids[4:]),
    ]
    s1 = evaluate_solution(instance, r1, penalty_tw=1000.0)
    s2 = evaluate_solution(instance, r2, penalty_tw=1000.0)
    assert dominates(s1, s2, penalty_tw=1000.0)


def test_ranking_prefers_smaller_route_time_on_tie_k() -> None:
    customers = {
        0: Customer(0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
        1: Customer(1, 1.0, 0.0, 1.0, 10.0, 200.0, 0.0),
        2: Customer(2, 2.0, 0.0, 1.0, 0.0, 200.0, 0.0),
    }
    distance_matrix, id_order, id_to_index = build_distance_matrix(customers)
    instance = Instance(
        name="TIE",
        capacity=10.0,
        depot_id=0,
        customers=customers,
        distance_matrix=distance_matrix,
        id_order=id_order,
        id_to_index=id_to_index,
    )

    r1 = [Route(customers=[1, 2])]
    r2 = [Route(customers=[2, 1])]
    s1 = evaluate_solution(instance, r1, penalty_tw=1000.0)
    s2 = evaluate_solution(instance, r2, penalty_tw=1000.0)
    assert s2.total_route_time < s1.total_route_time
    assert dominates(s2, s1, penalty_tw=1000.0)


def test_repair_invalid_offspring() -> None:
    customers = {
        0: Customer(0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
        1: Customer(1, 1.0, 0.0, 2.0, 0.0, 100.0, 0.0),
        2: Customer(2, 2.0, 0.0, 2.0, 0.0, 100.0, 0.0),
        3: Customer(3, 3.0, 0.0, 2.0, 0.0, 100.0, 0.0),
    }
    distance_matrix, id_order, id_to_index = build_distance_matrix(customers)
    instance = Instance(
        name="INVALID",
        capacity=2.0,
        depot_id=0,
        customers=customers,
        distance_matrix=distance_matrix,
        id_order=id_order,
        id_to_index=id_to_index,
    )
    rng = np.random.default_rng(0)
    routes, valid = repair_and_insert_unrouted(instance, [Route(customers=[1])], rng)
    assert routes
    assert valid is False
