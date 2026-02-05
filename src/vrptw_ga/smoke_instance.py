from __future__ import annotations

from typing import Dict, List

import numpy as np

from .distance import build_distance_matrix
from .model import Customer, Instance


def build_smoke_instance() -> Instance:
    """Small deterministic instance for tests.

    - Capacity forces >=2 routes
    - Time windows allow a known good permutation with zero timewarp
    """

    customers: Dict[int, Customer] = {
        0: Customer(0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
        1: Customer(1, -20.0, 0.0, 2.0, 0.0, 200.0, 0.0),
        2: Customer(2, -21.0, 0.0, 2.0, 0.0, 200.0, 0.0),
        3: Customer(3, 5.0, 0.0, 2.0, 0.0, 200.0, 0.0),
        4: Customer(4, 6.0, 0.0, 2.0, 0.0, 200.0, 0.0),
        5: Customer(5, 20.0, 0.0, 2.0, 0.0, 200.0, 0.0),
        6: Customer(6, 21.0, 0.0, 2.0, 0.0, 200.0, 0.0),
    }
    distance_matrix, id_order, id_to_index = build_distance_matrix(customers)
    return Instance(
        name="SMOKE",
        capacity=6.0,
        depot_id=0,
        customers=customers,
        distance_matrix=distance_matrix,
        id_order=id_order,
        id_to_index=id_to_index,
    )


def good_permutation() -> List[int]:
    return [5, 6, 3, 4, 1, 2]


def bad_permutation() -> List[int]:
    return [1, 2, 5, 6, 3, 4]
