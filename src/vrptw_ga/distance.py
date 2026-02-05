from __future__ import annotations

from typing import Dict, List

import numpy as np

from .model import Customer


def build_distance_matrix(customers: Dict[int, Customer]) -> tuple[np.ndarray, List[int], Dict[int, int]]:
    """Build Euclidean distance matrix ordered by sorted customer ids."""

    ids = sorted(customers.keys())
    index = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    mat = np.zeros((n, n), dtype=float)
    coords = np.array([(customers[cid].x, customers[cid].y) for cid in ids], dtype=float)

    for i in range(n):
        diff = coords[i] - coords
        mat[i, :] = np.sqrt((diff ** 2).sum(axis=1))

    return mat, ids, index


def distance_between(matrix: np.ndarray, id_to_index: Dict[int, int], id_a: int, id_b: int) -> float:
    return float(matrix[id_to_index[id_a], id_to_index[id_b]])
