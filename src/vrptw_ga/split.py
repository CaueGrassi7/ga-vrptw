from __future__ import annotations

from typing import List

import numpy as np

from .distance import distance_between
from .model import Instance, Route


def split_dp(instance: Instance, perm: List[int]) -> List[Route]:
    """Classic SPLIT DP (route-first, cluster-second) ignoring time windows.

    Minimizes total distance with capacity-feasible routes.
    """

    n = len(perm)
    if n == 0:
        return []

    # Precompute route costs for feasible segments
    cost = np.full((n + 1, n + 1), np.inf, dtype=float)
    for i in range(1, n + 1):
        load = 0.0
        prev = instance.depot_id
        for j in range(i, n + 1):
            cid = perm[j - 1]
            load += instance.customers[cid].demand
            if load > instance.capacity:
                break
            # Distance from prev to cid
            dist = 0.0
            if j == i:
                dist += distance_between(instance.distance_matrix, instance.id_to_index, instance.depot_id, cid)
            else:
                dist += distance_between(instance.distance_matrix, instance.id_to_index, prev, cid)
            prev = cid

            # Add distance from last to depot
            if j == i:
                cost[i, j] = dist + distance_between(
                    instance.distance_matrix, instance.id_to_index, cid, instance.depot_id
                )
            else:
                # Extend previous segment cost incrementally
                cost[i, j] = cost[i, j - 1] - distance_between(
                    instance.distance_matrix, instance.id_to_index, perm[j - 2], instance.depot_id
                )
                cost[i, j] += distance_between(
                    instance.distance_matrix, instance.id_to_index, perm[j - 2], cid
                )
                cost[i, j] += distance_between(
                    instance.distance_matrix, instance.id_to_index, cid, instance.depot_id
                )

    # DP
    dp = [0.0] + [np.inf] * n
    pred = [-1] * (n + 1)
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            if cost[i, j] == np.inf:
                continue
            candidate = dp[i - 1] + cost[i, j]
            if candidate < dp[j]:
                dp[j] = candidate
                pred[j] = i - 1

    # Reconstruct routes
    routes: List[Route] = []
    j = n
    while j > 0:
        i = pred[j] + 1
        routes.append(Route(customers=perm[i - 1 : j]))
        j = pred[j]

    routes.reverse()
    return routes
