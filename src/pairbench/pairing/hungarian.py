"""Global assignment pairing (Hungarian).

cost(i, j) = |folded angle diff| / TWIST_SCALE
           + min(YZ centroid distance / DIST_SCALE, DIST_CAP)
           + LEVEL_PENALTY * |z_level_i - z_level_j|
           + Z_OUTLIER_PENALTY * (z_outlier_i != z_outlier_j)

Padded with UNMATCH_COST dummies so instances may stay unmatched.
`diagnostics["pair_margins"]`: per pair, cost gap to the row's next-best column.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detect.ccorr import Detection
from .common import (
    SYMMETRY_PERIOD_DEG,
    PairingResult,
    angle_difference,
    group_by_z_level,
    yz_distance,
    z_outlier_flags,
)

TWIST_SCALE_DEG = 2.5
DIST_SCALE_MM = 45.0
DIST_CAP = 3.0       # saturation of the distance term
LEVEL_PENALTY = 2.0
Z_OUTLIER_PENALTY = 2.0
UNMATCH_COST = 6.0   # a pair worse than this stays unmatched


def _levels_of(dets: list[Detection]) -> list[int]:
    levels = group_by_z_level(dets)
    out = [0] * len(dets)
    for lv, row in enumerate(levels):
        for i in row:
            out[i] = lv
    return out


def pair_cost_matrix(
    dets1: list[Detection],
    dets2: list[Detection],
    period_deg: float = SYMMETRY_PERIOD_DEG,
    dist_cap: float = DIST_CAP,
) -> np.ndarray:
    """(n1, n2) pair costs per the module's cost model."""
    lv1 = _levels_of(dets1)
    lv2 = _levels_of(dets2)
    ol1 = z_outlier_flags(dets1)
    ol2 = z_outlier_flags(dets2)
    cost = np.zeros((len(dets1), len(dets2)), dtype=np.float64)
    for i in range(len(dets1)):
        for j in range(len(dets2)):
            cost[i, j] = (
                angle_difference(dets1[i], dets2[j], period_deg) / TWIST_SCALE_DEG
                + min(yz_distance(dets1[i], dets2[j]) / DIST_SCALE_MM, dist_cap)
                + LEVEL_PENALTY * abs(lv1[i] - lv2[j])
                + Z_OUTLIER_PENALTY * float(ol1[i] != ol2[j])
            )
    return cost


def _assignment_margin(cost: np.ndarray, r: int, c: int, n2: int,
                       unmatch_cost: float) -> float:
    """Cost gap between column c and the next-best real column in row r
    (or `unmatch_cost` if there is none)."""
    if n2 > 1:
        row = cost[r, :n2].copy()
        row[c] = np.inf
        alt = float(row.min())
    else:
        alt = unmatch_cost
    return alt - float(cost[r, c])


def pair_hungarian(
    dets1: list[Detection],
    dets2: list[Detection],
    period_deg: float = SYMMETRY_PERIOD_DEG,
    unmatch_cost: float = UNMATCH_COST,
    dist_cap: float = DIST_CAP,
    **_ignored,
) -> PairingResult:
    res = PairingResult(strategy="hungarian")
    n1, n2 = len(dets1), len(dets2)
    if n1 == 0 or n2 == 0:
        res.unmatched_1 = list(range(n1))
        res.unmatched_2 = list(range(n2))
        return res

    size = n1 + n2  # square padding: every instance can go unmatched
    cost = np.full((size, size), unmatch_cost, dtype=np.float64)
    cost[:n1, :n2] = pair_cost_matrix(dets1, dets2, period_deg, dist_cap)

    rows, cols = linear_sum_assignment(cost)
    matched1: set[int] = set()
    matched2: set[int] = set()
    costs = []
    margins = []
    for r, c in zip(rows, cols):
        if r < n1 and c < n2 and cost[r, c] < unmatch_cost:
            res.pairs.append((int(r), int(c)))
            matched1.add(int(r))
            matched2.add(int(c))
            costs.append(float(cost[r, c]))
            margins.append(_assignment_margin(cost, r, c, n2, unmatch_cost))
    res.unmatched_1 = [i for i in range(n1) if i not in matched1]
    res.unmatched_2 = [j for j in range(n2) if j not in matched2]
    res.diagnostics = {"pair_costs": costs, "pair_margins": margins}
    return res
