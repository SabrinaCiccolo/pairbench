"""Order-preserving (monotone) assignment with the `hungarian` cost model.

Both sides are sorted by Y and aligned with a Needleman-Wunsch-style DP over
the cost matrix; skipping an element costs unmatch_cost / 2, so a pair is
taken when its cost is below unmatch_cost. Cannot express a crossed bundle.
"""

from __future__ import annotations

import numpy as np

from ..detect.ccorr import Detection
from .common import PairingResult, SYMMETRY_PERIOD_DEG, estimate_translation_ransac
from .hungarian import pair_cost_matrix
from .hungarian_aligned import (ALIGNED_DIST_CAP, ALIGNED_UNMATCH_COST,
                                CONSENSUS_GATE_MM, _shifted)

_MATCH, _SKIP1, _SKIP2 = 0, 1, 2


def monotone_assignment(cost: np.ndarray, unmatch_cost: float
                        ) -> list[tuple[int, int]]:
    """Cheapest order-preserving matching over a cost matrix whose rows and
    columns are already sorted. O(n1 * n2)."""
    n1, n2 = cost.shape
    skip = unmatch_cost / 2.0
    dp = np.zeros((n1 + 1, n2 + 1), dtype=np.float64)
    back = np.zeros((n1 + 1, n2 + 1), dtype=np.int8)
    for i in range(1, n1 + 1):
        dp[i, 0] = dp[i - 1, 0] + skip
        back[i, 0] = _SKIP1
    for j in range(1, n2 + 1):
        dp[0, j] = dp[0, j - 1] + skip
        back[0, j] = _SKIP2
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            options = (dp[i - 1, j - 1] + cost[i - 1, j - 1],
                       dp[i - 1, j] + skip,
                       dp[i, j - 1] + skip)
            move = int(np.argmin(options))
            dp[i, j] = options[move]
            back[i, j] = move

    pairs = []
    i, j = n1, n2
    while i > 0 or j > 0:
        move = back[i, j]
        if move == _MATCH:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == _SKIP1:
            i -= 1
        else:
            j -= 1
    return sorted(pairs)


def pair_monotone_aligned(
    dets1: list[Detection],
    dets2: list[Detection],
    period_deg: float = SYMMETRY_PERIOD_DEG,
    unmatch_cost: float = ALIGNED_UNMATCH_COST,
    dist_cap: float = ALIGNED_DIST_CAP,
    gate_mm: float = CONSENSUS_GATE_MM,
    refit: bool = True,
    t_init: np.ndarray | None = None,
    **_ignored,
) -> PairingResult:
    """Translation compensation (as in hungarian_aligned) + monotone DP.
    `t_init` supplies the translation instead of estimating it here."""
    res = PairingResult(strategy="monotone_aligned")
    n1, n2 = len(dets1), len(dets2)
    if n1 == 0 or n2 == 0:
        res.unmatched_1 = list(range(n1))
        res.unmatched_2 = list(range(n2))
        return res

    order1 = sorted(range(n1), key=lambda i: dets1[i].centroid_yz_mm[0])
    order2 = sorted(range(n2), key=lambda j: dets2[j].centroid_yz_mm[0])
    s1 = [dets1[i] for i in order1]
    s2 = [dets2[j] for j in order2]
    c1 = np.array([d.centroid_yz_mm for d in s1], dtype=np.float64)
    c2 = np.array([d.centroid_yz_mm for d in s2], dtype=np.float64)
    t = (np.asarray(t_init, dtype=np.float64) if t_init is not None
         else estimate_translation_ransac(c1, c2, gate_mm=gate_mm))

    def solve(t):
        cost = pair_cost_matrix(s1, _shifted(s2, t), period_deg, dist_cap)
        return monotone_assignment(cost, unmatch_cost), cost

    pairs, cost = solve(t)
    if refit and pairs:
        resid = np.array([c1[i] - (c2[j] + t) for i, j in pairs])
        t = t + np.median(resid, axis=0)
        pairs, cost = solve(t)

    pairs = [(i, j) for i, j in pairs if cost[i, j] < unmatch_cost]
    res.pairs = [(order1[i], order2[j]) for i, j in pairs]
    matched1 = {i for i, _ in res.pairs}
    matched2 = {j for _, j in res.pairs}
    res.unmatched_1 = [i for i in range(n1) if i not in matched1]
    res.unmatched_2 = [j for j in range(n2) if j not in matched2]
    res.diagnostics = {
        "t_yz_mm": [float(t[0]), float(t[1])],
        "pair_costs": [float(cost[i, j]) for i, j in pairs],
        "residuals_after_t_mm": [float(np.linalg.norm(c1[i] - (c2[j] + t)))
                                 for i, j in pairs],
    }
    return res


__all__ = ["monotone_assignment", "pair_monotone_aligned"]
