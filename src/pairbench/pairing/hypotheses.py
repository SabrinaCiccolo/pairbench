"""Enumerating and costing the scene translations a bundle admits.

Each cross-side detection pair proposes a translation; each hypothesis is
re-costed with the order-constrained objective (`monotone`) at its own median
translation. `shift_hypothesis` builds the shift-by-k family directly.
Used by `pairbench.experiments.cost_gap`.
"""

from __future__ import annotations

import numpy as np

from ..detect.ccorr import Detection
from .common import SYMMETRY_PERIOD_DEG, group_by_z_level
from .hungarian import pair_cost_matrix
from .hungarian_aligned import (ALIGNED_DIST_CAP, ALIGNED_UNMATCH_COST,
                                _shifted)
from .monotone import monotone_assignment

T_DEDUP_MM = 4.0   # anchors closer than this propose the same hypothesis


def translation_of(pairs, dets1: list[Detection], dets2: list[Detection]
                   ) -> np.ndarray:
    """Median YZ residual over a hypothesis' own pairs (dets1 ~= dets2 + t)."""
    resid = np.array([np.array(dets1[i].centroid_yz_mm)
                      - np.array(dets2[j].centroid_yz_mm) for i, j in pairs])
    return np.median(resid, axis=0)


def hypothesis_cost(pairs, dets1: list[Detection], dets2: list[Detection],
                    period_deg: float = SYMMETRY_PERIOD_DEG,
                    unmatch_cost: float = ALIGNED_UNMATCH_COST,
                    dist_cap: float = ALIGNED_DIST_CAP) -> float:
    """Order-constrained objective of one hypothesis at its own median
    translation; each unmatched detection pays unmatch_cost / 2."""
    if not pairs:
        return float("nan")
    n1, n2 = len(dets1), len(dets2)
    t = translation_of(pairs, dets1, dets2)
    cm = pair_cost_matrix(dets1, _shifted(dets2, t), period_deg, dist_cap)
    return (sum(float(cm[i, j]) for i, j in pairs)
            + (unmatch_cost / 2.0) * (n1 + n2 - 2 * len(pairs)))


def enumerate_hypotheses(dets1: list[Detection], dets2: list[Detection],
                         period_deg: float, unmatch_cost: float,
                         dist_cap: float):
    """Order-constrained solutions under every translation a cross-side pair
    proposes, deduplicated, cheapest first, as (cost, translation, pairs)."""
    n1, n2 = len(dets1), len(dets2)
    c1 = np.array([d.centroid_yz_mm for d in dets1], dtype=np.float64)
    c2 = np.array([d.centroid_yz_mm for d in dets2], dtype=np.float64)
    anchors: list[np.ndarray] = []
    for i in range(n1):
        for j in range(n2):
            t = c1[i] - c2[j]
            if all(np.linalg.norm(t - u) > T_DEDUP_MM for u in anchors):
                anchors.append(t)

    seen: set[tuple] = set()
    out = []
    for t in anchors:
        cost = pair_cost_matrix(dets1, _shifted(dets2, t), period_deg, dist_cap)
        pairs = [(i, j) for i, j in monotone_assignment(cost, unmatch_cost)
                 if cost[i, j] < unmatch_cost]
        if not pairs or tuple(pairs) in seen:
            continue
        seen.add(tuple(pairs))
        tm = translation_of(pairs, dets1, dets2)
        dp = hypothesis_cost(pairs, dets1, dets2, period_deg, unmatch_cost,
                             dist_cap)
        out.append((dp, tm, pairs))
    out.sort(key=lambda h: h[0])
    return out


def shift_hypothesis(dets1: list[Detection], dets2: list[Detection],
                     k: int) -> list[tuple[int, int]]:
    """Shift-family member `pi_k`: within each stacking level, pair the i-th
    face (by Y) on side 1 with the (i+k)-th on side 2. Levels are matched by
    index if both sides have the same count, else one level. Sorted pairs."""
    lv1, lv2 = group_by_z_level(dets1), group_by_z_level(dets2)
    if len(lv1) != len(lv2):
        lv1 = [sorted(range(len(dets1)),
                      key=lambda i: -dets1[i].centroid_yz_mm[0])]
        lv2 = [sorted(range(len(dets2)),
                      key=lambda j: -dets2[j].centroid_yz_mm[0])]
    pairs = []
    for row1, row2 in zip(lv1, lv2):
        for i, a in enumerate(row1):
            if 0 <= i + k < len(row2):
                pairs.append((a, row2[i + k]))
    return sorted(pairs)


__all__ = ["translation_of", "hypothesis_cost", "enumerate_hypotheses",
           "shift_hypothesis", "T_DEDUP_MM"]
