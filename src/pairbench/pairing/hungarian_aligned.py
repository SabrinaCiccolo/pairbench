"""Translation-compensated global assignment.

Estimates the scene's cross-view YZ translation by RANSAC consensus (each
cross-side centroid pair proposes t = c1 - c2; most nearest-neighbour inliers
wins), runs `hungarian` on the shifted detections, refits t as the median pair
residual and assigns once more.
"""

from __future__ import annotations

import numpy as np

from ..detect.ccorr import Detection
from .common import (PairingResult, SYMMETRY_PERIOD_DEG,
                     estimate_translation_ransac)
from .hungarian import pair_hungarian

CONSENSUS_GATE_MM = 30.0   # absorbs per-bar skew spread across a long bar
ALIGNED_DIST_CAP = 4.5     # above the noise floor of a true pair's cost
ALIGNED_UNMATCH_COST = 7.5


def _shifted(dets: list[Detection], t: np.ndarray) -> list[Detection]:
    """Copies with centroid_yz_mm shifted by t (assignment costs only)."""
    out = []
    for d in dets:
        e = Detection(**{**d.__dict__})
        e.centroid_yz_mm = (d.centroid_yz_mm[0] + t[0],
                            d.centroid_yz_mm[1] + t[1])
        out.append(e)
    return out


def pair_hungarian_aligned(
    dets1: list[Detection],
    dets2: list[Detection],
    period_deg: float = SYMMETRY_PERIOD_DEG,
    unmatch_cost: float = ALIGNED_UNMATCH_COST,
    dist_cap: float = ALIGNED_DIST_CAP,
    gate_mm: float = CONSENSUS_GATE_MM,
    refit: bool = True,
    **_ignored,
) -> PairingResult:
    res = PairingResult(strategy="hungarian_aligned")
    n1, n2 = len(dets1), len(dets2)
    if n1 == 0 or n2 == 0:
        res.unmatched_1 = list(range(n1))
        res.unmatched_2 = list(range(n2))
        return res

    c1 = np.array([d.centroid_yz_mm for d in dets1], dtype=np.float64)
    c2 = np.array([d.centroid_yz_mm for d in dets2], dtype=np.float64)
    t = estimate_translation_ransac(c1, c2, gate_mm=gate_mm)

    def assign(t):
        return pair_hungarian(dets1, _shifted(dets2, t),
                              period_deg=period_deg,
                              unmatch_cost=unmatch_cost, dist_cap=dist_cap)

    inner = assign(t)
    if refit and inner.pairs:
        resid = np.array([c1[i] - (c2[j] + t) for i, j in inner.pairs])
        t = t + np.median(resid, axis=0)
        inner = assign(t)

    res.pairs = inner.pairs
    res.unmatched_1 = inner.unmatched_1
    res.unmatched_2 = inner.unmatched_2
    res.diagnostics = {
        "t_yz_mm": [float(t[0]), float(t[1])],
        "pair_costs": inner.diagnostics.get("pair_costs", []),
        "pair_margins": inner.diagnostics.get("pair_margins", []),
        "residuals_after_t_mm": [
            float(np.linalg.norm(c1[i] - (c2[j] + t)))
            for i, j in inner.pairs],
    }
    return res
