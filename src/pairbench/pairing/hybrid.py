"""Order-constrained pairing plus evidence-gated crossing hypotheses.

1. Bars elevated above the bundle's median height in both views, in equal
   numbers, that break the scene's rigid cross-view translation are treated
   as crossing bars: paired among themselves (Hungarian), the rest solved by
   `_solve_bed` (monotone DP per stacking level).
2. Bars elevated in one view only, with no partner within the gate under the
   scene translation, are peeled out; after the bed solve they are matched to
   the other side's leftovers on folded angle and translated YZ distance only.
3. Otherwise the arm is the order-constrained assignment alone.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detect.ccorr import Detection
from .common import (PairingResult, SYMMETRY_PERIOD_DEG, Z_OUTLIER_MM,
                     angle_difference, estimate_translation_ransac,
                     group_by_z_level)
from .hungarian import DIST_SCALE_MM, TWIST_SCALE_DEG, pair_cost_matrix
from .hungarian_aligned import (ALIGNED_DIST_CAP, ALIGNED_UNMATCH_COST,
                                CONSENSUS_GATE_MM)
from .monotone import pair_monotone_aligned

LEVEL_MATCH_MM = 20.0   # max median-height difference of matched levels
                        # (half the thinnest section's height, 40 mm)
MIN_BED_DETS = 3        # min detections for a bed-height estimate


def bed_height_mm(dets1: list[Detection], dets2: list[Detection]) -> float | None:
    """Median end-face centroid height over both views, or None if fewer
    than MIN_BED_DETS detections. Used only as a relative reference."""
    z = [d.centroid_yz_mm[1] for d in dets1] + [d.centroid_yz_mm[1] for d in dets2]
    if len(z) < MIN_BED_DETS:
        return None
    return float(np.median(z))


def elevated_outliers(dets: list[Detection], bed_mm: float | None,
                      threshold_mm: float = Z_OUTLIER_MM) -> list[int]:
    """Indices of bars more than threshold_mm above the bed height (one-sided)."""
    if bed_mm is None:
        return []
    return [i for i, d in enumerate(dets)
            if d.centroid_yz_mm[1] - bed_mm > threshold_mm]


def _match_crossing(dets1: list[Detection], dets2: list[Detection],
                    idx1: list[int], idx2: list[int],
                    period_deg: float, dist_cap: float) -> list[tuple[int, int]]:
    """Pair the peeled crossing bars among themselves (k x k, k >= 1)."""
    if len(idx1) == 1:
        return [(idx1[0], idx2[0])]
    sub1 = [dets1[i] for i in idx1]
    sub2 = [dets2[j] for j in idx2]
    cost = pair_cost_matrix(sub1, sub2, period_deg, dist_cap)
    rows, cols = linear_sum_assignment(cost)
    return [(idx1[r], idx2[c]) for r, c in zip(rows, cols)]


def moves_with_the_bundle(dets1: list[Detection], dets2: list[Detection],
                          idx1: list[int], idx2: list[int],
                          gate_mm: float = CONSENSUS_GATE_MM) -> bool:
    """True if the elevated bars are consistent with the bundle's own motion
    (a stacked reading) rather than breaking it (a crossing)."""
    if not idx1 or len(idx1) != len(idx2):
        return True
    c1 = np.array([d.centroid_yz_mm for d in dets1], dtype=np.float64)
    c2 = np.array([d.centroid_yz_mm for d in dets2], dtype=np.float64)
    t = estimate_translation_ransac(c1, c2, gate_mm=gate_mm)
    e1 = c1[idx1]
    e2 = c2[idx2] + t
    d = np.linalg.norm(e1[:, None, :] - e2[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(d)
    return bool(d[rows, cols].max() <= gate_mm)


def breaks_rigid_motion(dets1: list[Detection], dets2: list[Detection],
                        idx1: list[int], idx2: list[int],
                        gate_mm: float = CONSENSUS_GATE_MM):
    """For bars elevated in one view only (exactly one of idx1, idx2 non-empty):
    the scene translation if none of them has an other-side detection within
    `gate_mm` after translation, else None."""
    if bool(idx1) == bool(idx2):
        return None
    c1 = np.array([d.centroid_yz_mm for d in dets1], dtype=np.float64)
    c2 = np.array([d.centroid_yz_mm for d in dets2], dtype=np.float64)
    t = estimate_translation_ransac(c1, c2, gate_mm=gate_mm)
    if idx1:
        near = [np.linalg.norm(c1[i] - (c2 + t), axis=1).min() for i in idx1]
    else:
        near = [np.linalg.norm(c2[j] + t - c1, axis=1).min() for j in idx2]
    return t if min(near) > gate_mm else None


def _match_one_view(dets1: list[Detection], dets2: list[Detection],
                    free1: list[int], free2: list[int], t, period_deg: float,
                    unmatch_cost: float, dist_cap: float
                    ) -> list[tuple[int, int]]:
    """Match peeled one-view crossing bars against the other side's leftovers
    on folded angle and translated YZ distance only (see module docstring)."""
    if not free1 or not free2:
        return []
    c1 = np.array([dets1[i].centroid_yz_mm for i in free1], dtype=np.float64)
    c2 = np.array([dets2[j].centroid_yz_mm for j in free2], dtype=np.float64) + t
    cost = np.array([[
        angle_difference(dets1[i], dets2[j], period_deg) / TWIST_SCALE_DEG
        + min(float(np.linalg.norm(c1[a] - c2[b])) / DIST_SCALE_MM, dist_cap)
        for b, j in enumerate(free2)] for a, i in enumerate(free1)])
    rows, cols = linear_sum_assignment(cost)
    return [(free1[r], free2[c]) for r, c in zip(rows, cols)
            if cost[r, c] < unmatch_cost]


def _solve_ordered(dets1: list[Detection], dets2: list[Detection],
                   period_deg: float, unmatch_cost: float,
                   dist_cap: float, gate_mm: float,
                   t_scene=None) -> list[tuple[int, int]]:
    """Order-constrained assignment. Indices are local to the lists given."""
    if not dets1 or not dets2:
        return []
    res = pair_monotone_aligned(dets1, dets2, period_deg=period_deg,
                                unmatch_cost=unmatch_cost, dist_cap=dist_cap,
                                gate_mm=gate_mm, t_init=t_scene,
                                refit=t_scene is None)
    return res.pairs


def _solve_bed(dets1: list[Detection], dets2: list[Detection],
               period_deg: float, unmatch_cost: float,
               dist_cap: float, gate_mm: float,
               t_scene=None) -> list[tuple[int, int]]:
    """Order-constrained solve per stacking level (group_by_z_level).

    Levels are matched by index when both views have the same count, else by
    median height after translation within LEVEL_MATCH_MM; unmatched levels
    are solved together. Fewer than two levels in either view: one global solve.
    """
    if not dets1 or not dets2:
        return []
    lv1 = group_by_z_level(dets1)
    lv2 = group_by_z_level(dets2)
    if min(len(lv1), len(lv2)) < 2:
        return _solve_ordered(dets1, dets2, period_deg,
                              unmatch_cost, dist_cap, gate_mm, t_scene)
    if t_scene is None:
        # one estimate for the whole bed, held fixed per level: a single
        # level is too small a sample to re-run consensus on
        t_scene = estimate_translation_ransac(
            np.array([d.centroid_yz_mm for d in dets1], dtype=np.float64),
            np.array([d.centroid_yz_mm for d in dets2], dtype=np.float64),
            gate_mm=gate_mm)
    if len(lv1) == len(lv2):
        matched = list(zip(lv1, lv2))
    else:
        c1 = np.array([d.centroid_yz_mm for d in dets1], dtype=np.float64)
        c2 = np.array([d.centroid_yz_mm for d in dets2], dtype=np.float64)
        z1 = [float(np.median(c1[row, 1])) for row in lv1]
        z2 = [float(np.median(c2[row, 1])) + float(t_scene[1]) for row in lv2]
        dz = np.abs(np.subtract.outer(z1, z2))
        rows, cols = linear_sum_assignment(dz)
        matched = [(lv1[a], lv2[b]) for a, b in zip(rows, cols)
                   if dz[a, b] <= LEVEL_MATCH_MM]
        seen1 = {i for r, _ in matched for i in r}
        seen2 = {j for _, r in matched for j in r}
        rest1 = [i for i in range(len(dets1)) if i not in seen1]
        rest2 = [j for j in range(len(dets2)) if j not in seen2]
        if rest1 and rest2:
            matched.append((rest1, rest2))
    pairs = []
    for row1, row2 in matched:
        idx1, idx2 = sorted(row1), sorted(row2)   # dets are Y-sorted already
        local = _solve_ordered([dets1[i] for i in idx1], [dets2[j] for j in idx2],
                               period_deg, unmatch_cost,
                               dist_cap, gate_mm, t_scene)
        pairs += [(idx1[i], idx2[j]) for i, j in local]
    return pairs


def pair_hybrid_aligned(
    dets1: list[Detection],
    dets2: list[Detection],
    period_deg: float = SYMMETRY_PERIOD_DEG,
    unmatch_cost: float = ALIGNED_UNMATCH_COST,
    dist_cap: float = ALIGNED_DIST_CAP,
    gate_mm: float = CONSENSUS_GATE_MM,
    **_ignored,
) -> PairingResult:
    res = PairingResult(strategy="hybrid_aligned")
    n1, n2 = len(dets1), len(dets2)
    if n1 == 0 or n2 == 0:
        res.unmatched_1 = list(range(n1))
        res.unmatched_2 = list(range(n2))
        return res

    bed_mm = bed_height_mm(dets1, dets2)
    out1 = elevated_outliers(dets1, bed_mm)
    out2 = elevated_outliers(dets2, bed_mm)
    corroborated = bool(out1) and len(out1) == len(out2)
    stacked = moves_with_the_bundle(dets1, dets2, out1, out2, gate_mm)
    crossing_used = corroborated and not stacked

    t_one_view = (None if crossing_used
                  else breaks_rigid_motion(dets1, dets2, out1, out2, gate_mm))
    one_view_used = t_one_view is not None

    if crossing_used:
        pairs = _match_crossing(dets1, dets2, out1, out2, period_deg, dist_cap)
        bed1 = [i for i in range(n1) if i not in set(out1)]
        bed2 = [j for j in range(n2) if j not in set(out2)]
    elif one_view_used:
        pairs = []
        bed1 = [i for i in range(n1) if i not in set(out1)]
        bed2 = [j for j in range(n2) if j not in set(out2)]
    else:
        pairs = []
        bed1, bed2 = list(range(n1)), list(range(n2))

    bed_pairs = _solve_bed([dets1[i] for i in bed1], [dets2[j] for j in bed2],
                           period_deg, unmatch_cost, dist_cap,
                           gate_mm)
    pairs += [(bed1[i], bed2[j]) for i, j in bed_pairs]

    one_view_pairs = []
    if one_view_used:
        used1 = {i for i, _ in pairs}
        used2 = {j for _, j in pairs}
        free1 = [i for i in (out1 or range(n1)) if i not in used1]
        free2 = [j for j in (out2 or range(n2)) if j not in used2]
        one_view_pairs = _match_one_view(dets1, dets2, free1, free2,
                                         t_one_view, period_deg,
                                         unmatch_cost, dist_cap)
        pairs += one_view_pairs

    res.pairs = sorted(pairs)
    matched1 = {i for i, _ in res.pairs}
    matched2 = {j for _, j in res.pairs}
    res.unmatched_1 = [i for i in range(n1) if i not in matched1]
    res.unmatched_2 = [j for j in range(n2) if j not in matched2]
    res.diagnostics = {
        "crossing_hypothesis": crossing_used or one_view_used,
        "one_view_crossing": one_view_used,
        "one_view_crossing_pairs": [(int(i), int(j)) for i, j in one_view_pairs],
        "elevation_corroborated": corroborated,
        "moves_with_bundle": stacked,
        "crossing_pairs": [(int(i), int(j)) for i, j in pairs[:len(out1)]]
                          if crossing_used else [],
        "n_elevated_1": len(out1),
        "n_elevated_2": len(out2),
    }
    return res


__all__ = ["pair_hybrid_aligned", "elevated_outliers", "bed_height_mm",
           "breaks_rigid_motion",
           "moves_with_the_bundle"]
