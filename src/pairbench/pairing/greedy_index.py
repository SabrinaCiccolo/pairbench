"""Positional baseline pairing: Z-level grouping + index matching + 3 checks.

Levels and clusters within a level are paired by index; a pair is accepted
when the enabled twist, length and tilt checks pass. `length_bias_mm` centres
the length tolerance on an estimated bias.
"""

from __future__ import annotations

from ..detect.ccorr import Detection
from .common import (
    LENGTH_TOLERANCE_MM,
    PROFILE_LENGTH_MM,
    SYMMETRY_PERIOD_DEG,
    TWIST_TOLERANCE_DEG,
    PairingResult,
    angle_difference,
    group_by_z_level,
    length_residual,
    normals_aligned_xy,
)


def pair_greedy_index(
    dets1: list[Detection],
    dets2: list[Detection],
    period_deg: float = SYMMETRY_PERIOD_DEG,
    profile_length_mm: float = PROFILE_LENGTH_MM,
    check_length: bool = True,
    length_bias_mm: float = 0.0,
    check_twist: bool = True,
    check_tilt: bool = True,
    **_ignored,
) -> PairingResult:
    res = PairingResult(strategy="greedy_index")
    levels1 = group_by_z_level(dets1)
    levels2 = group_by_z_level(dets2)
    matched1: set[int] = set()
    matched2: set[int] = set()
    residuals = []
    fail_counts = {"twist": 0, "length": 0, "tilt": 0}

    for lv in range(min(len(levels1), len(levels2))):
        row1, row2 = levels1[lv], levels2[lv]
        for j in range(min(len(row1), len(row2))):
            i1, i2 = row1[j], row2[j]
            d1, d2 = dets1[i1], dets2[i2]
            ok = True
            if check_twist and angle_difference(d1, d2, period_deg) > TWIST_TOLERANCE_DEG:
                fail_counts["twist"] += 1
                ok = False
            resid = length_residual(d1, d2, profile_length_mm)
            residuals.append(resid)
            if ok and check_length and abs(resid - length_bias_mm) > LENGTH_TOLERANCE_MM:
                fail_counts["length"] += 1
                ok = False
            if ok and check_tilt and not normals_aligned_xy(d1, d2):
                fail_counts["tilt"] += 1
                ok = False
            if ok:
                res.pairs.append((i1, i2))
                matched1.add(i1)
                matched2.add(i2)

    res.unmatched_1 = [i for i in range(len(dets1)) if i not in matched1]
    res.unmatched_2 = [i for i in range(len(dets2)) if i not in matched2]
    res.diagnostics = {"length_residuals_mm": residuals, "fail_counts": fail_counts}
    return res
