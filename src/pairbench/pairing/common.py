"""Shared geometry for the pairing strategies.

Pair checks (twist: folded angle difference; length: centroid distance vs the
nominal bar length; tilt: face-normal alignment), Z-level grouping, and the
RANSAC cross-view translation estimate. Length residuals are reported raw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detect.ccorr import Detection

TWIST_TOLERANCE_DEG = 2.5
LENGTH_TOLERANCE_MM = 10.0
TILT_TOLERANCE_DEG = 2.5
PROFILE_LENGTH_MM = 6005.0
SYMMETRY_PERIOD_DEG = 360.0   # fallback; per-family period from pairbench.symmetry


@dataclass
class PairingResult:
    strategy: str
    pairs: list = field(default_factory=list)   # (idx_side1, idx_side2)
    unmatched_1: list = field(default_factory=list)
    unmatched_2: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def fold_angle(angle_deg: float, period_deg: float = SYMMETRY_PERIOD_DEG) -> float:
    """Maps an angle into [-period/2, +period/2)."""
    half = period_deg / 2.0
    wrapped = angle_deg % period_deg
    if wrapped >= half:
        wrapped -= period_deg
    return wrapped


def normalize_signed(angle_deg: float) -> float:
    """Map into (-180, 180]."""
    a = angle_deg % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def angle_difference(d1: Detection, d2: Detection,
                     period_deg: float = SYMMETRY_PERIOD_DEG) -> float:
    raw = normalize_signed(d1.angle_deg - d2.angle_deg)
    folded = fold_angle(raw, period_deg)
    diff = abs(folded)
    # 0/180 alias fallback when the period exceeds 180
    if period_deg > 180.0 and diff > TWIST_TOLERANCE_DEG:
        diff = min(diff, abs(fold_angle(raw, 180.0)))
    return diff


def yz_distance(d1: Detection, d2: Detection) -> float:
    return float(np.hypot(d1.centroid_yz_mm[0] - d2.centroid_yz_mm[0],
                          d1.centroid_yz_mm[1] - d2.centroid_yz_mm[1]))


def length_residual(d1: Detection, d2: Detection,
                    profile_length_mm: float = PROFILE_LENGTH_MM) -> float:
    c1 = np.asarray(d1.centroid_world_mm)
    c2 = np.asarray(d2.centroid_world_mm)
    return float(np.linalg.norm(c1 - c2) - profile_length_mm)


def normals_aligned_xy(d1: Detection, d2: Detection,
                       tol_deg: float = TILT_TOLERANCE_DEG) -> bool:
    """XY projections of face normals within tol (both normals oriented
    toward +X by the detector, so compare directly)."""
    n1 = np.asarray(d1.normal_world[:2], dtype=np.float64)
    n2 = np.asarray(d2.normal_world[:2], dtype=np.float64)
    l1, l2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if l1 == 0 or l2 == 0:
        return True  # degenerate normal: don't fail the pair on it
    cosang = np.clip(np.dot(n1, n2) / (l1 * l2), -1.0, 1.0)
    return bool(np.degrees(np.arccos(cosang)) <= tol_deg)


def greedy_nn(c1: np.ndarray, c2_shifted: np.ndarray, gate_mm: float):
    """Greedy mutual matching by ascending distance within the gate.

    Returns (i, j, dist) triples over centroid arrays (side 2 pre-shifted)."""
    if len(c1) == 0 or len(c2_shifted) == 0:
        return []
    d = np.linalg.norm(c1[:, None, :] - c2_shifted[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    used1: set[int] = set()
    used2: set[int] = set()
    pairs = []
    for i, j in order:
        if d[i, j] > gate_mm:
            break
        if i in used1 or j in used2:
            continue
        pairs.append((int(i), int(j), float(d[i, j])))
        used1.add(int(i))
        used2.add(int(j))
    return pairs


def estimate_translation_ransac(c1: np.ndarray, c2: np.ndarray,
                                gate_mm: float) -> np.ndarray:
    """Consensus YZ translation: every cross-side centroid pair proposes
    t = c1 - c2; the anchor with the most greedy-NN inliers wins (ties:
    smaller mean residual, then lower anchor indices — deterministic)."""
    best = None
    best_t = np.zeros(2)
    for ai in range(len(c1)):
        for aj in range(len(c2)):
            t = c1[ai] - c2[aj]
            pairs = greedy_nn(c1, c2 + t, gate_mm)
            if not pairs:
                continue
            mean_resid = float(np.mean([p[2] for p in pairs]))
            key = (len(pairs), -mean_resid, -ai, -aj)
            if best is None or key > best:
                best = key
                best_t = t
    return best_t


def group_by_z_level(dets: list[Detection]) -> list[list[int]]:
    """Iterative stacking levels from Y/Z extents: a detection is one level
    above another if its centroid Y falls inside the other's Y extent and its
    centroid Z exceeds the other's max Z.

    Returns index lists, outer ascending level, inner descending centroid Y.
    """
    n = len(dets)
    levels = [0] * n
    changed = True
    guard = 0
    while changed and guard < n + 2:
        changed = False
        guard += 1
        for a in range(n):
            best = 0
            for b in range(n):
                if a == b:
                    continue
                ymin, ymax = dets[b].y_extent_mm
                if (ymin <= dets[a].centroid_yz_mm[0] <= ymax
                        and dets[a].centroid_yz_mm[1] > dets[b].z_extent_mm[1]):
                    best = max(best, levels[b] + 1)
            if best != levels[a]:
                levels[a] = best
                changed = True
    out: dict[int, list[int]] = {}
    for i, lv in enumerate(levels):
        out.setdefault(lv, []).append(i)
    grouped = []
    for lv in sorted(out):
        grouped.append(sorted(out[lv], key=lambda i: -dets[i].centroid_yz_mm[0]))
    return grouped


Z_OUTLIER_MM = 8.0  # above the same-view height noise of non-crossed bars


def z_outlier_flags(dets: list[Detection],
                    threshold_mm: float = Z_OUTLIER_MM) -> list[bool]:
    """True per detection whose centroid Z is more than threshold_mm from
    this view's median height (catches bars lying across their neighbours)."""
    if not dets:
        return []
    z = np.array([d.centroid_yz_mm[1] for d in dets])
    med = float(np.median(z))
    return [bool(abs(v - med) > threshold_mm) for v in z]
