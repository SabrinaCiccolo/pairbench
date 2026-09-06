"""Count-proxy evaluation metrics shared by the experiment scripts."""

from __future__ import annotations

import numpy as np

from .detect.ccorr import Detection
from .pairing.common import group_by_z_level


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap_resample_f1(counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """counts: (n,3) tp/fp/fn per resampling unit. idx: (n_boot,n) resample
    indices. Returns (n_boot,) pooled F1 per replicate."""
    resampled = counts[idx].sum(axis=1)
    tp, fp, fn = resampled[:, 0], resampled[:, 1], resampled[:, 2]
    denom_p = tp + fp
    denom_r = tp + fn
    p = np.divide(tp, denom_p, out=np.zeros_like(tp), where=denom_p > 0)
    r = np.divide(tp, denom_r, out=np.zeros_like(tp), where=denom_r > 0)
    denom_f1 = p + r
    return np.divide(2 * p * r, denom_f1, out=np.zeros_like(tp), where=denom_f1 > 0)


def bootstrap_f1_ci(per_scene_tp_fp_fn: list[tuple[int, int, int]],
                    n_boot: int = 2000, seed: int = 0,
                    ) -> tuple[float, float]:
    """Percentile 95% CI on pooled F1, resampling scenes (not pairs) with
    replacement."""
    n = len(per_scene_tp_fp_fn)
    if n == 0:
        return (0.0, 0.0)
    counts = np.array(per_scene_tp_fp_fn, dtype=np.float64)  # (n, 3)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1 = bootstrap_resample_f1(counts, idx)
    lo, hi = np.percentile(f1, [2.5, 97.5])
    return (float(lo), float(hi))


def count_inversions(pairs, d1: list[Detection], d2: list[Detection]) -> int:
    """Swap proxy: pairs on the same stacking level in both views whose Y
    order disagrees across views. On non-crossed scenes these are almost
    certainly wrong-instance assignments that a count-proxy F1 cannot see."""
    lv1, lv2 = {}, {}
    for lv, row in enumerate(group_by_z_level(d1)):
        for i in row:
            lv1[i] = lv
    for lv, row in enumerate(group_by_z_level(d2)):
        for j in row:
            lv2[j] = lv
    n = 0
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i, j = pairs[a]
            k, l = pairs[b]
            if lv1[i] == lv1[k] and lv2[j] == lv2[l]:
                dy1 = d1[i].centroid_yz_mm[0] - d1[k].centroid_yz_mm[0]
                dy2 = d2[j].centroid_yz_mm[0] - d2[l].centroid_yz_mm[0]
                if dy1 * dy2 < 0:
                    n += 1
    return n
