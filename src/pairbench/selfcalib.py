"""Joint estimation of the cross-view transform and the assignment.

The per-scan carriage X position s of each scanner is unknown per scan; it is
fitted from matched pairs with the constraint |T1(p1, s1) - T2(p2, s2)| = 6005 mm.
Assignment and transform are alternated until the assignment stops changing.
Only the separation s1 - s2 is observable (moving both together leaves every
pair length unchanged); `observability` measures that conditioning.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from .detect.ccorr import Detection
from .io.calib import DualCalibration
from .io.calib_full import FullCalibration
from .pairing.common import PROFILE_LENGTH_MM

MAX_ITERS = 12


def to_scanner_frame(dets: list[Detection], side: int,
                     offline: DualCalibration) -> np.ndarray:
    """(N,3) centroids mapped back out of the single-knot offline calibration
    into scanner-local coordinates, without re-reading the source PLYs."""
    sc = offline.for_side(side)
    if not dets:
        return np.zeros((0, 3))
    c = np.array([d.centroid_world_mm for d in dets], dtype=np.float64)
    return (c - sc.t) @ sc.R


def recalibrated(dets: list[Detection], side: int, s: float,
                 offline: DualCalibration, full: FullCalibration,
                 t_yz: tuple[float, float] = (0.0, 0.0)) -> list[Detection]:
    """Copies re-projected at carriage position s (+ a YZ shift on side 2).
    Extents are shifted by the centroid's delta, not re-derived."""
    if not dets:
        return []
    p_scan = to_scanner_frame(dets, side, offline)
    w = full.for_side(side).transform(p_scan, s)
    if side == 2:
        w = w + np.array([0.0, t_yz[0], t_yz[1]])
    out = []
    for d, new_w in zip(dets, w):
        dy = float(new_w[1]) - d.centroid_yz_mm[0]
        dz = float(new_w[2]) - d.centroid_yz_mm[1]
        out.append(replace(
            d,
            centroid_world_mm=tuple(float(v) for v in new_w),
            centroid_yz_mm=(float(new_w[1]), float(new_w[2])),
            y_extent_mm=(d.y_extent_mm[0] + dy, d.y_extent_mm[1] + dy),
            z_extent_mm=(d.z_extent_mm[0] + dz, d.z_extent_mm[1] + dz),
        ))
    return out


def length_residuals(p1: np.ndarray, p2: np.ndarray, full: FullCalibration,
                     s1: float, s2: float,
                     t_yz: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """|w1 - w2| - nominal bar length, over matched scanner-frame centroids."""
    if len(p1) == 0:
        return np.zeros(0)
    w1 = full.scanner1.transform(p1, s1)
    w2 = full.scanner2.transform(p2, s2) + np.array([0.0, t_yz[0], t_yz[1]])
    return np.linalg.norm(w1 - w2, axis=1) - PROFILE_LENGTH_MM


def fit_steps(p1: np.ndarray, p2: np.ndarray, full: FullCalibration,
              x0: tuple[float, float] | None = None,
              t_yz: tuple[float, float] = (0.0, 0.0),
              coarse: int = 43) -> tuple[float, float]:
    """(s1, s2) minimizing median |length residual|: coarse grid over the
    calibrated range, then Nelder-Mead, clamped to the knot range."""
    def objective(x) -> float:
        return float(np.median(np.abs(
            length_residuals(p1, p2, full, x[0], x[1], t_yz))))

    lo1, hi1 = full.scanner1.step_positions[[0, -1]]
    lo2, hi2 = full.scanner2.step_positions[[0, -1]]
    if x0 is None:
        g1 = np.linspace(lo1, hi1, coarse)
        g2 = np.linspace(lo2, hi2, coarse)
        x0 = min(((objective((a, b)), a, b) for a in g1 for b in g2))[1:]
    opt = minimize(objective, x0=list(x0), method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 0.01})
    return (float(np.clip(opt.x[0], lo1, hi1)),
            float(np.clip(opt.x[1], lo2, hi2)))


def observability(p1: np.ndarray, p2: np.ndarray, full: FullCalibration,
                  s1: float, s2: float, h: float = 5.0) -> dict:
    """Gauss-Newton conditioning of the length constraint in (s1, s2):
    eigenvalues of J^T J (descending), their ratio, and the well-observed
    eigenvector (expected ~(1, -1)/sqrt(2), the separation direction)."""
    n = len(p1)
    if n == 0:
        return {"n": 0, "eigvals": (0.0, 0.0), "ratio": float("inf"),
                "observed_dir": (float("nan"), float("nan"))}
    r0 = length_residuals(p1, p2, full, s1, s2)
    j1 = (length_residuals(p1, p2, full, s1 + h, s2) - r0) / h
    j2 = (length_residuals(p1, p2, full, s1, s2 + h) - r0) / h
    J = np.column_stack([j1, j2])
    JtJ = J.T @ J / n
    vals, vecs = np.linalg.eigh(JtJ)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ratio = float(vals[0] / vals[1]) if vals[1] > 0 else float("inf")
    v = vecs[:, 0]
    if v[0] < 0:
        v = -v
    return {"n": int(n), "eigvals": (float(vals[0]), float(vals[1])),
            "ratio": ratio, "observed_dir": (float(v[0]), float(v[1]))}


def refit_translation(p1: np.ndarray, p2: np.ndarray, full: FullCalibration,
                      s1: float, s2: float) -> tuple[float, float]:
    """Median YZ residual (side 1 - side 2) over the current pairs. Only Z is
    meaningful as a global constant; Y is left to the per-scene estimate."""
    if len(p1) == 0:
        return (0.0, 0.0)
    w1 = full.scanner1.transform(p1, s1)
    w2 = full.scanner2.transform(p2, s2)
    d = np.median(w1 - w2, axis=0)
    return (float(d[1]), float(d[2]))


def bootstrap_separation(p1: np.ndarray, p2: np.ndarray,
                         full: FullCalibration, s1: float, s2: float,
                         scene_of_pair: np.ndarray | None = None,
                         n_boot: int = 2000, span: float = 60.0,
                         step: float = 0.25, seed: int = 0) -> np.ndarray:
    """Bootstrap estimates of the separation s1 - s2, with s2 held fixed
    (only the separation is identified). Resamples scenes when
    `scene_of_pair` is given, otherwise pairs. Each replicate minimizes the
    median |length residual| on a 1-D grid around s1."""
    if len(p1) == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    grid = np.arange(s1 - span, s1 + span + step, step)
    resid = np.array([length_residuals(p1, p2, full, g, s2) for g in grid])
    if scene_of_pair is None:
        groups = [np.arange(resid.shape[1])]
    else:
        labels = np.asarray(scene_of_pair)
        groups = [np.flatnonzero(labels == u) for u in np.unique(labels)]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        obj = np.median(np.abs(resid[:, idx]), axis=1)
        out[b] = grid[int(np.argmin(obj))] - s2
    return out
