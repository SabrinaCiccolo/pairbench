"""Scoring a synthetic scene against its exact ground truth.

`detection_report` gives localization error in mm; `pairing_report` scores
correspondence against true bar identity. Detections are matched to true
faces by Hungarian assignment on YZ distance under a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detect.ccorr import Detection, sort_left_to_right
from ..detect.pool import detect_faces_pool
from ..io.dxf_template import render_template
from ..io.loader import load_scene
from ..pairing import STRATEGIES
from ..symmetry import period_for_dxf
from .generator import load_truth

MATCH_GATE_MM = 25.0   # farther than this from every true face = false positive


@dataclass
class SceneDetections:
    truth: dict
    confirmed: dict   # side -> list[Detection], sorted left to right
    lost: dict
    bar_of: dict      # side -> list[int|None], true bar index per detection


def detect_synth_scene(scene_dir, gate_mm: float = MATCH_GATE_MM) -> SceneDetections:
    """Runs the pool detector on a synthetic scene and labels each
    detection with the bar it landed on."""
    truth = load_truth(scene_dir)
    template = render_template(truth["dxf"])
    clouds = load_scene(scene_dir)
    confirmed, lost, bar_of = {}, {}, {}
    for side, world in ((1, clouds.side1_world), (2, clouds.side2_world)):
        dets, lost_dets, _ = detect_faces_pool(world, template, side)
        dets = sort_left_to_right(dets)
        confirmed[side] = dets
        lost[side] = lost_dets
        bar_of[side] = assign_to_truth(dets, truth["faces"][f"side{side}"], gate_mm)
    return SceneDetections(truth=truth, confirmed=confirmed, lost=lost, bar_of=bar_of)


def assign_to_truth(dets: list[Detection], faces: list[dict], gate_mm: float):
    """Returns a bar index (or None) per detection via Hungarian match on YZ distance."""
    out: list = [None] * len(dets)
    if not dets or not faces:
        return out
    d = np.array([det.centroid_yz_mm for det in dets], dtype=float)
    f = np.array([[fa["y_mm"], fa["z_mm"]] for fa in faces], dtype=float)
    cost = np.linalg.norm(d[:, None, :] - f[None, :, :], axis=2)
    big = gate_mm * 10.0
    rows, cols = linear_sum_assignment(np.where(cost <= gate_mm, cost, big))
    for i, j in zip(rows, cols):
        if cost[i, j] <= gate_mm:
            out[i] = int(faces[j]["bar"])
    return out


def detection_report(sd: SceneDetections) -> dict:
    """Returns per-scene detection counts and localization error (mm)."""
    tp = fp = fn = 0
    errors, dy, dz = [], [], []
    for side in (1, 2):
        dets = sd.confirmed[side]
        faces = sd.truth["faces"][f"side{side}"]
        matched = [b for b in sd.bar_of[side] if b is not None]
        tp += len(matched)
        fp += len(dets) - len(matched)
        fn += len(faces) - len(matched)
        by_bar = {fa["bar"]: fa for fa in faces}
        for det, bar in zip(dets, sd.bar_of[side]):
            if bar is None:
                continue
            fa = by_bar[bar]
            dy.append(float(det.centroid_yz_mm[0] - fa["y_mm"]))
            dz.append(float(det.centroid_yz_mm[1] - fa["z_mm"]))
            errors.append(float(np.hypot(dy[-1], dz[-1])))
    return {"tp": tp, "fp": fp, "fn": fn, "loc_errors_mm": errors,
            "loc_dy_mm": dy, "loc_dz_mm": dz,
            "n_faces": sum(len(sd.truth["faces"][f"side{s}"]) for s in (1, 2))}


def _min_margin(diag) -> float | None:
    """Smallest per-pair assignment margin in the scene, or None."""
    m = diag.get("pair_margins") or []
    return float(min(m)) if m else None


def _t_y(diag) -> float | None:
    """The arm's fitted cross-view translation along Y (mm), or None."""
    t = diag.get("t_yz_mm")
    return float(t[0]) if t else None


def pairing_report(sd: SceneDetections, arms=None) -> dict:
    """Per-arm correspondence scores against true bar identity: `correct`
    (same bar), `resolved` (both sides on a true bar), `n_pairable` (bars
    detected on both sides)."""
    arms = arms or list(STRATEGIES)
    period = period_for_dxf(sd.truth["dxf"])
    bars1 = {b for b in sd.bar_of[1] if b is not None}
    bars2 = {b for b in sd.bar_of[2] if b is not None}
    pairable = bars1 & bars2
    out = {"n_pairable": len(pairable)}
    for arm in arms:
        res = STRATEGIES[arm](sd.confirmed[1], sd.confirmed[2], period_deg=period)
        correct = resolved = 0
        for i, j in res.pairs:
            b1, b2 = sd.bar_of[1][i], sd.bar_of[2][j]
            if b1 is None or b2 is None:
                continue
            resolved += 1
            correct += int(b1 == b2)
        diag = res.diagnostics
        out[arm] = {
            "correct": correct, "resolved": resolved, "n_pairs": len(res.pairs),
            "n_elevated_1": diag.get("n_elevated_1"),
            "n_elevated_2": diag.get("n_elevated_2"),
            "elevation_corroborated": diag.get("elevation_corroborated"),
            "moves_with_bundle": diag.get("moves_with_bundle"),
            "crossing": bool(diag.get("crossing_hypothesis", False)),
            "min_margin": _min_margin(diag),
            "t_y_mm": _t_y(diag),
        }
    return out
