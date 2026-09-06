"""Cross-view guided re-detection of missed faces.

For each confirmed detection with no cross-view partner, the scene translation
(`pairing.hungarian_aligned`) predicts the partner's position; a local
template match at the partner's angle +/- one step runs there on the
confirmed-erased image. A candidate is kept if it lies within RESCUE_RADIUS_MM,
has enough points and Z extent, clears RESCUE_CCORR_MIN and is not a re-find
of an existing detection. Asymmetric profiles fall back to half-template
matching (left or right half) when no full-template match passes.
"""

from __future__ import annotations

import cv2
import numpy as np

from .ccorr import (
    ANGLE_STEP_DEG,
    OVERLAP_SCORE_WEIGHT,
    WEAK_PEAK_THRESHOLD,
    Detection,
    _attach_points,
    _angle_packet,
    _erase_footprints,
    _overlap_scores,
)
from ..io.dxf_template import Template
from ..io.projection import Projection, project_yz
from ..pairing.common import SYMMETRY_PERIOD_DEG
from ..pairing.hungarian_aligned import pair_hungarian_aligned

RESCUE_RADIUS_MM = 100.0
RESCUE_MIN_POINTS = 500
RESCUE_MIN_Z_EXTENT_MM = 15.0
RESCUE_CCORR_MIN = WEAK_PEAK_THRESHOLD
RESCUE_CCORR_MIN_HALF = 0.30
RESCUE_REFIND_MM = 20.0

# Min Y and Z extent overlap for two detections to be the same face.
REFIND_Y_OVERLAP = 0.5
REFIND_Z_OVERLAP = 0.5


def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Returns the overlap of two intervals as a fraction of the shorter one."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))


def _covers_same_material(det: Detection, other: Detection) -> bool:
    """True if det's Y and Z extents both overlap other's (a re-find)."""
    return (_interval_overlap(det.y_extent_mm, other.y_extent_mm)
            >= REFIND_Y_OVERLAP
            and _interval_overlap(det.z_extent_mm, other.z_extent_mm)
            >= REFIND_Z_OVERLAP)


def _half_template(rot: np.ndarray, keep: str) -> tuple[np.ndarray, float]:
    """Returns (rot with the other half zeroed along width, self-CCorr
    reference for the visible half only)."""
    masked = rot.copy()
    w = masked.shape[1]
    if keep == "left":
        masked[:, w // 2:] = 0
    else:
        masked[:, : w // 2] = 0
    ref = float(np.sum(masked.astype(np.float64) ** 2))
    return masked, ref


def _redetect_at(
    erased: np.ndarray,
    proj: Projection,
    world: np.ndarray,
    template: Template,
    side: int,
    pred_yz: np.ndarray,
    partner_angle_deg: float,
    radius_mm: float,
    allow_half: bool = True,
) -> Detection | None:
    """Returns the best face-like template match near pred_yz, or None."""
    px = proj.padding_px + (pred_yz[0] - proj.min_y) * proj.eff_scale
    py = proj.padding_px + (proj.max_z - pred_yz[1]) * proj.eff_scale
    half_win = int(round(radius_mm * proj.eff_scale + template.diagonal_px / 2.0))
    x0 = max(0, int(round(px)) - half_win)
    y0 = max(0, int(round(py)) - half_win)
    x1 = min(erased.shape[1], int(round(px)) + half_win)
    y1 = min(erased.shape[0], int(round(py)) + half_win)
    crop = erased[y0:y1, x0:x1].astype(np.float32)

    def gate(det: Detection) -> bool:
        _attach_points(det, template, proj, world)
        if det.n_points < RESCUE_MIN_POINTS:
            return False
        if det.z_extent_mm[1] - det.z_extent_mm[0] < RESCUE_MIN_Z_EXTENT_MM:
            return False
        if np.linalg.norm(np.array(det.centroid_yz_mm) - pred_yz) > radius_mm:
            return False
        return True

    best: Detection | None = None
    for da in (-ANGLE_STEP_DEG, 0.0, ANGLE_STEP_DEG):
        angle = (partner_angle_deg + da) % 360.0
        packet = _angle_packet(template, float(angle))
        rot = packet.rot
        th, tw = rot.shape
        if th > crop.shape[0] or tw > crop.shape[1]:
            continue
        match = cv2.matchTemplate(crop, rot.astype(np.float32), cv2.TM_CCORR)
        _, max_val, _, max_loc = cv2.minMaxLoc(match)
        frac = max_val / template.max_val_ref
        if frac < RESCUE_CCORR_MIN:
            continue
        gx, gy = x0 + max_loc[0], y0 + max_loc[1]
        overlap, ring = _overlap_scores(erased, packet, gx, gy)
        det = Detection(
            side=side, angle_deg=float(angle),
            score=(OVERLAP_SCORE_WEIGHT * overlap
                   + (1.0 - OVERLAP_SCORE_WEIGHT) * ring),
            ccorr_frac=float(frac), kind="rescued",
            obb=((gx + tw / 2.0, gy + th / 2.0),
                 (float(template.image.shape[1]),
                  float(template.image.shape[0])),
                 -float(angle)))
        if not gate(det):
            continue
        if best is None or det.ccorr_frac > best.ccorr_frac:
            best = det
    if best is not None or not allow_half:
        return best

    for da in (-ANGLE_STEP_DEG, 0.0, ANGLE_STEP_DEG):
        angle = (partner_angle_deg + da) % 360.0
        packet = _angle_packet(template, float(angle))
        rot = packet.rot
        th, tw = rot.shape
        if th > crop.shape[0] or tw > crop.shape[1]:
            continue
        for keep in ("left", "right"):
            masked, ref = _half_template(rot, keep)
            match = cv2.matchTemplate(crop, masked.astype(np.float32), cv2.TM_CCORR)
            _, max_val, _, max_loc = cv2.minMaxLoc(match)
            frac = max_val / ref
            if frac < RESCUE_CCORR_MIN_HALF:
                continue
            gx, gy = x0 + max_loc[0], y0 + max_loc[1]
            overlap, ring = _overlap_scores(erased, packet, gx, gy)
            det = Detection(
                side=side, angle_deg=float(angle),
                score=(OVERLAP_SCORE_WEIGHT * overlap
                       + (1.0 - OVERLAP_SCORE_WEIGHT) * ring),
                ccorr_frac=float(frac), kind="rescued_half",
                obb=((gx + tw / 2.0, gy + th / 2.0),
                     (float(template.image.shape[1]),
                      float(template.image.shape[0])),
                     -float(angle)))
            if not gate(det):
                continue
            if best is None or det.ccorr_frac > best.ccorr_frac:
                best = det
    return best


def rescue_misses(
    world1: np.ndarray,
    world2: np.ndarray,
    template: Template,
    conf1: list[Detection],
    conf2: list[Detection],
    radius_mm: float = RESCUE_RADIUS_MM,
    period_deg: float = SYMMETRY_PERIOD_DEG,
) -> tuple[list[Detection], list[Detection], dict]:
    """Returns (rescued_side1, rescued_side2, diagnostics). `period_deg` is
    used by the internal pairing step only."""
    diagnostics: dict = {"t_yz_mm": None, "unmatched_partners": 0,
                         "attempted": 0}
    if not conf1 or not conf2:
        return [], [], diagnostics

    pairing = pair_hungarian_aligned(conf1, conf2, period_deg=period_deg)
    t = np.array(pairing.diagnostics["t_yz_mm"], dtype=np.float64)
    diagnostics["t_yz_mm"] = [float(t[0]), float(t[1])]
    diagnostics["unmatched_partners"] = (len(pairing.unmatched_1)
                                         + len(pairing.unmatched_2))
    if not pairing.unmatched_1 and not pairing.unmatched_2:
        return [], [], diagnostics

    padding = template.default_padding
    rescued: dict[int, list[Detection]] = {1: [], 2: []}
    sides = (
        (1, world1, conf1, conf2, pairing.unmatched_2, +1.0),  # pred = c2 + t
        (2, world2, conf2, conf1, pairing.unmatched_1, -1.0),  # pred = c1 - t
    )
    for side, world, conf_here, partners, unmatched, sgn in sides:
        if not unmatched:
            continue
        proj = project_yz(world, padding_px=padding)
        kept_world = world[proj.keep_mask]
        erased = _erase_footprints(proj.image, template, conf_here)
        for uo in unmatched:
            partner = partners[uo]
            pred = np.array(partner.centroid_yz_mm) + sgn * t
            diagnostics["attempted"] += 1
            det = _redetect_at(erased, proj, kept_world, template, side, pred,
                               partner.angle_deg, radius_mm,
                               allow_half=(period_deg >= 360.0))
            if det is None:
                continue
            taken = conf_here + rescued[side]
            if any(np.linalg.norm(np.array(det.centroid_yz_mm)
                                  - np.array(k.centroid_yz_mm))
                   < RESCUE_REFIND_MM or _covers_same_material(det, k)
                   for k in taken):
                continue
            rescued[side].append(det)
    return rescued[1], rescued[2], diagnostics


def merge_rescued_into_confirmed(data: dict, rescued: dict) -> dict:
    """Merges rescued faces into each scene's confirmed detections, side by side.

    `data`/`rescued` are detection-report dicts of the form
    `{"scenes": {sid: {"side1": {"confirmed": [...], "lost": [...]},
    "side2": {...}}}, ...}`. Returns just the merged `scenes` dict.
    """
    merged = {}
    for sid, scene in data["scenes"].items():
        r = rescued["scenes"].get(sid, {})
        merged[sid] = {
            "side1": {"confirmed": scene["side1"]["confirmed"]
                      + r.get("rescued_1", []),
                      "lost": scene["side1"]["lost"]},
            "side2": {"confirmed": scene["side2"]["confirmed"]
                      + r.get("rescued_2", []),
                      "lost": scene["side2"]["lost"]},
        }
    return merged
