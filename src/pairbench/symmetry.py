"""Rotational symmetry period of a profile cross-section, measured from its DXF.

The pairing cost folds the cross-view angle difference into this period. The
cross-section is rasterized at RENDER_SCALE and rotated by 360/k about its
centroid; the smallest period whose mask IoU passes the threshold wins.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .io.dxf_template import load_loops
from .io.loader import dxf_stem_for_scene
from .io.projection import RENDER_SCALE

# k = number of rotational folds tested; 1 (i.e. 360 deg) is the fallback.
SYMMETRY_FOLD_CANDIDATES = (2, 3, 4, 5, 6, 8, 12)
SYMMETRY_IOU_THRESHOLD = 0.90


def _rasterize_centered(loops_mm: list[np.ndarray], scale: float) -> np.ndarray:
    """Filled cross-section mask on a square canvas, area centroid centred,
    large enough for any rotation about the centre."""
    all_pts = np.vstack(loops_mm)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    extent_px = (max_xy - min_xy) * scale
    side = int(np.ceil(np.hypot(*extent_px))) + 4  # diagonal fits at any angle

    polys = []
    for loop in loops_mm:
        px = (loop[:, 0] - min_xy[0]) * scale
        py = (loop[:, 1] - min_xy[1]) * scale
        polys.append(np.round(np.column_stack([px, py])).astype(np.int32))
    rough = np.zeros((side, side), dtype=np.uint8)
    cv2.fillPoly(rough, polys, 255)  # even-odd: nested loops become holes

    m = cv2.moments(rough, binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("empty cross-section raster")
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    shift = np.float32([[1, 0, side / 2.0 - cx], [0, 1, side / 2.0 - cy]])
    return cv2.warpAffine(rough, shift, (side, side), flags=cv2.INTER_NEAREST)


def _self_iou_at(mask: np.ndarray, angle_deg: float) -> float:
    side = mask.shape[0]
    m = cv2.getRotationMatrix2D((side / 2.0 - 0.5, side / 2.0 - 0.5), angle_deg, 1.0)
    rot = cv2.warpAffine(mask, m, (side, side), flags=cv2.INTER_NEAREST)
    base_b = mask > 127
    rot_b = rot > 127
    union = np.logical_or(base_b, rot_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(base_b, rot_b).sum() / union)


def measure_symmetry_period_deg(
    loops_mm: list[np.ndarray],
    scale: float = RENDER_SCALE,
    iou_threshold: float = SYMMETRY_IOU_THRESHOLD,
) -> tuple[float, dict[int, float]]:
    """(period_deg, {k: self-IoU at 360/k}). Smallest passing period wins."""
    mask = _rasterize_centered(loops_mm, scale)
    ious = {k: _self_iou_at(mask, 360.0 / k) for k in SYMMETRY_FOLD_CANDIDATES}
    passing = [k for k in SYMMETRY_FOLD_CANDIDATES if ious[k] >= iou_threshold]
    period = 360.0 / max(passing) if passing else 360.0
    return period, ious


@lru_cache(maxsize=None)
def period_for_dxf(dxf_stem: str) -> float:
    """Rotational symmetry period (deg) of a profile, cached per DXF stem."""
    period, _ = measure_symmetry_period_deg(load_loops(dxf_stem))
    return period


def period_for_scene(scene_id: str) -> float:
    """Symmetry period of the profile the scene's family is made of."""
    return period_for_dxf(dxf_stem_for_scene(scene_id))
