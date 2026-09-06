"""Shared overlay drawing for the experiment scripts."""

from __future__ import annotations

import cv2
import numpy as np


def draw_obbs(vis: np.ndarray, dets, color, thickness: int,
              shift_px: int = 0, prefer_display: bool = True) -> None:
    """Draws oriented bounding boxes of Detection objects or their dicts,
    preferring `obb_display` over `obb` when `prefer_display`. `shift_px`
    offsets every box horizontally."""
    for d in dets:
        if isinstance(d, dict):
            obb = (d.get("obb_display") or d["obb"]) if prefer_display else d["obb"]
        else:
            obb = (d.obb_display or d.obb) if prefer_display else d.obb
        box = cv2.boxPoints((tuple(obb[0]), tuple(obb[1]),
                             obb[2])).astype(np.int32)
        box[:, 0] += shift_px
        cv2.polylines(vis, [box], True, color, thickness)


def shared_y_origin_shifts(projs) -> list[int]:
    """Per-projection left-pad (px) so stacked panels share one world-Y origin."""
    origin_y = min(p.min_y for p in projs)
    return [int(round((p.min_y - origin_y) * p.eff_scale)) for p in projs]


def pad_panel_left(vis: np.ndarray, shift_px: int) -> np.ndarray:
    if not shift_px:
        return vis
    return cv2.copyMakeBorder(vis, 0, 0, shift_px, 0,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    """Pad BGR panels to a common size and stack them vertically."""
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    padded = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, w - p.shape[1],
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
              for p in panels]
    return np.vstack(padded)
