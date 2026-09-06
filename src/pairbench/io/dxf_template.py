"""DXF cross-section -> filled template image for 2D matching.

DXF entities are flattened and chained into closed loops (nested loops become
holes), rasterized at RENDER_SCALE px/mm and downscaled by SCALE_FACTOR to
2.5 px/mm. The template's self-correlation is stored; matching thresholds are
fractions of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import ezdxf
import numpy as np

from .projection import RENDER_SCALE, SCALE_FACTOR

DXF_DIR = Path(__file__).resolve().parents[3] / "data" / "dxf"

_CHAIN_TOL_MM = 0.05      # endpoint snap distance when chaining segments
_ARC_SAGITTA_MM = 0.02    # flattening tolerance for arcs/splines
_TEMPLATE_CACHE_MAX = 8   # distinct profiles kept cached at once


@dataclass(frozen=True)
class Template:
    image: np.ndarray        # uint8 grayscale, effective scale (2.5 px/mm)
    max_val_ref: float       # template self-correlation (CCorr peak)
    loops_mm: list           # closed loops, list of (N,2) float arrays (mm)
    eff_scale: float         # px/mm

    @property
    def diagonal_px(self) -> float:
        h, w = self.image.shape
        return float(np.hypot(h, w))

    @property
    def default_padding(self) -> int:
        """Projection padding so any template rotation fits at the border."""
        return int(np.ceil(self.diagonal_px / 2.0)) + 1


def _flatten_entity(entity) -> np.ndarray | None:
    """Entity -> (N,2) polyline in mm, or None for unsupported types."""
    kind = entity.dxftype()
    if kind == "LINE":
        s, e = entity.dxf.start, entity.dxf.end
        return np.array([[s.x, s.y], [e.x, e.y]], dtype=np.float64)
    if kind in ("ARC", "CIRCLE", "ELLIPSE", "SPLINE"):
        pts = [(v.x, v.y) for v in entity.flattening(_ARC_SAGITTA_MM)]
        return np.asarray(pts, dtype=np.float64) if len(pts) >= 2 else None
    if kind == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in entity.get_points()]
        arr = np.asarray(pts, dtype=np.float64)
        if entity.closed and len(arr) >= 2:
            arr = np.vstack([arr, arr[:1]])
        return arr if len(arr) >= 2 else None
    if kind == "POLYLINE":
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        arr = np.asarray(pts, dtype=np.float64)
        if entity.is_closed and len(arr) >= 2:
            arr = np.vstack([arr, arr[:1]])
        return arr if len(arr) >= 2 else None
    return None


def _chain_loops(segments: list[np.ndarray], tol: float = _CHAIN_TOL_MM) -> list[np.ndarray]:
    """Chain open segments into closed loops by endpoint proximity."""
    loops: list[np.ndarray] = []
    open_segs: list[np.ndarray] = []
    for seg in segments:
        if len(seg) >= 3 and np.linalg.norm(seg[0] - seg[-1]) < tol:
            loops.append(seg[:-1])
        else:
            open_segs.append(seg)

    while open_segs:
        chain = open_segs.pop(0)
        extended = True
        while extended:
            extended = False
            if np.linalg.norm(chain[0] - chain[-1]) < tol and len(chain) >= 3:
                break
            for i, seg in enumerate(open_segs):
                if np.linalg.norm(chain[-1] - seg[0]) < tol:
                    chain = np.vstack([chain, seg[1:]])
                elif np.linalg.norm(chain[-1] - seg[-1]) < tol:
                    chain = np.vstack([chain, seg[::-1][1:]])
                elif np.linalg.norm(chain[0] - seg[-1]) < tol:
                    chain = np.vstack([seg, chain[1:]])
                elif np.linalg.norm(chain[0] - seg[0]) < tol:
                    chain = np.vstack([seg[::-1], chain[1:]])
                else:
                    continue
                open_segs.pop(i)
                extended = True
                break
        if np.linalg.norm(chain[0] - chain[-1]) < tol and len(chain) >= 3:
            loops.append(chain[:-1] if np.allclose(chain[0], chain[-1]) else chain)
        else:
            raise ValueError(
                f"open chain left after loop chaining ({len(chain)} pts, "
                f"gap {np.linalg.norm(chain[0] - chain[-1]):.3f} mm)"
            )
    return loops


def load_loops(dxf_stem: str, dxf_dir: Path = DXF_DIR) -> list[np.ndarray]:
    doc = ezdxf.readfile(dxf_dir / f"{dxf_stem}.dxf")
    segments = []
    for entity in doc.modelspace():
        arr = _flatten_entity(entity)
        if arr is not None:
            segments.append(arr)
    if not segments:
        raise ValueError(f"no drawable entities in {dxf_stem}.dxf")
    return _chain_loops(segments)


def loops_to_polys(loops_mm: list[np.ndarray], min_xy: np.ndarray, height: int,
                    scale: float) -> list[np.ndarray]:
    """mm-space closed loops -> int32 pixel polygons for cv2.fillPoly,
    image-Y flipped (DXF y grows up, image y grows down)."""
    polys = []
    for loop in loops_mm:
        px = (loop[:, 0] - min_xy[0]) * scale
        py = (height - 1) - (loop[:, 1] - min_xy[1]) * scale
        polys.append(np.round(np.column_stack([px, py])).astype(np.int32))
    return polys


@lru_cache(maxsize=_TEMPLATE_CACHE_MAX)
def render_template(
    dxf_stem: str,
    dxf_dir: Path = DXF_DIR,
    render_scale: float = RENDER_SCALE,
    scale_factor: float = SCALE_FACTOR,
) -> Template:
    loops = load_loops(dxf_stem, dxf_dir)
    all_pts = np.vstack(loops)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    width = int(np.ceil((max_xy[0] - min_xy[0]) * render_scale)) + 1
    height = int(np.ceil((max_xy[1] - min_xy[1]) * render_scale)) + 1

    polys = loops_to_polys(loops, min_xy, height, render_scale)

    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, polys, 255)  # even-odd: nested loops become holes

    image = cv2.resize(
        canvas, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
    )
    ref = cv2.matchTemplate(
        image.astype(np.float32), image.astype(np.float32), cv2.TM_CCORR
    )
    max_val_ref = float(ref.max())
    return Template(
        image=image,
        max_val_ref=max_val_ref,
        loops_mm=[loop - min_xy for loop in loops],
        eff_scale=render_scale * scale_factor,
    )
