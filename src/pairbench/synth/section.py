"""DXF cross-section to extruded prism mesh.

The section (holes included, even-odd fill) is triangulated by Delaunay over
the resampled boundary plus an interior grid, keeping triangles whose centroid
is inside the filled mask; the lateral surface is a quad strip per loop.

Local frame: +X = extrusion axis, +Y = section x (world Y), +Z = section y
(world Z); origin at the section's area centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial import Delaunay

from ..io.dxf_template import load_loops

BOUNDARY_STEP_MM = 0.5     # arc-length resampling of the section boundary
INTERIOR_STEP_MM = 1.0     # grid pitch for interior triangulation points
INTERIOR_MARGIN_MM = 0.45  # keep interior points this far off the boundary
RASTER_SCALE = 10.0        # px/mm for the even-odd fill mask (0.1 mm)


@dataclass(frozen=True)
class SectionMesh:
    """Triangulated cross-section in section coordinates, centroid-centered."""

    vertices: np.ndarray   # (V,2) mm, (section-x, section-y), centroid at 0
    triangles: np.ndarray  # (T,3) int
    loops: list            # resampled closed loops, (Ni,2) mm, centroid-centered
    area_mm2: float        # exact triangulated area
    bbox_mm: tuple         # (width, height) of the section bounding box


def resample_loop(loop: np.ndarray, step_mm: float = BOUNDARY_STEP_MM) -> np.ndarray:
    """Resamples a closed polyline by arc length. Returns an open ring."""
    ring = np.vstack([loop, loop[:1]])
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return loop
    n = max(3, int(np.ceil(total / step_mm)))
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2))
    for axis in (0, 1):
        out[:, axis] = np.interp(targets, s, ring[:, axis])
    return out


def _fill_mask(loops: list[np.ndarray], scale: float = RASTER_SCALE):
    """Returns (even-odd filled mask of the section, mm-to-pixel origin, height)."""
    pts = np.vstack(loops)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    w = int(np.ceil((max_xy[0] - min_xy[0]) * scale)) + 3
    h = int(np.ceil((max_xy[1] - min_xy[1]) * scale)) + 3
    canvas = np.zeros((h, w), dtype=np.uint8)
    polys = []
    for loop in loops:
        px = (loop[:, 0] - min_xy[0]) * scale + 1
        py = (h - 2) - (loop[:, 1] - min_xy[1]) * scale
        polys.append(np.round(np.column_stack([px, py])).astype(np.int32))
    cv2.fillPoly(canvas, polys, 255)
    return canvas, min_xy, h


def _inside(mask, min_xy, h, xy: np.ndarray, scale: float = RASTER_SCALE) -> np.ndarray:
    col = np.round((xy[:, 0] - min_xy[0]) * scale + 1).astype(np.int64)
    row = np.round((h - 2) - (xy[:, 1] - min_xy[1]) * scale).astype(np.int64)
    ok = (col >= 0) & (col < mask.shape[1]) & (row >= 0) & (row < mask.shape[0])
    out = np.zeros(len(xy), dtype=bool)
    out[ok] = mask[row[ok], col[ok]] > 0
    return out


def build_section_mesh(
    dxf_stem: str,
    boundary_step_mm: float = BOUNDARY_STEP_MM,
    interior_step_mm: float = INTERIOR_STEP_MM,
) -> SectionMesh:
    """Builds a triangulated, centroid-centered cross-section from a DXF stem."""
    loops = [resample_loop(loop, boundary_step_mm) for loop in load_loops(dxf_stem)]
    mask, min_xy, h = _fill_mask(loops)

    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    pts = np.vstack(loops)
    gx = np.arange(pts[:, 0].min(), pts[:, 0].max() + interior_step_mm, interior_step_mm)
    gy = np.arange(pts[:, 1].min(), pts[:, 1].max() + interior_step_mm, interior_step_mm)
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    col = np.clip(np.round((grid[:, 0] - min_xy[0]) * RASTER_SCALE + 1).astype(np.int64),
                  0, mask.shape[1] - 1)
    row = np.clip(np.round((h - 2) - (grid[:, 1] - min_xy[1]) * RASTER_SCALE).astype(np.int64),
                  0, mask.shape[0] - 1)
    keep = dist[row, col] > INTERIOR_MARGIN_MM * RASTER_SCALE
    verts = np.vstack([pts, grid[keep]])

    tri = Delaunay(verts)
    centres = verts[tri.simplices].mean(axis=1)
    tris = tri.simplices[_inside(mask, min_xy, h, centres)]
    if len(tris) == 0:
        raise ValueError(f"{dxf_stem}: section triangulation is empty")

    t = verts[tris]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    tri_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    area_mm2 = float(tri_area.sum())
    centroid = (t.mean(axis=1) * tri_area[:, None]).sum(axis=0) / tri_area.sum()

    raster_area = float((mask > 0).sum()) / RASTER_SCALE**2
    if not (0.90 * raster_area <= area_mm2 <= raster_area):
        raise ValueError(
            f"{dxf_stem}: triangulated area {area_mm2:.1f} mm2 disagrees with "
            f"the even-odd fill {raster_area:.1f} mm2 -- a hole was probably lost")

    bbox = tuple(pts.max(axis=0) - pts.min(axis=0))
    return SectionMesh(
        vertices=verts - centroid,
        triangles=tris,
        loops=[loop - centroid for loop in loops],
        area_mm2=area_mm2,
        bbox_mm=(float(bbox[0]), float(bbox[1])),
    )


@lru_cache(maxsize=8)
def section_mesh(dxf_stem: str) -> SectionMesh:
    """Returns a cached build_section_mesh result."""
    return build_section_mesh(dxf_stem)


def extrude_prism(section: SectionMesh, length_mm: float):
    """Returns (vertices (V,3), triangles (T,3)) for a closed prism mesh in
    bar-local coordinates. Local +X is the extrusion axis, caps at
    x = +/- length/2; local (Y,Z) carry the section, centroid at the origin.
    Face winding is not made consistent.
    """
    half = length_mm / 2.0
    v2 = section.vertices
    n = len(v2)
    verts = np.vstack([
        np.column_stack([np.full(n, +half), v2]),
        np.column_stack([np.full(n, -half), v2]),
    ])
    tris = [section.triangles, section.triangles[:, ::-1] + n]

    for loop in section.loops:
        base = len(verts)
        m = len(loop)
        ring = np.vstack([
            np.column_stack([np.full(m, +half), loop]),
            np.column_stack([np.full(m, -half), loop]),
        ])
        verts = np.vstack([verts, ring])
        i = np.arange(m)
        j = (i + 1) % m
        a, b = base + i, base + j
        c, d = base + m + i, base + m + j
        tris.append(np.column_stack([a, b, c]))
        tris.append(np.column_stack([b, d, c]))

    return verts, np.vstack(tris).astype(np.int32)
