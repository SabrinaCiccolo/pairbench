"""PLY reading (open3d). Scene PLYs are ~47k points, mm, scanner frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply_points(path: Path | str) -> np.ndarray:
    """Read a PLY file -> (N,3) float64 array (mm, scanner frame)."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"no points read from {path}")
    return pts

