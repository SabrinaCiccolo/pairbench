"""Scene loading: PLY pair -> radial filter -> calibration -> world clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import data_root as _data_root
from .calib import DualCalibration, load_calibration, radial_filter
from .ply_io import read_ply_points

DATA_ROOT = _data_root()

# family folder -> DXF stem of its cross-section (next to the folder's scenes)
FAMILY_DXF = {
    "l-profile": "l-profile",
    "l-profile-reshoot": "l-profile",
    "square-profile": "square-profile",
    "complex-profile": "complex-profile",
    "heavy-profile": "heavy-profile",
}


def dxf_stem_for_scene(scene_id: str) -> str:
    """scene_id's leading family folder -> DXF stem of its cross-section."""
    family = scene_id.split("/")[0]
    return FAMILY_DXF[family]


@dataclass(frozen=True)
class SceneClouds:
    scene_dir: Path
    side1_world: np.ndarray  # (N,3) mm, world frame (X not metric truth)
    side2_world: np.ndarray
    n_raw_1: int
    n_raw_2: int


def load_scene(
    scene_dir: Path | str,
    calib: DualCalibration | None = None,
    apply_radial_filter: bool = True,
) -> SceneClouds:
    scene_dir = Path(scene_dir)
    if calib is None:
        calib = load_calibration()
    worlds = {}
    raw_counts = {}
    for side in (1, 2):
        ply = scene_dir / f"scanner-{side}.ply"
        if not ply.is_file():
            raise FileNotFoundError(ply)
        pts = read_ply_points(ply)
        raw_counts[side] = len(pts)
        if apply_radial_filter:
            pts = radial_filter(pts, calib.max_distance_mm)
        worlds[side] = calib.for_side(side).apply(pts)
    return SceneClouds(
        scene_dir=scene_dir,
        side1_world=worlds[1],
        side2_world=worlds[2],
        n_raw_1=raw_counts[1],
        n_raw_2=raw_counts[2],
    )
