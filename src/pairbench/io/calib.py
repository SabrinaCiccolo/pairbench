"""Offline dual-scanner calibration.

Loads calibration.json and applies p_world = R @ p_scanner + t per scanner,
with one fixed translation knot per scanner: YZ is metric, world-X carries a
constant bias. See calibration.json's `_comment`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "calibration.json"

MAX_DISTANCE_MM = 850.0  # radial pre-filter from scanner origin, scanner frame


@dataclass(frozen=True)
class ScannerCalibration:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Scanner-frame (N,3) mm -> world-frame (N,3) mm."""
        return points @ self.R.T + self.t


@dataclass(frozen=True)
class DualCalibration:
    scanner1: ScannerCalibration
    scanner2: ScannerCalibration
    max_distance_mm: float

    def for_side(self, side: int) -> ScannerCalibration:
        if side == 1:
            return self.scanner1
        if side == 2:
            return self.scanner2
        raise ValueError(f"side must be 1 or 2, got {side}")


def load_calibration(path: Path | str = DEFAULT_CALIBRATION_PATH) -> DualCalibration:
    raw = json.loads(Path(path).read_text())
    scanners = {}
    for key in ("scanner1", "scanner2"):
        entry = raw[key]
        R = np.asarray(entry["R"], dtype=np.float64)
        idx = int(entry["step_idx"])
        t = np.array(
            [entry["t_knots_x"][idx], entry["t_knots_y"][idx], entry["t_knots_z"][idx]],
            dtype=np.float64,
        )
        scanners[key] = ScannerCalibration(R=R, t=t)
    return DualCalibration(
        scanner1=scanners["scanner1"],
        scanner2=scanners["scanner2"],
        max_distance_mm=float(raw.get("max_distance_mm", MAX_DISTANCE_MM)),
    )


def radial_filter(points: np.ndarray, max_distance_mm: float = MAX_DISTANCE_MM) -> np.ndarray:
    """Keep points within max_distance_mm of the scanner origin (scanner
    frame), removing conveyor structure and background."""
    dist_sq = np.einsum("ij,ij->i", points, points)
    return points[dist_sq < max_distance_mm * max_distance_mm]
