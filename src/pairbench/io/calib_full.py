"""Full multi-knot calibration: interpolated p_world = R(s)*p + t(s).

Loads data/calibration_full.json: per scanner, rotations and translations at
a set of carriage-X step positions. Translation is interpolated linearly and
rotation by SLERP, clamped outside the calibrated range.

The per-scan carriage position is not stored in the calibration file;
`pairbench.experiments.fit_step_positions` estimates it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DEFAULT_FULL_CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "data" / "calibration_full.json"


@dataclass(frozen=True)
class ScannerSpline:
    step_positions: np.ndarray   # (K,) ascending, mm
    translations: np.ndarray     # (K,3) mm
    slerp: Slerp

    def transform(self, points: np.ndarray, s: float) -> np.ndarray:
        """Scanner-frame (N,3) mm -> world (N,3) mm at carriage position s."""
        s = float(np.clip(s, self.step_positions[0], self.step_positions[-1]))
        t = np.array([np.interp(s, self.step_positions, self.translations[:, k])
                      for k in range(3)])
        R = self.slerp([s])[0].as_matrix()
        return points @ R.T + t


@dataclass(frozen=True)
class FullCalibration:
    scanner1: ScannerSpline
    scanner2: ScannerSpline

    def for_side(self, side: int) -> ScannerSpline:
        if side == 1:
            return self.scanner1
        if side == 2:
            return self.scanner2
        raise ValueError(f"side must be 1 or 2, got {side}")


def load_full_calibration(
    path: Path | str = DEFAULT_FULL_CALIBRATION_PATH,
) -> FullCalibration:
    raw = json.loads(Path(path).read_text())
    scanners = {}
    for key in ("scanner1", "scanner2"):
        entry = raw[key]
        steps = np.asarray(entry["step_positions"], dtype=np.float64)
        order = np.argsort(steps)
        steps = steps[order]
        trans = np.asarray(entry["translations"], dtype=np.float64)[order]
        rots = Rotation.from_matrix(
            np.asarray(entry["rotations"], dtype=np.float64)[order]
        )
        scanners[key] = ScannerSpline(
            step_positions=steps,
            translations=trans,
            slerp=Slerp(steps, rots),
        )
    return FullCalibration(scanner1=scanners["scanner1"],
                           scanner2=scanners["scanner2"])
