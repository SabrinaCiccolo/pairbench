"""Scene generator: bundles of posed bars, rendered to two scanner PLYs.

Scenes are built in the reconstructed frame: the offline calibration's
world-X bias makes a 6005 mm bar appear `APPARENT_LENGTH_MM` long, so the twin
does not exercise self-calibration or the length check.

- Bars are rigid prisms placed by a 6-DOF pose (centre = section area
  centroid; yaw about Z, pitch about Y, roll about the bar axis).
- A crossed bar has a small yaw and is raised by one section height.
- Side 2 sees the scene translated along Y by `transport_y_mm`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..io.calib import DualCalibration, load_calibration
from .section import extrude_prism, section_mesh
from .sensor import SensorModel, raycast, sensor_for_side

# Reconstructed (not physical) end-face separation: 6005 mm nominal minus the
# offline calibration's world-X bias.
APPARENT_LENGTH_MM = 4928.0

# Bundle placement defaults, from the real scenes' annotated geometry.
BED_Z_MM = 918.9
BUNDLE_Y_MM = 1727.0
TRANSPORT_Y_MM = 11.2

# Section roll giving each family's resting orientation on the conveyor.
FAMILY_BASE_ROLL_DEG = {
    "l-profile": 180.0,       # flange down, box above it
    "heavy-profile": 90.0,    # notches left/right, not top/bottom
    "square-profile": 0.0,
    "complex-profile": 0.0,
}

# Median consecutive-bar Y gap per family.
FAMILY_PITCH_MM = {
    "l-profile": 69.9,
    "square-profile": 42.1,
    "complex-profile": 178.4,
    "heavy-profile": 118.8,
}


@dataclass(frozen=True)
class BarPose:
    """6-DOF pose of one bar, translation = its section area centroid."""

    y_mm: float
    z_mm: float
    x_mm: float = 0.0
    yaw_deg: float = 0.0     # about world Z: lying diagonally
    pitch_deg: float = 0.0   # about world Y: nose up/down
    roll_deg: float = 0.0    # about the bar's own axis: the face's twist

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x_mm, self.y_mm, self.z_mm])

    def rotation(self) -> np.ndarray:
        cy, sy = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        cp, sp = np.cos(np.radians(self.pitch_deg)), np.sin(np.radians(self.pitch_deg))
        cr, sr = np.cos(np.radians(self.roll_deg)), np.sin(np.radians(self.roll_deg))
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
        rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx

    def to_world(self, local_pts: np.ndarray) -> np.ndarray:
        return local_pts @ self.rotation().T + self.centre

    def face_centre(self, side: int, length_mm: float) -> np.ndarray:
        """Returns the world position of the end face this side looks at."""
        half = length_mm / 2.0 if side == 1 else -length_mm / 2.0
        return self.to_world(np.array([[half, 0.0, 0.0]]))[0]


@dataclass
class SyntheticScene:
    scene_id: str
    dxf_stem: str
    bars: list = field(default_factory=list)
    transport_y_mm: float = TRANSPORT_Y_MM
    transport_z_mm: float = 0.0
    length_mm: float = APPARENT_LENGTH_MM
    seed: int = 0
    params: dict = field(default_factory=dict)


def scene_mesh(scene: SyntheticScene):
    """Returns the whole bundle as one world-frame triangle soup plus a
    per-triangle bar-index owner array."""
    sec = section_mesh(scene.dxf_stem)
    verts_l, tris_l = extrude_prism(sec, scene.length_mm)
    verts, tris, owner = [], [], []
    for k, bar in enumerate(scene.bars):
        base = sum(len(v) for v in verts)
        verts.append(bar.to_world(verts_l))
        tris.append(tris_l + base)
        owner.append(np.full(len(tris_l), k))
    return np.vstack(verts), np.vstack(tris), np.concatenate(owner)


def face_truth(scene: SyntheticScene) -> dict:
    """Returns exact per-side, per-bar face-center ground truth in world mm.

    Side 2 carries the transport offset, since that is where the bundle is
    when the second scanner fires.
    """
    shift = np.array([0.0, scene.transport_y_mm, scene.transport_z_mm])
    out = {}
    for side in (1, 2):
        faces = []
        for k, bar in enumerate(scene.bars):
            c = bar.face_centre(side, scene.length_mm)
            if side == 2:
                c = c + shift
            faces.append({
                "bar": k,
                "y_mm": float(c[1]),
                "z_mm": float(c[2]),
                "x_mm": float(c[0]),
                "roll_deg": float(bar.roll_deg),
                "yaw_deg": float(bar.yaw_deg),
            })
        out[f"side{side}"] = faces
    return out


def render_scene(
    scene: SyntheticScene,
    out_dir: Path | str,
    calib: DualCalibration | None = None,
    sensor_overrides: dict | None = None,
) -> Path:
    """Raycasts both views and writes scanner-N.ply (scanner frame, readable
    by `io.loader.load_scene`) plus truth.json into out_dir."""
    import open3d as o3d

    calib = calib or load_calibration()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verts, tris, _ = scene_mesh(scene)
    shift = np.array([0.0, scene.transport_y_mm, scene.transport_z_mm])

    for side in (1, 2):
        rng = np.random.default_rng((scene.seed, side))
        sensor: SensorModel = sensor_for_side(side, **(sensor_overrides or {}))
        cal = calib.for_side(side)
        # Side 2 sees the bundle after it has moved: cast from an origin
        # shifted the other way and add the shift back to the hits.
        origin = cal.t - (shift if side == 2 else 0.0)
        pts_scanner = raycast(verts, tris, cal.R, origin, sensor, rng)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_scanner)
        o3d.io.write_point_cloud(str(out_dir / f"scanner-{side}.ply"), pcd,
                                 write_ascii=False)

    truth = {
        "scene_id": scene.scene_id,
        "dxf": scene.dxf_stem,
        "length_mm": scene.length_mm,
        "transport_y_mm": scene.transport_y_mm,
        "transport_z_mm": scene.transport_z_mm,
        "seed": scene.seed,
        "params": scene.params,
        "bars": [asdict(b) for b in scene.bars],
        "faces": face_truth(scene),
    }
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    return out_dir


def load_truth(scene_dir: Path | str) -> dict:
    return json.loads((Path(scene_dir) / "truth.json").read_text())


# ---------------------------------------------------------------- layouts


def bundle_row(
    dxf_stem: str,
    n_bars: int,
    scene_id: str,
    pitch_mm: float | None = None,
    pitch_jitter_mm: float = 0.0,
    roll_jitter_deg: float = 0.0,
    x_jitter_mm: float = 0.0,
    transport_y_mm: float = TRANSPORT_Y_MM,
    seed: int = 0,
    n_stacked: int = 0,
    crossing_yaw_deg: float = 0.0,
    n_crossing: int = 0,
    z_mm: float = BED_Z_MM,
    y_centre_mm: float = BUNDLE_Y_MM,
    base_roll_deg: float | None = None,
    **params,
) -> SyntheticScene:
    """Builds one conveyor bed of n_bars, optionally with stacked and crossed bars.

    - `pitch_jitter_mm` breaks the regular bar spacing.
    - `n_stacked` bars are lifted one section height onto a second layer,
      sitting between two bed bars.
    - `n_crossing` bars are lifted the same way and yawed by
      `crossing_yaw_deg`, lying diagonally across the bundle.
    """
    rng = np.random.default_rng(seed)
    sec = section_mesh(dxf_stem)
    width, height = sec.bbox_mm
    pitch = pitch_mm if pitch_mm is not None else FAMILY_PITCH_MM.get(dxf_stem, width * 1.1)
    if base_roll_deg is None:
        base_roll_deg = FAMILY_BASE_ROLL_DEG.get(dxf_stem, 0.0)

    def roll():
        return base_roll_deg + (float(rng.normal(0.0, roll_jitter_deg))
                                if roll_jitter_deg else 0.0)

    n_bed = n_bars - n_stacked - n_crossing
    if n_bed < 1:
        raise ValueError("n_bars must exceed n_stacked + n_crossing")

    offsets = np.arange(n_bed) * pitch
    if pitch_jitter_mm > 0:
        offsets = offsets + rng.normal(0.0, pitch_jitter_mm, size=n_bed)
    offsets = offsets - offsets.mean()

    bars = []
    for off in offsets:
        bars.append(BarPose(
            y_mm=y_centre_mm + float(off),
            z_mm=z_mm,
            x_mm=float(rng.normal(0.0, x_jitter_mm)) if x_jitter_mm else 0.0,
            roll_deg=roll(),
        ))

    upper_slots = np.linspace(offsets.min() + pitch / 2, offsets.max() - pitch / 2,
                              max(n_stacked + n_crossing, 1))
    for k in range(n_stacked):
        bars.append(BarPose(
            y_mm=y_centre_mm + float(upper_slots[k]),
            z_mm=z_mm + height,
            roll_deg=roll(),
        ))
    for k in range(n_crossing):
        bars.append(BarPose(
            y_mm=y_centre_mm + float(upper_slots[n_stacked + k]),
            z_mm=z_mm + height,
            yaw_deg=crossing_yaw_deg,
            roll_deg=roll(),
        ))

    return SyntheticScene(
        scene_id=scene_id,
        dxf_stem=dxf_stem,
        bars=bars,
        transport_y_mm=transport_y_mm,
        seed=seed,
        params=dict(n_bars=n_bars, pitch_mm=float(pitch),
                    pitch_jitter_mm=pitch_jitter_mm, n_stacked=n_stacked,
                    n_crossing=n_crossing, crossing_yaw_deg=crossing_yaw_deg,
                    roll_jitter_deg=roll_jitter_deg, x_jitter_mm=x_jitter_mm,
                    base_roll_deg=base_roll_deg,
                    section_width_mm=width, section_height_mm=height, **params),
    )
