"""Virtual scanner: a pinhole range sensor raycast against a triangle scene.

Parameters measured from the real scans (see docs/method.md):
- field of view in tangent space u = x/z (hard limit), v = y/z (observed envelope);
- 1544 x 2064 px sensor, stepping over the 1544 axis (`n_u`);
- range noise sigma 0.12 mm along the ray;
- range gate 1400 mm (the loader's 850 mm radial filter still applies);
- incidence cutoff 55 deg (end faces return at ~23 deg, flanks do not at ~68 deg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Tangent-space FOV limits: u = x/z (hard limit), v = y/z (observed envelope).
_U_LIMITS = {1: (-0.350988, +0.300222), 2: (-0.347865, +0.312564)}
_V_LIMITS = {1: (-0.333307, +0.337289), 2: (-0.311154, +0.341779)}
_N_U = 1544

NOISE_SIGMA_MM = 0.12
RANGE_MAX_MM = 1400.0
INCIDENCE_MAX_DEG = 55.0


@dataclass(frozen=True)
class SensorModel:
    """Pinhole range sensor in its own frame (+Z is the optical axis)."""

    u_lo: float
    u_hi: float
    v_lo: float
    v_hi: float
    n_u: int = _N_U
    noise_sigma_mm: float = NOISE_SIGMA_MM
    range_max_mm: float = RANGE_MAX_MM
    incidence_max_deg: float = INCIDENCE_MAX_DEG
    dropout_prob: float = 0.0

    @property
    def pitch(self) -> float:
        """Tangent-space step per pixel; both axes share it (square pixels)."""
        return (self.u_hi - self.u_lo) / (self.n_u - 1)

    @property
    def n_v(self) -> int:
        return int(round((self.v_hi - self.v_lo) / self.pitch)) + 1

    def uv_grid(self, u_range: tuple[int, int], v_range: tuple[int, int]):
        iu = np.arange(u_range[0], u_range[1] + 1)
        iv = np.arange(v_range[0], v_range[1] + 1)
        u = self.u_lo + iu * self.pitch
        v = self.v_lo + iv * self.pitch
        return u, v


def sensor_for_side(side: int, **overrides) -> SensorModel:
    """Returns the SensorModel for scanner side 1 or 2."""
    if side not in _U_LIMITS:
        raise ValueError(f"side must be 1 or 2, got {side}")
    u_lo, u_hi = _U_LIMITS[side]
    v_lo, v_hi = _V_LIMITS[side]
    return SensorModel(u_lo=u_lo, u_hi=u_hi, v_lo=v_lo, v_hi=v_hi, **overrides)


def _pixel_window(sensor: SensorModel, corners_scanner: np.ndarray, margin: int = 8):
    """Returns the index window of the ray grid that can see corners_scanner."""
    z = corners_scanner[:, 2]
    if not np.any(z > 1e-6):
        return None
    ok = z > 1e-6
    u = corners_scanner[ok, 0] / z[ok]
    v = corners_scanner[ok, 1] / z[ok]
    iu0 = int(np.floor((u.min() - sensor.u_lo) / sensor.pitch)) - margin
    iu1 = int(np.ceil((u.max() - sensor.u_lo) / sensor.pitch)) + margin
    iv0 = int(np.floor((v.min() - sensor.v_lo) / sensor.pitch)) - margin
    iv1 = int(np.ceil((v.max() - sensor.v_lo) / sensor.pitch)) + margin
    iu0, iu1 = max(iu0, 0), min(iu1, sensor.n_u - 1)
    iv0, iv1 = max(iv0, 0), min(iv1, sensor.n_v - 1)
    if iu1 < iu0 or iv1 < iv0:
        return None
    return (iu0, iu1), (iv0, iv1)


def raycast(
    mesh_vertices: np.ndarray,
    mesh_triangles: np.ndarray,
    R: np.ndarray,
    origin_world: np.ndarray,
    sensor: SensorModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scans a world-frame triangle soup. Returns (N,3) points in scanner frame.

    `R` maps scanner-frame directions into world coordinates; `origin_world`
    is the center of projection in world coordinates.
    """
    import open3d as o3d

    verts_s = (mesh_vertices - origin_world) @ R  # world -> scanner frame
    window = _pixel_window(sensor, verts_s)
    if window is None:
        return np.zeros((0, 3))
    u, v = sensor.uv_grid(*window)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    dirs_s = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    dirs_s /= np.linalg.norm(dirs_s, axis=1, keepdims=True)
    dirs_w = dirs_s @ R.T

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(mesh_vertices, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(mesh_triangles, dtype=np.uint32)),
    )
    rays = np.empty((len(dirs_w), 6), dtype=np.float32)
    rays[:, :3] = origin_world
    rays[:, 3:] = dirs_w
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = ans["t_hit"].numpy()
    normals = ans["primitive_normals"].numpy()

    keep = np.isfinite(t_hit) & (t_hit <= sensor.range_max_mm)
    if sensor.incidence_max_deg < 90.0:
        cos_inc = np.abs(np.einsum("ij,ij->i", normals, dirs_w))
        keep &= cos_inc >= np.cos(np.radians(sensor.incidence_max_deg))
    if sensor.dropout_prob > 0:
        keep &= rng.random(len(keep)) >= sensor.dropout_prob
    if not keep.any():
        return np.zeros((0, 3))

    rng_hit = t_hit[keep].astype(np.float64)
    if sensor.noise_sigma_mm > 0:
        rng_hit = rng_hit + rng.normal(0.0, sensor.noise_sigma_mm, size=len(rng_hit))
    return dirs_s[keep] * rng_hit[:, None]
