"""World-space point cloud -> YZ-plane grayscale projection.

Render chain:

1. optional statistical filter: drop points beyond nsigma std devs of the
   mean in world-Y or world-Z (stabilizes the bounding box against stragglers)
2. rasterize at render_scale px/mm: col = floor((Y-minY)*s),
   row = (H-1) - floor((Z-minZ)*s)  [image Y grows downward, world Z upward]
3. dilate each point to a disk of radius ceil(point_radius_mm * render_scale) px
4. downscale by scale_factor (bilinear) -> effective 2.5 px/mm
5. Gaussian blur (blur_kernel, sigma 0)
6. constant black border of padding_px

Image X axis <- world Y (transport axis), image Y axis <- world Z (height,
flipped so up is up).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RENDER_SCALE = 10.0        # px/mm on the high-res canvas
POINT_RADIUS_MM = 0.5      # disk radius per point (5 px at 10 px/mm)
SCALE_FACTOR = 0.25        # downscale -> 2.5 px/mm effective
BLUR_KERNEL = 3
PADDING_PX = 95
NSIGMA = 3.0

_MAX_CANVAS_PX = 30000     # hard guard against degenerate extents


@dataclass(frozen=True)
class Projection:
    image: np.ndarray        # uint8 grayscale, padded
    min_y: float             # world-Y at unpadded left edge (mm)
    max_z: float             # world-Z at unpadded top edge (mm)
    eff_scale: float         # px/mm of the final image
    padding_px: int
    n_points: int            # points surviving the statistical filter
    keep_mask: np.ndarray    # bool per input point: True if rasterized; map
                             # back to 3D via world_points[proj.keep_mask]

    def pixel_to_yz(self, px: float, py: float) -> tuple[float, float]:
        """Final-image pixel -> (world Y, world Z) in mm."""
        y = (px - self.padding_px) / self.eff_scale + self.min_y
        z = self.max_z - (py - self.padding_px) / self.eff_scale
        return y, z

    def yz_to_pixel(self, y: float, z: float) -> tuple[float, float]:
        """(world Y, world Z) in mm -> final-image pixel. Inverse of
        pixel_to_yz."""
        px = self.padding_px + (y - self.min_y) * self.eff_scale
        py = self.padding_px + (self.max_z - z) * self.eff_scale
        return px, py


@dataclass(frozen=True)
class DepthChannels:
    """Per-pixel depth/density/spread on a Projection's pixel grid, without
    dilation or blur."""
    near_x_mm: np.ndarray     # float32 (H,W); nearest-to-scanner world-X, nan if empty
    density: np.ndarray       # float32 (H,W); point count per pixel
    x_spread_mm: np.ndarray   # float32 (H,W); max_x - min_x per pixel, 0 if density <= 1
    observed: np.ndarray      # bool (H,W); density > 0


def depth_channels(world_points: np.ndarray, proj: Projection, side: int) -> DepthChannels:
    """Depth/density/spread channels aligned to `proj`. `world_points` is the
    unfiltered array `proj` was built from (keep_mask is re-applied). Side 1:
    larger X is nearer the scanner; side 2: smaller X."""
    if side not in (1, 2):
        raise ValueError(f"side must be 1 or 2, got {side}")
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"expected (N,3) points, got {world_points.shape}")
    kept = world_points[proj.keep_mask]
    x, y, z = kept[:, 0], kept[:, 1], kept[:, 2]

    h, w = proj.image.shape
    cols = np.clip(
        np.round(proj.padding_px + (y - proj.min_y) * proj.eff_scale).astype(np.int64),
        0, w - 1,
    )
    rows = np.clip(
        np.round(proj.padding_px + (proj.max_z - z) * proj.eff_scale).astype(np.int64),
        0, h - 1,
    )
    flat = (rows * w + cols).astype(np.int64)

    density_flat = np.zeros(h * w, dtype=np.float32)
    np.add.at(density_flat, flat, 1.0)

    x_min_flat = np.full(h * w, np.inf, dtype=np.float32)
    x_max_flat = np.full(h * w, -np.inf, dtype=np.float32)
    np.minimum.at(x_min_flat, flat, x.astype(np.float32))
    np.maximum.at(x_max_flat, flat, x.astype(np.float32))

    observed_flat = density_flat > 0
    x_spread_flat = np.where(observed_flat, x_max_flat - x_min_flat, 0.0).astype(np.float32)

    extreme_flat = x_max_flat if side == 1 else x_min_flat
    near_x_flat = np.where(observed_flat, extreme_flat, np.float32(np.nan))

    return DepthChannels(
        near_x_mm=near_x_flat.reshape(h, w),
        density=density_flat.reshape(h, w),
        x_spread_mm=x_spread_flat.reshape(h, w),
        observed=observed_flat.reshape(h, w),
    )


def project_yz(
    world_points: np.ndarray,
    render_scale: float = RENDER_SCALE,
    point_radius_mm: float = POINT_RADIUS_MM,
    scale_factor: float = SCALE_FACTOR,
    blur_kernel: int = BLUR_KERNEL,
    padding_px: int = PADDING_PX,
    nsigma: float | None = NSIGMA,
) -> Projection:
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"expected (N,3) points, got {world_points.shape}")
    y = world_points[:, 1]
    z = world_points[:, 2]
    keep_mask = np.ones(len(y), dtype=bool)

    if nsigma is not None and len(y) >= 3:
        dev_y = y.std()
        dev_z = z.std()
        if dev_y > 0 and dev_z > 0:
            keep_mask = (np.abs(y - y.mean()) < nsigma * dev_y) & (
                np.abs(z - z.mean()) < nsigma * dev_z
            )
            y, z = y[keep_mask], z[keep_mask]
    if len(y) < 3:
        raise ValueError("fewer than 3 points after filtering")

    min_y, max_y = float(y.min()), float(y.max())
    min_z, max_z = float(z.min()), float(z.max())
    width = int(np.ceil((max_y - min_y) * render_scale)) + 1
    height = int(np.ceil((max_z - min_z) * render_scale)) + 1
    if width > _MAX_CANVAS_PX or height > _MAX_CANVAS_PX:
        raise ValueError(f"canvas {width}x{height} exceeds guard {_MAX_CANVAS_PX}")

    cols = np.clip(np.floor((y - min_y) * render_scale).astype(np.int64), 0, width - 1)
    rows = np.clip(
        (height - 1) - np.floor((z - min_z) * render_scale).astype(np.int64),
        0,
        height - 1,
    )

    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[rows, cols] = 255

    radius_px = max(1, int(np.ceil(point_radius_mm * render_scale)))
    ksize = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    canvas = cv2.dilate(canvas, kernel)

    down = cv2.resize(
        canvas, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
    )
    if blur_kernel and blur_kernel >= 3:
        down = cv2.GaussianBlur(down, (blur_kernel, blur_kernel), 0.0)
    padded = cv2.copyMakeBorder(
        down, padding_px, padding_px, padding_px, padding_px,
        cv2.BORDER_CONSTANT, value=0,
    )
    return Projection(
        image=padded,
        min_y=min_y,
        max_z=max_z,
        eff_scale=render_scale * scale_factor,
        padding_px=padding_px,
        n_points=int(len(y)),
        keep_mask=keep_mask,
    )
