"""Synthetic scene suites. Each is deterministic given its parameters.

- `realism_suite`: mirrors real single-bundle scenes (family, bar count).
- `pitch_suite`: bar-pitch jitter crossed with cross-view transport.
- `ambiguity_suite`: transport swept over three whole pitches.
- `crossing_suite`: elevation and yaw of a probe bar as independent factors.
- `one_end_suite`: a bar raised at one end only, elevated in one view.

`WINDOW_Y_MM` is a placement aid for `placement="edge"` (bundle truncated by
the window edge) vs `"centre"` (no truncation); the raycaster does the
actual truncation.
"""

from __future__ import annotations

import numpy as np

from .generator import (
    APPARENT_LENGTH_MM,
    BED_Z_MM,
    BUNDLE_Y_MM,
    FAMILY_BASE_ROLL_DEG,
    FAMILY_PITCH_MM,
    TRANSPORT_Y_MM,
    BarPose,
    SyntheticScene,
    bundle_row,
)
from .section import section_mesh

# Scan window in world Y: measured bundle centre +/- half the widest window
# over the real clouds after the 850 mm radial gate
# (`results/pairbench/synthetic_sensor_model.json`, `scan_window_mm.gated`).
WINDOW_CENTRE_Y_MM = 1728.9
WINDOW_WIDTH_MM = 414.0
WINDOW_Y_MM = (WINDOW_CENTRE_Y_MM - WINDOW_WIDTH_MM / 2.0,
               WINDOW_CENTRE_Y_MM + WINDOW_WIDTH_MM / 2.0)


def realism_suite(specs) -> list[SyntheticScene]:
    """specs: iterable of (scene_id, dxf_stem, n_bars) mirroring real scenes."""
    scenes = []
    for k, (scene_id, dxf_stem, n_bars) in enumerate(specs):
        scenes.append(bundle_row(
            dxf_stem, n_bars, f"realism/{scene_id.replace('/', '__')}",
            pitch_jitter_mm=5.0, roll_jitter_deg=2.0, x_jitter_mm=12.0,
            transport_y_mm=TRANSPORT_Y_MM, seed=1000 + k,
            mirrors=scene_id,
        ))
    return scenes


def pitch_suite(
    dxf_stem: str = "l-profile",
    n_bars: int = 6,
    jitters=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    transports=(0.0, 17.5, 35.0, 52.5, 70.0, 87.5, 105.0),
    seeds=(0, 1, 2),
    placement: str = "edge",
    arm: str = "pitch",
) -> list[SyntheticScene]:
    """Sweeps pitch jitter crossed with cross-view transport.

    `placement="edge"`: bundle against the left window edge (leftmost bar
    truncated; transport moves bars in/out of view). `"centre"`: centred
    with clearance, so no bar crosses the window edge at any transport.
    """
    pitch = FAMILY_PITCH_MM[dxf_stem]
    width = section_mesh(dxf_stem).bbox_mm[0]
    if placement == "edge":
        y_left = WINDOW_Y_MM[0] - width * 0.25
        y_centre = y_left + (n_bars - 1) * pitch / 2.0
    elif placement == "centre":
        y_centre = 0.5 * (WINDOW_Y_MM[0] + WINDOW_Y_MM[1])
        span = (n_bars - 1) * pitch + width
        margin = (WINDOW_Y_MM[1] - WINDOW_Y_MM[0]) - span - max(transports)
        if margin < 0:
            raise ValueError(
                f"{n_bars} bars at {pitch:g} mm pitch plus "
                f"{max(transports):g} mm of transport do not fit the "
                f"{WINDOW_Y_MM[1] - WINDOW_Y_MM[0]:g} mm window without "
                "truncation; reduce n_bars or the transport range")
    else:
        raise ValueError(f"placement must be 'edge' or 'centre', got {placement!r}")
    scenes = []
    for jitter in jitters:
        for transport in transports:
            for seed in seeds:
                sid = f"{arm}/j{jitter:g}_t{transport:g}_s{seed}"
                scenes.append(bundle_row(
                    dxf_stem, n_bars, sid, pitch_mm=pitch,
                    pitch_jitter_mm=jitter, roll_jitter_deg=1.0,
                    transport_y_mm=transport, y_centre_mm=y_centre,
                    seed=seed, arm=arm, placement=placement,
                    transport_over_pitch=round(transport / pitch, 3),
                ))
    return scenes


def ambiguity_suite(
    dxf_stem: str = "square-profile",
    n_bars: int = 5,
    jitters=(0.0, 2.0, 10.0),
    transport_over_pitch=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                          2.0, 2.25, 2.5, 2.75, 3.0),
    seeds=(0, 1, 2),
    placements=("centre", "edge"),
    arm: str = "ambiguity",
) -> list[SyntheticScene]:
    """Cross-view transport swept over three whole bar pitches, at both
    placements. The default `square-profile` has the smallest pitch and
    section (so the sweep fits the window when centred) and a 90 deg
    symmetry period (weakest angle term)."""
    pitch = FAMILY_PITCH_MM[dxf_stem]
    transports = tuple(round(f * pitch, 4) for f in transport_over_pitch)
    width = section_mesh(dxf_stem).bbox_mm[0]
    span = (n_bars - 1) * pitch + width
    scenes = []
    for placement in placements:
        if placement == "centre":
            margin = (WINDOW_Y_MM[1] - WINDOW_Y_MM[0]) - span - max(transports)
            if margin < 0:
                raise ValueError(
                    f"{n_bars} bars at {pitch:g} mm pitch plus "
                    f"{max(transports):g} mm of transport do not fit the "
                    f"{WINDOW_Y_MM[1] - WINDOW_Y_MM[0]:g} mm window without "
                    "truncation; reduce n_bars or the transport range")
            y_centre = 0.5 * (WINDOW_Y_MM[0] + WINDOW_Y_MM[1])
        else:
            y_left = WINDOW_Y_MM[0] - width * 0.25
            y_centre = y_left + (n_bars - 1) * pitch / 2.0
        for jitter in jitters:
            for frac, transport in zip(transport_over_pitch, transports):
                for seed in seeds:
                    sid = (f"{arm}/{placement}_j{jitter:g}_t{frac:g}_s{seed}")
                    scenes.append(bundle_row(
                        dxf_stem, n_bars, sid, pitch_mm=pitch,
                        pitch_jitter_mm=jitter, roll_jitter_deg=1.0,
                        transport_y_mm=transport, y_centre_mm=y_centre,
                        seed=seed, arm=arm, placement=placement,
                        transport_over_pitch=round(frac, 3),
                    ))
    return scenes


def crossing_scene(
    scene_id: str,
    dxf_stem: str = "l-profile",
    n_bed: int = 3,
    yaw_deg: float = 0.0,
    elevated: bool = True,
    seed: int = 0,
    transport_y_mm: float = TRANSPORT_Y_MM,
    pitch_mm: float | None = None,
) -> SyntheticScene:
    """n_bed bars in a row plus one probe bar, optionally yawed and/or
    elevated. Elevated probes sit one section height up between two bed bars;
    flat probes sit one extra pitch beyond the row's end."""
    rng = np.random.default_rng(seed)
    sec = section_mesh(dxf_stem)
    width, height = sec.bbox_mm
    pitch = pitch_mm if pitch_mm is not None else FAMILY_PITCH_MM[dxf_stem]
    base_roll = FAMILY_BASE_ROLL_DEG.get(dxf_stem, 0.0)
    offsets = (np.arange(n_bed) - (n_bed - 1) / 2.0) * pitch
    bars = [BarPose(y_mm=BUNDLE_Y_MM + float(o), z_mm=BED_Z_MM,
                    roll_deg=base_roll + float(rng.normal(0.0, 1.0)))
            for o in offsets]
    if elevated:
        probe_y = BUNDLE_Y_MM + float(offsets[0] + pitch / 2.0)
        probe_z = BED_Z_MM + height
    else:
        probe_y = BUNDLE_Y_MM + float(offsets[-1] + 2.0 * pitch)
        probe_z = BED_Z_MM
    bars.append(BarPose(y_mm=probe_y, z_mm=probe_z, yaw_deg=yaw_deg,
                        roll_deg=base_roll + float(rng.normal(0.0, 1.0))))
    return SyntheticScene(
        scene_id=scene_id, dxf_stem=dxf_stem, bars=bars,
        transport_y_mm=transport_y_mm, seed=seed,
        params=dict(arm="crossing", n_bars=n_bed + 1, n_bed=n_bed,
                    yaw_deg=yaw_deg, elevated=int(elevated), pitch_mm=pitch,
                    probe_index=n_bed, section_height_mm=height,
                    section_width_mm=width,
                    # how far the probe's two end faces separate in
                    # transport-Y purely because of the yaw
                    yaw_face_offset_mm=round(
                        APPARENT_LENGTH_MM * np.sin(np.radians(yaw_deg)), 2)),
    )


def crossing_suite(
    yaws=(0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0),
    elevations=(True, False),
    seeds=(0, 1, 2),
) -> list[SyntheticScene]:
    scenes = []
    for elevated in elevations:
        for yaw in yaws:
            for seed in seeds:
                tag = "up" if elevated else "flat"
                scenes.append(crossing_scene(
                    f"crossing/{tag}_y{yaw:g}_s{seed}", yaw_deg=yaw,
                    elevated=elevated, seed=seed))
    return scenes


def one_end_scene(
    scene_id: str,
    lateral_mm: float,
    raised_side: int,
    seed: int = 0,
    dxf_stem: str = "l-profile",
    n_bed: int = 3,
    transport_y_mm: float = TRANSPORT_Y_MM,
) -> SyntheticScene:
    """n_bed bars in a row plus one probe bar raised at one end only: the
    raised end rests one section height up between the last two bed bars, the
    other lies on the bed `lateral_mm` further along Y. `raised_side` is the
    scanner that sees the raised end (side 1 sees the +X end)."""
    rng = np.random.default_rng(seed)
    width, height = section_mesh(dxf_stem).bbox_mm
    pitch = FAMILY_PITCH_MM[dxf_stem]
    base_roll = FAMILY_BASE_ROLL_DEG.get(dxf_stem, 0.0)
    offsets = (np.arange(n_bed) - (n_bed - 1) / 2.0) * pitch
    bars = [BarPose(y_mm=BUNDLE_Y_MM + float(o), z_mm=BED_Z_MM,
                    roll_deg=base_roll + float(rng.normal(0.0, 1.0)))
            for o in offsets]
    y_top = BUNDLE_Y_MM + float(offsets[-1] - pitch / 2.0)
    y_low = y_top + lateral_mm
    # +X end raised (seen by side 1) needs a negative pitch and a yaw that
    # puts the +X end at y_top; side 2 is the mirror image
    sign = -1.0 if raised_side == 1 else 1.0
    yaw = float(np.degrees(np.arctan2(sign * lateral_mm, APPARENT_LENGTH_MM)))
    tilt = float(np.degrees(np.arctan2(sign * height, APPARENT_LENGTH_MM)))
    bars.append(BarPose(y_mm=0.5 * (y_top + y_low), z_mm=BED_Z_MM + height / 2.0,
                        yaw_deg=yaw, pitch_deg=tilt,
                        roll_deg=base_roll + float(rng.normal(0.0, 1.0))))
    return SyntheticScene(
        scene_id=scene_id, dxf_stem=dxf_stem, bars=bars,
        transport_y_mm=transport_y_mm, seed=seed,
        params=dict(arm="one_end", n_bars=n_bed + 1, n_bed=n_bed,
                    lateral_mm=lateral_mm, raised_side=raised_side,
                    pitch_mm=pitch, probe_index=n_bed,
                    section_height_mm=height, section_width_mm=width),
    )


def one_end_suite(
    laterals=(120.0, 170.0, 220.0),
    raised_sides=(1, 2),
    seeds=(0, 1, 2),
) -> list[SyntheticScene]:
    return [one_end_scene(f"one_end/l{lat:g}_side{side}_s{seed}", lat, side, seed)
            for lat in laterals for side in raised_sides for seed in seeds]

