"""Tests for pairbench.synth: section extrusion, pose algebra, sensor model,
PLY round trip through `load_scene`, and truth scoring. Needs only the DXFs.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pairbench.io.calib import load_calibration, radial_filter
from pairbench.io.loader import load_scene
from pairbench.io.ply_io import read_ply_points
from pairbench.synth import (
    APPARENT_LENGTH_MM,
    BarPose,
    SyntheticScene,
    bundle_row,
    extrude_prism,
    face_truth,
    load_truth,
    raycast,
    render_scene,
    scene_mesh,
    section_mesh,
    sensor_for_side,
)
from pairbench.synth.evaluate import assign_to_truth
from pairbench.synth.section import build_section_mesh, resample_loop
from pairbench.synth.suites import crossing_scene, pitch_suite

STEMS = ("l-profile", "complex-profile", "square-profile", "heavy-profile")

# Section areas from the same even-odd fill dxf_template.render_template uses,
# so a triangulation that quietly loses a hole would show up here.
EXPECTED_AREA_MM2 = {"l-profile": 280.04, "complex-profile": 999.46,
                     "square-profile": 304.0, "heavy-profile": 2039.36}


def _tri_areas(mesh):
    """Areas of a 2D triangulation."""
    v = mesh.vertices[mesh.triangles]
    a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


# --------------------------------------------------------------- section


def test_resample_loop_preserves_perimeter():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    out = resample_loop(square, 0.5)
    ring = np.vstack([out, out[:1]])
    perimeter = np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()
    assert abs(perimeter - 40.0) < 1e-6
    assert len(out) == 80


def test_triangulated_area_is_exact():
    """Holes must survive into 3D: a filled-in tube would score ~3x here. The
    square tube has a closed-form area (40^2 - 36^2 = 304), so it pins the
    triangulation exactly rather than to a tolerance."""
    for stem in STEMS:
        mesh = section_mesh(stem)
        area = _tri_areas(mesh).sum()
        assert area == pytest.approx(mesh.area_mm2)
        assert abs(area - EXPECTED_AREA_MM2[stem]) / EXPECTED_AREA_MM2[stem] < 0.005, stem
    assert section_mesh("square-profile").area_mm2 == pytest.approx(304.0, abs=1e-9)


def test_section_is_centroid_centred():
    for stem in STEMS:
        mesh = section_mesh(stem)
        v = mesh.vertices[mesh.triangles]
        w = _tri_areas(mesh)
        centroid = (v.mean(axis=1) * w[:, None]).sum(axis=0) / w.sum()
        assert np.abs(centroid).max() < 0.5, (stem, centroid)


def test_prism_caps_sit_at_plus_minus_half_length():
    mesh = section_mesh("square-profile")
    verts, tris = extrude_prism(mesh, 1000.0)
    assert abs(verts[:, 0].min() + 500.0) < 1e-9
    assert abs(verts[:, 0].max() - 500.0) < 1e-9
    assert tris.max() < len(verts)


# --------------------------------------------------------------- poses


def test_yaw_separates_the_two_end_faces_by_l_sin_yaw():
    """The quantity the rigid-motion crossing gate thresholds on."""
    for yaw in (0.5, 2.0, 5.0):
        bar = BarPose(y_mm=1700.0, z_mm=918.9, yaw_deg=yaw)
        c1 = bar.face_centre(1, APPARENT_LENGTH_MM)
        c2 = bar.face_centre(2, APPARENT_LENGTH_MM)
        expected = APPARENT_LENGTH_MM * np.sin(np.radians(yaw))
        assert abs((c1[1] - c2[1]) - expected) < 1e-6


def test_roll_leaves_the_face_centre_alone():
    """Roll is about the bar's own axis through the section centroid, so it
    changes the face's twist and nothing else."""
    a = BarPose(y_mm=1700.0, z_mm=918.9).face_centre(1, APPARENT_LENGTH_MM)
    b = BarPose(y_mm=1700.0, z_mm=918.9, roll_deg=137.0).face_centre(1, APPARENT_LENGTH_MM)
    assert np.allclose(a, b)


def test_transport_offset_lands_on_side_2_only():
    scene = SyntheticScene(scene_id="t", dxf_stem="l-profile",
                           bars=[BarPose(y_mm=1700.0, z_mm=918.9)],
                           transport_y_mm=42.0)
    truth = face_truth(scene)
    assert truth["side1"][0]["y_mm"] == pytest.approx(1700.0)
    assert truth["side2"][0]["y_mm"] == pytest.approx(1742.0)


# --------------------------------------------------------------- sensor


def _flat_quad(x_mm, half=200.0, y0=1727.0, z0=918.9):
    """A single square facing the +X scanner, centred on the bundle."""
    verts = np.array([[x_mm, y0 - half, z0 - half], [x_mm, y0 + half, z0 - half],
                      [x_mm, y0 + half, z0 + half], [x_mm, y0 - half, z0 + half]])
    return verts, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def test_raycast_recovers_a_known_plane():
    calib = load_calibration().for_side(1)
    verts, tris = _flat_quad(2464.0)
    sensor = sensor_for_side(1, noise_sigma_mm=0.0)
    pts = raycast(verts, tris, calib.R, calib.t, sensor,
                  np.random.default_rng(0))
    world = pts @ calib.R.T + calib.t
    assert len(world) > 10000
    assert np.abs(world[:, 0] - 2464.0).max() < 1e-3   # float32 raycaster


def test_noise_is_along_the_ray_not_across_it():
    calib = load_calibration().for_side(1)
    verts, tris = _flat_quad(2464.0)
    pts = raycast(verts, tris, calib.R, calib.t,
                  sensor_for_side(1, noise_sigma_mm=0.5),
                  np.random.default_rng(0))
    world = pts @ calib.R.T + calib.t
    # the rays run mostly along world X, so the plane thickness picks up ~all
    # of the noise and the in-plane spread picks up almost none
    assert 0.4 < world[:, 0].std() < 0.6


def test_incidence_cutoff_sits_on_a_wide_plateau():
    """Faces are seen at ~23 deg and flanks at ~68 deg, so every threshold in
    between gives the same cloud."""
    scene = bundle_row("l-profile", 2, "t/plateau", seed=0)
    verts, tris, _ = scene_mesh(scene)
    calib = load_calibration().for_side(1)
    counts = []
    for cutoff in (40.0, 50.0, 60.0):
        pts = raycast(verts, tris, calib.R, calib.t,
                      sensor_for_side(1, incidence_max_deg=cutoff,
                                      noise_sigma_mm=0.0),
                      np.random.default_rng(0))
        counts.append(len(radial_filter(pts)))
    assert len(set(counts)) == 1, counts
    wide = raycast(verts, tris, calib.R, calib.t,
                   sensor_for_side(1, incidence_max_deg=80.0, noise_sigma_mm=0.0),
                   np.random.default_rng(0))
    assert len(radial_filter(wide)) > 5 * counts[0]


def test_scan_window_truncates_a_bundle_pushed_past_its_edge():
    """A partially visible bar at one end must be producible, and by the
    same mechanism a real scan window truncates one."""
    calib = load_calibration().for_side(1)
    sensor = sensor_for_side(1, noise_sigma_mm=0.0)
    inside, _ = _flat_quad(2464.0, half=60.0, y0=1727.0)
    outside, tris = _flat_quad(2464.0, half=60.0, y0=1490.0)
    n_in = len(raycast(inside, tris, calib.R, calib.t, sensor,
                       np.random.default_rng(0)))
    n_out = len(raycast(outside, tris, calib.R, calib.t, sensor,
                        np.random.default_rng(0)))
    assert n_in > 100000
    assert 0 < n_out < 0.6 * n_in


# --------------------------------------------------------------- round trip


def test_render_round_trips_through_the_unmodified_loader():
    """The twin is read by exactly the code path real scenes are: scanner-
    frame PLY -> radial filter -> calibration."""
    scene = bundle_row("square-profile", 2, "t/roundtrip", seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        out = render_scene(scene, Path(tmp) / "scene")
        clouds = load_scene(out)
        calib = load_calibration()
        for side, world in ((1, clouds.side1_world), (2, clouds.side2_world)):
            cal = calib.for_side(side)
            raw = read_ply_points(out / f"scanner-{side}.ply")
            kept = radial_filter(raw, calib.max_distance_mm)
            assert np.allclose(world, kept @ cal.R.T + cal.t, atol=1e-6)
            assert len(world) > 1000
        truth = load_truth(out)
        assert len(truth["faces"]["side1"]) == 2
        assert truth["length_mm"] == APPARENT_LENGTH_MM


def test_rendered_faces_land_where_the_truth_says():
    """truth.json must agree with the geometry that was actually scanned."""
    scene = bundle_row("l-profile", 3, "t/truth", seed=11, pitch_jitter_mm=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        out = render_scene(scene, Path(tmp) / "scene")
        clouds = load_scene(out)
        for side, world in ((1, clouds.side1_world), (2, clouds.side2_world)):
            for face in load_truth(out)["faces"][f"side{side}"]:
                near = world[(np.abs(world[:, 1] - face["y_mm"]) < 20)
                             & (np.abs(world[:, 2] - face["z_mm"]) < 20)]
                assert len(near) > 500, face
                assert abs(np.median(near[:, 1]) - face["y_mm"]) < 3.0
                assert abs(np.median(near[:, 0]) - face["x_mm"]) < 3.0


def test_render_is_deterministic_from_the_seed():
    scene = bundle_row("l-profile", 2, "t/seed", seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        a = read_ply_points(render_scene(scene, Path(tmp) / "a") / "scanner-1.ply")
        b = read_ply_points(render_scene(scene, Path(tmp) / "b") / "scanner-1.ply")
    assert np.array_equal(a, b)


# --------------------------------------------------------------- scoring


class _Det:
    def __init__(self, y, z):
        self.centroid_yz_mm = (y, z)


def test_assign_to_truth_refuses_a_detection_on_the_wrong_bar():
    """A count-only proxy would score this as a true positive; bar identity
    must not."""
    faces = [{"bar": 0, "y_mm": 1700.0, "z_mm": 918.9},
             {"bar": 1, "y_mm": 1770.0, "z_mm": 918.9}]
    assert assign_to_truth([_Det(1700.5, 919.0)], faces, 25.0) == [0]
    assert assign_to_truth([_Det(1735.0, 919.0)], faces, 25.0) == [None]
    both = assign_to_truth([_Det(1701.0, 919.0), _Det(1769.0, 919.0)], faces, 25.0)
    assert both == [0, 1]


def test_suites_are_deterministic_and_carry_their_parameters():
    a = pitch_suite(jitters=(0.0, 5.0), transports=(0.0,), seeds=(0,))
    b = pitch_suite(jitters=(0.0, 5.0), transports=(0.0,), seeds=(0,))
    assert [s.scene_id for s in a] == [s.scene_id for s in b]
    assert [bar.y_mm for bar in a[0].bars] == [bar.y_mm for bar in b[0].bars]
    assert a[0].params["pitch_jitter_mm"] == 0.0
    assert a[1].params["pitch_jitter_mm"] == 5.0
    # zero jitter must actually be a regular grid, or the ablation is a no-op
    ys = sorted(bar.y_mm for bar in a[0].bars)
    gaps = np.diff(ys)
    assert gaps.std() < 1e-9
    assert np.std(np.diff(sorted(bar.y_mm for bar in a[1].bars))) > 1.0


def test_crossing_probe_is_elevated_xor_flat_and_never_interpenetrates():
    up = crossing_scene("t/up", yaw_deg=2.0, elevated=True)
    flat = crossing_scene("t/flat", yaw_deg=2.0, elevated=False)
    height = section_mesh("l-profile").bbox_mm[1]
    bed_z = up.bars[0].z_mm
    assert up.bars[-1].z_mm == pytest.approx(bed_z + height)
    assert flat.bars[-1].z_mm == pytest.approx(bed_z)
    # the flat probe clears the row, so a yawed flat bar swings into free space
    row_max = max(b.y_mm for b in flat.bars[:-1])
    assert flat.bars[-1].y_mm - row_max > flat.params["pitch_mm"]


def test_truth_json_is_plain_serialisable():
    scene = crossing_scene("t/json", yaw_deg=2.0)
    with tempfile.TemporaryDirectory() as tmp:
        out = render_scene(scene, Path(tmp) / "s")
        payload = json.loads((out / "truth.json").read_text())
    assert payload["params"]["yaw_deg"] == 2.0
    assert payload["params"]["yaw_face_offset_mm"] == pytest.approx(
        APPARENT_LENGTH_MM * np.sin(np.radians(2.0)), abs=0.01)
    assert len(payload["bars"]) == 4


def test_build_section_mesh_is_cache_independent():
    a = build_section_mesh("l-profile")
    b = section_mesh("l-profile")
    assert a.area_mm2 == pytest.approx(b.area_mm2)
    assert a.bbox_mm == b.bbox_mm
