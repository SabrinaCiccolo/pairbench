"""Self-calibration. Synthetic calibration only, no scan data."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pairbench.io.calib import DualCalibration, ScannerCalibration
from pairbench.io.calib_full import FullCalibration, ScannerSpline
from pairbench.detect.ccorr import Detection
from pairbench.selfcalib import (
    PROFILE_LENGTH_MM,
    bootstrap_separation,
    fit_steps,
    length_residuals,
    observability,
    recalibrated,
    refit_translation,
    to_scanner_frame,
)

KNOTS = np.linspace(0.0, 4000.0, 5)


def _spline(x_at_zero: float, sign: float) -> ScannerSpline:
    """Carriage travelling along X: t_x(s) = x_at_zero + sign * s."""
    rots = Rotation.from_euler("z", np.zeros((len(KNOTS), 1)))
    trans = np.zeros((len(KNOTS), 3))
    trans[:, 0] = x_at_zero + sign * KNOTS
    return ScannerSpline(step_positions=KNOTS, translations=trans,
                         slerp=Slerp(KNOTS, rots))


def _full() -> FullCalibration:
    return FullCalibration(scanner1=_spline(0.0, 1.0),
                           scanner2=_spline(0.0, -1.0))


def _offline() -> DualCalibration:
    eye = np.eye(3)
    return DualCalibration(
        scanner1=ScannerCalibration(R=eye, t=np.zeros(3)),
        scanner2=ScannerCalibration(R=eye, t=np.zeros(3)),
        max_distance_mm=850.0)


def det(y, z, x=0.0):
    return Detection(side=1, angle_deg=180.0, score=0.9, ccorr_frac=0.9,
                     kind="confirmed", obb=((y, z), (40.0, 12.0), 0.0),
                     centroid_yz_mm=(y, z), centroid_world_mm=(x, y, z),
                     normal_world=(1.0, 0.0, 0.0), n_points=100,
                     z_extent_mm=(z - 6.0, z + 6.0),
                     y_extent_mm=(y - 20.0, y + 20.0))


def test_to_scanner_frame_inverts_the_offline_calibration():
    off = DualCalibration(
        scanner1=ScannerCalibration(
            R=Rotation.from_euler("z", 30, degrees=True).as_matrix(),
            t=np.array([10.0, -5.0, 2.0])),
        scanner2=ScannerCalibration(R=np.eye(3), t=np.zeros(3)),
        max_distance_mm=850.0)
    p = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]])
    sc = off.for_side(1)
    world = p @ sc.R.T + sc.t
    dets = [det(0, 0, 0) for _ in world]
    for d, w in zip(dets, world):
        d.centroid_world_mm = tuple(w)
    assert np.allclose(to_scanner_frame(dets, 1, off), p)


def test_recalibrated_shifts_extents_with_the_centroid():
    d = det(1600.0, 917.0)
    out = recalibrated([d], 1, 500.0, _offline(), _full())[0]
    assert out.centroid_world_mm[0] == 500.0        # carriage X applied
    assert out.centroid_yz_mm == d.centroid_yz_mm   # this rig moves X only
    assert out.y_extent_mm == d.y_extent_mm
    # a Y shift on side 2 must carry the extents with it
    out2 = recalibrated([d], 2, 500.0, _offline(), _full(), t_yz=(25.0, 0.0))[0]
    assert out2.centroid_yz_mm[0] == 1625.0
    assert out2.y_extent_mm == (d.y_extent_mm[0] + 25.0,
                                d.y_extent_mm[1] + 25.0)


def test_fit_steps_recovers_the_separation():
    # side-1 faces at x=0 in scanner frame, side-2 faces at x=0 too; the
    # carriages are what place them 6005 mm apart in world.
    rng = np.random.default_rng(0)
    p1 = np.column_stack([np.zeros(12), rng.uniform(1500, 1900, 12),
                          rng.uniform(910, 925, 12)])
    p2 = p1.copy()
    full = _full()
    # this rig's scanners travel in opposite directions, so
    # w1.x - w2.x = s1 + s2, which must equal 6005 for a zero residual
    s1_true = 3000.0
    s2_true = PROFILE_LENGTH_MM - s1_true
    assert abs(length_residuals(p1, p2, full, s1_true, s2_true)).max() < 1e-6
    s1_fit, s2_fit = fit_steps(p1, p2, full)
    assert abs((s1_fit + s2_fit) - PROFILE_LENGTH_MM) < 1.0


def test_only_the_separation_is_observable():
    rng = np.random.default_rng(1)
    p1 = np.column_stack([np.zeros(20), rng.uniform(1500, 1900, 20),
                          rng.uniform(910, 925, 20)])
    full = _full()
    obs = observability(p1, p1.copy(), full, 3000.0, PROFILE_LENGTH_MM - 3000.0)
    # this rig's scanners travel in opposite directions, so the observed
    # direction is (+1, +1) here and the null space is (+1, -1); the real
    # calibration has both travelling +X, which flips the labels. What the
    # test pins is the RANK: one direction only.
    assert obs["ratio"] > 1e6, obs
    v = np.array(obs["observed_dir"])
    assert abs(abs(v[0]) - abs(v[1])) < 1e-3, v


def test_refit_translation_reads_the_cross_view_offset():
    rng = np.random.default_rng(2)
    p1 = np.column_stack([np.zeros(15), rng.uniform(1500, 1900, 15),
                          rng.uniform(910, 925, 15)])
    p2 = p1 - np.array([0.0, 12.0, -3.0])   # side 2 sits 12 mm low in Y
    ty, tz = refit_translation(p1, p2, _full(), 3000.0, 3000.0)
    assert abs(ty - 12.0) < 1e-6 and abs(tz + 3.0) < 1e-6


def test_bootstrap_spread_tracks_the_noise():
    rng = np.random.default_rng(3)
    p1 = np.column_stack([np.zeros(40), rng.uniform(1500, 1900, 40),
                          rng.uniform(910, 925, 40)])
    full = _full()
    s1, s2 = 3000.0, PROFILE_LENGTH_MM - 3000.0
    clean = bootstrap_separation(p1, p1.copy(), full, s1, s2, n_boot=40)
    noisy_p2 = p1 + rng.normal(0, 6.0, p1.shape)
    noisy = bootstrap_separation(p1, noisy_p2, full, s1, s2, n_boot=40)
    assert len(clean) == 40
    assert clean.std() <= noisy.std() + 1e-9, (clean.std(), noisy.std())
