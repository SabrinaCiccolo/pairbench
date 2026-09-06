import numpy as np

from pairbench.io.calib import load_calibration, radial_filter


def test_rotations_orthonormal():
    calib = load_calibration()
    for sc in (calib.scanner1, calib.scanner2):
        assert np.allclose(sc.R @ sc.R.T, np.eye(3), atol=1e-4)
        assert np.isclose(np.linalg.det(sc.R), 1.0, atol=1e-4)


def test_sides_face_each_other():
    # scanner1 sits at +X, scanner2 at -X: fixed translation knots must have
    # opposite-sign world X and near-equal Y/Z (facing scanners)
    calib = load_calibration()
    t1, t2 = calib.scanner1.t, calib.scanner2.t
    assert t1[0] > 0 > t2[0]
    assert abs(t1[1] - t2[1]) < 20.0
    assert abs(t1[2] - t2[2]) < 20.0


def test_apply_shape_and_rigidity():
    calib = load_calibration()
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(100, 3)) * 100
    out = calib.scanner1.apply(pts)
    assert out.shape == (100, 3)
    # rigid transform preserves pairwise distances (R stored at ~1e-8 orthonormality)
    d_in = np.linalg.norm(pts[0] - pts[1])
    d_out = np.linalg.norm(out[0] - out[1])
    assert np.isclose(d_in, d_out, rtol=1e-6)


def test_radial_filter():
    pts = np.array([[0, 0, 0], [849, 0, 0], [851, 0, 0], [600, 601, 0]], dtype=float)
    kept = radial_filter(pts, 850.0)  # (600,601) norm ~ 849.2 -> kept
    assert len(kept) == 3
