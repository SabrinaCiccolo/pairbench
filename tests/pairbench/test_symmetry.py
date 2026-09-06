"""Tests for pairbench.symmetry on synthetic shapes and the four DXF sections."""

import numpy as np

from pairbench.symmetry import (
    SYMMETRY_IOU_THRESHOLD,
    measure_symmetry_period_deg,
    period_for_dxf,
    period_for_scene,
)


def _square(side_mm=40.0):
    return [np.array([[0.0, 0.0], [side_mm, 0.0],
                      [side_mm, side_mm], [0.0, side_mm]])]


def _rect(w=80.0, h=20.0):
    return [np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])]


def _l_shape(a=40.0, b=12.0):
    return [np.array([[0.0, 0.0], [a, 0.0], [a, b], [b, b], [b, a], [0.0, a]])]


def test_square_is_four_fold():
    period, ious = measure_symmetry_period_deg(_square())
    assert period == 90.0
    assert ious[4] >= SYMMETRY_IOU_THRESHOLD
    assert ious[3] < SYMMETRY_IOU_THRESHOLD


def test_rectangle_is_two_fold():
    period, ious = measure_symmetry_period_deg(_rect())
    assert period == 180.0
    assert ious[2] >= SYMMETRY_IOU_THRESHOLD
    assert ious[4] < SYMMETRY_IOU_THRESHOLD


def test_l_shape_has_no_rotational_symmetry():
    period, ious = measure_symmetry_period_deg(_l_shape())
    assert period == 360.0
    assert max(ious.values()) < SYMMETRY_IOU_THRESHOLD


def test_smallest_passing_period_wins():
    """A square passes both k=2 and k=4; the tighter period must be returned."""
    period, ious = measure_symmetry_period_deg(_square())
    assert ious[2] >= SYMMETRY_IOU_THRESHOLD  # 180 deg is a superset period
    assert period == 90.0


def test_measured_periods_of_the_dataset_profiles():
    assert period_for_dxf("l-profile") == 360.0          # l-profile / l-profile-reshoot (chiral)
    assert period_for_dxf("complex-profile") == 360.0          # complex-profile
    assert period_for_dxf("square-profile") == 90.0      # square-profile (square tube)
    assert period_for_dxf("heavy-profile") == 180.0        # heavy-profile


def test_period_for_scene_follows_the_family():
    assert period_for_scene("square-profile/1") == 90.0
    assert period_for_scene("l-profile-reshoot/L1") == 360.0
    assert period_for_scene("heavy-profile/whatever") == 180.0
