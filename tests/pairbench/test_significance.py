"""Tests for pairbench.significance: paired bootstrap, Holm correction,
bundle aggregation."""

import numpy as np
import pytest

from pairbench.significance import (
    sum_by_bundle,
    bootstrap_p_value,
    compare_arms,
    holm_reject,
    paired_bootstrap,
    paired_bootstrap_f1,
    percentile_ci,
)


def _units(per_scene_claims):
    return np.array(per_scene_claims, dtype=np.float64)


# ------------------------------------------------------------ paired bootstrap

def test_identical_arms_have_an_exactly_zero_delta():
    """Two arms that agree on every claim must differ by exactly zero at
    every replicate, not by bootstrap noise, since the resampling indices
    are shared."""
    units = _units([5, 3, 8, 2])
    a = np.array([4.0, 3.0, 6.0, 1.0])
    boots, _ = paired_bootstrap(units, {"a": a, "b": a.copy()}, n_boot=200, seed=1)
    assert np.array_equal(boots["a"], boots["b"])
    assert bootstrap_p_value(boots["a"] - boots["b"]) == 1.0


def test_paired_delta_is_tighter_than_the_unpaired_difference():
    """For positively correlated arms, the CI on the paired delta is
    narrower than differencing two independently resampled arms."""
    rng = np.random.default_rng(0)
    units = _units(rng.integers(2, 9, size=40))
    a = np.array([rng.integers(0, int(u) + 1) for u in units], dtype=np.float64)
    b = np.clip(a - 1.0, 0.0, None)
    boots, _ = paired_bootstrap(units, {"a": a, "b": b}, n_boot=2000, seed=2)
    paired = percentile_ci(boots["a"] - boots["b"])
    shifted, _ = paired_bootstrap(units, {"b": b}, n_boot=2000, seed=99)
    unpaired = percentile_ci(boots["a"] - shifted["b"])
    assert (paired[1] - paired[0]) < (unpaired[1] - unpaired[0])


def test_scene_resampling_is_wider_than_claim_resampling_on_clustered_data():
    """Resampling individual claims when outcomes are clustered by scene
    treats correlated outcomes as independent and understates the spread."""
    per_scene = [6] * 20
    correct = [6.0] * 14 + [0.0] * 6      # whole scenes flip together
    scene_boots, _ = paired_bootstrap(_units(per_scene), {"x": np.array(correct)},
                                      n_boot=4000, seed=3)
    flat = np.repeat([1.0] * 14 + [0.0] * 6, 6)
    claim_boots, _ = paired_bootstrap(np.ones_like(flat), {"x": flat},
                                      n_boot=4000, seed=3)
    s_lo, s_hi = percentile_ci(scene_boots["x"])
    c_lo, c_hi = percentile_ci(claim_boots["x"])
    assert (s_hi - s_lo) > 1.5 * (c_hi - c_lo)


def test_compare_arms_recovers_the_point_estimates_and_flags_a_real_gap():
    units = _units([10] * 30)
    good = np.array([9.0] * 30)
    bad = np.array([5.0] * 30)
    rows = {r["arm"]: r for r in compare_arms(units, {"good": good, "bad": bad},
                                              baseline="good", n_boot=500)}
    assert rows["good"]["value"] == pytest.approx(0.9)
    assert rows["bad"]["value"] == pytest.approx(0.5)
    assert rows["bad"]["delta"] == pytest.approx(-0.4)
    assert rows["bad"]["significant"]
    assert not rows["good"]["significant"]      # a baseline cannot beat itself


def test_compare_arms_rejects_a_missing_baseline():
    with pytest.raises(KeyError):
        compare_arms(_units([1]), {"a": np.array([1.0])}, baseline="nope")


def test_empty_input_does_not_raise():
    boots, den = paired_bootstrap(np.zeros(0), {"a": np.zeros(0)})
    assert boots["a"].size == 0 and den.size == 0
    assert np.isnan(bootstrap_p_value(np.zeros(0)))


# ---------------------------------------------------------- paired F1 bootstrap

def _tpfpfn(rows):
    return np.array(rows, dtype=np.int64)


def test_paired_f1_point_estimate_is_the_pooled_f1():
    """The point estimate is the pooled F1, not the mean of per-scene F1s."""
    arm = _tpfpfn([[3, 1, 0], [5, 0, 2]])   # tp 8, fp 1, fn 2
    row = {r["arm"]: r for r in
           paired_bootstrap_f1({"a": arm}, baseline="a", n_boot=50)}["a"]
    p, r_ = 8 / 9, 8 / 10
    assert row["f1"] == pytest.approx(2 * p * r_ / (p + r_))


def test_paired_f1_gives_identical_arms_an_exactly_zero_delta():
    arm = _tpfpfn([[3, 1, 0], [5, 0, 2], [2, 0, 1]])
    rows = {r["arm"]: r for r in
            paired_bootstrap_f1({"a": arm, "b": arm.copy()}, baseline="a",
                                n_boot=300, seed=2)}
    assert rows["b"]["delta"] == pytest.approx(0.0)
    assert rows["b"]["ci_lo"] == 0.0 and rows["b"]["ci_hi"] == 0.0
    assert rows["b"]["p_value"] == 1.0
    assert rows["b"]["significant"] is False


def test_paired_f1_separates_a_clearly_worse_arm():
    good = _tpfpfn([[4, 0, 0]] * 12)
    bad = _tpfpfn([[1, 3, 3]] * 12)
    rows = {r["arm"]: r for r in
            paired_bootstrap_f1({"good": good, "bad": bad}, baseline="good",
                                n_boot=500, seed=1)}
    assert rows["bad"]["delta"] < -0.4
    assert rows["bad"]["significant"] is True
    assert rows["bad"]["ci_hi"] < 0.0


def test_paired_f1_rejects_a_missing_baseline_and_ragged_arms():
    a = _tpfpfn([[1, 0, 0], [1, 0, 0]])
    with pytest.raises(KeyError):
        paired_bootstrap_f1({"a": a}, baseline="nope")
    with pytest.raises(ValueError):
        paired_bootstrap_f1({"a": a, "b": _tpfpfn([[1, 0, 0]])}, baseline="a")


def test_paired_f1_handles_an_all_empty_arm_without_dividing_by_zero():
    rows = {r["arm"]: r for r in
            paired_bootstrap_f1({"a": _tpfpfn([[1, 0, 1], [2, 0, 0]]),
                                 "none": _tpfpfn([[0, 0, 2], [0, 0, 2]])},
                                baseline="a", n_boot=100)}
    assert rows["none"]["f1"] == 0.0
    assert not np.isnan(rows["none"]["p_value"])


# --------------------------------------------------------------- Holm

def test_holm_is_stricter_than_uncorrected_and_stops_at_the_first_failure():
    p = {"a": 0.0001, "b": 0.004, "c": 0.030, "d": 0.031, "e": 0.040}
    out = holm_reject(p, alpha=0.05)
    assert out["a"] and out["b"]        # 0.0001 <= .01, 0.004 <= .0125
    assert not out["c"]                 # 0.030 > 0.05/3 -> stop
    assert not out["d"] and not out["e"]


def test_holm_keeps_a_single_test_uncorrected():
    assert holm_reject({"only": 0.04}, alpha=0.05) == {"only": True}


def test_sum_by_bundle_groups_repeat_captures_and_keeps_pooled_ratios():
    """Repeat captures of one bundle collapse into one resampling unit; a
    scene outside any group stays its own unit; pooled ratios do not move."""
    bundles = {"a/1": "A", "a/2": "A"}
    scenes = ["a/1", "b/1", "a/2"]
    den = np.array([2.0, 3.0, 4.0])
    num = np.array([1.0, 3.0, 2.0])
    labels, d, n = sum_by_bundle(scenes, den, num, bundles=bundles)
    assert labels == ["A", "b/1"]
    assert d.tolist() == [6.0, 3.0] and n.tolist() == [3.0, 3.0]
    assert n.sum() / d.sum() == num.sum() / den.sum()
    # 2-D rows (per-scene tp, fp, fn) are summed row-wise
    _, rows = sum_by_bundle(scenes, np.array([[1, 0, 1], [2, 1, 0], [1, 1, 1]]),
                            bundles=bundles)
    assert rows.tolist() == [[2, 1, 2], [2, 1, 0]]
