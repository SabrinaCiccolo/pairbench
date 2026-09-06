from pairbench.detect.ccorr import Detection
from pairbench.metrics import bootstrap_f1_ci, count_inversions, precision_recall_f1


def det(side, y, z, angle):
    """Synthetic detection; only side/centroid_yz_mm/extents matter here."""
    return Detection(
        side=side, angle_deg=angle, score=0.9, ccorr_frac=0.9, kind="confirmed",
        obb=((y, z), (40.0, 12.0), -angle),
        centroid_yz_mm=(y, z),
        centroid_world_mm=(3002.5 if side == 1 else -3002.5, y, z),
        normal_world=(1.0, 0.0, 0.0),
        n_points=1000,
        z_extent_mm=(z - 6.0, z + 6.0),
        y_extent_mm=(y - 20.0, y + 20.0),
    )


def test_precision_recall_f1_normal_case():
    p, r, f1 = precision_recall_f1(tp=3, fp=1, fn=2)
    assert abs(p - 0.75) < 1e-9
    assert abs(r - 0.6) < 1e-9
    assert abs(f1 - (2 * 0.75 * 0.6 / (0.75 + 0.6))) < 1e-9


def test_precision_recall_f1_perfect():
    assert precision_recall_f1(tp=5, fp=0, fn=0) == (1.0, 1.0, 1.0)


def test_precision_recall_f1_all_zero():
    # no predictions, no ground truth: both ratios are 0/0 -> defined as 0.0
    assert precision_recall_f1(tp=0, fp=0, fn=0) == (0.0, 0.0, 0.0)


def test_precision_recall_f1_no_predictions_but_gt_exists():
    # tp+fp == 0 -> precision 0/0 -> 0.0; recall well-defined at 0.0
    p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=4)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_precision_recall_f1_no_gt_but_predictions_exist():
    # tp+fn == 0 -> recall 0/0 -> 0.0; precision well-defined at 0.0
    p, r, f1 = precision_recall_f1(tp=0, fp=4, fn=0)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_precision_recall_f1_zero_f1_when_p_xor_r_zero():
    # precision 0, recall > 0 (or vice versa): harmonic mean must stay 0,
    # not divide by (p + r) == r and silently return something nonzero.
    p, r, f1 = precision_recall_f1(tp=0, fp=3, fn=2)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_count_inversions_empty_pairs():
    d1 = [det(1, 100, 20, 0)]
    d2 = [det(2, 100, 20, 0)]
    assert count_inversions([], d1, d2) == 0


def test_count_inversions_consistent_order_no_inversion():
    d1 = [det(1, 100, 20, 0), det(1, 200, 20, 0)]
    d2 = [det(2, 100, 20, 0), det(2, 200, 20, 0)]
    # identity pairing: Y order agrees across both views
    assert count_inversions([(0, 0), (1, 1)], d1, d2) == 0


def test_count_inversions_swapped_pair_counts_one():
    d1 = [det(1, 100, 20, 0), det(1, 200, 20, 0)]
    d2 = [det(2, 100, 20, 0), det(2, 200, 20, 0)]
    # side-2 indices swapped relative to side-1 Y order -> one inversion
    assert count_inversions([(0, 1), (1, 0)], d1, d2) == 1


def test_count_inversions_different_z_levels_not_counted():
    # third bar in d1 stacked directly above bar 0 (same Y span, higher Z):
    # different stacking level on side 1 -> pair (0,_) and (2,_) must never
    # be compared for inversion even though their Y order disagrees.
    d1 = [det(1, 100, 20, 0), det(1, 200, 20, 0), det(1, 100, 60, 0)]
    d2 = [det(2, 100, 20, 0), det(2, 200, 20, 0), det(2, 300, 20, 0)]
    pairs = [(0, 0), (1, 1), (2, 2)]
    assert count_inversions(pairs, d1, d2) == 0


def test_bootstrap_f1_ci_empty_returns_zero():
    assert bootstrap_f1_ci([]) == (0.0, 0.0)


def test_bootstrap_f1_ci_perfect_scenes_gives_tight_interval_at_one():
    # every scene is a perfect match: no amount of resampling can produce
    # anything but F1=1.0, so the interval collapses to a point.
    scenes = [(5, 0, 0)] * 20
    lo, hi = bootstrap_f1_ci(scenes, n_boot=500, seed=0)
    assert abs(lo - 1.0) < 1e-9
    assert abs(hi - 1.0) < 1e-9


def test_bootstrap_f1_ci_brackets_the_point_estimate():
    # mixed scenes: point estimate must fall inside its own CI.
    scenes = [(3, 1, 0), (2, 0, 1), (4, 0, 0), (1, 2, 1), (3, 0, 2)]
    tp = sum(v[0] for v in scenes)
    fp = sum(v[1] for v in scenes)
    fn = sum(v[2] for v in scenes)
    _, _, f1 = precision_recall_f1(tp, fp, fn)
    lo, hi = bootstrap_f1_ci(scenes, n_boot=2000, seed=0)
    assert lo <= f1 <= hi


def test_bootstrap_f1_ci_narrows_with_more_scenes():
    # two different scene "types" alternating, more of each -> the
    # resampled mix varies less, so the CI tightens. (An identical-tuple
    # dataset resamples to the exact same sum regardless of scene count,
    # so this needs genuine scene-to-scene variation to be a real test.)
    pattern = [(3, 1, 1), (1, 3, 3)]
    few = pattern * 3    # 6 scenes
    many = pattern * 30  # 60 scenes
    lo_few, hi_few = bootstrap_f1_ci(few, n_boot=2000, seed=0)
    lo_many, hi_many = bootstrap_f1_ci(many, n_boot=2000, seed=0)
    assert (hi_many - lo_many) < (hi_few - lo_few)
