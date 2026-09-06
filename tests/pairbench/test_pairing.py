from pairbench.detect.ccorr import Detection
from pairbench.pairing.common import (
    angle_difference,
    fold_angle,
    group_by_z_level,
)
from pairbench.pairing.greedy_index import pair_greedy_index
from pairbench.pairing.hungarian import pair_hungarian
from pairbench.pairing.hungarian_aligned import pair_hungarian_aligned
from pairbench.pairing.hybrid import (
    bed_height_mm,
    elevated_outliers,
    pair_hybrid_aligned,
)
from pairbench.pairing.monotone import (
    monotone_assignment,
    pair_monotone_aligned,
)
from pairbench.pairing.hypotheses import (
    enumerate_hypotheses,
    hypothesis_cost,
    shift_hypothesis,
    translation_of,
)


def det(side, y, z, angle, x=None, y_ext=40.0, z_ext=12.0):
    """Synthetic detection; world X = +-3002.5 so pair distance ~ 6005 mm."""
    if x is None:
        x = 3002.5 if side == 1 else -3002.5
    return Detection(
        side=side, angle_deg=angle, score=0.9, ccorr_frac=0.9, kind="confirmed",
        obb=((y, z), (40.0, 12.0), -angle),
        centroid_yz_mm=(y, z),
        centroid_world_mm=(x, y, z),
        normal_world=(1.0, 0.0, 0.0),
        n_points=1000,
        z_extent_mm=(z - z_ext / 2, z + z_ext / 2),
        y_extent_mm=(y - y_ext / 2, y + y_ext / 2),
    )


def test_fold_angle():
    assert fold_angle(359.0, 360.0) == -1.0
    assert fold_angle(180.0, 360.0) == -180.0
    assert fold_angle(92.0, 90.0) == 2.0
    assert angle_difference(det(1, 0, 0, 0.0), det(2, 0, 0, 358.0)) == 2.0


def test_z_level_grouping():
    # two flat bars + one stacked on the first
    dets = [det(1, 100, 20, 0), det(1, 200, 20, 0), det(1, 100, 60, 0)]
    levels = group_by_z_level(dets)
    assert len(levels) == 2
    assert set(levels[0]) == {0, 1}
    assert levels[1] == [2]


def test_flat_scene_all_strategies_pair_all():
    d1 = [det(1, y, 20, 180.0) for y in (100, 200, 300)]
    d2 = [det(2, y, 20, 180.0) for y in (100, 200, 300)]
    for fn in (pair_greedy_index, pair_hungarian, pair_hungarian_aligned):
        res = fn(d1, d2)
        assert len(res.pairs) == 3, (res.strategy, res.pairs)
    resid = pair_greedy_index(d1, d2).diagnostics["length_residuals_mm"]
    assert all(abs(r) < 10.0 for r in resid)


def test_count_mismatch_leaves_unmatched():
    d1 = [det(1, y, 20, 180.0) for y in (100, 200, 300)]
    d2 = [det(2, y, 20, 180.0) for y in (100, 200)]
    for fn in (pair_greedy_index, pair_hungarian, pair_hungarian_aligned):
        res = fn(d1, d2)
        assert len(res.pairs) == 2, res.strategy
        assert len(res.unmatched_1) == 1, res.strategy


def test_crossed_scene_hungarian_beats_greedy():
    # bars A(0 deg), B(0 deg) at the same Y in both views; C(30 deg) sits at
    # Y=100 in side 1 but Y=400 in side 2 (cross-view position swap)
    d1 = [det(1, 300, 20, 0.0), det(1, 200, 20, 0.0), det(1, 100, 20, 30.0)]
    d2 = [det(2, 300, 20, 0.0), det(2, 200, 20, 0.0), det(2, 400, 20, 30.0)]
    hung = pair_hungarian(d1, d2)
    assert len(hung.pairs) == 3
    assert (2, 2) in hung.pairs  # the crossed bar is paired by angle
    greedy = pair_greedy_index(d1, d2)
    # index pairing misaligns the whole row: twist check kills the aliased
    # pairs, so greedy pairs strictly fewer than hungarian
    assert len(greedy.pairs) < len(hung.pairs)


def test_crossed_scene_aligned_keeps_crossed_pair():
    # same scene as above: translation compensation must not break the
    # genuinely crossed bar (its residual distance saturates at the cap but
    # stays below the unmatch cost)
    d1 = [det(1, 300, 20, 0.0), det(1, 200, 20, 0.0), det(1, 100, 20, 30.0)]
    d2 = [det(2, 300, 20, 0.0), det(2, 200, 20, 0.0), det(2, 400, 20, 30.0)]
    res = pair_hungarian_aligned(d1, d2)
    assert len(res.pairs) == 3
    assert (2, 2) in res.pairs


def test_hungarian_pair_margins_present_and_positive_for_unambiguous_scene():
    d1 = [det(1, 0, 0, 0.0), det(1, 1000, 0, 0.0)]
    d2 = [det(2, 0, 0, 0.0), det(2, 1000, 0, 0.0)]
    res = pair_hungarian(d1, d2)
    margins = res.diagnostics["pair_margins"]
    assert len(margins) == len(res.pairs) == 2
    assert all(m > 0 for m in margins)  # unambiguous: own-index beats the swap


def test_hungarian_pair_margin_near_zero_for_near_tie():
    # side-1 bar 0 sits ~equidistant from two side-2 candidates: whichever
    # the solver picks, the alternative was nearly as good.
    d1 = [det(1, 0, 0, 0.0)]
    d2 = [det(2, 0, 0, 0.0), det(2, 0.01, 0, 0.0)]
    res = pair_hungarian(d1, d2)
    margins = res.diagnostics["pair_margins"]
    assert len(margins) == 1
    assert 0.0 <= margins[0] < 0.1


def test_hungarian_pair_margin_single_candidate_uses_unmatch_cost():
    # no alternative real column exists: the margin is against leaving the
    # bar unmatched, not against another bar.
    d1 = [det(1, 0, 0, 0.0)]
    d2 = [det(2, 0, 0, 0.0)]
    res = pair_hungarian(d1, d2)
    margins = res.diagnostics["pair_margins"]
    assert len(margins) == 1
    assert margins[0] > 3.0  # near UNMATCH_COST (6.0): a near-zero real pair cost


def test_hungarian_aligned_forwards_pair_margins():
    d1 = [det(1, 0, 0, 0.0), det(1, 1000, 0, 0.0)]
    d2 = [det(2, 0, 0, 0.0), det(2, 1000, 0, 0.0)]
    res = pair_hungarian_aligned(d1, d2)
    assert len(res.diagnostics["pair_margins"]) == len(res.pairs)


def test_hungarian_aligned_recovers_from_offset_and_quantized_angle():
    # a ~+19 mm cross-view offset plus a quantized angle mismatch (178 vs
    # 180) on one bar: the aligned arm must recover the sorted-order mapping.
    y1 = (1655, 1583, 1770, 1721, 1803)
    a1 = (180.0, 180.0, 178.0, 178.0, 178.0)
    y2 = (1601, 1674, 1822, 1751, 1790)
    a2 = (180.0, 180.0, 178.0, 180.0, 180.0)
    d1 = [det(1, y, 20, a) for y, a in zip(y1, a1)]
    d2 = [det(2, y, 20, a) for y, a in zip(y2, a2)]
    truth = {(1, 0), (0, 1), (3, 3), (2, 4), (4, 2)}  # sorted-Y order
    assert set(pair_hungarian_aligned(d1, d2).pairs) == truth


def test_aligned_impostor_does_not_beat_noisy_true_pair():
    # after compensation the true partner carries a 6 deg angle error (cost
    # 2.4), an unpairable leftover sits far away (capped). The cap must
    # exceed noise so the true pair wins; the leftovers pair with each other
    # (both capped) rather than stealing the true partner.
    d1 = [det(1, 1616, 20, 180.0), det(1, 1720, 20, 180.0),
          det(1, 1685, 20, 180.0)]
    d2 = [det(2, 1649, 20, 180.0), det(2, 1719, 20, 180.0),
          det(2, 1823, 20, 180.0), det(2, 1781, 20, 174.0)]
    # true global t ~ -103: 1616<-1719, 1720<-1823, 1685<-1781(6deg off)
    res = pair_hungarian_aligned(d1, d2)
    assert (0, 1) in res.pairs and (1, 2) in res.pairs
    assert (2, 3) in res.pairs, res.pairs  # not the capped impostor (2, 0)


def test_greedy_length_check_toggle():
    d1 = [det(1, 100, 20, 180.0, x=4000.0)]  # wrong X -> |c1-c2| far from 6005
    d2 = [det(2, 100, 20, 180.0)]
    assert len(pair_greedy_index(d1, d2).pairs) == 0
    assert len(pair_greedy_index(d1, d2, check_length=False).pairs) == 1


def test_greedy_length_bias_restores_the_check():
    # a constant world-X bias shortens every pair's |c1-c2|, so the plain
    # check rejects everything. Compensating that one scalar must restore it
    # without making the check vacuous -- a genuinely wrong-length bar still
    # fails.
    bias = -1000.0
    d1 = [det(1, 100, 20, 180.0, x=3002.5 + bias / 2)]
    d2 = [det(2, 100, 20, 180.0, x=-3002.5 - bias / 2)]
    assert len(pair_greedy_index(d1, d2).pairs) == 0
    assert len(pair_greedy_index(d1, d2, length_bias_mm=bias).pairs) == 1
    off = [det(1, 100, 20, 180.0, x=3002.5 + bias / 2 + 50.0)]
    assert len(pair_greedy_index(off, d2, length_bias_mm=bias).pairs) == 0


def test_greedy_order_only_ignores_every_check():
    # a pair the twist check rejects (angles 20 deg apart) and whose length is
    # wrong: the plain arm drops it, the order-only arm keeps it, since with
    # all three checks off the arm is instance order and nothing else
    d1 = [det(1, 100, 20, 180.0, x=4000.0)]
    d2 = [det(2, 100, 20, 160.0)]
    assert len(pair_greedy_index(d1, d2, check_length=False).pairs) == 0
    order_only = pair_greedy_index(d1, d2, check_length=False,
                                   check_twist=False, check_tilt=False)
    assert order_only.pairs == [(0, 0)]


def test_monotone_refuses_the_crossing_free_assignment_takes():
    # Two bars whose quantized angles point at each other's partner: the free
    # assignment buys the angle agreement with a crossing (both distances are
    # capped, so the crossing is nearly free). Same-level bars on a conveyor
    # cannot interleave, and the order constraint says so.
    d1 = [det(1, 100, 20, 0.0), det(1, 200, 20, 30.0)]
    d2 = [det(2, 110, 20, 30.0), det(2, 210, 20, 0.0)]
    free = set(pair_hungarian_aligned(d1, d2).pairs)
    mono = sorted(pair_monotone_aligned(d1, d2).pairs)
    assert free == {(0, 1), (1, 0)}, free  # crossed
    # order preserved: side-2 indices ascend with side-1 indices (an offset
    # mapping is fine, an interleaving is not)
    assert [j for _, j in mono] == sorted(j for _, j in mono), mono


def test_monotone_cannot_express_a_crossed_bundle():
    # Same fixture the aligned arm passes: on a crossed bundle the order
    # constraint loses the crossed pair.
    d1 = [det(1, 300, 20, 0.0), det(1, 200, 20, 0.0), det(1, 100, 20, 30.0)]
    d2 = [det(2, 300, 20, 0.0), det(2, 200, 20, 0.0), det(2, 400, 20, 30.0)]
    assert (2, 2) in pair_hungarian_aligned(d1, d2).pairs
    assert (2, 2) not in pair_monotone_aligned(d1, d2).pairs


def test_monotone_assignment_preserves_order():
    import numpy as np

    # cheapest free assignment would be the anti-diagonal; the constraint
    # forbids it, so the DP must fall back to the ordered matching.
    cost = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert monotone_assignment(cost, unmatch_cost=6.0) == [(0, 0), (1, 1)]


def test_monotone_can_leave_elements_unmatched():
    import numpy as np

    cost = np.array([[0.0, 9.0], [9.0, 0.0], [9.0, 9.0]])
    assert monotone_assignment(cost, unmatch_cost=6.0) == [(0, 0), (1, 1)]


def test_hybrid_recovers_the_crossed_bundle():
    # one bar lies across the bundle, so it reads Y=1871 in side 1 and
    # Y=1647 in side 2 while the three bars under it shift by a uniform
    # ~56 mm. It is elevated (+17 / +25 mm) in BOTH views -- the
    # corroboration the crossing hypothesis needs. Sorted left-to-right the
    # true mapping looks cyclic.
    d1 = [det(1, y, z, 180.0) for y, z in
          ((1647, 916.9), (1716, 916.4), (1785, 917.2), (1871, 933.9))]
    d2 = [det(2, y, z, 180.0) for y, z in
          ((1647, 941.9), (1706, 916.5), (1772, 917.2), (1838, 918.2))]
    truth = {(0, 1), (1, 2), (2, 3), (3, 0)}
    assert set(pair_hybrid_aligned(d1, d2).pairs) == truth
    # both bracketing arms miss it: the free one lets the crossing bar's
    # 224 mm offset corrupt the consensus translation, the constrained one
    # cannot express a crossing at all
    assert set(pair_hungarian_aligned(d1, d2).pairs) != truth
    assert set(pair_monotone_aligned(d1, d2).pairs) != truth


def test_hybrid_needs_both_views_to_see_the_elevation():
    # Same scene with side 2's crossing bar sitting at bed height: one view's
    # elevation is not corroboration, so no crossing may be hypothesized and
    # the arm must fall back to the order-constrained assignment.
    d1 = [det(1, y, z, 180.0) for y, z in
          ((1647, 916.9), (1716, 916.4), (1785, 917.2), (1871, 933.9))]
    d2 = [det(2, y, z, 180.0) for y, z in
          ((1647, 917.0), (1706, 916.5), (1772, 917.2), (1838, 918.2))]
    res = pair_hybrid_aligned(d1, d2)
    assert res.diagnostics["crossing_hypothesis"] is False
    assert [j for _, j in res.pairs] == sorted(j for _, j in res.pairs), res.pairs


def test_hybrid_recovers_a_bar_raised_at_one_end_only():
    # A bar lying across the bundle with one end on top of its neighbours
    # (side 1, Z ~951) and the other on the bed beyond them (side 2, Z ~918),
    # after l-profile/crossed-07. Elevated in one view only, so there is no
    # corroboration, but its raised end has no partner within the consensus
    # gate under the scene translation: it breaks the rigid motion alone.
    # Its true pair carries the level and height-outlier penalties plus a
    # saturated distance, above the unmatch cost, so without the one-view
    # path the arm leaves it unpaired.
    d1 = [det(1, y, z, a) for y, z, a in
          ((1651.4, 920.2, 0.0), (1751.1, 950.7, 180.0), (1752.8, 917.6, 180.0))]
    d2 = [det(2, y, z, a) for y, z, a in
          ((1568.3, 918.7, 0.0), (1690.6, 916.1, 180.0), (1858.7, 917.6, 180.0))]
    res = pair_hybrid_aligned(d1, d2)
    assert res.diagnostics["one_view_crossing"] is True
    assert set(res.pairs) == {(0, 0), (1, 2), (2, 1)}, res.pairs


def test_elevation_alone_is_not_a_crossing():
    # A bar merely stacked on top of the bundle is elevated in both views but
    # still travels with it. Peeling it out as a "crossing" would break an
    # otherwise perfect pairing, so the arm must check that the elevated bars
    # actually broke the bundle's rigid motion before believing them.
    d1 = [det(1, 100, 20, 0.0), det(1, 200, 20, 0.0), det(1, 300, 20, 0.0),
          det(1, 205, 60, 0.0)]
    d2 = [det(2, 150, 20, 0.0), det(2, 250, 20, 0.0), det(2, 350, 20, 0.0),
          det(2, 255, 60, 0.0)]   # same +50 shift as the bed: stacked, not crossed
    res = pair_hybrid_aligned(d1, d2)
    assert res.diagnostics["elevation_corroborated"] is True
    assert res.diagnostics["moves_with_bundle"] is True
    assert res.diagnostics["crossing_hypothesis"] is False
    assert set(res.pairs) == {(0, 0), (1, 1), (2, 2), (3, 3)}, res.pairs
    # the same scene with that bar NOT following the bundle is a crossing
    d2[3] = det(2, 700, 60, 0.0)
    assert pair_hybrid_aligned(d1, d2).diagnostics["crossing_hypothesis"] is True


def test_order_constraint_is_per_stacking_level():
    # Two layers: bars 7/9 rest on top in side 1, 4/6 in side 2. A bed bar
    # and the bar stacked above it read in opposite order between the views
    # (1795/1800 vs 1732/1731), which a global order constraint cannot
    # express and a per-level one can.
    s1 = ((1549, 924.7, 0), (1583, 925.1, 0), (1624, 925.0, 0),
          (1671, 925.9, 182), (1713, 926.2, 0), (1753, 927.0, 0),
          (1795, 925.6, 0), (1800, 964.2, 88), (1838, 924.9, 268),
          (1843, 967.0, 88))
    s2 = ((1566, 925.0, 0), (1603, 924.5, 0), (1648, 925.0, 182),
          (1689, 925.1, 182), (1731, 967.3, 0), (1732, 926.6, 176),
          (1771, 966.3, 0), (1776, 926.0, 0))
    d1 = [det(1, y, z, a) for y, z, a in s1]
    d2 = [det(2, y, z, a) for y, z, a in s2]
    truth = {(2, 0), (3, 1), (4, 2), (5, 3), (6, 5), (7, 4), (8, 7), (9, 6)}
    assert set(pair_hybrid_aligned(d1, d2, period_deg=90.0).pairs) == truth
    # ... and the global constraint provably cannot: it loses the two
    # level-crossing pairs
    mono = set(pair_monotone_aligned(d1, d2, period_deg=90.0).pairs)
    assert (7, 4) not in mono and (9, 6) not in mono, mono


def test_order_constraint_per_level_survives_an_extra_level_in_one_view():
    # The fixture above with one more bar stacked on top in side 2 only, so
    # the views report 2 and 3 levels. Levels are then matched by height
    # rather than by index, and the per-level solution must still hold;
    # the extra bar has no partner and stays unmatched.
    s1 = ((1549, 924.7, 0), (1583, 925.1, 0), (1624, 925.0, 0),
          (1671, 925.9, 182), (1713, 926.2, 0), (1753, 927.0, 0),
          (1795, 925.6, 0), (1800, 964.2, 88), (1838, 924.9, 268),
          (1843, 967.0, 88))
    s2 = ((1566, 925.0, 0), (1603, 924.5, 0), (1648, 925.0, 182),
          (1689, 925.1, 182), (1731, 967.3, 0), (1732, 926.6, 176),
          (1771, 966.3, 0), (1776, 926.0, 0), (1772, 1006.0, 0))
    d1 = [det(1, y, z, a) for y, z, a in s1]
    d2 = [det(2, y, z, a) for y, z, a in s2]
    assert len(group_by_z_level(d1)) == 2 and len(group_by_z_level(d2)) == 3
    truth = {(2, 0), (3, 1), (4, 2), (5, 3), (6, 5), (7, 4), (8, 7), (9, 6)}
    pairs = set(pair_hybrid_aligned(d1, d2, period_deg=90.0).pairs)
    assert truth <= pairs, pairs
    assert all(j != 8 for _, j in pairs), pairs


def test_hybrid_keeps_refusing_the_impostor_crossing():
    # monotone's fixture: a crossing the angles make cheap, with no elevation
    # anywhere. Order still rules -- the crossing hypothesis is
    # evidence-gated, not a licence to interleave whenever the cost says so.
    d1 = [det(1, 100, 20, 0.0), det(1, 200, 20, 30.0)]
    d2 = [det(2, 110, 20, 30.0), det(2, 210, 20, 0.0)]
    res = pair_hybrid_aligned(d1, d2)
    assert res.diagnostics["crossing_hypothesis"] is False
    assert [j for _, j in res.pairs] == sorted(j for _, j in res.pairs), res.pairs


def test_bed_height_is_pooled_across_views():
    # side 2 sees only 2 faces, one of them the crossing bar, so its own
    # median lands halfway up that bar and would call the OTHER (bed) bar a
    # low outlier. Pooling both views puts the bed where the four
    # bed-height faces are.
    d1 = [det(1, 1581, 915.5, 180.0), det(1, 1790, 914.3, 180.0),
          det(1, 1803, 940.8, 180.0)]
    d2 = [det(2, 1710, 940.1, 180.0), det(2, 1736, 916.9, 180.0)]
    bed = bed_height_mm(d1, d2)
    assert abs(bed - 916.9) < 1e-6, bed
    assert elevated_outliers(d1, bed) == [2]
    assert elevated_outliers(d2, bed) == [0]
    assert bed_height_mm(d1[:1], d2[:1]) is None  # under-determined


def test_symmetry_period_folds_the_angle_cost():
    # Square tube (period 90): the same face read 90 deg apart across views
    # is the same pose and must still pair. At the 360 deg fallback the same
    # pair is charged the full angle penalty and stays unmatched.
    d1 = [det(1, 100, 20, 0.0)]
    d2 = [det(2, 100, 20, 90.0)]
    assert len(pair_hungarian(d1, d2, period_deg=360.0).pairs) == 0
    assert len(pair_hungarian(d1, d2, period_deg=90.0).pairs) == 1
    assert len(pair_hungarian_aligned(d1, d2, period_deg=90.0).pairs) == 1
    assert len(pair_greedy_index(d1, d2, period_deg=90.0,
                                 check_length=False).pairs) == 1


def bar(side, y, y_lo, y_hi, z=917.0, angle=180.0):
    """Detection whose claimed Y-extent is given explicitly -- a partially
    occluded bar's extent is not centred on its centroid."""
    d = det(side, y, z, angle)
    d.y_extent_mm = (y_lo, y_hi)
    return d


def test_hypothesis_cost_scores_at_the_hypothesis_own_translation():
    """A hypothesis must not be charged for the anchor that proposed it, or
    the enumeration ranks by anchor luck instead of by fit."""
    d1 = [bar(1, y, y - 20.0, y + 20.0) for y in (1600.0, 1670.0, 1740.0)]
    d2 = [bar(2, y + 12.0, y - 8.0, y + 32.0) for y in (1600.0, 1670.0, 1740.0)]
    pairs = [(0, 0), (1, 1), (2, 2)]
    t = translation_of(pairs, d1, d2)
    assert abs(float(t[0]) + 12.0) < 1e-9
    # every pair sits exactly on the compensated position, so the objective
    # charges only the unmatch-free assignment: zero
    assert hypothesis_cost(pairs, d1, d2, period_deg=360.0) < 1e-6


def test_enumerate_hypotheses_returns_rivals_cheapest_first():
    """`cost_gap` asks whether the objective ranks the true correspondence
    first, which needs the rivals enumerated and sorted by their own cost."""
    d1 = [bar(1, y, y - 20.0, y + 20.0) for y in (1600.0, 1670.0, 1740.0)]
    d2 = [bar(2, y, y - 20.0, y + 20.0)
          for y in (1600.0, 1670.0, 1740.0, 1810.0)]
    hyps = enumerate_hypotheses(d1, d2, 360.0, 6.0, 3.0)
    assert len(hyps) > 1
    costs = [c for c, _t, _p in hyps]
    assert costs == sorted(costs)
    # the identity correspondence is among the enumerated rivals
    assert any(pairs == [(0, 0), (1, 1), (2, 2)] for _c, _t, pairs in hyps)


def test_shift_hypothesis_builds_the_shift_family_per_level():
    """pi_k pairs face i with face i+k inside each level and leaves |k| faces
    over at one end of each view; pi_0 is the identity."""
    d1 = [bar(1, y, y - 20.0, y + 20.0) for y in (1600.0, 1670.0, 1740.0)]
    d2 = [bar(2, y, y - 20.0, y + 20.0) for y in (1600.0, 1670.0, 1740.0)]
    # faces inside a level are ordered by descending Y: index 2 comes first
    assert shift_hypothesis(d1, d2, 0) == [(0, 0), (1, 1), (2, 2)]
    assert shift_hypothesis(d1, d2, 1) == [(1, 0), (2, 1)]
    assert shift_hypothesis(d1, d2, -1) == [(0, 1), (1, 2)]
    assert shift_hypothesis(d1, d2, 3) == []
    # a shift pays the unmatch cost for its two leftovers, so on a regular
    # bundle the truth is cheaper
    assert (hypothesis_cost(shift_hypothesis(d1, d2, 1), d1, d2, 360.0)
            > hypothesis_cost(shift_hypothesis(d1, d2, 0), d1, d2, 360.0))
