"""Tests for detect/pool.py selection and de-duplication on hand-built
candidate pools."""

import numpy as np

from pairbench.detect.pool import (
    _dedupe_pool,
    _jaccard,
    select_greedy,
    select_quality_aware,
)


def _fake_pool(n):
    """n placeholder Detection-like objects; only .score is read by
    _dedupe_pool's ranking."""
    from pairbench.detect.ccorr import Detection

    return [
        Detection(side=1, angle_deg=0.0, score=float(n - i), ccorr_frac=0.5,
                   kind="pool", obb=((0.0, 0.0), (1.0, 1.0), 0.0))
        for i in range(n)
    ]


# --- select_greedy -----------------------------------------------------


# Claim sizes are well above POOL_INSTANCE_COST_DEFAULT (300), the default
# cost of select_greedy (detect_faces_pool uses a scene-relative cost).


def test_greedy_selects_two_disjoint_real_candidates():
    """Two candidates whose claimed points are entirely disjoint (two
    genuinely distinct bars, adjacent enough that their footprints could
    overlap in 2D but whose actual 3D data does not) must BOTH be selected
    -- the core property an IoU-only NMS threshold cannot express."""
    n_points = 16_000
    footprints = [np.arange(0, 8_000), np.arange(8_000, 16_000)]
    claims = [np.arange(0, 8_000), np.arange(8_000, 16_000)]
    selected = select_greedy(footprints, claims, n_points)
    assert set(selected) == {0, 1}


def test_greedy_rejects_near_duplicate_of_same_instance():
    """Two candidates claiming nearly the SAME points (a same-instance
    re-detection at a slightly different angle) -- only one should be
    selected, since the second's marginal gain over the first is ~0."""
    n_points = 8_000
    footprints = [np.arange(0, 8_000), np.arange(0, 8_000)]
    claims = [np.arange(0, 8_000), np.arange(0, 7_600)]  # heavy overlap
    selected = select_greedy(footprints, claims, n_points)
    assert len(selected) == 1


def test_greedy_rejects_weak_mostly_foreign_candidate():
    """A candidate whose footprint is mostly unexplained/foreign data (a
    junk peak that happens to sit near real bars) must not be selected --
    its own claimed-point count is too small relative to the leftover
    foreign residual to clear POOL_INSTANCE_COST."""
    n_points = 8_000
    footprints = [np.arange(0, 8_000)]
    claims = [np.arange(0, 50)]  # claims almost nothing of its own footprint
    selected = select_greedy(footprints, claims, n_points)
    assert selected == []


def test_greedy_occluder_reduces_partial_candidates_penalty():
    """The joint-explanation property: a truncated candidate whose visible
    slice is real but whose footprint is mostly a NEIGHBOUR's claimed data
    should be selected once the neighbour (occluder) is already selected --
    the neighbour's claim removes those points from "foreign" in the
    truncated candidate's own penalty term."""
    n_points = 16_000
    # candidate 0: the occluder, a full strong face
    fp0 = np.arange(0, 7_500)
    cl0 = np.arange(0, 7_500)
    # candidate 1: truncated face -- claims only a small genuine slice
    # (7500-8500), but its FOOTPRINT overlaps candidate 0's claimed region
    # (0-7500) heavily, which is real data belonging to the occluder, not
    # foreign/absent once candidate 0 is selected.
    fp1 = np.arange(0, 8_500)
    cl1 = np.arange(7_500, 8_500)
    footprints = [fp0, fp1]
    claims = [cl0, cl1]
    selected = select_greedy(footprints, claims, n_points)
    assert set(selected) == {0, 1}, (
        "the occluder must be picked, and picking it must make the "
        "truncated candidate's own footprint residual small enough to "
        "clear a positive marginal gain too"
    )


def test_greedy_partial_candidate_alone_not_selected_without_occluder():
    """Sanity counterpart: the same truncated candidate from the test
    above, evaluated WITHOUT its occluder ever being a candidate at all --
    its footprint's foreign residual is real (unexplained, unclaimed
    elsewhere) and must not be swept under the rug."""
    n_points = 16_000
    fp1 = np.arange(0, 8_500)
    cl1 = np.arange(7_500, 8_500)
    selected = select_greedy([fp1], [cl1], n_points)
    assert selected == []


def test_greedy_per_candidate_cost_accepts_small_real_candidate_scalar_would_reject():
    """`instance_cost` as a per-candidate array (scaled to each candidate's
    own claim size) versus a single scene-relative scalar (scaled to the
    scene's largest claim): a genuinely real but small candidate, disjoint
    from a much larger one, is rejected under the scalar but accepted under
    the per-candidate array -- the mechanism difference under test."""
    n_points = 8_400
    footprints = [np.arange(0, 8_000), np.arange(8_000, 8_400)]
    claims = [np.arange(0, 8_000), np.arange(8_000, 8_400)]

    scalar_cost = 0.4 * 8_000  # scene-relative: scaled to the larger claim
    assert select_greedy(footprints, claims, n_points,
                          instance_cost=scalar_cost) == [0]

    per_candidate_cost = 0.4 * np.array([8_000, 400], dtype=np.float64)
    assert set(select_greedy(footprints, claims, n_points,
                              instance_cost=per_candidate_cost)) == {0, 1}


def test_quality_aware_per_candidate_cost_indexes_by_original_id_not_position():
    """select_quality_aware's post-hoc correction re-runs select_greedy on a
    kept subset of the pool. When instance_cost is a per-candidate array,
    the subset passed to that second call must be re-indexed by the kept
    candidates' ORIGINAL ids, not by their new position in the subset list
    -- otherwise a candidate silently gets a different candidate's cost.

    Built directly: a low per-candidate cost makes a weak, low-ccorr
    candidate (0) win the first select_greedy pass over a strong, heavily
    overlapping candidate (1) it should have lost to; the quality-dominance
    correction must then swap them, which requires candidate 1's OWN cost
    (5000, index 1 in the original array), not candidate 0's (10, what an
    off-by-position subset would read at local index 0)."""
    from pairbench.detect.ccorr import Detection

    n_points = 8_000
    fp0 = cl0 = np.arange(0, 8_000)          # weak, claims everything
    fp1 = cl1 = np.arange(0, 7_800)          # strong, near-total overlap with 0
    footprints = [fp0, fp1]
    claims = [cl0, cl1]
    pool = [
        Detection(side=1, angle_deg=0.0, score=0.5, ccorr_frac=0.5,
                   kind="pool", obb=((0.0, 0.0), (1.0, 1.0), 0.0)),
        Detection(side=1, angle_deg=0.0, score=0.9, ccorr_frac=0.9,
                   kind="pool", obb=((0.0, 0.0), (1.0, 1.0), 0.0)),
    ]
    instance_cost = np.array([10.0, 5000.0])

    # Sanity: plain select_greedy picks the weak candidate first, and the
    # strong one never recovers (its claim is a subset of the weak one's).
    assert select_greedy(footprints, claims, n_points,
                          instance_cost=instance_cost) == [0]

    selected = select_quality_aware(pool, footprints, claims, n_points,
                                     instance_cost=instance_cost)
    assert selected == [1], (
        "the dominance correction must drop the weak candidate and re-run "
        "on the strong one with ITS OWN cost (5000, not 10)"
    )


# --- _dedupe_pool / _jaccard ----------------------------------------------


def test_jaccard_identical_and_disjoint():
    a = np.arange(0, 10)
    b = np.arange(0, 10)
    c = np.arange(10, 20)
    assert _jaccard(a, b) == 1.0
    assert _jaccard(a, c) == 0.0


def test_dedupe_collapses_near_identical_claims_keeps_best_score():
    pool = _fake_pool(3)  # scores 3.0, 2.0, 1.0 (index 0 highest)
    footprints = [np.arange(0, 50)] * 3
    claims = [np.arange(0, 50), np.arange(0, 48), np.arange(100, 150)]
    kept_pool, kept_fp, kept_cl = _dedupe_pool(pool, footprints, claims)
    # candidate 0 and 1 are near-identical (jaccard 48/50=0.96 > 0.6) -> only
    # the higher-scoring (index 0) survives; candidate 2 is disjoint -> kept.
    assert len(kept_pool) == 2
    scores = sorted(d.score for d in kept_pool)
    assert scores == [1.0, 3.0]
