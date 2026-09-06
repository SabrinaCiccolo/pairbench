"""IPAA / IPAA-X aggregation (pairbench.experiments.correspondence.ipaa_stats),
pinned on a hand-built toy scene set.

IPAA is the unweighted mean of each scene's own recall (index 3 of the
per-scene tuple `(correct, n_resolved, n_gt, recall, precision)`); IPAA-X is
the fraction of scenes at or above X.
"""

from pairbench.experiments.correspondence import ipaa_stats


def test_ipaa_is_the_unweighted_mean_of_per_scene_recall():
    # scene A: 1/1 correct (recall 1.0); scene B: 1/4 (recall 0.25) -- a small
    # scene must NOT be swamped by a big one, unlike the pooled recall table.
    per_scene_arm = {
        "A": (1, 1, 1, 1.0, 1.0),
        "B": (1, 4, 4, 0.25, 0.25),
    }
    stats = ipaa_stats(per_scene_arm)
    assert stats["ipaa"] == 0.625  # mean(1.0, 0.25), NOT pooled 2/5 = 0.4
    assert stats["n_scenes"] == 2


def test_ipaa_x_is_the_fraction_of_scenes_clearing_the_threshold():
    per_scene_arm = {
        "A": (1, 1, 1, 1.0, 1.0),   # clears every threshold
        "B": (3, 4, 4, 0.75, 0.75),  # clears 0.5 not 0.8/0.9/1.0
        "C": (0, 4, 4, 0.0, 0.0),   # clears nothing
    }
    stats = ipaa_stats(per_scene_arm)
    ix = stats["ipaa_x"]
    assert ix[1.0] == 1 / 3
    assert ix[0.9] == 1 / 3
    assert ix[0.8] == 1 / 3
    assert ix[0.5] == 2 / 3


def test_unresolved_scene_counts_as_a_complete_miss():
    """A scene with zero resolved claims forces recall=0.0; IPAA must count
    it as scored-and-wrong, not drop it from the average."""
    per_scene_arm = {
        "A": (1, 1, 1, 1.0, 1.0),
        "B": (0, 0, 5, 0.0, 0.0),  # fully unresolved
    }
    stats = ipaa_stats(per_scene_arm)
    assert stats["n_scenes"] == 2
    assert stats["ipaa"] == 0.5


def test_empty_input_is_nan_not_a_crash():
    stats = ipaa_stats({})
    assert stats["n_scenes"] == 0
    import math
    assert math.isnan(stats["ipaa"])
    assert all(math.isnan(v) for v in stats["ipaa_x"].values())
