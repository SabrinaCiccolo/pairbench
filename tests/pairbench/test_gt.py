import tempfile
from pathlib import Path

from pairbench.detect.ccorr import Detection
from pairbench.gt import (
    expand_shorthand_pairs,
    format_anchors,
    format_correspondence_pairs,
    load_gt_correspondence,
    load_gt,
    parse_anchors,
    parse_correspondence_pairs,
    resolve_anchor_distances,
    resolve_anchor_indices,
)


def det(y, z=920.0):
    return Detection(side=1, angle_deg=180.0, score=0.9, ccorr_frac=0.9,
                     kind="confirmed", obb=((y, z), (40.0, 12.0), -180.0),
                     centroid_yz_mm=(y, z))


def test_parse_correspondence_pairs_empty():
    assert parse_correspondence_pairs("") == []
    assert parse_correspondence_pairs("   ") == []


def test_parse_correspondence_pairs_basic():
    assert parse_correspondence_pairs("0-1,1-0,2-2") == [(0, 1), (1, 0), (2, 2)]


def test_parse_correspondence_pairs_tolerates_whitespace():
    assert parse_correspondence_pairs(" 0-1 , 1-0 ") == [(0, 1), (1, 0)]


def test_format_correspondence_pairs_sorts_output():
    # format is the round-trip partner of parse; output must be sorted
    # regardless of input order (gt_correspondence.csv rows are diffable).
    assert format_correspondence_pairs([(2, 0), (0, 1)]) == "0-1,2-0"


def test_format_parse_roundtrip():
    pairs = [(0, 2), (1, 3), (2, 0)]
    assert parse_correspondence_pairs(format_correspondence_pairs(pairs)) == sorted(pairs)


def test_expand_shorthand_blank_is_identity():
    # blank -> identity over min(n1, n2), matching the common non-crossed case
    assert expand_shorthand_pairs("", 3, 4) == [(0, 0), (1, 1), (2, 2)]
    assert expand_shorthand_pairs("", 4, 3) == [(0, 0), (1, 1), (2, 2)]


def test_expand_shorthand_positive_shift():
    # a missing left-edge detection on one side shifts every remaining
    # index by +1 against the other side
    assert expand_shorthand_pairs("shift:+1", 3, 4) == [(0, 1), (1, 2), (2, 3)]


def test_expand_shorthand_negative_shift():
    assert expand_shorthand_pairs("shift:-1", 3, 4) == [(1, 0), (2, 1)]


def test_expand_shorthand_no_correspondence_tokens():
    # 'i-x' / 'x-j' exclude that index from the identity default entirely
    # (not stored as a claimed pair), rather than defaulting to (i, i).
    assert expand_shorthand_pairs("1-x", 3, 3) == [(0, 0), (2, 2)]
    assert expand_shorthand_pairs("x-1", 3, 3) == [(0, 0), (2, 2)]


def test_expand_shorthand_explicit_overrides_shift_default():
    # explicit '0-3' consumes side-1 index 0 (skips the shift default for
    # it) AND side-2 index 3 (so index 2's shift default, which would also
    # land on side-2 index 3, is dropped rather than double-claiming it).
    assert expand_shorthand_pairs("shift:+1,0-3", 3, 4) == [(0, 3), (1, 2)]


def test_load_gt_keyed_by_scene_id():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gt.csv"
        path.write_text(
            "scene_id,gt_A,gt_B,gt_pairs,category,confidence,notes\n"
            "l-profile/crossed-02,4,4,4,crossed,high,\n"
        )
        rows = load_gt(path)
        assert set(rows) == {"l-profile/crossed-02"}
        assert rows["l-profile/crossed-02"]["category"] == "crossed"


def test_anchor_roundtrip():
    dets = [det(1583.09, 916.22), det(1654.76, 916.97)]
    anchors = parse_anchors(format_anchors(dets))
    assert anchors == [(1583.09, 916.22), (1654.76, 916.97)]
    assert parse_anchors("") == []


def test_anchors_relink_unchanged_detections_exactly():
    dets = [det(1583.0), det(1654.0), det(1721.0)]
    anchors = parse_anchors(format_anchors(dets))
    assert resolve_anchor_indices(anchors, dets) == {0: 0, 1: 1, 2: 2}


def test_anchors_survive_an_inserted_detection():
    """A newly-recovered face shifts later indices; the anchor must follow
    the physical bar, not the index."""
    annotated = [det(1583.0), det(1721.0)]
    anchors = parse_anchors(format_anchors(annotated))
    rescued_in_between = [det(1583.0), det(1654.0), det(1721.0)]
    assert resolve_anchor_indices(anchors, rescued_in_between) == {0: 0, 1: 2}


def test_anchor_of_an_undetected_bar_is_unresolved_not_reassigned():
    # middle bar is gone from this detection run: its anchor must resolve to
    # nothing rather than stealing a neighbour.
    anchors = parse_anchors(format_anchors([det(1583.0), det(1654.0),
                                            det(1721.0)]))
    dets = [det(1583.0), det(1721.0)]
    assert resolve_anchor_indices(anchors, dets) == {0: 0, 2: 1}


def test_anchor_radius_shrinks_with_local_spacing():
    # bars 6 mm apart: the tolerance must fall below half that spacing, so a
    # 4 mm drift cannot cross onto the neighbour.
    anchors = parse_anchors(format_anchors([det(1000.0), det(1006.0)]))
    assert resolve_anchor_indices(anchors, [det(1000.0), det(1006.0)]) == {0: 0, 1: 1}
    drifted = [det(1004.0)]  # 4 mm from anchor 0, 2 mm from anchor 1
    assert resolve_anchor_indices(anchors, drifted) == {}


def test_anchor_distances_zero_for_unchanged_detections():
    dets = [det(1583.0), det(1654.0), det(1721.0)]
    anchors = parse_anchors(format_anchors(dets))
    distances = resolve_anchor_distances(anchors, dets)
    assert set(distances) == {0, 1, 2}
    assert all(abs(d) < 1e-9 for d in distances.values())


def test_anchor_distances_track_drift():
    anchors = parse_anchors(format_anchors([det(1000.0)]))
    drifted = [det(1004.0)]  # 4 mm Y drift, same Z
    distances = resolve_anchor_distances(anchors, drifted)
    assert abs(distances[0] - 4.0) < 1e-9


def test_anchor_distances_and_indices_agree_on_resolved_set():
    # both views resolve the same matching rule to the same detection
    annotated = [det(1583.0), det(1721.0)]
    anchors = parse_anchors(format_anchors(annotated))
    rescued_in_between = [det(1583.0), det(1654.0), det(1721.0)]
    indices = resolve_anchor_indices(anchors, rescued_in_between)
    distances = resolve_anchor_distances(anchors, rescued_in_between)
    assert set(indices) == set(distances) == {0, 1}
    assert all(d < 1e-9 for d in distances.values())


def test_anchor_distances_unresolved_anchor_absent():
    anchors = parse_anchors(format_anchors([det(1000.0), det(1006.0)]))
    drifted = [det(1004.0)]  # ambiguous: absent from resolve_anchor_indices too
    assert resolve_anchor_distances(anchors, drifted) == {}


def test_load_gt_correspondence_missing_file_returns_empty():
    assert load_gt_correspondence(Path("/nonexistent/gt_correspondence.csv")) == {}


def test_load_gt_correspondence_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gt_correspondence.csv"
        path.write_text(
            "scene_id,n_det_1,n_det_2,pairs,anchors_1,anchors_2,"
            "confidence,notes\n"
            "l-profile/crossed-02,3,4,\"0-2,1-3,2-0\",1000.00|900.00,"
            "1010.00|900.00,high,\n"
        )
        rows = load_gt_correspondence(path)
        assert rows["l-profile/crossed-02"]["n_det_2"] == "4"
        assert parse_correspondence_pairs(rows["l-profile/crossed-02"]["pairs"]) == [
            (0, 2), (1, 3), (2, 0),
        ]
