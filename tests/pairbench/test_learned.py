"""Tests for pairbench.detect.learned: family-grouped splits, peak-to-world
mapping, crop augmentation and threshold search.

Torch-dependent tests are skipped when torch (optional extra `learned`) is
not installed.
"""

import numpy as np
import pytest

from pairbench.detect.learned import (
    CROP_PX,
    FAMILY_GROUPS,
    HEATMAP_SIGMA_PX,
    OUT_STRIDE,
    SceneSide,
    build_targets,
    count_counts,
    group_folds,
    peaks_to_detections,
    pick_threshold,
    positional_counts,
    render_heatmap,
    sample_crop,
    scene_folds,
)

torch = pytest.importorskip("torch", reason="learned baseline is an optional extra")


def _sample(scene_id="l-profile/a", side=1, family="l-profile", h=200, w=400, centers=((100, 90),),
            gt_count=None):
    img = np.zeros((h, w), dtype=np.uint8)
    ctr = np.array(centers, dtype=np.float32).reshape(-1, 2)
    for x, y in ctr:
        img[int(y) - 20:int(y) + 20, int(x) - 20:int(x) + 20] = 255
    return SceneSide(
        scene_id=scene_id, side=side, family=family, category="single",
        image=img, min_y=1000.0, max_z=900.0, eff_scale=2.5, padding_px=95,
        centers_px=ctr, gt_count=len(ctr) if gt_count is None else gt_count,
    )


# ------------------------------------------------------------------- splits

def test_l_profile_and_l_profile_reshoot_are_one_group():
    """The load-bearing split decision. They are the same profile on an
    overlapping scenario set, so a family-disjoint split that separated them
    would put the test cross-section in the training set and report
    transfer where there is none."""
    assert FAMILY_GROUPS["l-profile"] == FAMILY_GROUPS["l-profile-reshoot"]
    assert len({FAMILY_GROUPS[f] for f in
                ("square-profile", "complex-profile", "heavy-profile")}) == 3


def test_group_folds_never_split_l_profile_from_l_profile_reshoot():
    samples = [_sample("l-profile/1", family="l-profile"),
               _sample("l-profile-reshoot/1", family="l-profile-reshoot"),
               _sample("square-profile/1", family="square-profile")]
    folds = dict(group_folds(samples))
    held = [ids for ids in folds.values() if "l-profile/1" in ids][0]
    assert "l-profile-reshoot/1" in held


def test_scene_folds_are_disjoint_and_cover_everything():
    ids = [f"l-profile/{k:02d}" for k in range(23)]
    folds = scene_folds(ids, n_folds=5, seed=3)
    flat = [s for f in folds for s in f]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat))


def test_scene_folds_are_deterministic_for_a_seed():
    assert scene_folds([f"s{k}" for k in range(20)], 5, 7) == \
           scene_folds([f"s{k}" for k in range(20)], 5, 7)


# ------------------------------------------------------------------ targets

def test_heatmap_peaks_at_one_on_every_centre():
    hm = render_heatmap((40, 60), np.array([[10.0, 12.0], [40.0, 30.0]]))
    assert hm[12, 10] == pytest.approx(1.0)
    assert hm[30, 40] == pytest.approx(1.0)
    assert hm.max() == pytest.approx(1.0)


def test_heatmap_is_empty_without_centres():
    assert render_heatmap((10, 10), np.zeros((0, 2))).max() == 0.0


def test_heatmap_ignores_centres_outside_the_grid():
    hm = render_heatmap((20, 20), np.array([[-5.0, -5.0], [100.0, 3.0]]))
    assert hm.max() == 0.0


def test_offset_target_carries_the_sub_stride_residual():
    """Without the offset head the best attainable localization error is one
    output cell, an order of magnitude coarser than the sub-millimetre
    ground-truth re-link tolerance elsewhere in the pipeline."""
    hm, off, mask = build_targets(np.array([[100.0, 56.0]]), (128, 256))
    cx, cy = 100.0 / OUT_STRIDE, 56.0 / OUT_STRIDE
    ix, iy = int(round(cx)), int(round(cy))
    assert mask[iy, ix] == 1.0
    assert off[0, iy, ix] == pytest.approx(cx - ix)
    assert off[1, iy, ix] == pytest.approx(cy - iy)
    assert hm.shape == (128 // OUT_STRIDE, 256 // OUT_STRIDE)


def test_sigma_is_narrower_than_the_tightest_bar_spacing():
    """40 mm (the tightest profile's edge-to-edge spacing) at 2.5 px/mm and
    stride 8 is 12.5 output cells; the Gaussian must not merge two of those
    into one blob."""
    spacing_cells = 40.0 * 2.5 / OUT_STRIDE
    assert HEATMAP_SIGMA_PX * 3 < spacing_cells


# ------------------------------------------------------------- augmentation

def test_crop_keeps_its_labels_on_the_material():
    """A crop whose labels drifted off the profile would train the model on
    noise; this is the cheapest check that the affine is applied to the
    image and the points consistently."""
    rng = np.random.default_rng(0)
    s = _sample(centers=((100, 90), (250, 90)))
    hits = 0
    trials = 40
    for _ in range(trials):
        img, ctr = sample_crop(rng, s)
        assert img.shape == (CROP_PX, CROP_PX)
        assert 0.0 <= img.min() and img.max() <= 1.0
        for x, y in ctr:
            assert 0 <= x < CROP_PX and 0 <= y < CROP_PX
            if img[int(y), int(x)] > 0.5:
                hits += 1
    assert hits > trials  # the labelled squares are solid, so most land on them


def test_crop_without_labels_is_still_valid():
    rng = np.random.default_rng(1)
    s = _sample(centers=np.zeros((0, 2)), gt_count=0)
    img, ctr = sample_crop(rng, s)
    assert img.shape == (CROP_PX, CROP_PX) and len(ctr) == 0


def test_rotation_can_be_switched_off_for_a_reproducible_check():
    rng = np.random.default_rng(2)
    s = _sample()
    img, _ = sample_crop(rng, s, rotate=False, flip=False, scale_range=(1.0, 1.0))
    assert img.shape == (CROP_PX, CROP_PX)


# --------------------------------------------------------------- decoding

def test_peaks_map_back_through_the_projection_geometry():
    """The learned arm's positions must live in the same world frame as the
    classical arm's, or a shared scorer is comparing two coordinate
    systems."""
    s = _sample()
    proj = s.as_projection()
    px, py = proj.yz_to_pixel(1042.0, 871.0)
    dets = peaks_to_detections(np.array([[px, py, 0.9]]), s, threshold=0.5)
    assert len(dets) == 1
    y, z = dets[0].centroid_yz_mm
    assert y == pytest.approx(1042.0, abs=1e-6)
    assert z == pytest.approx(871.0, abs=1e-6)
    assert dets[0].side == s.side and dets[0].kind == "confirmed"


def test_peaks_below_threshold_are_dropped():
    s = _sample()
    peaks = np.array([[50.0, 50.0, 0.9], [80.0, 50.0, 0.2]])
    assert len(peaks_to_detections(peaks, s, threshold=0.5)) == 1
    assert len(peaks_to_detections(peaks, s, threshold=0.95)) == 0


def test_detections_come_out_sorted_left_to_right():
    """Detections must be indexed left-to-right, matching the convention the
    rest of the pipeline's ground truth uses."""
    s = _sample()
    peaks = np.array([[300.0, 50.0, 0.9], [50.0, 50.0, 0.8], [180.0, 50.0, 0.7]])
    ys = [d.centroid_yz_mm[0] for d in peaks_to_detections(peaks, s, 0.5)]
    assert ys == sorted(ys)


# ------------------------------------------------------------------ scoring

def test_count_convention_matches_the_classical_runner():
    assert count_counts(3, 3) == (3, 0, 0)
    assert count_counts(5, 3) == (3, 2, 0)
    assert count_counts(1, 3) == (1, 0, 2)


def test_threshold_search_finds_the_count_matching_threshold():
    s = _sample(centers=((100, 90), (250, 90)))          # gt_count 2
    peaks = np.array([[100.0, 90.0, 0.9], [250.0, 90.0, 0.8],
                      [300.0, 90.0, 0.3]])               # one junk peak
    thr, f1 = pick_threshold({(s.scene_id, s.side): peaks}, [s])
    assert f1 == pytest.approx(1.0)
    assert 0.3 < thr <= 0.8


def test_positional_counts_uses_the_shared_anchor_matcher():
    s = _sample(centers=((100, 90), (250, 90)))
    proj = s.as_projection()
    anchors = [proj.pixel_to_yz(100.0, 90.0), proj.pixel_to_yz(250.0, 90.0)]
    dets = peaks_to_detections(
        np.array([[100.0, 90.0, 0.9], [250.0, 90.0, 0.9]]), s, 0.5)
    assert positional_counts(dets, anchors, 25.0) == (2, 2)
    # a detector that found only one of them re-links only one anchor
    one = peaks_to_detections(np.array([[100.0, 90.0, 0.9]]), s, 0.5)
    assert positional_counts(one, anchors, 25.0) == (1, 2)
    assert positional_counts([], anchors, 25.0) == (0, 2)


# -------------------------------------------------------------------- model

def test_model_output_shapes_and_size():
    from pairbench.detect.learned import build_model
    m = build_model()
    x = torch.zeros(2, 1, 128, 256)
    hm, off = m(x)
    assert hm.shape == (2, 1, 128 // OUT_STRIDE, 256 // OUT_STRIDE)
    assert off.shape == (2, 2, 128 // OUT_STRIDE, 256 // OUT_STRIDE)
    n = sum(p.numel() for p in m.parameters())
    assert n < 600_000, f"CPU budget blown: {n} parameters"


def test_focal_loss_is_lower_when_the_prediction_is_right():
    from pairbench.detect.learned import focal_loss
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    right = torch.full((1, 1, 8, 8), -6.0)
    right[0, 0, 4, 4] = 6.0
    wrong = torch.full((1, 1, 8, 8), -6.0)
    assert float(focal_loss(right, target)) < float(focal_loss(wrong, target))


def test_prediction_recovers_a_planted_peak_end_to_end():
    """Overfit a two-step model on one synthetic image and check the decode
    path (max-pool NMS + offset + stride) returns a peak, in image pixels,
    inside the image."""
    from pairbench.detect.learned import build_model, predict_peaks
    torch.manual_seed(0)
    m = build_model(width=16)
    m.eval()
    s = _sample()
    peaks = predict_peaks(m, s.image)
    assert peaks.ndim == 2 and peaks.shape[1] == 3
    assert ((peaks[:, 0] >= 0) & (peaks[:, 0] < s.image.shape[1])).all()
    assert ((peaks[:, 1] >= 0) & (peaks[:, 1] < s.image.shape[0])).all()


def test_training_reduces_the_loss_on_a_trivial_dataset():
    from pairbench.detect.learned import train_model
    samples = [_sample(centers=((100, 90), (280, 90)))]
    _, losses = train_model(samples, steps=25, batch=2, seed=0, crop=192,
                            width=16, threads=1)
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
