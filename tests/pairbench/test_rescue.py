import numpy as np

from pairbench.detect.ccorr import Detection
from pairbench.io.dxf_template import Template
from pairbench.io.projection import project_yz
from pairbench.detect.rescue import (RESCUE_REFIND_MM,
                              _covers_same_material,
                              rescue_misses)

FACE_W_MM = 60.0
FACE_H_MM = 24.0
EFF_SCALE = 2.5


def face_points(y0: float, z0: float = 20.0, x: float = 0.0,
                step: float = 0.5) -> np.ndarray:
    ys = np.arange(y0, y0 + FACE_W_MM, step)
    zs = np.arange(z0, z0 + FACE_H_MM, step)
    yy, zz = np.meshgrid(ys, zs)
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def solid_template() -> Template:
    img = np.full((int(FACE_H_MM * EFF_SCALE), int(FACE_W_MM * EFF_SCALE)),
                  255, dtype=np.uint8)
    return Template(image=img,
                    max_val_ref=float((img.astype(np.float32) ** 2).sum()),
                    loops_mm=[], eff_scale=EFF_SCALE)


def conf_det(side: int, y_center: float, proj, template,
             z_center: float = 32.0) -> Detection:
    """Confirmed detection whose obb pixels agree with the projection."""
    cx = proj.padding_px + (y_center - proj.min_y) * proj.eff_scale
    cy = proj.padding_px + (proj.max_z - z_center) * proj.eff_scale
    return Detection(
        side=side, angle_deg=0.0, score=0.9, ccorr_frac=0.9, kind="confirmed",
        obb=((cx, cy), (float(template.image.shape[1]),
                        float(template.image.shape[0])), 0.0),
        centroid_yz_mm=(y_center, z_center),
        centroid_world_mm=(0.0, y_center, z_center),
        normal_world=(1.0, 0.0, 0.0), n_points=1000,
        z_extent_mm=(z_center - FACE_H_MM / 2, z_center + FACE_H_MM / 2),
        y_extent_mm=(y_center - FACE_W_MM / 2, y_center + FACE_W_MM / 2))


def build_scene(face_ys_world, conf_ys_1, conf_ys_2):
    template = solid_template()
    world1 = np.vstack([face_points(y0) for y0 in face_ys_world])
    world2 = world1.copy()
    padding = template.default_padding
    proj1 = project_yz(world1, padding_px=padding)
    conf1 = [conf_det(1, y, proj1, template) for y in conf_ys_1]
    conf2 = [conf_det(2, y, proj1, template) for y in conf_ys_2]
    return world1, world2, template, conf1, conf2


def test_rescues_missing_face_at_partner_prediction():
    # faces at 100/200/300; side 1 confirmed only the first two, side 2 all
    # three -> the unmatched side-2 partner predicts (330, 32) on side 1
    world1, world2, template, conf1, conf2 = build_scene(
        (100.0, 200.0, 300.0), (130.0, 230.0), (130.0, 230.0, 330.0))
    r1, r2, diag = rescue_misses(world1, world2, template, conf1, conf2)
    assert len(r1) == 1 and not r2, (len(r1), len(r2))
    det = r1[0]
    assert det.kind == "rescued"
    assert abs(det.centroid_yz_mm[0] - 330.0) < 10.0, det.centroid_yz_mm
    assert abs(det.centroid_yz_mm[1] - 32.0) < 10.0, det.centroid_yz_mm
    assert det.z_extent_mm[1] - det.z_extent_mm[0] >= 15.0
    assert diag["attempted"] == 1


def test_no_unmatched_partner_no_rescue():
    world1, world2, template, conf1, conf2 = build_scene(
        (100.0, 200.0, 300.0), (130.0, 230.0, 330.0), (130.0, 230.0, 330.0))
    r1, r2, diag = rescue_misses(world1, world2, template, conf1, conf2)
    assert not r1 and not r2
    assert diag["attempted"] == 0


def test_no_face_in_window_no_rescue():
    # side-2 partner predicts (430, 32) but side 1 has no points there:
    # the local re-match must come back empty, not invent a detection
    world1, world2, template, conf1, conf2 = build_scene(
        (100.0, 200.0), (130.0, 230.0), (130.0, 230.0, 430.0))
    r1, r2, diag = rescue_misses(world1, world2, template, conf1, conf2)
    assert not r1 and not r2, (r1, r2)
    assert diag["attempted"] == 1


def _det(y_lo, y_hi, z_lo, z_hi, centroid=None):
    """Centroid is passed separately: a real detection's centroid is the
    mean of its claimed points, not the midpoint of their extent."""
    y, z = centroid if centroid else ((y_lo + y_hi) / 2.0, (z_lo + z_hi) / 2.0)
    return Detection(side=1, angle_deg=180.0, score=0.5, ccorr_frac=0.5,
                     kind="rescued", obb=((0.0, 0.0), (10.0, 10.0), 0.0),
                     centroid_yz_mm=(y, z), centroid_world_mm=(0.0, y, z),
                     normal_world=(1.0, 0.0, 0.0), n_points=1000,
                     z_extent_mm=(z_lo, z_hi), y_extent_mm=(y_lo, y_hi))


def test_same_material_guard_sees_what_the_centroid_gate_cannot():
    """A confirmed detection covering only part of a face and a rescue
    covering the whole face can have centroids well outside a plain
    re-find-distance gate, while their claimed extents overlap almost
    completely in both axes."""
    confirmed = _det(1843.0, 1867.0, 906.0, 940.0, (1854.2, 922.9))
    rescue = _det(1841.0, 1866.0, 906.0, 975.0, (1860.5, 956.4))
    d = np.linalg.norm(np.array(rescue.centroid_yz_mm)
                       - np.array(confirmed.centroid_yz_mm))
    assert d > RESCUE_REFIND_MM, d          # documents why a centroid-only gate fails
    assert _covers_same_material(rescue, confirmed)


def test_same_material_guard_keeps_genuinely_different_bars():
    """Both axes are required: adjacent faces sharing Z almost exactly, and
    a bar stacked directly on a detected one sharing Y, must both survive
    as distinct."""
    confirmed = _det(1600.0, 1669.0, 906.0, 930.0)
    adjacent = _det(1531.0, 1600.0, 906.0, 930.0)     # y_overlap ~ 0
    stacked = _det(1600.0, 1669.0, 931.0, 955.0)      # z_overlap ~ 0
    assert not _covers_same_material(adjacent, confirmed)
    assert not _covers_same_material(stacked, confirmed)
