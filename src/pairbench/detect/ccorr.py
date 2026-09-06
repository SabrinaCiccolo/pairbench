"""Template-matching baseline detector for bar end faces.

Normalized cross-correlation of the DXF template against the YZ projection
over a rotation sweep (ANGLE_STEP_DEG), erase of accepted footprints and a
second sweep, then attachment of the 3D points under each footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..io.dxf_template import Template
from ..io.projection import Projection, project_yz
from ..timing import stage

ANGLE_STEP_DEG = 2.0           # deg between coarse candidates
MAX_MATCHES_PER_ANGLE = 5      # accepted matches per angle
PEAK_THRESHOLD = 0.4           # valid peak, fraction of template self-correlation
WEAK_PEAK_THRESHOLD = 0.2      # ignored below, fraction of template self-correlation
MIN_PEAK_SEPARATION_PX = 4
OVERLAP_KERNEL = 5             # footprint dilation (px)
OUTER_RING_KERNEL = 9          # clutter-ring dilation (px)
OVERLAP_THRESHOLD = 0.65       # min footprint overlap to confirm a peak
OVERLAP_SCORE_WEIGHT = 0.75
IOU_THRESHOLD = 0.2            # OBB NMS
COVERED_THRESHOLD = 0.7        # drop a match this much covered by better ones
CONTOUR_KERNEL = 5             # erase-mask dilation (px)
ERASE_DILATE_ITERATIONS = 3

_LOST_NMS_CAP = 300         # lost-band peaks kept (by score) before NMS
_MAX_PEAKS_PER_ANGLE = 50   # hard ceiling on accept attempts per angle


@dataclass
class Detection:
    side: int
    angle_deg: float                  # coarse matching angle
    score: float                      # 0.75*overlap + 0.25*outer_ring
    ccorr_frac: float                 # CCorr peak / template self-correlation
    kind: str                         # "confirmed" | "lost"
    obb: tuple                        # ((cx,cy),(w,h),deg) in final scene px
    centroid_yz_mm: tuple = (0.0, 0.0)
    centroid_world_mm: tuple = (0.0, 0.0, 0.0)
    normal_world: tuple = (0.0, 0.0, 0.0)
    n_points: int = 0
    z_extent_mm: tuple = (0.0, 0.0)   # (minZ, maxZ) of supporting points
    y_extent_mm: tuple = (0.0, 0.0)
    obb_display: tuple | None = None  # ((cx,cy),(w,h),deg), display-only OBB;
                                       # None unless a detector sets it


def sort_left_to_right(dets: list[Detection]) -> list[Detection]:
    """Sorts detections by ascending world-Y (= ascending image-X)."""
    return sorted(dets, key=lambda d: d.centroid_yz_mm[0])


def dets_from_json(entries: list[dict]) -> list[Detection]:
    """Builds Detection objects from a list of dict records."""
    out = []
    for e in entries:
        e = dict(e)
        e["obb"] = (tuple(e["obb"][0]), tuple(e["obb"][1]), e["obb"][2])
        if e.get("obb_display") is not None:
            od = e["obb_display"]
            e["obb_display"] = (tuple(od[0]), tuple(od[1]), od[2])
        for k in ("centroid_yz_mm", "centroid_world_mm", "normal_world",
                  "z_extent_mm", "y_extent_mm"):
            e[k] = tuple(e[k])
        out.append(Detection(**e))
    return out


def _rotate_trim(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotates image about its center into a full canvas, trims to content."""
    h, w = image.shape
    diag = int(np.ceil(np.hypot(h, w))) + 2
    canvas = np.zeros((diag, diag), dtype=image.dtype)
    oy, ox = (diag - h) // 2, (diag - w) // 2
    canvas[oy : oy + h, ox : ox + w] = image
    m = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle_deg, 1.0)
    rot = cv2.warpAffine(canvas, m, (diag, diag), flags=cv2.INTER_LINEAR)
    nz = cv2.findNonZero((rot > 0).astype(np.uint8))
    if nz is None:
        return rot
    x, y, bw, bh = cv2.boundingRect(nz)
    return rot[y : y + bh, x : x + bw]


def _ellipse(ksize: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


_ANGLE_CACHE_ATTR = "_angle_packets"
_ANGLE_CACHE_MAX = 512
_ANGLE_CACHE_MAX_BYTES = 192 * 1024 * 1024


class _AnglePacket:
    """Cached rotated template raster plus its lazily built overlap mask,
    outer ring and erase mask for one (template, angle)."""

    __slots__ = ("rot", "_core", "_core_bool", "_mask", "_mask_bool",
                 "_ring_bool", "_mask_px", "_ring_px", "_erase_bool",
                 "_cache")

    def __init__(self, image: np.ndarray, angle_deg: float) -> None:
        self.rot = np.ascontiguousarray(_rotate_trim(image, angle_deg))
        self._core: np.ndarray | None = None
        self._core_bool: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._mask_bool: np.ndarray | None = None
        self._ring_bool: np.ndarray | None = None
        self._mask_px = -1
        self._ring_px = -1
        self._erase_bool: np.ndarray | None = None
        self._cache: dict | None = None

    @property
    def core_bool(self) -> np.ndarray:
        """The exact, undilated matched silhouette."""
        if self._core_bool is None:
            self._core = (self.rot > 0).astype(np.uint8)
            self._core_bool = self._core.view(np.bool_)
        return self._core_bool

    def _build_masks(self) -> None:
        self.core_bool
        self._mask = cv2.dilate(self._core, _ellipse(OVERLAP_KERNEL))
        self._mask_bool = self._mask.view(np.bool_)
        ring = cv2.dilate(self._mask, _ellipse(OUTER_RING_KERNEL))
        ring[self._mask_bool] = 0
        self._ring_bool = ring.view(np.bool_)
        self._mask_px = int(self._mask.sum())
        self._ring_px = int(ring.sum())
        self._enforce_budget()

    @property
    def mask_bool(self) -> np.ndarray:
        """The dilated footprint: which pixels this candidate claims."""
        if self._mask_bool is None:
            self._build_masks()
        return self._mask_bool

    @property
    def ring_bool(self) -> np.ndarray:
        """The annulus just outside the footprint, used to score clutter."""
        if self._ring_bool is None:
            self._build_masks()
        return self._ring_bool

    @property
    def mask_px(self) -> int:
        if self._mask_px < 0:
            self._build_masks()
        return self._mask_px

    @property
    def ring_px(self) -> int:
        if self._ring_px < 0:
            self._build_masks()
        return self._ring_px

    def _enforce_budget(self) -> None:
        """Evicts the oldest packets from this packet's cache under the byte budget."""
        cache = self._cache
        if cache is None:
            return
        total = sum(p.nbytes for p in cache.values())
        while total > _ANGLE_CACHE_MAX_BYTES and len(cache) > 1:
            oldest = cache[next(iter(cache))]
            if oldest is self:
                break
            total -= oldest.nbytes
            del cache[next(iter(cache))]

    @property
    def nbytes(self) -> int:
        """Resident size of this packet, for the cache's byte budget."""
        return sum(a.nbytes for a in
                   (self.rot, self._core, self._mask, self._ring_bool,
                    self._erase_bool) if a is not None)

    @property
    def erase_bool(self) -> np.ndarray:
        """The footprint the erase step blanks out."""
        if self._erase_bool is None:
            self.core_bool
            self._erase_bool = cv2.dilate(
                self._core, _ellipse(CONTOUR_KERNEL),
                iterations=ERASE_DILATE_ITERATIONS).view(np.bool_)
        return self._erase_bool


def _angle_packet(template: Template, angle_deg: float) -> _AnglePacket:
    """Returns the cached `_AnglePacket` for this template and angle."""
    cache = getattr(template, _ANGLE_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        object.__setattr__(template, _ANGLE_CACHE_ATTR, cache)
    key = float(angle_deg)
    packet = cache.get(key)
    if packet is None:
        packet = _AnglePacket(template.image, key)
        while len(cache) >= _ANGLE_CACHE_MAX:
            del cache[next(iter(cache))]
        cache[key] = packet
        packet._cache = cache
    return packet


def _overlap_scores(
    scene: np.ndarray, packet: _AnglePacket, x: int, y: int
) -> tuple[float, float]:
    """Returns (overlap_score, outer_ring_score) for a candidate window."""
    if packet.mask_px == 0:
        return 0.0, 0.0
    mask = packet.mask_bool
    th, tw = mask.shape
    crop_nz = scene[y : y + th, x : x + tw] > 0
    overlap_px = int((crop_nz & mask).sum())
    clutter_px = int((crop_nz & packet.ring_bool).sum())
    overlap_score = overlap_px / packet.mask_px
    ring_score = 1.0 - (clutter_px / packet.ring_px if packet.ring_px else 0.0)
    return overlap_score, ring_score


def _obb_iou(a: tuple, b: tuple) -> float:
    if (np.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1])
            > (np.hypot(*a[1]) + np.hypot(*b[1])) / 2.0):
        return 0.0
    ra = (tuple(a[0]), tuple(a[1]), a[2])
    rb = (tuple(b[0]), tuple(b[1]), b[2])
    ok, inter = cv2.rotatedRectangleIntersection(ra, rb)
    if ok == cv2.INTERSECT_NONE or inter is None:
        return 0.0
    inter_area = cv2.contourArea(inter)
    area_a = a[1][0] * a[1][1]
    area_b = b[1][0] * b[1][1]
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _nms(dets: list[Detection], iou_thr: float = IOU_THRESHOLD) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(_obb_iou(d.obb, k.obb) <= iou_thr for k in kept):
            kept.append(d)
    return kept


def _covered_fraction(d: Detection, kept: list[Detection]) -> float:
    """Fraction of d's OBB area covered by the union of the kept OBBs."""
    box_pts = cv2.boxPoints((tuple(d.obb[0]), tuple(d.obb[1]), d.obb[2]))
    x0, y0 = np.floor(box_pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(box_pts.max(axis=0)).astype(int)
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    origin = np.array([x0, y0], dtype=np.float32)

    d_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(d_mask, np.round(box_pts - origin).astype(np.int32), 1)
    total_px = int(d_mask.sum())
    if total_px == 0:
        return 0.0

    union_mask = np.zeros((h, w), dtype=np.uint8)
    for k in kept:
        if _obb_iou(d.obb, k.obb) <= 0.0:
            continue
        k_pts = cv2.boxPoints((tuple(k.obb[0]), tuple(k.obb[1]), k.obb[2]))
        cv2.fillConvexPoly(union_mask, np.round(k_pts - origin).astype(np.int32), 1)

    covered_px = int(((d_mask > 0) & (union_mask > 0)).sum())
    return covered_px / total_px


def _remove_fully_covered(
    dets: list[Detection], thr: float = COVERED_THRESHOLD
) -> list[Detection]:
    """Drops a match when >= thr of its OBB area is covered by higher-score OBBs."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if d.obb[1][0] * d.obb[1][1] <= 0:
            continue
        if kept and _covered_fraction(d, kept) >= thr:
            continue
        kept.append(d)
    return sorted(kept, key=lambda d: -d.score)


def _sweep(
    scene_img: np.ndarray, template: Template, side: int
) -> tuple[list[Detection], list[Detection]]:
    """Runs one coarse pass over all angles. Returns (confirmed, lost)."""
    scene_f32 = scene_img.astype(np.float32)
    confirmed: list[Detection] = []
    lost: list[Detection] = []
    for angle in np.arange(0.0, 360.0, ANGLE_STEP_DEG):
        packet = _angle_packet(template, float(angle))
        rot = packet.rot
        th, tw = rot.shape
        if th > scene_img.shape[0] or tw > scene_img.shape[1]:
            continue
        match = cv2.matchTemplate(scene_f32, rot.astype(np.float32), cv2.TM_CCORR)
        accepted_here: list[tuple[int, int]] = []
        for _ in range(_MAX_PEAKS_PER_ANGLE):
            if len(accepted_here) >= MAX_MATCHES_PER_ANGLE:
                break
            _, max_val, _, max_loc = cv2.minMaxLoc(match)
            if max_val <= WEAK_PEAK_THRESHOLD * template.max_val_ref:
                break
            x, y = max_loc
            # Blank a template-sized box centered on the peak so the next
            # iteration does not re-find the same instance.
            y0, y1 = max(0, y - th // 2), min(match.shape[0], y + th // 2 + 1)
            x0, x1 = max(0, x - tw // 2), min(match.shape[1], x + tw // 2 + 1)
            match[y0:y1, x0:x1] = -1.0
            if any(np.hypot(x - ax, y - ay) < MIN_PEAK_SEPARATION_PX
                   for ax, ay in accepted_here):
                continue
            frac = max_val / template.max_val_ref
            obb = ((x + tw / 2.0, y + th / 2.0),
                   (float(template.image.shape[1]), float(template.image.shape[0])),
                   -float(angle))
            if frac > PEAK_THRESHOLD:
                overlap, ring = _overlap_scores(scene_img, packet, x, y)
                score = (OVERLAP_SCORE_WEIGHT * overlap
                         + (1.0 - OVERLAP_SCORE_WEIGHT) * ring)
                det = Detection(side=side, angle_deg=float(angle), score=score,
                                ccorr_frac=float(frac), kind="confirmed", obb=obb)
                if overlap > OVERLAP_THRESHOLD:
                    confirmed.append(det)
                    accepted_here.append((x, y))
                else:
                    det.kind = "lost"
                    lost.append(det)
            else:
                lost.append(Detection(side=side, angle_deg=float(angle),
                                      score=float(frac), ccorr_frac=float(frac),
                                      kind="lost", obb=obb))
    return confirmed, lost


def _erase_footprints(
    scene_img: np.ndarray, template: Template, dets: list[Detection]
) -> np.ndarray:
    """Returns a copy of scene_img with each detection's dilated footprint zeroed."""
    erased = scene_img.copy()
    for d in dets:
        mask = _angle_packet(template, d.angle_deg).erase_bool
        th, tw = mask.shape
        cx, cy = d.obb[0]
        x0 = int(round(cx - tw / 2.0))
        y0 = int(round(cy - th / 2.0))
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1 = min(erased.shape[1], x0 + tw)
        sy1 = min(erased.shape[0], y0 + th)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        sub = mask[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
        region = erased[sy0:sy1, sx0:sx1]
        region[sub] = 0
    return erased


def _project_to_pixels(
    proj: Projection, world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Converts already keep_mask-filtered world points to projection pixel coordinates."""
    px = proj.padding_px + (world[:, 1] - proj.min_y) * proj.eff_scale
    py = proj.padding_px + (proj.max_z - world[:, 2]) * proj.eff_scale
    return px, py


def _template_local_index(
    shape: tuple[int, int], obb_center: tuple[float, float],
    px: np.ndarray, py: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (inside, ix_inside, iy_inside) for a template-sized window
    centered on obb_center: which points land in the window, and their
    integer template-local coordinates."""
    th, tw = shape
    x0 = obb_center[0] - tw / 2.0
    y0 = obb_center[1] - th / 2.0
    ix = np.round(px - x0).astype(np.int64)
    iy = np.round(py - y0).astype(np.int64)
    inside = (ix >= 0) & (ix < tw) & (iy >= 0) & (iy < th)
    return inside, ix[inside], iy[inside]


def _mask_select(
    mask: np.ndarray, inside: np.ndarray, ix_in: np.ndarray, iy_in: np.ndarray
) -> np.ndarray:
    """Returns a boolean mask over the whole cloud selecting points whose
    template-local pixel is set in mask."""
    sel = inside.copy()
    sel[inside] = mask[iy_in, ix_in]
    return sel


def _footprint_mask(
    det: Detection, template: Template, proj: Projection, world: np.ndarray
) -> np.ndarray:
    """Returns a boolean mask over world (already keep_mask-filtered)
    selecting points under det's dilated template footprint."""
    mask = _angle_packet(template, det.angle_deg).mask_bool
    px, py = _project_to_pixels(proj, world)
    inside, ix_in, iy_in = _template_local_index(mask.shape, det.obb[0], px, py)
    return _mask_select(mask, inside, ix_in, iy_in)


def _fill_geometry(det: Detection, pts: np.ndarray) -> None:
    """Fills a Detection's 3D fields (centroid, extents, normal) from a point subset."""
    det.n_points = int(len(pts))
    if len(pts) < 3:
        return
    centroid = pts.mean(axis=0)
    det.centroid_world_mm = tuple(float(v) for v in centroid)
    det.centroid_yz_mm = (float(centroid[1]), float(centroid[2]))
    det.z_extent_mm = (float(pts[:, 2].min()), float(pts[:, 2].max()))
    det.y_extent_mm = (float(pts[:, 1].min()), float(pts[:, 1].max()))
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[0] < 0:  # orient toward +X
        normal = -normal
    det.normal_world = tuple(float(v) for v in normal)


def _attach_points(
    det: Detection, template: Template, proj: Projection, world: np.ndarray
) -> None:
    """Fills a Detection's 3D fields from world points under its footprint."""
    sel = _footprint_mask(det, template, proj, world)
    _fill_geometry(det, world[sel])


def detect_faces(
    world_points: np.ndarray, template: Template, side: int,
) -> tuple[list[Detection], list[Detection], Projection]:
    """Runs detection on one side's world cloud: sweep, erase the accepted
    footprints, sweep again; both passes' confirmed detections are kept.
    Returns (confirmed, lost, projection)."""
    padding = template.default_padding
    with stage("projection"):
        proj = project_yz(world_points, padding_px=padding)

    with stage("template_match"):
        pass1_conf, pass1_lost = _sweep(proj.image, template, side)
        pass1_conf = _remove_fully_covered(_nms(pass1_conf))

        erased = _erase_footprints(proj.image, template, pass1_conf)
        pass2_conf, pass2_lost = _sweep(erased, template, side)
        pass2_conf = _remove_fully_covered(_nms(pass2_conf))

    confirmed: list[Detection] = list(pass1_conf) + list(pass2_conf)

    lost_all = sorted(pass1_lost + pass2_lost, key=lambda d: -d.score)[:_LOST_NMS_CAP]
    lost = _nms(lost_all)
    lost = [d for d in lost
            if all(_obb_iou(d.obb, k.obb) <= IOU_THRESHOLD for k in confirmed)]

    with stage("point_attach"):
        kept_world = world_points[proj.keep_mask]
        for d in confirmed + lost:
            _attach_points(d, template, proj, kept_world)
    confirmed = [d for d in confirmed if d.n_points >= 3]
    lost = [d for d in lost if d.n_points >= 3]
    return confirmed, lost, proj
