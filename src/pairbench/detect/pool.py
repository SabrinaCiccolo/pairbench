"""Candidate-pool detector.

1. `_pool_sweep`: every correlation peak above WEAK_PEAK_THRESHOLD at every
   angle, with no accept cap and no NMS.
2. `_build_claims`: each candidate seeds a depth from the median world-X under
   its exact silhouette and claims footprint points within
   OWNERSHIP_PLANE_TOL_MM of it.
3. `_dedupe_pool`: collapses same-instance candidates (claim Jaccard, or close
   centre plus matching depth).
4. `select_quality_aware`: greedy coverage selection (points explained minus
   unexplained footprint points minus a per-instance cost), no shared points.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import sparse

from .ccorr import (
    ANGLE_STEP_DEG,
    IOU_THRESHOLD,
    MIN_PEAK_SEPARATION_PX,
    OVERLAP_SCORE_WEIGHT,
    WEAK_PEAK_THRESHOLD,
    Detection,
    _angle_packet,
    _attach_points,
    _fill_geometry,
    _LOST_NMS_CAP,
    _mask_select,
    _MAX_PEAKS_PER_ANGLE,
    _nms,
    _obb_iou,
    _overlap_scores,
    _project_to_pixels,
    _remove_fully_covered,
    _template_local_index,
)
from ..io.dxf_template import Template
from ..io.projection import Projection, project_yz
from ..timing import stage

# Objective weights, in units of raw 3D-point counts.
POOL_ALPHA = 1.0                    # reward per newly explained point
POOL_BETA = 0.5                     # penalty per footprint point left unexplained
POOL_INSTANCE_COST_DEFAULT = 300.0  # default fixed per-candidate cost
POOL_INSTANCE_COST_FRACTION = 0.4   # per-candidate cost, as a fraction of the
                                     # 85th-percentile claim size of the scene side

# Depth seeding for claim membership.
OWNERSHIP_PLANE_TOL_MM = 5.0
MIN_FACE_ESTIMATE_PX = 3            # minimum core points to seed a depth
FOOTPRINT_DEPTH_BAND_MM = 30.0      # wider band used for the unexplained-footprint penalty
MIN_CLAIMED_POINTS = 3              # minimum claimed points to keep a candidate

DEDUP_JACCARD_THRESHOLD = 0.6       # claimed-point-set similarity to collapse

# Same face matched at different angles: close centres and matching depth seed.
DEDUP_DEPTH_TOL_MM = 15.0
DEDUP_CENTER_FRACTION = 0.15        # fraction of the candidate's own template diagonal

# Quality-dominated candidate suppression.
QUALITY_OVERLAP_FRACTION = 0.4      # shared-claim fraction (of the smaller candidate) to call a conflict
QUALITY_CCORR_GAP = 0.15            # ccorr_frac lead required to suppress the weaker candidate

# Spatial pre-filter applied before claim-building, to bound its cost.
PREFILTER_BIN_PX = 12
PREFILTER_TOP_K = 8


def _pool_sweep(
    scene_img: np.ndarray, template: Template, side: int
) -> list[Detection]:
    """Returns every local maximum above WEAK_PEAK_THRESHOLD at every
    coarse angle against the scene image, with no accept cap and no NMS."""
    scene_f32 = scene_img.astype(np.float32)
    pool: list[Detection] = []
    for angle in np.arange(0.0, 360.0, ANGLE_STEP_DEG):
        packet = _angle_packet(template, float(angle))
        rot = packet.rot
        th, tw = rot.shape
        if th > scene_img.shape[0] or tw > scene_img.shape[1]:
            continue
        match = cv2.matchTemplate(scene_f32, rot.astype(np.float32), cv2.TM_CCORR)
        accepted_here: list[tuple[int, int]] = []
        for _ in range(_MAX_PEAKS_PER_ANGLE):
            _, max_val, _, max_loc = cv2.minMaxLoc(match)
            if max_val <= WEAK_PEAK_THRESHOLD * template.max_val_ref:
                break
            x, y = max_loc
            y0, y1 = max(0, y - th // 2), min(match.shape[0], y + th // 2 + 1)
            x0, x1 = max(0, x - tw // 2), min(match.shape[1], x + tw // 2 + 1)
            match[y0:y1, x0:x1] = -1.0
            if any(np.hypot(x - ax, y - ay) < MIN_PEAK_SEPARATION_PX
                   for ax, ay in accepted_here):
                continue
            accepted_here.append((x, y))
            frac = max_val / template.max_val_ref
            obb = ((x + tw / 2.0, y + th / 2.0),
                   (float(template.image.shape[1]), float(template.image.shape[0])),
                   -float(angle))
            overlap, ring = _overlap_scores(scene_img, packet, x, y)
            score = (OVERLAP_SCORE_WEIGHT * overlap
                     + (1.0 - OVERLAP_SCORE_WEIGHT) * ring)
            pool.append(Detection(side=side, angle_deg=float(angle), score=score,
                                   ccorr_frac=float(frac), kind="pool", obb=obb))
    return pool


def _pool_prefilter(
    pool: list[Detection], bin_px: int = PREFILTER_BIN_PX, top_k: int = PREFILTER_TOP_K
) -> list[Detection]:
    """Keeps only the top-k candidates by ccorr_frac per bin_px-square
    spatial cell (by OBB center), independent of angle."""
    buckets: dict[tuple[int, int], list[Detection]] = {}
    for d in pool:
        cx, cy = d.obb[0]
        key = (int(cx // bin_px), int(cy // bin_px))
        buckets.setdefault(key, []).append(d)
    kept: list[Detection] = []
    for cell in buckets.values():
        cell.sort(key=lambda d: -d.ccorr_frac)
        kept.extend(cell[:top_k])
    return kept


def _core_mask(
    det: Detection, template: Template, proj: Projection, world: np.ndarray
) -> np.ndarray:
    """Returns a boolean mask over world selecting points under det's exact
    (undilated) matched silhouette, used only to seed a depth estimate."""
    core = _angle_packet(template, det.angle_deg).core_bool
    px, py = _project_to_pixels(proj, world)
    inside, ix_in, iy_in = _template_local_index(core.shape, det.obb[0], px, py)
    return _mask_select(core, inside, ix_in, iy_in)


def _build_claims(
    pool: list[Detection], template: Template, proj: Projection, kept_world: np.ndarray
) -> tuple[list[Detection], list[np.ndarray], list[np.ndarray], list[float]]:
    """Depth-seeds each candidate from its exact-core median world-X, then
    splits its dilated footprint into a penalizable footprint set and a
    claim set by distance from that seed depth.

    Returns (surviving pool, footprint index list, claim index list, seed_x
    list); index lists are into kept_world.
    """
    survivors: list[Detection] = []
    footprints: list[np.ndarray] = []
    claims: list[np.ndarray] = []
    seeds: list[float] = []
    px, py = _project_to_pixels(proj, kept_world)
    world_x = kept_world[:, 0]
    for d in pool:
        packet = _angle_packet(template, d.angle_deg)
        inside, ix_in, iy_in = _template_local_index(
            packet.core_bool.shape, d.obb[0], px, py)
        core_sel = _mask_select(packet.core_bool, inside, ix_in, iy_in)
        if int(core_sel.sum()) < MIN_FACE_ESTIMATE_PX:
            continue
        seed_x = float(np.median(world_x[core_sel]))

        footprint_sel = _mask_select(packet.mask_bool, inside, ix_in, iy_in)
        dist = np.abs(world_x[footprint_sel] - seed_x)
        footprint_idx = np.nonzero(footprint_sel)[0]
        claim_idx = footprint_idx[dist <= OWNERSHIP_PLANE_TOL_MM]
        penalizable_idx = footprint_idx[dist <= FOOTPRINT_DEPTH_BAND_MM]

        if claim_idx.size < MIN_CLAIMED_POINTS:
            continue
        d.n_points = int(claim_idx.size)  # placeholder; _finalize refits
        survivors.append(d)
        footprints.append(penalizable_idx)
        claims.append(claim_idx)
        seeds.append(seed_x)
    return survivors, footprints, claims, seeds


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    if inter == 0:
        return 0.0
    union = a.size + b.size - inter
    return inter / union if union else 0.0


def _dedupe_pool(
    pool: list[Detection], footprints: list[np.ndarray], claims: list[np.ndarray],
    seeds: list[float] | None = None,
    jaccard_thr: float = DEDUP_JACCARD_THRESHOLD,
    max_center_dist_px: float | None = None,
    depth_tol_mm: float = DEDUP_DEPTH_TOL_MM,
    center_fraction: float = DEDUP_CENTER_FRACTION,
) -> tuple[list[Detection], list[np.ndarray], list[np.ndarray]]:
    """Collapses candidates that are the same physical instance, keeping the
    highest-scoring representative. Either of two tests is sufficient:

    1. Claimed-point sets are near-identical (Jaccard > jaccard_thr).
    2. Centers are close (within center_fraction of the candidate's own
       template diagonal) and depth seeds agree (within depth_tol_mm). Only
       applied when `seeds` is given.

    `max_center_dist_px`, when given, is a cheap center-distance reject
    before the more expensive set-intersection test 1.
    """
    order = sorted(range(len(pool)), key=lambda i: -pool[i].score)
    kept_idx: list[int] = []
    kept_cx = np.empty(len(pool), dtype=np.float64)
    kept_cy = np.empty(len(pool), dtype=np.float64)
    kept_seed = np.empty(len(pool), dtype=np.float64)
    for i in order:
        ci = pool[i].obb[0]
        m = len(kept_idx)
        center_dist = np.hypot(kept_cx[:m] - ci[0], kept_cy[:m] - ci[1])
        conflict = bool(
            seeds is not None
            and np.any((center_dist <= center_fraction * np.hypot(*pool[i].obb[1]))
                       & (np.abs(kept_seed[:m] - seeds[i]) <= depth_tol_mm)))
        if not conflict:
            near = (np.nonzero(center_dist <= max_center_dist_px)[0]
                    if max_center_dist_px is not None else range(m))
            for j_pos in near:
                if _jaccard(claims[i], claims[kept_idx[j_pos]]) > jaccard_thr:
                    conflict = True
                    break
        if conflict:
            continue
        kept_cx[m] = ci[0]
        kept_cy[m] = ci[1]
        if seeds is not None:
            kept_seed[m] = seeds[i]
        kept_idx.append(i)
    return ([pool[i] for i in kept_idx],
            [footprints[i] for i in kept_idx],
            [claims[i] for i in kept_idx])


def select_quality_aware(
    pool: list[Detection], footprints: list[np.ndarray], claims: list[np.ndarray],
    n_points: int,
    alpha: float = POOL_ALPHA, beta: float = POOL_BETA,
    instance_cost: float | np.ndarray = POOL_INSTANCE_COST_DEFAULT,
    overlap_fraction: float = QUALITY_OVERLAP_FRACTION,
    ccorr_gap: float = QUALITY_CCORR_GAP,
) -> list[int]:
    """Runs `select_greedy`, drops any selected candidate out-scored by
    >= `ccorr_gap` in ccorr_frac by an unselected one sharing >=
    `overlap_fraction` of its claimed points, and re-runs selection once."""
    selected_idx = select_greedy(footprints, claims, n_points,
                                  alpha=alpha, beta=beta, instance_cost=instance_cost)
    selected = set(selected_idx)
    dominated = set()
    for s in selected_idx:
        cs = claims[s]
        if cs.size == 0:
            continue
        for u in range(len(pool)):
            if u in selected:
                continue
            cu = claims[u]
            if cu.size == 0:
                continue
            inter = np.intersect1d(cs, cu, assume_unique=True).size
            if inter == 0:
                continue
            overlap = inter / min(cs.size, cu.size)
            if overlap >= overlap_fraction and pool[u].ccorr_frac - pool[s].ccorr_frac >= ccorr_gap:
                dominated.add(s)
                break
    if dominated:
        keep_idx = [i for i in range(len(pool)) if i not in dominated]
        sub_footprints = [footprints[i] for i in keep_idx]
        sub_claims = [claims[i] for i in keep_idx]
        sub_cost = (instance_cost[keep_idx] if isinstance(instance_cost, np.ndarray)
                    else instance_cost)
        sub_selected = select_greedy(sub_footprints, sub_claims, n_points,
                                      alpha=alpha, beta=beta, instance_cost=sub_cost)
        selected_idx = [keep_idx[i] for i in sub_selected]
    return selected_idx


def select_greedy(
    footprints: list[np.ndarray], claims: list[np.ndarray], n_points: int,
    alpha: float = POOL_ALPHA, beta: float = POOL_BETA,
    instance_cost: float | np.ndarray = POOL_INSTANCE_COST_DEFAULT,
) -> list[int]:
    """Greedy marginal-gain selection; returns candidate indices in order.

    Gain = alpha * newly_explained - beta * foreign_unexplained - instance_cost
    (scalar or per-candidate array), recomputed each step via sparse
    candidate x point matrices. Stops when no gain is positive; ties go to
    the lowest index.
    """
    n = len(footprints)
    if n == 0:
        return []

    foreign = [np.setdiff1d(footprints[i], claims[i], assume_unique=False)
               for i in range(n)]

    def _to_csr(sets: list[np.ndarray]) -> sparse.csr_matrix:
        rows = np.concatenate([np.full(s.size, i, dtype=np.int64)
                                for i, s in enumerate(sets)])
        cols = np.concatenate(sets)
        data = np.ones(cols.size, dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_points))

    claim_mat = _to_csr(claims)
    foreign_mat = _to_csr(foreign)

    cost_arr = (instance_cost.astype(np.float64) if isinstance(instance_cost, np.ndarray)
                else np.full(n, float(instance_cost)))

    explained = np.zeros(n_points, dtype=np.float32)
    alive = np.ones(n, dtype=bool)
    selected: list[int] = []

    while alive.any():
        unexplained = 1.0 - explained
        new_counts = claim_mat.dot(unexplained)
        foreign_counts = foreign_mat.dot(unexplained)
        gains = alpha * new_counts - beta * foreign_counts - cost_arr
        gains[~alive] = -np.inf
        best_i = int(np.argmax(gains))
        if gains[best_i] <= 0.0:
            break
        selected.append(best_i)
        if claims[best_i].size:
            explained[claims[best_i]] = 1.0
        alive[best_i] = False

    return selected


def _refit_obb_display(det: Detection, proj: Projection, claimed_world: np.ndarray) -> None:
    """Sets det.obb_display to det.obb tightened to the pixel extent of the
    claimed points (det.obb is left unchanged)."""
    if claimed_world.shape[0] < 3:
        return
    px = proj.padding_px + (claimed_world[:, 1] - proj.min_y) * proj.eff_scale
    py = proj.padding_px + (proj.max_z - claimed_world[:, 2]) * proj.eff_scale
    pts = np.stack([px, py], axis=1)

    box_pts = cv2.boxPoints((det.obb[0], det.obb[1], det.obb[2]))
    u = box_pts[1] - box_pts[0]
    u_norm = np.linalg.norm(u)
    v = box_pts[3] - box_pts[0]
    v_norm = np.linalg.norm(v)
    if u_norm == 0 or v_norm == 0:
        return
    u /= u_norm
    v /= v_norm

    rel = pts - box_pts[0]
    lu = rel @ u
    lv = rel @ v
    extent_u = float(lu.max() - lu.min())
    extent_v = float(lv.max() - lv.min())
    if extent_u <= 0 or extent_v <= 0:
        return
    mid_u = (lu.max() + lu.min()) / 2.0
    mid_v = (lv.max() + lv.min()) / 2.0
    new_center = box_pts[0] + mid_u * u + mid_v * v

    # map extents to (w, h) by matching u_norm to the original edge lengths
    orig_w, orig_h = det.obb[1]
    if abs(u_norm - orig_w) <= abs(u_norm - orig_h):
        w, h = extent_u, extent_v
    else:
        w, h = extent_v, extent_u
    det.obb_display = ((float(new_center[0]), float(new_center[1])), (w, h), det.obb[2])


def _finalize(det: Detection, proj: Projection, kept_world: np.ndarray, idx: np.ndarray) -> Detection:
    """Recomputes a selected candidate's geometry from only its claimed
    points, and sets its display-only tightened OBB."""
    claimed_world = kept_world[idx]
    _fill_geometry(det, claimed_world)
    _refit_obb_display(det, proj, claimed_world)
    det.kind = "confirmed"
    return det


def detect_faces_pool(
    world_points: np.ndarray, template: Template, side: int,
) -> tuple[list[Detection], list[Detection], Projection]:
    """Runs the candidate-pool detector on one side's world cloud.
    Returns (confirmed, lost, projection)."""
    padding = template.default_padding
    with stage("projection"):
        proj = project_yz(world_points, padding_px=padding)
        kept_world = world_points[proj.keep_mask]

    with stage("template_match"):
        pool = _pool_sweep(proj.image, template, side)
        pool = _pool_prefilter(pool)

    with stage("ownership"):
        pool, footprints, claims, seeds = _build_claims(pool, template, proj, kept_world)
        pool, footprints, claims = _dedupe_pool(
            pool, footprints, claims, seeds, max_center_dist_px=template.diagonal_px)

    # Scene-relative instance cost: 85th percentile of post-dedupe claim sizes.
    scale = float(np.percentile([c.size for c in claims], 85)) if claims else 0.0
    instance_cost = POOL_INSTANCE_COST_FRACTION * scale

    with stage("selection"):
        selected_idx = select_quality_aware(pool, footprints, claims, len(kept_world),
                                             instance_cost=instance_cost)
        selected = set(selected_idx)

        confirmed = [_finalize(pool[i], proj, kept_world, claims[i]) for i in selected_idx]

    # Lost band: unselected candidates above the weak-peak floor (reported only).
    lost = [pool[i] for i in range(len(pool))
            if i not in selected and pool[i].ccorr_frac > WEAK_PEAK_THRESHOLD]
    for d in lost:
        d.kind = "lost"
        _attach_points(d, template, proj, kept_world)
    lost = [d for d in lost if d.n_points >= 3]
    lost = _remove_fully_covered(_nms(lost))[:_LOST_NMS_CAP]
    lost = [d for d in lost
            if all(_obb_iou(d.obb, k.obb) <= IOU_THRESHOLD for k in confirmed)]

    return confirmed, lost, proj
