"""Learned detection baseline: a small CPU-trainable center-point CNN.

Outputs per-face positions (a CenterNet-style center heatmap plus a
sub-stride offset), not boxes, matching what the shared count and
correspondence metrics consume. `obb` in the resulting `Detection` objects
is a nominal fixed-size square for overlay rendering only; this detector
does not estimate extent or angle.

torch is an optional dependency and is imported lazily; nothing else in
this package requires it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ccorr import Detection
from ..io.projection import Projection

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

OUT_STRIDE = 8              # heatmap is input/8
HEATMAP_SIGMA_PX = 1.5      # gaussian target sigma, on the heatmap grid
CROP_PX = 384               # square training crop
PEAK_NMS_KERNEL = 3         # heatmap-grid max-pool NMS window
MAX_PEAKS = 64

THRESHOLD_GRID = tuple(np.round(np.arange(0.05, 0.96, 0.025), 3))

# Family grouping for splits: profiles that are not independent samples of
# the same cross-section share one group.
FAMILY_GROUPS = {
    "l-profile": "l-profile+l-profile-reshoot",
    "l-profile-reshoot": "l-profile+l-profile-reshoot",
    "square-profile": "square-profile",
    "complex-profile": "complex-profile",
    "heavy-profile": "heavy-profile",
}


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

@dataclass
class SceneSide:
    """One projection, its face-center labels, and its ground-truth count."""
    scene_id: str
    side: int
    family: str
    category: str
    image: np.ndarray            # uint8 (H,W)
    min_y: float
    max_z: float
    eff_scale: float
    padding_px: int
    centers_px: np.ndarray       # (N,2) float32, (x, y) in image pixels
    gt_count: int

    @property
    def group(self) -> str:
        return FAMILY_GROUPS[self.family]

    def as_projection(self) -> Projection:
        """Returns a Projection carrying this sample's pixel<->world mapping."""
        return Projection(
            image=self.image, min_y=self.min_y, max_z=self.max_z,
            eff_scale=self.eff_scale, padding_px=self.padding_px,
            n_points=0, keep_mask=np.zeros(0, dtype=bool),
        )


def save_dataset(samples: list[SceneSide], path: Path) -> None:
    """Writes the dataset to one .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    meta = []
    for k, s in enumerate(samples):
        payload[f"img_{k}"] = s.image
        payload[f"ctr_{k}"] = s.centers_px.astype(np.float32)
        meta.append((s.scene_id, s.side, s.family, s.category, s.min_y,
                     s.max_z, s.eff_scale, s.padding_px, s.gt_count))
    payload["meta"] = np.array(meta, dtype=object)
    np.savez_compressed(path, **payload)


def load_dataset(path: Path) -> list[SceneSide]:
    """Loads a dataset written by save_dataset."""
    with np.load(path, allow_pickle=True) as z:
        meta = z["meta"]
        out = []
        for k, row in enumerate(meta):
            out.append(SceneSide(
                scene_id=str(row[0]), side=int(row[1]), family=str(row[2]),
                category=str(row[3]), image=z[f"img_{k}"],
                min_y=float(row[4]), max_z=float(row[5]),
                eff_scale=float(row[6]), padding_px=int(row[7]),
                centers_px=z[f"ctr_{k}"], gt_count=int(row[8]),
            ))
    return out


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def scene_folds(scene_ids: list[str], n_folds: int = 5, seed: int = 0
                ) -> list[list[str]]:
    """Returns scene-disjoint folds; both sides of a scene stay in the same fold."""
    ids = sorted(set(scene_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for k, i in enumerate(perm):
        folds[k % n_folds].append(ids[i])
    return [sorted(f) for f in folds]


def group_folds(samples: list[SceneSide]) -> list[tuple[str, list[str]]]:
    """Returns leave-one-group-out folds as (group name, held-out scene ids)."""
    by_group: dict[str, set[str]] = {}
    for s in samples:
        by_group.setdefault(s.group, set()).add(s.scene_id)
    return [(g, sorted(ids)) for g, ids in sorted(by_group.items())]


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------

def render_heatmap(shape_hw: tuple[int, int], centers_hm: np.ndarray,
                   sigma: float = HEATMAP_SIGMA_PX) -> np.ndarray:
    """Returns the CenterNet target heatmap: max of a Gaussian at each center."""
    h, w = shape_hw
    hm = np.zeros((h, w), dtype=np.float32)
    if len(centers_hm) == 0:
        return hm
    radius = int(np.ceil(3 * sigma))
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    gy, gx = np.meshgrid(ax, ax, indexing="ij")
    kernel = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    for cx, cy in centers_hm:
        ix, iy = int(round(float(cx))), int(round(float(cy)))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        kx0, kx1 = x0 - (ix - radius), kernel.shape[1] - ((ix + radius + 1) - x1)
        ky0, ky1 = y0 - (iy - radius), kernel.shape[0] - ((iy + radius + 1) - y1)
        np.maximum(hm[y0:y1, x0:x1], kernel[ky0:ky1, kx0:kx1], out=hm[y0:y1, x0:x1])
    return hm


def build_targets(centers_px: np.ndarray, shape_hw: tuple[int, int],
                  stride: int = OUT_STRIDE, sigma: float = HEATMAP_SIGMA_PX):
    """Returns (heatmap, offset, offset_mask) targets for one crop."""
    h, w = shape_hw
    hh, hw = h // stride, w // stride
    c = np.asarray(centers_px, dtype=np.float32).reshape(-1, 2) / stride
    hm = render_heatmap((hh, hw), c, sigma)
    off = np.zeros((2, hh, hw), dtype=np.float32)
    mask = np.zeros((hh, hw), dtype=np.float32)
    for cx, cy in c:
        ix, iy = int(round(float(cx))), int(round(float(cy)))
        if not (0 <= ix < hw and 0 <= iy < hh):
            continue
        off[0, iy, ix] = float(cx) - ix
        off[1, iy, ix] = float(cy) - iy
        mask[iy, ix] = 1.0
    return hm, off, mask


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def sample_crop(rng: np.random.Generator, s: SceneSide, crop: int = CROP_PX,
                rotate: bool = True, flip: bool = True,
                scale_range: tuple[float, float] = (0.85, 1.18),
                ) -> tuple[np.ndarray, np.ndarray]:
    """Returns one augmented (crop_image, centers_in_crop) pair.

    Crops are centered on a labeled face 70% of the time and uniformly at
    random otherwise. Rotation, scale, and horizontal flip are applied via
    one affine transform.
    """
    h, w = s.image.shape
    if len(s.centers_px) and rng.random() < 0.7:
        cx, cy = s.centers_px[rng.integers(len(s.centers_px))]
        cx += rng.normal(0, crop * 0.15)
        cy += rng.normal(0, crop * 0.15)
    else:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)

    angle = float(rng.uniform(0, 360)) if rotate else 0.0
    scale = float(rng.uniform(*scale_range))
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, scale)
    m[0, 2] += crop / 2.0 - cx
    m[1, 2] += crop / 2.0 - cy
    if flip and rng.random() < 0.5:
        flip_m = np.array([[-1.0, 0.0, crop - 1.0], [0.0, 1.0, 0.0]])
        m = flip_m @ np.vstack([m, [0, 0, 1]])

    img = cv2.warpAffine(s.image, m, (crop, crop), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if len(s.centers_px):
        pts = np.hstack([s.centers_px, np.ones((len(s.centers_px), 1), np.float32)])
        out = (m @ pts.T).T.astype(np.float32)
        keep = ((out[:, 0] >= 0) & (out[:, 0] < crop)
                & (out[:, 1] >= 0) & (out[:, 1] < crop))
        out = out[keep]
    else:
        out = np.zeros((0, 2), dtype=np.float32)

    img = img.astype(np.float32) / 255.0
    if rng.random() < 0.5:
        img = np.clip(img * rng.uniform(0.8, 1.2), 0.0, 1.0)
    return img, out


# --------------------------------------------------------------------------
# model (torch imported lazily)
# --------------------------------------------------------------------------

BODY_DILATIONS = (1, 2, 4, 8, 16, 1)


def build_model(width: int = 64):
    """Returns a small dilated fully-convolutional center detector.

    A stride-8 stem feeds a dilated residual-free stack (dilations
    1, 2, 4, 8, 16, 1) sized to reach a receptive field wide enough to cover
    the largest supported cross-section, at roughly 0.25M parameters.
    """
    import torch.nn as nn

    def block(cin, cout, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class CenterNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.stem = nn.Sequential(
                block(1, w // 4, stride=2),          # /2
                block(w // 4, w // 2, stride=2),     # /4
                block(w // 2, w, stride=2),          # /8
            )
            self.body = nn.Sequential(
                *[block(w, w, dilation=d) for d in BODY_DILATIONS])
            self.hm = nn.Conv2d(w, 1, 1)
            self.off = nn.Conv2d(w, 2, 1)
            nn.init.constant_(self.hm.bias, -4.6)

        def forward(self, x):
            f = self.body(self.stem(x))
            return self.hm(f), self.off(f)

    return CenterNet()


def focal_loss(logits, target):
    """Returns the CenterNet penalty-reduced focal loss (Law & Deng), on logits."""
    import torch
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    neg_w = torch.pow(1.0 - target, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    n = pos.sum()
    total = -(pos_loss.sum() + neg_loss.sum())
    return total / n if n > 0 else -neg_loss.sum()


def train_model(samples: list[SceneSide], steps: int = 2000, batch: int = 8,
                lr: float = 2e-3, seed: int = 0, crop: int = CROP_PX,
                width: int = 64, log_every: int = 0, threads: int | None = None):
    """Trains one fold's model. Returns (model, loss history)."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(width)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    # Sample scene-sides in proportion to their labeled faces.
    weights = np.array([max(len(s.centers_px), 1) for s in samples], dtype=np.float64)
    weights /= weights.sum()

    losses = []
    for step in range(steps):
        imgs, hms, offs, masks = [], [], [], []
        for _ in range(batch):
            s = samples[rng.choice(len(samples), p=weights)]
            img, ctr = sample_crop(rng, s, crop=crop)
            hm, off, mask = build_targets(ctr, img.shape)
            imgs.append(img[None]); hms.append(hm[None])
            offs.append(off); masks.append(mask[None])
        x = torch.from_numpy(np.stack(imgs))
        t_hm = torch.from_numpy(np.stack(hms))
        t_off = torch.from_numpy(np.stack(offs))
        t_mask = torch.from_numpy(np.stack(masks))

        p_hm, p_off = model(x)
        loss = focal_loss(p_hm, t_hm)
        n_pos = t_mask.sum()
        if n_pos > 0:
            loss = loss + 1.0 * ((p_off - t_off).abs() * t_mask).sum() / n_pos
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{steps} loss {np.mean(losses[-log_every:]):.4f}",
                  flush=True)
    model.eval()
    return model, losses


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

def predict_peaks(model, image: np.ndarray, stride: int = OUT_STRIDE,
                  max_peaks: int = MAX_PEAKS) -> np.ndarray:
    """Runs a full-image forward pass. Returns an (M,3) array of (x_px, y_px, score).

    All peaks are returned with their score; thresholding is left to the caller.
    """
    import torch
    import torch.nn.functional as F

    h, w = image.shape
    pad_h = (-h) % (stride * 4)
    pad_w = (-w) % (stride * 4)
    img = np.pad(image.astype(np.float32) / 255.0, ((0, pad_h), (0, pad_w)))
    with torch.no_grad():
        x = torch.from_numpy(img)[None, None]
        logits, off = model(x)
        hm = torch.sigmoid(logits)
        pooled = F.max_pool2d(hm, PEAK_NMS_KERNEL, stride=1,
                              padding=PEAK_NMS_KERNEL // 2)
        keep = (hm == pooled).float() * hm
        flat = keep.reshape(-1)
        k = min(max_peaks, flat.numel())
        scores, idx = torch.topk(flat, k)
        hw = hm.shape[-1]
        ys = (idx // hw).float()
        xs = (idx % hw).float()
        ox = off[0, 0].reshape(-1)[idx].clamp(-0.5, 0.5)
        oy = off[0, 1].reshape(-1)[idx].clamp(-0.5, 0.5)
        px = ((xs + ox) * stride).numpy()
        py = ((ys + oy) * stride).numpy()
        sc = scores.numpy()
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (sc > 0)
    return np.stack([px[ok], py[ok], sc[ok]], axis=1)


def peaks_to_detections(peaks: np.ndarray, s: SceneSide, threshold: float,
                        ) -> list[Detection]:
    """Converts thresholded peaks into Detection objects.

    Only `centroid_yz_mm`, `side`, `kind`, and `score` are meaningful; `obb`
    is a nominal square at the peak, since this detector does not estimate
    extent or angle.
    """
    proj = s.as_projection()
    out = []
    for px, py, sc in peaks:
        if sc < threshold:
            continue
        y, z = proj.pixel_to_yz(float(px), float(py))
        out.append(Detection(
            side=s.side, angle_deg=0.0, score=float(sc), ccorr_frac=float(sc),
            kind="confirmed", obb=((float(px), float(py)), (40.0, 40.0), 0.0),
            centroid_yz_mm=(float(y), float(z)),
        ))
    out.sort(key=lambda d: d.centroid_yz_mm[0])
    return out


def pick_threshold(preds: dict[tuple[str, int], np.ndarray],
                   samples: list[SceneSide],
                   grid=THRESHOLD_GRID) -> tuple[float, float]:
    """Returns (threshold, f1) maximizing count-F1 on a validation set."""
    best_thr, best_f1 = float(grid[0]), -1.0
    for thr in grid:
        tp = fp = fn = 0
        for s in samples:
            p = preds[(s.scene_id, s.side)]
            n = int((p[:, 2] >= thr).sum()) if len(p) else 0
            tp += min(s.gt_count, n)
            fp += max(n - s.gt_count, 0)
            fn += max(s.gt_count - n, 0)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    return best_thr, best_f1


# --------------------------------------------------------------------------
# shared scoring helpers
# --------------------------------------------------------------------------

def count_counts(n_pred: int, gt: int) -> tuple[int, int, int]:
    """Returns (tp, fp, fn) under the min(pred, gt) count convention."""
    return min(gt, n_pred), max(n_pred - gt, 0), max(gt - n_pred, 0)


def positional_counts(dets, anchors: list[tuple[float, float]],
                      gate_mm: float) -> tuple[int, int]:
    """Returns (#anchors re-linked to a detection, #anchors) for one scene-side."""
    from ..gt import resolve_anchor_indices
    if not anchors:
        return 0, 0
    links = resolve_anchor_indices(anchors, dets, gate_mm=gate_mm)
    return len(links), len(anchors)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
