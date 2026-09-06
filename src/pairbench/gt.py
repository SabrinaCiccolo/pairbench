"""Loading and parsing of the hand-annotated ground truth
(`results/pairbench/gt.csv`, `results/pairbench/gt_correspondence.csv`)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Scenes excluded from pooled metrics. l-profile-reshoot/overlap-01 is a
# byte-identical copy of l-profile/overlap-01; it stays annotated.
FLAGGED_SCENES: set[str] = {
    "l-profile-reshoot/overlap-01",
}

# Repeat captures: separate scans of one unchanged bundle. They are not
# independent draws, so every bootstrap resamples bundles, not scenes.
REPEAT_CAPTURES_CSV = Path(__file__).resolve().parents[2] / "data" / "repeat_captures.csv"


def load_bundles(path: Path = REPEAT_CAPTURES_CSV) -> dict[str, str]:
    """scene_id -> bundle label for the scenes in a repeat group. A scene not
    listed is its own bundle; use `bundle_of` rather than indexing this."""
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        return {r["scene_id"]: r["bundle"] for r in csv.DictReader(f)}


def bundle_of(scene_id: str, bundles: dict[str, str] | None = None) -> str:
    """The resampling unit a scene belongs to."""
    if bundles is None:
        bundles = load_bundles()
    return bundles.get(scene_id, scene_id)


def load_gt(path: Path) -> dict[str, dict]:
    """scene_id -> raw gt row (string values, exactly as annotated)."""
    with open(path, newline="") as f:
        return {r["scene_id"]: r for r in csv.DictReader(f)}


# Ceiling for re-linking a stored anchor to a detection in a later detection
# set. Also capped per anchor at half the distance to its nearest neighbouring
# anchor, so a link stays unambiguous relative to local bar spacing.
ANCHOR_GATE_MM = 25.0


def parse_correspondence_pairs(pairs_field: str) -> list[tuple[int, int]]:
    """'0-1,1-0,2-2' -> [(0,1),(1,0),(2,2)]; '' -> []."""
    pairs_field = pairs_field.strip()
    if not pairs_field:
        return []
    out = []
    for tok in pairs_field.split(","):
        i, j = tok.strip().split("-")
        out.append((int(i), int(j)))
    return out


def format_correspondence_pairs(pairs: list[tuple[int, int]]) -> str:
    return ",".join(f"{i}-{j}" for i, j in sorted(pairs))


def expand_shorthand_pairs(raw: str, n1: int, n2: int) -> list[tuple[int, int]]:
    """Expands annotation shorthand into a full (i,j) pair list.

    Default is the identity (i,i), or (i,i+N) with a 'shift:N' token.
    Tokens 'i-j' (explicit pair), 'i-x' / 'x-j' (unmatched index) override
    the default. '' is the identity. Raises ValueError on malformed tokens.
    """
    consumed1: set[int] = set()
    consumed2: set[int] = set()
    explicit: list[tuple[int, int]] = []
    shift = 0
    raw = raw.strip()
    if raw:
        for tok in raw.split(","):
            tok = tok.strip()
            if tok.lower().startswith("shift:"):
                shift = int(tok.split(":", 1)[1])
                continue
            left, right = tok.split("-")
            left, right = left.strip(), right.strip()
            if left != "x":
                consumed1.add(int(left))
            if right != "x":
                consumed2.add(int(right))
            if left != "x" and right != "x":
                explicit.append((int(left), int(right)))
    if shift:
        defaults = [(i, i + shift) for i in range(n1)
                    if 0 <= i + shift < n2 and i not in consumed1
                    and i + shift not in consumed2]
    else:
        n = min(n1, n2)
        defaults = [(i, i) for i in range(n)
                    if i not in consumed1 and i not in consumed2]
    return sorted(defaults + explicit)


def format_anchors(dets) -> str:
    """Detection list -> 'y0|z0;y1|z1;...' (YZ centroids, in index order).
    Lets a later run re-link annotated pairs to detections by position."""
    return ";".join(f"{d.centroid_yz_mm[0]:.2f}|{d.centroid_yz_mm[1]:.2f}"
                    for d in dets)


def parse_anchors(field: str) -> list[tuple[float, float]]:
    """Inverse of format_anchors; '' -> []."""
    field = (field or "").strip()
    if not field:
        return []
    out = []
    for tok in field.split(";"):
        y, z = tok.split("|")
        out.append((float(y), float(z)))
    return out


def _resolve_anchor_links(anchors: list[tuple[float, float]], dets,
                          gate_mm: float) -> dict[int, tuple[int, float]]:
    """{annotation-time index -> (current index, distance_mm)} for anchors
    with an unambiguous detection: mutual nearest neighbour, within
    min(gate_mm, half the nearest-anchor distance), runner-up anchor >= 2x farther."""
    if not anchors or not dets:
        return {}
    a = np.asarray(anchors, dtype=np.float64)
    c = np.asarray([d.centroid_yz_mm for d in dets], dtype=np.float64)
    dist = np.linalg.norm(a[:, None, :] - c[None, :, :], axis=2)

    if len(a) > 1:
        aa = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
        np.fill_diagonal(aa, np.inf)
        radius = np.minimum(gate_mm, aa.min(axis=1) / 2.0)
    else:
        radius = np.full(len(a), gate_mm)

    best_for_anchor = dist.argmin(axis=1)
    best_for_det = dist.argmin(axis=0)
    out = {}
    for ai, di in enumerate(best_for_anchor):
        if dist[ai, di] > radius[ai] or best_for_det[di] != ai:
            continue
        if len(a) > 1:
            runner_up = float(np.partition(dist[:, di], 1)[1])
            if dist[ai, di] * 2.0 >= runner_up:
                continue  # detection sits between two anchors: ambiguous
        out[ai] = (int(di), float(dist[ai, di]))
    return out


def resolve_anchor_indices(anchors: list[tuple[float, float]], dets,
                           gate_mm: float = ANCHOR_GATE_MM) -> dict[int, int]:
    """{annotation-time index -> current index} for anchors that still have an
    unambiguous detection near them. See `_resolve_anchor_links`."""
    return {ai: di for ai, (di, _) in _resolve_anchor_links(anchors, dets, gate_mm).items()}


def resolve_anchor_distances(anchors: list[tuple[float, float]], dets,
                             gate_mm: float = ANCHOR_GATE_MM) -> dict[int, float]:
    """{annotation-time index -> distance_mm} to the detection it resolved
    to. Same matching rule as `resolve_anchor_indices`."""
    return {ai: dist for ai, (_, dist) in _resolve_anchor_links(anchors, dets, gate_mm).items()}


def load_gt_correspondence(path: Path) -> dict[str, dict]:
    """scene_id -> raw gt_correspondence row. `pairs` still needs
    parse_correspondence_pairs."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["scene_id"]: r for r in csv.DictReader(f)}
