# Method

## Setup

Two 3D scanners face each other across a conveyor. Each scanner sits on its
own carriage that moves along the cross-conveyor axis and sees only the end
faces of the bars on its side. The two views share no surface, so matching a
bar's two ends has to be solved from geometry alone.

| Axis | Meaning |
|---|---|
| X | cross-conveyor axis, along the bars; scanner 1 on +X, scanner 2 on −X |
| Y | transport axis |
| Z | height |

Parameters used throughout the code:

| Parameter | Value | Where |
|---|---|---|
| Nominal bar length | 6005 mm | `pairing/common.py::PROFILE_LENGTH_MM` |
| Radial capture gate | 850 mm from each scanner origin | `calibration.json::max_distance_mm` |
| Carriage separation | ≈ 5403 mm (fitted, see `results/pairbench/a2_selfcalib.md`) | `selfcalib.py` |
| Effective image resolution | 2.5 px/mm (10 px/mm render × 0.25) | `io/projection.py` |

## Calibration

Each scanner maps its points to a shared world frame with
`p_world = R · p_scanner + t`. Two calibrations are provided:

- `src/pairbench/calibration.json` (`io/calib.py`): one rotation and one
  translation knot per scanner. YZ is metric; absolute world-X carries a
  constant bias of about 1075 mm.
- `data/calibration_full.json` (`io/calib_full.py`): ten knots per scanner,
  interpolated along the carriage position (linear translation, SLERP rotation).

The carriage position of each scan is not recorded. `selfcalib.py` fits it
from matched pairs under the bar-length constraint. Only the separation
`s1 − s2` is observable; moving both carriages together leaves every bar
length unchanged.

## Projection

Points beyond the radial gate are dropped. The world-frame cloud of one
scanner is projected onto the YZ plane (image x ← world Y, image y ← world Z
flipped), rasterized at 10 px/mm with a small disk per point, downscaled to
2.5 px/mm and blurred (`io/projection.py`). A lookup table keeps the 3D points
behind every pixel, so a 2D detection recovers its 3D support.

## Detection

The template is the profile's DXF cross-section (`data/dxf/`), filled and
rendered at the same resolution (`io/dxf_template.py`). Thresholds are
fractions of the template's self-correlation.

**`ccorr` — template-matching baseline** (`detect/ccorr.py`)

1. Rotate the template in 2° steps and cross-correlate it with the image.
2. Keep peaks above 0.2; a peak above 0.4 whose footprint overlap is at least
   0.65 is a confirmed face, a weaker one is kept as "lost".
3. Non-max suppression on oriented-box IoU (0.2).
4. Erase the confirmed footprints and sweep a second time.
5. Attach the 3D points under each footprint.

**`pool` — candidate-pool detector** (`detect/pool.py`, default)

1. Collect every correlation peak above 0.2 at every angle, without NMS.
2. Each candidate claims the footprint points close to its own depth (median
   world-X under its silhouette).
3. Merge candidates that describe the same instance.
4. Select instances greedily by explained points minus unexplained footprint
   minus a per-instance cost, with no point shared between instances
   (`select_quality_aware`).

**Rescue** (`detect/rescue.py`): for a face with no partner in the other
view, the scene translation predicts where the partner should be, and a local
template match runs there.

## Pairing

Angles are folded into the profile's symmetry period, measured from the
template (`symmetry.py`): 360° for an asymmetric section, 180° or 90° for
symmetric ones. Faces are grouped into stacking levels by height.

| Arm | Rule |
|---|---|
| `greedy_index` | pair by index within each level; twist ≤ 2.5°, tilt ≤ 2.5°, \|length − 6005\| ≤ 10 mm |
| `greedy_index_nolen` | as above, without the length check |
| `greedy_index_order` | index order alone, no checks |
| `greedy_index_debiased` | length check after removing the global length bias |
| `hungarian` | global assignment on folded angle, saturating YZ distance and level cost |
| `hungarian_aligned` | RANSAC estimate of the cross-view YZ translation, then `hungarian` |
| `monotone_aligned` | same cost, assignment constrained to keep Y order within a level |
| `hybrid_aligned` | `monotone_aligned` plus crossing hypotheses admitted only on evidence (default) |

`pairing/hypotheses.py` enumerates the translations a bundle admits and costs
each one; `experiments/cost_gap.py` uses it to check whether the objective
ranks the true correspondence first. The equations are listed in
[`equations.md`](equations.md).

## Synthetic twin

`pairbench.synth` extrudes a DXF section into a prism, places bundles at known
poses, raycasts the two scanners from their calibrated positions and writes
PLY files plus `truth.json`. The output goes through the same loader,
detector and pairing code as the real scans.
