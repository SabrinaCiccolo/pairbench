"""Estimate the per-scanner carriage step positions from the bar-length prior.

The full calibration (data/calibration_full.json) is evaluated at the carriage
X position s at scan time, which is not recorded in the data. This script fits
one effective (s1, s2) for the whole dataset,

  minimize over (s1, s2):  median | |T1(c1, s1) - T2(c2, s2)| - 6005 mm |

over `hungarian` centroid pairs, and re-runs greedy pairing with the length
check on the recalibrated centroids. The objective has a flat valley along
s1 + s2 ~ const; the report shows it.

Outputs: results/pairbench/fitted_steps.json, results/pairbench/step_fit_report.md.

Usage: python -m pairbench.experiments.fit_step_positions [--detector pool]
"""

import argparse
import json

import numpy as np
from scipy.optimize import minimize

from ..io.calib import load_calibration
from ..io.calib_full import load_full_calibration
from ..detect.ccorr import dets_from_json
from ..gt import FLAGGED_SCENES, load_gt
from ..metrics import precision_recall_f1
from ..pairing.common import PROFILE_LENGTH_MM
from ..pairing.greedy_index import pair_greedy_index
from ..pairing.hungarian import pair_hungarian
from .common import GT, RESULTS

OUT_JSON = RESULTS / "fitted_steps.json"
OUT_MD = RESULTS / "step_fit_report.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="pool", choices=("pool", "ccorr"),
                    help="which b1_detections_<detector>.json to fit against")
    args = ap.parse_args()
    in_json = RESULTS / f"b1_detections_{args.detector}.json"

    offline = load_calibration()
    full = load_full_calibration()
    with open(in_json) as f:
        data = json.load(f)
    gt_rows = load_gt(GT)

    # scanner-frame centroid pairs from hungarian proposals
    pairs_scanner = {1: [], 2: []}
    scene_pair_index = []  # (scene_id, slice into the pair arrays)
    dets_by_scene = {}
    for sid in sorted(data["scenes"]):
        if sid in FLAGGED_SCENES:
            continue
        scene = data["scenes"][sid]
        d1 = dets_from_json(scene["side1"]["confirmed"])
        d2 = dets_from_json(scene["side2"]["confirmed"])
        dets_by_scene[sid] = (d1, d2)
        res = pair_hungarian(d1, d2)
        start = len(pairs_scanner[1])
        for i, j in res.pairs:
            for side, det in ((1, d1[i]), (2, d2[j])):
                sc = offline.for_side(side)
                c_world = np.asarray(det.centroid_world_mm)
                pairs_scanner[side].append(sc.R.T @ (c_world - sc.t))
        scene_pair_index.append((sid, start, len(pairs_scanner[1])))

    p1 = np.asarray(pairs_scanner[1])  # (M,3) scanner-1 frame
    p2 = np.asarray(pairs_scanner[2])
    print(f"{len(p1)} hungarian centroid pairs from {len(scene_pair_index)} scenes")

    def residuals(s1: float, s2: float) -> np.ndarray:
        w1 = full.scanner1.transform(p1, s1)
        w2 = full.scanner2.transform(p2, s2)
        return np.linalg.norm(w1 - w2, axis=1) - PROFILE_LENGTH_MM

    def objective(x) -> float:
        return float(np.median(np.abs(residuals(x[0], x[1]))))

    # coarse grid then Nelder-Mead refine
    g1 = np.linspace(full.scanner1.step_positions[0],
                     full.scanner1.step_positions[-1], 43)
    g2 = np.linspace(full.scanner2.step_positions[0],
                     full.scanner2.step_positions[-1], 43)
    grid = [(objective((a, b)), a, b) for a in g1 for b in g2]
    grid.sort()
    best0 = grid[0]
    print(f"coarse best: s1={best0[1]:.0f} s2={best0[2]:.0f} "
          f"median|resid|={best0[0]:.1f} mm")

    opt = minimize(objective, x0=[best0[1], best0[2]], method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 0.01})
    # transform() clamps s to the calibrated range; report the effective values
    s1 = float(np.clip(opt.x[0], full.scanner1.step_positions[0],
                       full.scanner1.step_positions[-1]))
    s2 = float(np.clip(opt.x[1], full.scanner2.step_positions[0],
                       full.scanner2.step_positions[-1]))
    resid = residuals(s1, s2)
    print(f"refined: s1={s1:.1f} s2={s2:.1f} median resid {np.median(resid):+.2f} mm "
          f"IQR [{np.percentile(resid, 25):+.2f}, {np.percentile(resid, 75):+.2f}]")

    # greedy with length check, recalibrated centroids
    tp = fp = fn = 0
    per_scene = []
    for sid, d1, d2 in ((s, *dets_by_scene[s]) for s, _, _ in scene_pair_index):
        for side, dets, s_fit in ((1, d1, s1), (2, d2, s2)):
            sc = offline.for_side(side)
            spline = full.for_side(side)
            for det in dets:
                c_world = np.asarray(det.centroid_world_mm)
                p_scan = sc.R.T @ (c_world - sc.t)
                det.centroid_world_mm = tuple(
                    float(v) for v in spline.transform(p_scan[None, :], s_fit)[0]
                )
        res = pair_greedy_index(d1, d2, check_length=True)
        n_pred = len(res.pairs)
        gt_pairs = int(gt_rows[sid]["gt_pairs"])
        tp += min(n_pred, gt_pairs)
        fp += max(n_pred - gt_pairs, 0)
        fn += max(gt_pairs - n_pred, 0)
        per_scene.append((sid, n_pred, gt_pairs))
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    print(f"greedy with length check, recalibrated: "
          f"P {p:.3f} R {r:.3f} F1 {f1:.3f}")

    OUT_JSON.write_text(json.dumps({
        "s1_mm": s1, "s2_mm": s2,
        "median_residual_mm": float(np.median(resid)),
        "iqr_mm": [float(np.percentile(resid, 25)), float(np.percentile(resid, 75))],
        "n_pairs": int(len(p1)),
        "note": "effective per-dataset step positions fitted from the 6005 mm "
                "bar-length prior; degenerate along the valley (see report)",
    }, indent=1))

    valley = [f"| {a:.0f} | {b:.0f} | {v:.2f} |" for v, a, b in grid[:8]]
    lines = [
        "# Step-position fit (full calibration, bar-length prior)",
        "",
        f"Input: {len(p1)} hungarian centroid pairs, "
        f"{len(scene_pair_index)} scenes.",
        "",
        f"**Fitted: s1 = {s1:.1f} mm, s2 = {s2:.1f} mm** -> length residual "
        f"median {np.median(resid):+.2f} mm, "
        f"IQR [{np.percentile(resid, 25):+.2f}, {np.percentile(resid, 75):+.2f}] mm.",
        "",
        f"Length check (±10 mm around 6005): "
        f"{int((np.abs(resid) <= 10).sum())}/{len(resid)} pairs pass.",
        "",
        f"Greedy pairing with the length check, on the recalibrated "
        f"centroids: P {p:.3f} R {r:.3f} F1 {f1:.3f}.",
        "",
        "## Degeneracy valley (top coarse-grid cells)",
        "",
        "The bar length constrains the X separation, not s1/s2 individually;",
        "the cells below are near-equivalent under the fitted objective.",
        "",
        "| s1 | s2 | median\\|resid\\| mm |", "|---|---|---|",
        *valley,
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
