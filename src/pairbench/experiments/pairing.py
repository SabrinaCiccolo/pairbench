"""Scores pairing strategies against the count ground truth (`gt_pairs`).

Input: results/pairbench/b1_detections_<detector>.json. Reports pair-count
precision/recall/F1 overall, by category and by profile family, with
bundle-bootstrap CIs, Y-order inversions and overlay PNGs. TP per scene =
min(predicted, gt_pairs) is a count proxy; correspondence correctness is
scored by `experiments.correspondence`.

Usage: python -m pairbench.experiments.pairing [--overlays N] [--detector pool]
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict

import cv2
import numpy as np

from ..detect.ccorr import Detection, dets_from_json, sort_left_to_right
from ..io.dxf_template import render_template
from ..gt import FLAGGED_SCENES, load_gt
from ..significance import sum_by_bundle
from ..io.loader import DATA_ROOT, FAMILY_DXF, dxf_stem_for_scene, load_scene
from ..metrics import (
    bootstrap_f1_ci,
    count_inversions,
    precision_recall_f1 as prf,
)
from ..pairing import STRATEGIES
from ..pairing.common import (LENGTH_TOLERANCE_MM, PROFILE_LENGTH_MM,
                              SYMMETRY_PERIOD_DEG)
from ..pairing.greedy_index import pair_greedy_index
from ..io.projection import project_yz
from ..symmetry import period_for_dxf, period_for_scene
from ..timing import report_lines, reset, snapshot, stage
from .common import GT, RESULTS

ARMS = ["greedy_index", "greedy_index_debiased", "greedy_index_nolen",
        "greedy_index_order", "hungarian", "hungarian_aligned",
        "monotone_aligned", "hybrid_aligned"]

# Per-pair assignment margin: computed only by the cost-matrix aligned arm.
MARGIN_ARM = "hungarian_aligned"
LOW_MARGIN_THRESHOLD = 0.5   # under half a tolerance unit safer than the
                             # next-best alternative


def run_arm(arm: str, d1: list[Detection], d2: list[Detection],
           period_deg: float = SYMMETRY_PERIOD_DEG,
           length_bias_mm: float = 0.0):
    if arm == "greedy_index_nolen":
        return pair_greedy_index(d1, d2, period_deg=period_deg,
                                 check_length=False)
    if arm == "greedy_index_debiased":
        return pair_greedy_index(d1, d2, period_deg=period_deg,
                                 length_bias_mm=length_bias_mm)
    return STRATEGIES[arm](d1, d2, period_deg=period_deg)


def draw_pairing_overlay(scene_id: str, d1, d2, results_by_arm, out_path):
    """Draws side-1 above side-2, one panel per arm, lines joining paired centroids."""
    template = render_template(dxf_stem_for_scene(scene_id))
    clouds = load_scene(DATA_ROOT / scene_id)
    padding = template.default_padding
    projs = {1: project_yz(clouds.side1_world, padding_px=padding),
             2: project_yz(clouds.side2_world, padding_px=padding)}

    def centroid_px(proj, det):
        px, py = proj.yz_to_pixel(*det.centroid_yz_mm)
        return int(round(px)), int(round(py))

    panels = []
    for arm in ARMS:
        res = results_by_arm[arm]
        vis1 = cv2.cvtColor(projs[1].image, cv2.COLOR_GRAY2BGR)
        vis2 = cv2.cvtColor(projs[2].image, cv2.COLOR_GRAY2BGR)
        w = max(vis1.shape[1], vis2.shape[1])
        vis1 = cv2.copyMakeBorder(vis1, 0, 0, 0, w - vis1.shape[1],
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
        vis2 = cv2.copyMakeBorder(vis2, 0, 0, 0, w - vis2.shape[1],
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
        stack = np.vstack([vis1, vis2])
        h1 = vis1.shape[0]
        rng = np.random.default_rng(0)
        for i, j in res.pairs:
            color = tuple(int(c) for c in rng.integers(80, 255, 3))
            p1 = centroid_px(projs[1], d1[i])
            p2 = centroid_px(projs[2], d2[j])
            cv2.circle(stack, p1, 6, color, 2)
            cv2.circle(stack, (p2[0], p2[1] + h1), 6, color, 2)
            cv2.line(stack, p1, (p2[0], p2[1] + h1), color, 2)
        cv2.putText(stack, arm, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        panels.append(stack)
    hmax = max(p.shape[0] for p in panels)
    padded = [cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 8,
                                 cv2.BORDER_CONSTANT, value=(40, 40, 40))
              for p in panels]
    cv2.imwrite(str(out_path), np.hstack(padded))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", type=int, default=4)
    ap.add_argument("--detector", default="pool", choices=("pool", "ccorr"),
                    help="which b1_detections_<detector>.json to pair "
                         "(matches experiments.detect's --detector)")
    args = ap.parse_args()

    IN_JSON = RESULTS / f"b1_detections_{args.detector}.json"
    OUT_CSV = RESULTS / "a1_pairing.csv"
    OUT_MD = RESULTS / "a1_report.md"
    OUT_PNG = RESULTS / "a1_f1_by_category.png"
    OVERLAY_DIR = RESULTS / "a1_overlays"

    gt_rows = load_gt(GT)
    with open(IN_JSON) as f:
        data = json.load(f)

    # global length bias for greedy_index_debiased: median raw residual of
    # every index-matched candidate over the whole dataset
    residuals_all = []
    for sid in sorted(data["scenes"]):
        scene = data["scenes"][sid]
        d1 = sort_left_to_right(dets_from_json(scene["side1"]["confirmed"]))
        d2 = sort_left_to_right(dets_from_json(scene["side2"]["confirmed"]))
        res = pair_greedy_index(d1, d2, check_length=False)
        residuals_all += res.diagnostics.get("length_residuals_mm", [])
    length_bias = float(np.median(residuals_all)) if residuals_all else 0.0

    rows_out = []
    agg = {a: defaultdict(lambda: [0, 0, 0]) for a in ARMS}  # cat -> [tp,fp,fn]
    agg_fam = {a: defaultdict(lambda: [0, 0, 0]) for a in ARMS}  # family strata
    inversions = {a: 0 for a in ARMS}  # non-crossed scenes only
    per_scene_counts = {a: [] for a in ARMS}  # [(tp,fp,fn), ...] -- bootstrap_f1_ci
    count_scenes = []  # scene id of each row of per_scene_counts
    hungarian_margins = []  # (scene_id, i, j, margin) -- see MARGIN_ARM below
    overlay_budget = args.overlays
    reset()
    if OVERLAY_DIR.exists():
        shutil.rmtree(OVERLAY_DIR)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    for sid in sorted(data["scenes"]):
        scene = data["scenes"][sid]
        g = gt_rows[sid]
        gt_pairs = int(g["gt_pairs"])
        # Left-to-right order everywhere (pairbench.detect.ccorr.sort_left_to_right).
        d1 = sort_left_to_right(dets_from_json(scene["side1"]["confirmed"]))
        d2 = sort_left_to_right(dets_from_json(scene["side2"]["confirmed"]))
        period = period_for_scene(sid)
        in_metrics = sid not in FLAGGED_SCENES
        if in_metrics:
            count_scenes.append(sid)
        results_by_arm = {}
        row = {"scene_id": sid, "family": sid.split("/")[0],
               "category": g["category"], "gt_pairs": gt_pairs,
               "n_det_1": len(d1), "n_det_2": len(d2),
               "symmetry_period_deg": period,
               "in_metrics": int(in_metrics)}
        for arm in ARMS:
            with stage(f"pairing/{arm}"):
                res = run_arm(arm, d1, d2,
                              period_deg=period, length_bias_mm=length_bias)
            results_by_arm[arm] = res
            n_pred = len(res.pairs)
            row[f"pairs_{arm}"] = n_pred
            if arm == MARGIN_ARM and in_metrics:
                margins = res.diagnostics.get("pair_margins", [])
                for (i, j), m in zip(res.pairs, margins):
                    hungarian_margins.append((sid, i, j, m))
            if in_metrics:
                tp = min(n_pred, gt_pairs)
                fp = max(n_pred - gt_pairs, 0)
                fn = max(gt_pairs - n_pred, 0)
                per_scene_counts[arm].append((tp, fp, fn))
                for bucket in (agg[arm][g["category"]],
                               agg_fam[arm][row["family"]]):
                    bucket[0] += tp
                    bucket[1] += fp
                    bucket[2] += fn
                if g["category"] != "crossed":
                    inversions[arm] += count_inversions(res.pairs, d1, d2)
        rows_out.append(row)
        if overlay_budget > 0 and g["category"] in ("crossed", "overlap", "tilted"):
            draw_pairing_overlay(sid, d1, d2, results_by_arm,
                                 OVERLAY_DIR / (sid.replace("/", "__") + ".png"))
            overlay_budget -= 1

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    categories = sorted({r["category"] for r in rows_out
                         if r["in_metrics"]})
    by_dxf = defaultdict(list)
    for fam, dxf in sorted(FAMILY_DXF.items()):
        by_dxf[dxf].append(fam)
    symmetry = "; ".join(f"{period_for_dxf(dxf):g}° for {', '.join(fams)}"
                         for dxf, fams in sorted(by_dxf.items()))
    lines = ["# Pairing strategies vs gt (B1 detections)",
             "",
             f"Detections: `{IN_JSON.name}`. {len(count_scenes)} scenes scored "
             "(`gt.FLAGGED_SCENES` excluded).",
             "",
             "gt_pairs is a count: TP = min(pred, gt) does not verify which "
             "bars are paired (see overlay PNGs and correspondence_report.md).",
             "",
             "Arms:",
             "",
             "- `greedy_index`: index order per level, twist, tilt and length "
             "checks",
             f"- `greedy_index_debiased`: as above, length check centred on a "
             f"global bias of {length_bias:.1f} mm (median residual over all "
             "index-matched candidates)",
             "- `greedy_index_nolen`: as above, no length check",
             "- `greedy_index_order`: index order only, no checks",
             "- `hungarian`: cost-matrix assignment",
             "- `hungarian_aligned`: `hungarian` after RANSAC YZ translation "
             "compensation",
             "- `monotone_aligned`: same cost and translation, order-constrained "
             "per level",
             "- `hybrid_aligned`: `monotone_aligned` per stacking level plus an "
             "evidence-gated crossing hypothesis",
             "",
             f"Angle costs fold at each profile's symmetry period "
             f"(`pairbench.symmetry`): {symmetry}.",
             ""]
    lines += ["## Overall", "",
              "`inv` = same-level Y-order inversions on non-crossed scenes, "
              "a swap proxy that count-F1 does not see. `F1 95% CI` resamples "
              "bundles with replacement (2000 reps, "
              "`pairbench.metrics.bootstrap_f1_ci`); repeat captures of one "
              "bundle (`data/repeat_captures.csv`) are resampled together.",
              "",
              "| strategy | P | R | F1 | F1 95% CI | inv |",
              "|---|---|---|---|---|---|"]
    print("\noverall (pair counts):")
    arm_ci = {}
    overall_json = {}
    for arm in ARMS:
        tp = sum(v[0] for v in agg[arm].values())
        fp = sum(v[1] for v in agg[arm].values())
        fn = sum(v[2] for v in agg[arm].values())
        p, r, f1 = prf(tp, fp, fn)
        _, by_bundle = sum_by_bundle(count_scenes, per_scene_counts[arm])
        ci_lo, ci_hi = bootstrap_f1_ci([tuple(r) for r in by_bundle])
        arm_ci[arm] = (ci_lo, ci_hi)
        overall_json[arm] = {"precision": p, "recall": r, "f1": f1,
                             "f1_ci95": [ci_lo, ci_hi],
                             "inversions": int(inversions[arm]),
                             "tp": int(tp), "fp": int(fp), "fn": int(fn)}
        lines.append(f"| {arm} | {p:.3f} | {r:.3f} | {f1:.3f} "
                     f"| [{ci_lo:.3f}, {ci_hi:.3f}] | {inversions[arm]} |")
        print(f"  {arm:22s} P {p:.3f} R {r:.3f} F1 {f1:.3f} "
              f"CI [{ci_lo:.3f}, {ci_hi:.3f}] inv {inversions[arm]}")

    def overlaps(a, b):
        lo1, hi1 = arm_ci[a]
        lo2, hi2 = arm_ci[b]
        return lo1 <= hi2 and lo2 <= hi1

    ref_arm = "hybrid_aligned"
    overlapping = [a for a in ARMS if a != ref_arm and overlaps(a, ref_arm)]
    if overlapping:
        lo, hi = arm_ci[ref_arm]
        lines += ["", f"`{ref_arm}`'s interval ([{lo:.3f}, {hi:.3f}]) "
                  "overlaps " + ", ".join(f"`{a}`" for a in overlapping)
                  + ": count-F1 does not separate them at this sample size "
                  f"({len(count_scenes)} scenes). See "
                  "`results/pairbench/correspondence_report.md` for "
                  "correspondence-level scores."]

    lines += ["", "## By category", "",
              "| strategy | " + " | ".join(categories) + " |",
              "|---|" + "---|" * len(categories)]
    fig_data = {}
    for arm in ARMS:
        cells = []
        f1s = []
        for cat in categories:
            tp, fp, fn = agg[arm][cat]
            _, _, f1 = prf(tp, fp, fn)
            n_gt = tp + fn
            cells.append(f"{f1:.3f} (n={n_gt})")
            f1s.append(f1)
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        fig_data[arm] = f1s

    families = sorted({r["family"] for r in rows_out if r["in_metrics"]})
    lines += ["", "## By profile family", "",
              "`complex-profile` and `heavy-profile` carry few pairs each. "
              "`l-profile-reshoot` re-shoots `l-profile`'s scenarios with the "
              "same profile, so the two are not independent samples.",
              "",
              "| strategy | " + " | ".join(families) + " |",
              "|---|" + "---|" * len(families)]
    for arm in ARMS:
        cells = []
        for fam in families:
            tp, fp, fn = agg_fam[arm][fam]
            _, _, f1 = prf(tp, fp, fn)
            cells.append(f"{f1:.3f} (n={tp + fn})")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    if residuals_all:
        resid = np.array(residuals_all)
        lines += ["", "## Length-check residuals (greedy, |c1-c2| - "
                  f"{PROFILE_LENGTH_MM:g} mm)", "",
                  f"n={len(resid)}, median {np.median(resid):.1f} mm, "
                  f"IQR [{np.percentile(resid, 25):.1f}, "
                  f"{np.percentile(resid, 75):.1f}] mm; the gate is "
                  f"±{LENGTH_TOLERANCE_MM:g} mm."]

    if hungarian_margins:
        m = np.array([v[3] for v in hungarian_margins])
        low = sorted((v for v in hungarian_margins if v[3] < LOW_MARGIN_THRESHOLD),
                    key=lambda v: v[3])
        lines += ["", f"## Assignment confidence ({MARGIN_ARM})", "",
                  "Per accepted pair, how much cheaper the assigned partner "
                  "was than the next-best alternative (own-index cost minus "
                  "runner-up cost). Not computed for the order-constrained "
                  "arms.",
                  "",
                  f"n={len(m)}, median {np.median(m):.2f}, IQR "
                  f"[{np.percentile(m, 25):.2f}, {np.percentile(m, 75):.2f}], "
                  f"{len(low)} pair(s) below the {LOW_MARGIN_THRESHOLD} "
                  "low-margin threshold (close calls, not necessarily "
                  "wrong pairs)."]
        if low:
            lines += ["", "| scene | side1 idx | side2 idx | margin |",
                      "|---|---|---|---|"]
            for sid, i, j, margin in low[:20]:
                lines.append(f"| {sid} | {i} | {j} | {margin:.3f} |")
            if len(low) > 20:
                lines.append(f"| ... | | | ({len(low) - 20} more) |")

    lines += ["", ""] + report_lines(snapshot(), title="Timing (per arm, per scene)")
    lines += ["", "Real-data scale only; see "
              "`results/pairbench/scaling_report.md` for the synthetic "
              "bar-count sweep."]

    OUT_MD.write_text("\n".join(lines) + "\n")
    # count-metric table in machine-readable form
    (RESULTS / "a1_overall.json").write_text(json.dumps(
        {"n_scenes": len(per_scene_counts[ARMS[0]]),
         "arms": overall_json}, indent=1))

    # PNG figure: grouped bars, F1 per category per strategy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arm_colors = plt.get_cmap("tab10").colors
    arm_hatches = ["", "//"]
    x = np.arange(len(categories))
    width = 0.8 / len(ARMS)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, arm in enumerate(ARMS):
        ax.bar(x + (k - (len(ARMS) - 1) / 2) * width, fig_data[arm], width,
               label=arm, color=arm_colors[k % len(arm_colors)],
               hatch=arm_hatches[k % len(arm_hatches)], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("pair-count F1")
    ax.set_ylim(0, 1.22)   # headroom so the legend row sits above the bars, not on them
    ax.set_title("Pairing F1 by category", fontsize=10, pad=38)
    ax.legend(fontsize=6.5, ncol=4, loc="upper center",
             bbox_to_anchor=(0.5, 1.17), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT_CSV.name}, {OUT_MD.name}, {OUT_PNG.name}, overlays")


if __name__ == "__main__":
    main()
