"""Cross-view guided re-detection (rescue) runner.

Applies `pairbench.detect.rescue.rescue_misses` to every scene of
results/pairbench/b1_detections_<detector>.json and reports detection counts
and `hungarian_aligned` pairing counts before vs after rescue (count proxy, as
in experiments.detect / experiments.pairing).

Writes results/pairbench/b2_rescued_<detector>.json, b2_counts_<detector>.csv,
b2_rescue_<detector>.md and b2_overlays_<detector>/ (confirmed green, lost
thin red, rescued thick orange).

Usage: python -m pairbench.experiments.rescue [--detector pool] [--overlays N]
       [--workers N]
"""

import argparse
import csv
import dataclasses
import json
import shutil
from collections import defaultdict
from multiprocessing import Pool

import cv2

from ..detect.ccorr import dets_from_json, sort_left_to_right
from ..detect.rescue import (RESCUE_MIN_POINTS, RESCUE_MIN_Z_EXTENT_MM,
                             RESCUE_RADIUS_MM, rescue_misses)
from ..io.dxf_template import render_template
from ..gt import FLAGGED_SCENES, load_gt
from ..io.loader import DATA_ROOT, dxf_stem_for_scene, load_scene
from ..metrics import count_inversions, precision_recall_f1 as prf
from ..pairing.hungarian_aligned import pair_hungarian_aligned
from ..io.projection import project_yz
from ..symmetry import period_for_scene
from ..timing import merge, report_lines, reset, snapshot, stage
from ..viz import draw_obbs, stack_panels
from .common import GT, RESULTS

_scenes_json = None  # worker global


def process_scene(sid: str):
    scene = _scenes_json[sid]
    template = render_template(dxf_stem_for_scene(sid))
    clouds = load_scene(DATA_ROOT / sid)
    conf1 = dets_from_json(scene["side1"]["confirmed"])
    conf2 = dets_from_json(scene["side2"]["confirmed"])
    reset()  # isolate this scene's timing (Pool workers are reused)
    with stage("rescue_total"):
        r1, r2, diag = rescue_misses(clouds.side1_world, clouds.side2_world,
                                     template, conf1, conf2,
                                     period_deg=period_for_scene(sid))
    return sid, r1, r2, diag, snapshot()


def _init_worker(scenes_json):
    global _scenes_json
    _scenes_json = scenes_json
    cv2.setNumThreads(1)


def draw_overlay(scene_id, scene, rescued, out_path) -> None:
    template = render_template(dxf_stem_for_scene(scene_id))
    clouds = load_scene(DATA_ROOT / scene_id)
    padding = template.default_padding
    panels = []
    for side, world in ((1, clouds.side1_world), (2, clouds.side2_world)):
        proj = project_yz(world, padding_px=padding)
        vis = cv2.cvtColor(proj.image, cv2.COLOR_GRAY2BGR)
        draw_obbs(vis, scene[f"side{side}"]["lost"], (0, 0, 255), 1)
        draw_obbs(vis, scene[f"side{side}"]["confirmed"], (0, 255, 0), 2)
        draw_obbs(vis, rescued[side], (0, 165, 255), 3)
        cv2.putText(vis, f"side {side}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        panels.append(vis)
    cv2.imwrite(str(out_path), stack_panels(panels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--detector", default="pool",
                    help="which experiments.detect output to rescue against: "
                         "reads b1_detections_NAME.json, writes "
                         "b2_rescued_NAME.json + b2_counts_NAME.csv + "
                         "b2_rescue_NAME.md + b2_overlays_NAME/.")
    args = ap.parse_args()

    sfx = args.detector
    IN_JSON = RESULTS / f"b1_detections_{sfx}.json"
    OUT_JSON = RESULTS / f"b2_rescued_{sfx}.json"
    OUT_CSV = RESULTS / f"b2_counts_{sfx}.csv"
    OUT_MD = RESULTS / f"b2_rescue_{sfx}.md"
    OVERLAY_DIR = RESULTS / f"b2_overlays_{sfx}"

    gt_rows = load_gt(GT)
    with open(IN_JSON) as f:
        data = json.load(f)

    scene_ids = sorted(data["scenes"])
    with Pool(args.workers, initializer=_init_worker,
              initargs=(data["scenes"],)) as pool:
        timing_snapshots = []
        rescue_results = {}
        for sid, r1, r2, diag, timing in pool.imap(process_scene, scene_ids):
            rescue_results[sid] = (r1, r2, diag)
            timing_snapshots.append(timing)

    det_before = [0, 0, 0]   # tp, fp, fn (count proxy, both sides)
    det_after = [0, 0, 0]
    det_before_fam = defaultdict(lambda: [0, 0, 0])
    det_after_fam = defaultdict(lambda: [0, 0, 0])
    pair_before = [0, 0, 0]
    pair_after = [0, 0, 0]
    inv_before = inv_after = 0
    inv_scenes_after = []
    n_rescued_total = 0
    rows_out = []
    out_scenes = {}
    overlay_budget = args.overlays
    if OVERLAY_DIR.exists():
        shutil.rmtree(OVERLAY_DIR)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    for sid in scene_ids:
        scene = data["scenes"][sid]
        g = gt_rows[sid]
        c1 = sort_left_to_right(dets_from_json(scene["side1"]["confirmed"]))
        c2 = sort_left_to_right(dets_from_json(scene["side2"]["confirmed"]))
        r1, r2, diag = rescue_results[sid]
        out_scenes[sid] = {
            "rescued_1": [dataclasses.asdict(d) for d in r1],
            "rescued_2": [dataclasses.asdict(d) for d in r2],
            "diagnostics": diag,
        }
        row = {"scene_id": sid, "family": sid.split("/")[0],
               "category": g["category"],
               "gt_A": g["gt_A"], "gt_B": g["gt_B"],
               "n_conf_1": len(c1), "n_conf_2": len(c2),
               "n_rescued_1": len(r1), "n_rescued_2": len(r2),
               "in_metrics": int(sid not in FLAGGED_SCENES)}
        rows_out.append(row)
        if (r1 or r2) and overlay_budget > 0:
            draw_overlay(sid, scene, {1: r1, 2: r2},
                         OVERLAY_DIR / (sid.replace("/", "__") + ".png"))
            overlay_budget -= 1
        if sid in FLAGGED_SCENES:
            continue
        n_rescued_total += len(r1) + len(r2)

        fam = row["family"]
        for gt_n, before_n, after_n in (
            (int(g["gt_A"]), len(c1), len(c1) + len(r1)),
            (int(g["gt_B"]), len(c2), len(c2) + len(r2)),
        ):
            for acc, n in ((det_before, before_n), (det_after, after_n)):
                acc[0] += min(gt_n, n)
                acc[1] += max(n - gt_n, 0)
                acc[2] += max(gt_n - n, 0)
            for acc_fam, n in ((det_before_fam[fam], before_n),
                               (det_after_fam[fam], after_n)):
                acc_fam[0] += min(gt_n, n)
                acc_fam[1] += max(n - gt_n, 0)
                acc_fam[2] += max(gt_n - n, 0)

        gt_pairs = int(g["gt_pairs"])
        period = period_for_scene(sid)
        for which, acc, d1, d2 in (
            ("before", pair_before, c1, c2),
            ("after", pair_after,
             sort_left_to_right(c1 + r1), sort_left_to_right(c2 + r2)),
        ):
            res = pair_hungarian_aligned(d1, d2, period_deg=period)
            n_pred = len(res.pairs)
            acc[0] += min(n_pred, gt_pairs)
            acc[1] += max(n_pred - gt_pairs, 0)
            acc[2] += max(gt_pairs - n_pred, 0)
            if g["category"] != "crossed":
                inv = count_inversions(res.pairs, d1, d2)
                if which == "before":
                    inv_before += inv
                else:
                    inv_after += inv
                    if inv:
                        inv_scenes_after.append((sid, inv))

    with open(OUT_JSON, "w") as f:
        json.dump({"scenes": out_scenes}, f, indent=1)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    lines = ["# Cross-view guided re-detection (rescue)", "",
             f"Detections: {IN_JSON.name}. Count "
             "proxy metrics as in the detection/pairing reports.",
             "",
             "An unmatched confirmed detection on one side plus "
             "hungarian_aligned's fitted translation predicts the partner "
             "position on the other side; a local template match on the "
             "confirmed-erased image (partner angle +/- one coarse step) is "
             "accepted only with a face-like 3D footprint (z-extent >= "
             f"{RESCUE_MIN_Z_EXTENT_MM:g} mm, >= {RESCUE_MIN_POINTS} points, "
             f"centroid within {RESCUE_RADIUS_MM:g} mm of the prediction). "
             "Faces with no partner in GT cannot be rescued.",
             "",
             "## Detection counts (both sides)", "",
             "| arm | P | R | F1 |", "|---|---|---|---|"]
    for name, acc in (("B1 confirmed", det_before),
                      ("B1 + rescued", det_after)):
        p, r, f1 = prf(*acc)
        lines.append(f"| {name} | {p:.3f} | {r:.3f} | {f1:.3f} |")
        print(f"{name:14s} P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    families = sorted(det_after_fam)
    lines += ["", "### By profile family (B1 + rescued)", "",
              "l-profile and l-profile-reshoot are the same profile with "
              "overlapping scenarios, not independent samples. "
              "complex-profile and heavy-profile have few scenes.",
              "",
              "| family | P | R | F1 | faces |", "|---|---|---|---|---|"]
    for fam in families:
        tp, fp, fn = det_after_fam[fam]
        p, r, f1 = prf(tp, fp, fn)
        lines.append(f"| {fam} | {p:.3f} | {r:.3f} | {f1:.3f} | {tp + fn} |")

    lines += ["", "## Pairing (hungarian_aligned)", "",
              "| arm | P | R | F1 | inv |", "|---|---|---|---|---|"]
    for name, acc, inv in (("before rescue", pair_before, inv_before),
                           ("after rescue", pair_after, inv_after)):
        p, r, f1 = prf(*acc)
        lines.append(f"| {name} | {p:.3f} | {r:.3f} | {f1:.3f} | {inv} |")
        print(f"pairing {name:14s} P {p:.3f} R {r:.3f} F1 {f1:.3f} inv {inv}")
    if inv_scenes_after:
        lines += ["", "Post-rescue inversions by scene (swap proxy): "
                  + ", ".join(f"{sid} ({n})" for sid, n in inv_scenes_after)
                  + "."]
    n_attempted = sum(d[2]["attempted"] for d in rescue_results.values())
    lines += ["", f"Rescue attempts (unmatched partners): {n_attempted}; "
              f"accepted: {n_rescued_total}. Overlays in {OVERLAY_DIR.name}/ "
              "(rescued boxes thick orange)."]
    timing = merge(*timing_snapshots)
    lines += ["", ""] + report_lines(timing, title=f"Timing ({sfx})")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"rescued {n_rescued_total}/{n_attempted} attempts; wrote "
          f"{OUT_JSON.name}, {OUT_CSV.name}, {OUT_MD.name}, overlays")


if __name__ == "__main__":
    main()
