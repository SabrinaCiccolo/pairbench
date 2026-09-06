"""Runs a face detector over the gt-annotated scenes.

For every annotated scene, runs the detector on both sides and writes
results/pairbench/b1_detections_<detector>.json, b1_counts_<detector>.csv,
b1_report_<detector>.md and overlay PNGs for the first N scenes to
b1_overlays_<detector>/. Scores count-based precision/recall/F1 vs gt (TP per
side = min(pred, gt)).

Usage: python -m pairbench.experiments.detect [--detector pool|ccorr]
       [--limit N] [--overlays N] [--workers N] [--include-flagged]
"""

import argparse
import csv
import dataclasses
import json
import shutil
from collections import defaultdict
from multiprocessing import Pool

import cv2

from ..detect.ccorr import detect_faces
from ..detect.pool import detect_faces_pool
from ..io.dxf_template import render_template
from ..gt import FLAGGED_SCENES, load_gt
from ..io.loader import DATA_ROOT, dxf_stem_for_scene, load_scene
from ..metrics import precision_recall_f1 as prf
from ..timing import merge, report_lines, reset, snapshot, stage
from ..viz import draw_obbs, stack_panels
from .common import GT, RESULTS

DETECTORS = {"ccorr": detect_faces, "pool": detect_faces_pool}


def _init_worker() -> None:
    """Pins OpenCV to one thread per worker process."""
    cv2.setNumThreads(1)


def process_scene(args_tuple):
    scene_id, dxf_stem, detector_name = args_tuple
    detect_fn = DETECTORS[detector_name]
    template = render_template(dxf_stem)
    clouds = load_scene(DATA_ROOT / scene_id)
    out = {"scene_id": scene_id}
    projections = {}
    reset()  # isolate this scene's timing (Pool workers are reused)
    for side, world in ((1, clouds.side1_world), (2, clouds.side2_world)):
        with stage("detect_total"):
            confirmed, lost, proj = detect_fn(world, template, side)
        out[f"side{side}"] = {
            "confirmed": [dataclasses.asdict(d) for d in confirmed],
            "lost": [dataclasses.asdict(d) for d in lost],
        }
        projections[side] = proj
    return out, projections, snapshot()


def draw_overlay(scene_result, projections, out_path) -> None:
    panels = []
    for side in (1, 2):
        vis = cv2.cvtColor(projections[side].image, cv2.COLOR_GRAY2BGR)
        draw_obbs(vis, scene_result[f"side{side}"]["confirmed"], (0, 255, 0), 2)
        draw_obbs(vis, scene_result[f"side{side}"]["lost"], (0, 0, 255), 1)
        cv2.putText(vis, f"side {side}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        panels.append(vis)
    cv2.imwrite(str(out_path), stack_panels(panels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N scenes")
    ap.add_argument("--overlays", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--detector", choices=sorted(DETECTORS), default="pool",
                    help="pool = candidate pool + global selection (default); "
                         "ccorr = template-matching baseline")
    args = ap.parse_args()

    suffix = f"_{args.detector}"
    OUT_JSON = RESULTS / f"b1_detections{suffix}.json"
    OUT_CSV = RESULTS / f"b1_counts{suffix}.csv"
    OUT_MD = RESULTS / f"b1_report{suffix}.md"
    OVERLAY_DIR = RESULTS / f"b1_overlays{suffix}"

    gt_rows = load_gt(GT)
    scene_ids = sorted(gt_rows)
    if args.limit:
        scene_ids = scene_ids[: args.limit]

    if OVERLAY_DIR.exists():
        shutil.rmtree(OVERLAY_DIR)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(sid, dxf_stem_for_scene(sid), args.detector) for sid in scene_ids]
    results = {}
    timing_snapshots = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for k, (scene_result, projections, timing) in enumerate(
            pool.imap(process_scene, jobs)
        ):
            sid = scene_result["scene_id"]
            results[sid] = scene_result
            timing_snapshots.append(timing)
            if k < args.overlays:
                draw_overlay(scene_result, projections,
                             OVERLAY_DIR / (sid.replace("/", "__") + ".png"))
            n1 = len(scene_result["side1"]["confirmed"])
            n2 = len(scene_result["side2"]["confirmed"])
            g = gt_rows[sid]
            print(f"[{k + 1}/{len(jobs)}] {sid}: conf {n1}/{n2} "
                  f"(gt {g['gt_A']}/{g['gt_B']}, {g['category']})")

    with open(OUT_JSON, "w") as f:
        json.dump({"scenes": results}, f, indent=1)

    tp = fp = fn = 0
    tp_l = fp_l = fn_l = 0
    agg_fam = defaultdict(lambda: [0, 0, 0])
    rows_out = []
    for sid in scene_ids:
        r = results[sid]
        g = gt_rows[sid]
        skip = (sid in FLAGGED_SCENES) and not args.include_flagged
        row = {
            "scene_id": sid, "family": sid.split("/")[0],
            "category": g["category"],
            "confidence": g["confidence"],
            "gt_A": g["gt_A"], "gt_B": g["gt_B"],
            "n_conf_1": len(r["side1"]["confirmed"]),
            "n_conf_2": len(r["side2"]["confirmed"]),
            "n_lost_1": len(r["side1"]["lost"]),
            "n_lost_2": len(r["side2"]["lost"]),
            "in_metrics": int(not skip),
        }
        rows_out.append(row)
        if skip:
            continue
        for gt_n, conf_n, lost_n in (
            (int(g["gt_A"]), row["n_conf_1"], row["n_lost_1"]),
            (int(g["gt_B"]), row["n_conf_2"], row["n_lost_2"]),
        ):
            tp += min(gt_n, conf_n)
            fp += max(conf_n - gt_n, 0)
            fn += max(gt_n - conf_n, 0)
            fam_acc = agg_fam[row["family"]]
            fam_acc[0] += min(gt_n, conf_n)
            fam_acc[1] += max(conf_n - gt_n, 0)
            fam_acc[2] += max(gt_n - conf_n, 0)
            both = conf_n + lost_n
            tp_l += min(gt_n, both)
            fp_l += max(both - gt_n, 0)
            fn_l += max(gt_n - both, 0)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    p_c, r_c, f1_c = prf(tp, fp, fn)
    print(f"\ncount metrics (confirmed only):     P {p_c:.3f} R {r_c:.3f} F1 {f1_c:.3f}")
    p_l, r_l, f1_l = prf(tp_l, fp_l, fn_l)
    print(f"count metrics (confirmed + lost):   P {p_l:.3f} R {r_l:.3f} F1 {f1_l:.3f}")

    n_flagged = sum(1 for sid in scene_ids if sid in FLAGGED_SCENES)
    if n_flagged and not args.include_flagged:
        scene_note = (f"{len(scene_ids) - n_flagged} scenes scored "
                      f"({len(scene_ids)} annotated, {n_flagged} excluded as a "
                      "byte-identical duplicate, `gt.FLAGGED_SCENES`)")
    else:
        scene_note = f"{len(scene_ids)} annotated scenes"
    lines = [f"# {args.detector} detector vs gt counts", "",
             f"Detections: `{OUT_JSON.name}`, "
             f"{scene_note}. Count proxy: TP per side = "
             "min(pred, gt); see the correspondence report for the "
             "bar-level check.",
             "",
             "| arm | P | R | F1 |", "|---|---|---|---|",
             f"| confirmed | {p_c:.3f} | {r_c:.3f} | {f1_c:.3f} |",
             f"| confirmed + lost band | {p_l:.3f} | {r_l:.3f} | {f1_l:.3f} |",
             ""]
    lines += ["## By profile family", "",
             "l-profile and l-profile-reshoot are the same profile with "
             "overlapping scenarios, not independent samples. "
             "complex-profile and heavy-profile have few scenes.",
             "",
             "| family | scenes | P | R | F1 | faces |", "|---|---|---|---|---|---|"]
    scenes_per_fam = defaultdict(int)
    for row in rows_out:
        if row["in_metrics"]:
            scenes_per_fam[row["family"]] += 1
    for fam in sorted(agg_fam):
        f_tp, f_fp, f_fn = agg_fam[fam]
        p, r, f1 = prf(f_tp, f_fp, f_fn)
        lines.append(f"| {fam} | {scenes_per_fam[fam]} | {p:.3f} | {r:.3f} "
                     f"| {f1:.3f} | {f_tp + f_fn} |")

    timing = merge(*timing_snapshots)
    lines += ["", ""] + report_lines(timing, title=f"Timing ({args.detector})")
    if timing:
        lines += ["", f"Run with `--workers {args.workers}`; per-stage means "
                  "are per-call wall time."]
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_MD.name}, "
          f"{args.overlays} overlays")


if __name__ == "__main__":
    main()
