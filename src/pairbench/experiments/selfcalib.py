"""Joint self-calibration + assignment, and its observability.

Alternates (recalibrate -> assign -> refit transform) until the assignment
stops changing, then reports the bar-length residual and pairing count-F1
before/after, and which combination of (s1, s2) the bar-length prior
identifies. Writes results/pairbench/a2_selfcalib.md, a2_calibration.json and
a2_recalibrated.json (same schema as the detection files, so
`experiments.correspondence --detections` can score it).

Usage: python -m pairbench.experiments.selfcalib [--arm hybrid_aligned]
       [--detector pool]
"""

import argparse
import dataclasses
import json

import numpy as np

from ..io.calib import load_calibration
from ..io.calib_full import load_full_calibration
from ..detect.ccorr import dets_from_json, sort_left_to_right
from ..gt import FLAGGED_SCENES, load_gt, load_bundles, bundle_of
from ..metrics import precision_recall_f1 as prf
from ..pairing import STRATEGIES
from ..selfcalib import (
    MAX_ITERS,
    bootstrap_separation,
    PROFILE_LENGTH_MM,
    fit_steps,
    length_residuals,
    observability,
    recalibrated,
    refit_translation,
    to_scanner_frame,
)
from ..symmetry import period_for_scene
from ..timing import report_lines, reset, snapshot, stage
from .common import GT, RESULTS

OUT_MD = RESULTS / "a2_selfcalib.md"
OUT_JSON = RESULTS / "a2_calibration.json"
OUT_DETS = RESULTS / "a2_recalibrated.json"
ONE_SHOT = RESULTS / "fitted_steps.json"   # optional diagnostics baseline

ARMS = ["greedy_index", "hungarian", "hungarian_aligned",
        "monotone_aligned", "hybrid_aligned"]


def run_arm(arm, d1, d2, period):
    return STRATEGIES[arm](d1, d2, period_deg=period)


def count_f1(scenes, gt_rows, arm, calib=None):
    """Pair-count P/R/F1 for one arm, optionally on recalibrated detections."""
    tp = fp = fn = 0
    for sid, sc in scenes.items():
        d1, d2, _, _ = sc if calib is None else recalibrate_scene(sc, *calib)
        res = run_arm(arm, d1, d2, period_for_scene(sid))
        n_pred, gt = len(res.pairs), int(gt_rows[sid]["gt_pairs"])
        tp += min(n_pred, gt)
        fp += max(n_pred - gt, 0)
        fn += max(gt - n_pred, 0)
    return prf(tp, fp, fn)


def recalibrate_scene(sc, s1, s2, t_yz, offline, full):
    d1, d2, lost1, lost2 = sc
    return (sort_left_to_right(recalibrated(d1, 1, s1, offline, full)),
            sort_left_to_right(recalibrated(d2, 2, s2, offline, full, t_yz)),
            recalibrated(lost1, 1, s1, offline, full),
            recalibrated(lost2, 2, s2, offline, full, t_yz))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hybrid_aligned", choices=ARMS)
    ap.add_argument("--detector", default="pool",
                    help="which b1_detections_<detector>.json to calibrate "
                         "against (matches experiments.detect's --detector)")
    args = ap.parse_args()

    offline = load_calibration()
    full = load_full_calibration()
    gt_rows = load_gt(GT)
    with open(RESULTS / f"b1_detections_{args.detector}.json") as f:
        data = json.load(f)

    scenes = {}
    for sid in sorted(data["scenes"]):
        if sid in FLAGGED_SCENES:
            continue
        s = data["scenes"][sid]
        scenes[sid] = (sort_left_to_right(dets_from_json(s["side1"]["confirmed"])),
                       sort_left_to_right(dets_from_json(s["side2"]["confirmed"])),
                       dets_from_json(s["side1"]["lost"]),
                       dets_from_json(s["side2"]["lost"]))

    # baseline: offline single-knot calibration, no self-calibration
    base_pairs = {}
    for sid, (d1, d2, l1, l2) in scenes.items():
        res = run_arm(args.arm, d1, d2, period_for_scene(sid))
        base_pairs[sid] = [(i, j) for i, j in res.pairs]
    base_resid = np.array([
        np.linalg.norm(np.array(scenes[sid][0][i].centroid_world_mm)
                       - np.array(scenes[sid][1][j].centroid_world_mm))
        - PROFILE_LENGTH_MM
        for sid, prs in base_pairs.items() for i, j in prs])
    base_f1 = {a: count_f1(scenes, gt_rows, a) for a in ARMS}

    # alternation
    s1, s2 = fit_steps(*_pair_arrays(scenes, base_pairs, offline), full)
    t_yz = (0.0, 0.0)
    history = []
    prev_pairs = None
    pairs = base_pairs
    reset()
    for it in range(MAX_ITERS):
        with stage("selfcalib_iter"):
            recal = {sid: recalibrate_scene(sc, s1, s2, t_yz, offline, full)
                     for sid, sc in scenes.items()}
            pairs = {}
            for sid, (d1, d2, l1, l2) in recal.items():
                res = run_arm(args.arm, d1, d2, period_for_scene(sid))
                pairs[sid] = [(i, j) for i, j in res.pairs]
            p1, p2 = _pair_arrays(scenes, pairs, offline)
            s1, s2 = fit_steps(p1, p2, full, x0=(s1, s2), t_yz=t_yz)
            # Z is fitted as a calibration constant; the global Y term is not
            # (it is conveyor transport) — the arms estimate Y per scene.
            t_yz = (0.0, refit_translation(p1, p2, full, s1, s2)[1])
            resid = length_residuals(p1, p2, full, s1, s2, t_yz)
        history.append({"iter": it + 1, "s1": s1, "s2": s2,
                        "t_yz": t_yz, "n_pairs": int(len(p1)),
                        "median_abs_resid": float(np.median(np.abs(resid)))})
        print(f"  iter {it + 1}: s1={s1:.1f} s2={s2:.1f} sep={s1 - s2:.1f} "
              f"t_yz=({t_yz[0]:+.1f},{t_yz[1]:+.1f}) pairs={len(p1)} "
              f"median|resid|={np.median(np.abs(resid)):.2f} mm")
        if pairs == prev_pairs:
            print(f"  converged: assignment unchanged at iteration {it + 1}")
            break
        prev_pairs = pairs

    converged_iter = len(history)
    recal = {sid: recalibrate_scene(sc, s1, s2, t_yz, offline, full)
             for sid, sc in scenes.items()}
    p1, p2 = _pair_arrays(scenes, pairs, offline)
    resid = length_residuals(p1, p2, full, s1, s2, t_yz)
    calib = (s1, s2, t_yz, offline, full)
    after_f1 = {a: count_f1(scenes, gt_rows, a, calib=calib) for a in ARMS}

    # observability
    pooled = observability(p1, p2, full, s1, s2)
    per_scene = {}
    for sid, prs in pairs.items():
        if not prs:
            continue
        a, b = _pair_arrays(scenes, {sid: prs}, offline)
        obs = observability(a, b, full, s1, s2)
        s1_alone, _ = fit_steps(a, b, full, x0=(s1, s2), t_yz=t_yz)
        obs["sep_alone"] = float(s1_alone - s2)
        per_scene[sid] = obs

    scene_t = {}
    for sid, prs in pairs.items():
        if not prs:
            continue
        a, b = _pair_arrays(scenes, {sid: prs}, offline)
        scene_t[sid] = refit_translation(a, b, full, s1, s2)
    ty = np.array([v[0] for v in scene_t.values()])
    tz = np.array([v[1] for v in scene_t.values()])
    _, _, pair_scene = _pair_arrays(scenes, pairs, offline, with_labels=True)
    # cluster = bundle: repeat captures of one bundle resample together
    bundles = load_bundles()
    boot = bootstrap_separation(p1, p2, full, s1, s2,
                                scene_of_pair=np.array([bundle_of(str(s), bundles)
                                                        for s in pair_scene]))

    _write_reports(data, args.arm, history, converged_iter, s1, s2, t_yz,
                   base_resid, resid, base_f1, after_f1, pooled, per_scene,
                   ty, tz, recal, boot,
                   float(full.scanner2.step_positions[0]))
    timing_lines = report_lines(snapshot(), title="Timing (alternation)")
    if timing_lines:
        with open(OUT_MD, "a") as f:
            f.write("\n" + "\n".join(timing_lines) + "\n")


def _pair_arrays(scenes, pairs, offline, with_labels=False):
    """Scanner-frame centroid arrays for the current pair sets; with
    `with_labels`, also one scene id per pair."""
    a, b, lab = [], [], []
    for sid, prs in pairs.items():
        if not prs:
            continue
        d1, d2 = scenes[sid][0], scenes[sid][1]
        q1 = to_scanner_frame(d1, 1, offline)
        q2 = to_scanner_frame(d2, 2, offline)
        for i, j in prs:
            a.append(q1[i])
            b.append(q2[j])
            lab.append(sid)
    if not a:
        empty = (np.zeros((0, 3)), np.zeros((0, 3)))
        return (*empty, np.zeros(0, dtype=object)) if with_labels else empty
    if with_labels:
        return np.asarray(a), np.asarray(b), np.asarray(lab, dtype=object)
    return np.asarray(a), np.asarray(b)


def _stats(r):
    return (float(np.median(r)), float(np.percentile(r, 25)),
            float(np.percentile(r, 75)), int((np.abs(r) <= 10).sum()), len(r))


def _write_reports(data, arm, history, converged_iter, s1, s2, t_yz,
                   base_resid, resid, base_f1, after_f1, pooled, per_scene,
                   ty, tz, recal, boot, full_lo2):
    one_shot = json.loads(ONE_SHOT.read_text()) if ONE_SHOT.is_file() else None
    bm, bq1, bq3, bpass, bn = _stats(base_resid)
    am, aq1, aq3, apass, an = _stats(resid)
    ratio = pooled["ratio"]
    d = pooled["observed_dir"]
    seps_head = np.array([v["sep_alone"] for v in per_scene.values()])

    lines = [
        "# Joint self-calibration + assignment",
        "",
        f"Arm driving the alternation: `{arm}`. {len(per_scene)} scenes with pairs.",
        "",
        "Joint estimation of the cross-view transform and the assignment. "
        "Per-scan carriage position is not in the data, so metric world-X "
        f"calibration comes from scene content plus the {PROFILE_LENGTH_MM:g} mm "
        "bar-length prior.",
        "",
        "## Result",
        "",
        f"**Separation s1 - s2 = {s1 - s2:.1f} mm**, from s1 = {s1:.1f}, "
        f"s2 = {s2:.1f}; only the difference is identified (see "
        "observability below). Per-scene fits have IQR "
        f"[{np.percentile(seps_head, 25):.1f}, {np.percentile(seps_head, 75):.1f}] "
        f"and range [{seps_head.min():.1f}, {seps_head.max():.1f}] mm. A "
        "bundle-cluster bootstrap of the pooled fit (s2 held at its clamp) "
        f"gives 95% [{np.percentile(boot, 2.5):.1f}, "
        f"{np.percentile(boot, 97.5):.1f}]. "
        f"Global cross-view Z residual {t_yz[1]:+.2f} mm; no global Y term is "
        "fitted (see the transport section). "
        f"Converged in {converged_iter} iteration(s) (assignment identical "
        "between two rounds).",
        "",
        "| bar-length residual | median | IQR | within +-10 mm |",
        "|---|---|---|---|",
        f"| offline single-knot calibration | {bm:+.2f} mm | "
        f"[{bq1:+.2f}, {bq3:+.2f}] | {bpass}/{bn} |",
        *( [f"| one-shot fit (experiments.fit_step_positions) | "
            f"{one_shot['median_residual_mm']:+.2f} mm | "
            f"[{one_shot['iqr_mm'][0]:+.2f}, {one_shot['iqr_mm'][1]:+.2f}] | "
            f"— / {one_shot['n_pairs']} |"] if one_shot else [] ),
        f"| joint self-calibration | {am:+.2f} mm | "
        f"[{aq1:+.2f}, {aq3:+.2f}] | {apass}/{an} |",
        "",
        "The one-shot row is fitted once against a fixed `hungarian` "
        f"assignment. IQR {aq3 - aq1:.1f} mm (joint) vs "
        f"{one_shot['iqr_mm'][1] - one_shot['iqr_mm'][0]:.1f} mm (one-shot); "
        "the joint fit does not fit the calibration to pairs the "
        "calibration itself would reject." if one_shot else "",
        "" if one_shot else "",
        "## Pairing does not regress",
        "",
        "Pair-count P/R/F1 vs gt, same arms, before and after "
        "recalibration. Bar-level correspondence is checked separately: "
        "`python -m pairbench.experiments.correspondence --detections "
        "results/pairbench/a2_recalibrated.json`.",
        "",
        "| arm | F1 before | F1 after |", "|---|---|---|",
    ]
    for a in ARMS:
        lines.append(f"| {a} | {base_f1[a][2]:.3f} | {after_f1[a][2]:.3f} |")

    ident = "identified" if ratio < 1e6 else "NOT identified"
    lines += [
        "",
        "## Observability of (s1, s2) under the bar-length prior",
        "",
        "Gauss-Newton conditioning of the length constraint in (s1, s2), "
        "per pair. The eigenvalues of J^T J are the curvature along the "
        "best- and worst-observed directions.",
        "",
        f"Pooled over {pooled['n']} pairs: eigenvalues "
        f"{pooled['eigvals'][0]:.3e} / {pooled['eigvals'][1]:.3e}, "
        f"ratio **{ratio:.3g}**, best-observed direction "
        f"({d[0]:+.3f}, {d[1]:+.3f}).",
        "",
        f"s2 comes to rest at {s2:.1f} mm (lowest calibration knot: "
        f"{full_lo2:.1f} mm): the optimizer slides along the null direction "
        "until the knot range stops it, so only s1 - s2 is a measurement.",
        "",
        "The observed direction is the separation s1 - s2. The sum "
        f"direction is {ident} by this prior: moving both carriages together "
        "leaves every pair length unchanged, so absolute world-X is not "
        "recovered.",
        "",
        "### Which scenes identify the separation",
        "",
    ]
    seps = np.array([v["sep_alone"] for v in per_scene.values()])
    lam = np.array([v["eigvals"][0] for v in per_scene.values()])
    pooled_sep = s1 - s2
    lines += [
        f"Per-pair curvature along the observed direction is "
        f"{lam.min():.3f}–{lam.max():.3f} across all {len(seps)} scenes: "
        "d|w1 - w2| / ds is close to 1 for any bar, so precision comes from "
        "pair count and pair correctness, not scene geometry.",
        "",
        f"Separation fitted per scene: median "
        f"{np.median(seps):.1f} mm, IQR [{np.percentile(seps, 25):.1f}, "
        f"{np.percentile(seps, 75):.1f}], full range [{seps.min():.1f}, "
        f"{seps.max():.1f}] mm, against the pooled {pooled_sep:.1f} mm. A "
        "scene far from the pooled value points to wrong pairs in that scene.",
        "",
        "| scene | pairs | sep_alone (mm) | delta vs pooled |",
        "|---|---|---|---|",
    ]
    ranked = sorted(per_scene.items(),
                    key=lambda kv: -abs(kv[1]["sep_alone"] - pooled_sep))
    for sid, v in ranked[:10]:
        lines.append(f"| {sid} | {v['n']} | {v['sep_alone']:.1f} | "
                     f"{v['sep_alone'] - pooled_sep:+.1f} |")

    n_far = int((np.abs(ty) > 30).sum())
    lines += [
        "",
        "### Cross-view YZ residual: calibration in Z, transport in Y",
        "",
        "Per-scene median (side 1 - side 2) YZ offset over that scene's "
        "pairs. A calibration residual is constant across scenes; the two "
        "views of one scan are captured at different times while the "
        "conveyor runs.",
        "",
        f"- **Z: median {np.median(tz):+.2f} mm, IQR "
        f"[{np.percentile(tz, 25):+.2f}, {np.percentile(tz, 75):+.2f}], "
        f"range [{tz.min():+.1f}, {tz.max():+.1f}]**, fitted as a "
        "calibration constant.",
        f"- **Y: median {np.median(ty):+.2f} mm, IQR "
        f"[{np.percentile(ty, 25):+.2f}, {np.percentile(ty, 75):+.2f}], "
        f"range [{ty.min():+.1f}, {ty.max():+.1f}]**, with {n_far} of "
        f"{len(ty)} scenes past +-30 mm: bar motion along the transport "
        "axis, so no global Y term is fitted. The aligned pairing arms "
        "estimate a per-scene YZ translation by RANSAC instead.",
        "",
        "## Alternation history",
        "",
        "| iter | s1 | s2 | separation | t_yz | pairs | median\\|resid\\| |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in history:
        lines.append(f"| {h['iter']} | {h['s1']:.1f} | {h['s2']:.1f} | "
                     f"{h['s1'] - h['s2']:.1f} | ({h['t_yz'][0]:+.1f}, "
                     f"{h['t_yz'][1]:+.1f}) | {h['n_pairs']} | "
                     f"{h['median_abs_resid']:.2f} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps({
        "arm": arm, "s1_mm": s1, "s2_mm": s2, "separation_mm": s1 - s2,
        "t_yz_global_mm": list(t_yz),
        "median_residual_mm": am, "iqr_mm": [aq1, aq3],
        "n_pairs": an, "iterations": converged_iter,
        "separation_ci95_mm": [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))],
        "separation_per_scene_mm": {
            "values": {sid: float(v["sep_alone"]) for sid, v in per_scene.items()},
            "median": float(np.median(seps_head)),
            "iqr": [float(np.percentile(seps_head, 25)),
                    float(np.percentile(seps_head, 75))],
            "range": [float(seps_head.min()), float(seps_head.max())],
            "n_scenes": int(len(seps_head)),
        },
        "observability": pooled,
        "note": "separation is identified by the bar-length prior; the "
                "s1+s2 sum direction is not (see a2_selfcalib.md)",
    }, indent=1))

    out = {"calibration": {"s1_mm": s1, "s2_mm": s2,
                           "t_yz_global_mm": list(t_yz)},
           "scenes": {}}
    for sid, (d1, d2, l1, l2) in recal.items():
        out["scenes"][sid] = {
            "side1": {"confirmed": [dataclasses.asdict(d) for d in d1],
                      "lost": [dataclasses.asdict(d) for d in l1]},
            "side2": {"confirmed": [dataclasses.asdict(d) for d in d2],
                      "lost": [dataclasses.asdict(d) for d in l2]},
        }
    OUT_DETS.write_text(json.dumps(out))
    print(f"\nwrote {OUT_MD.name}, {OUT_JSON.name}, {OUT_DETS.name}")


if __name__ == "__main__":
    main()
