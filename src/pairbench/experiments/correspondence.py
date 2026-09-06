"""Scores pairing strategies against correspondence-level ground truth.

`gt_correspondence.csv` lists, per annotated scene, which side-1 detection
corresponds to which side-2 detection, stored as YZ anchors and re-linked to
the scored detections (`pairbench.gt.resolve_anchor_indices`); claims whose
bar is not detected are reported as unresolved. Per scene: correct =
|predicted ∩ resolved claims|, recall = correct / resolved, precision =
correct / predicted pairs whose side-1 index a resolved claim judged.

Usage: python -m pairbench.experiments.correspondence [--arm b1|b2|both]
       [--detections results/pairbench/a2_recalibrated.json]

`--detections` scores another detection set with the same schema (confirmed
detections only; the report is written next to it).
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..detect.ccorr import dets_from_json, sort_left_to_right
from ..detect.rescue import merge_rescued_into_confirmed
from ..gt import (
    FLAGGED_SCENES,
    load_gt_correspondence,
    parse_anchors,
    parse_correspondence_pairs,
    resolve_anchor_distances,
    resolve_anchor_indices,
)
from ..pairing import STRATEGIES
from ..pairing.greedy_index import pair_greedy_index
from ..significance import N_BOOT, compare_arms, holm_reject, sum_by_bundle
from ..symmetry import period_for_scene
from .common import RESULTS

IN_JSON = RESULTS / "b1_detections_pool.json"
OTHER_JSON = RESULTS / "b1_detections_ccorr.json"
RESCUED_JSON = RESULTS / "b2_rescued_pool.json"
GT_CORR = RESULTS / "gt_correspondence.csv"
OUT_MD = RESULTS / "correspondence_report.md"

ARMS = ["greedy_index", "greedy_index_debiased", "greedy_index_nolen",
        "greedy_index_order", "hungarian", "hungarian_aligned",
        "monotone_aligned", "hybrid_aligned"]

BASELINE_ARM = "hybrid_aligned"


def run_arm(arm: str, d1, d2, period_deg, length_bias_mm=0.0):
    if arm == "greedy_index_nolen":
        return pair_greedy_index(d1, d2, period_deg=period_deg,
                                 check_length=False)
    if arm == "greedy_index_debiased":
        return pair_greedy_index(d1, d2, period_deg=period_deg,
                                 length_bias_mm=length_bias_mm)
    return STRATEGIES[arm](d1, d2, period_deg=period_deg)


def load_detections(with_rescue: bool, in_json: Path = IN_JSON):
    """Returns (scenes dict, label). With rescue, merges the rescued faces
    into each side's confirmed detections."""
    with open(in_json) as f:
        data = json.load(f)
    if not with_rescue:
        label = ("B1 confirmed" if in_json == IN_JSON
                 else f"{in_json.stem} confirmed")
        return data["scenes"], label
    if not RESCUED_JSON.is_file():
        return None, None
    with open(RESCUED_JSON) as f:
        rescued = json.load(f)
    merged = merge_rescued_into_confirmed(data, rescued)
    return merged, "B1 + B2 rescued"


def global_length_bias(scenes) -> float:
    """Median raw cross-view length residual over the whole detection set."""
    residuals = []
    for scene in scenes.values():
        d1 = sort_left_to_right(dets_from_json(scene["side1"]["confirmed"]))
        d2 = sort_left_to_right(dets_from_json(scene["side2"]["confirmed"]))
        res = pair_greedy_index(d1, d2, check_length=False)
        residuals += res.diagnostics.get("length_residuals_mm", [])
    return float(np.median(residuals)) if residuals else 0.0


def shift_offset(resolved, pred_pairs):
    """Modal offset `k = j_pred - j_true` over a scene's resolved claims.

    Returns `(k, fraction of claims at k)`, or `(None, 0.0)` when no claim
    has a prediction to compare."""
    by_i = {i: j for i, j in pred_pairs}
    offsets = [by_i[i] - j for i, j in resolved if i in by_i]
    if not offsets:
        return None, 0.0
    counts = Counter(offsets)
    k, n = counts.most_common(1)[0]
    return int(k), n / len(offsets)


def score(scenes, gt_rows, period_override_deg=None):
    """Returns (totals, per_scene, coverage, localization_errors_mm, claims,
    extra) for one detection set.

    `localization_errors_mm` is the distance between each resolved claim's
    anchor and the detection it resolved to
    (`pairbench.gt.resolve_anchor_distances`).

    `claims` is one record per resolved claim: {scene, family, pair,
    correct: {arm: bool}} — the per-claim outcomes a paired bootstrap
    resamples (see `pairbench.significance`).

    `extra` is one record per (arm, scene): the modal index offset of the
    prediction against the resolved GT pairs, and whether that arm's own
    gates fired on that scene.

    `period_override_deg`, when set, forces every scene to one symmetry
    period instead of its profile's measured one.
    """
    length_bias = global_length_bias(scenes)
    per_scene = {arm: {} for arm in ARMS}
    extra = {arm: {} for arm in ARMS}  # shift offset + arm diagnostics
    totals = {arm: [0, 0, 0] for arm in ARMS}  # correct, n_resolved, n_pred
    coverage = [0, 0]  # resolved claims, claimed
    localization_errors_mm = []
    claims = []

    for sid, row in sorted(gt_rows.items()):
        if sid not in scenes:
            continue
        gt_pairs = parse_correspondence_pairs(row["pairs"])
        if not gt_pairs:
            continue
        scene = scenes[sid]
        d1 = sort_left_to_right(dets_from_json(scene["side1"]["confirmed"]))
        d2 = sort_left_to_right(dets_from_json(scene["side2"]["confirmed"]))
        period = (period_for_scene(sid) if period_override_deg is None
                  else float(period_override_deg))

        anchors_1 = parse_anchors(row["anchors_1"])
        anchors_2 = parse_anchors(row["anchors_2"])
        map1 = resolve_anchor_indices(anchors_1, d1)
        map2 = resolve_anchor_indices(anchors_2, d2)
        localization_errors_mm += list(resolve_anchor_distances(anchors_1, d1).values())
        localization_errors_mm += list(resolve_anchor_distances(anchors_2, d2).values())
        resolved = {(map1[i], map2[j]) for i, j in gt_pairs
                    if i in map1 and j in map2}
        coverage[0] += len(resolved)
        coverage[1] += len(gt_pairs)
        if not resolved:
            for arm in ARMS:
                per_scene[arm][sid] = (0, 0, len(gt_pairs), 0.0, 0.0)
                extra[arm][sid] = {"shift": None, "shift_frac": 0.0,
                                   "crossing": False,
                                   "n_det_1": len(d1), "n_det_2": len(d2),
                                   "min_margin": None, "t_y_mm": None}
            continue

        witnessed_i = {i for i, _ in resolved}
        predicted = {}
        for arm in ARMS:
            res = run_arm(arm, d1, d2, period, length_bias_mm=length_bias)
            pred_pairs = set(res.pairs)
            predicted[arm] = pred_pairs
            pred_witnessed = {(i, j) for i, j in pred_pairs if i in witnessed_i}
            correct = len(resolved & pred_pairs)
            recall = correct / len(resolved)
            precision = correct / len(pred_witnessed) if pred_witnessed else 0.0
            per_scene[arm][sid] = (correct, len(resolved), len(gt_pairs),
                                   recall, precision)
            k, kfrac = shift_offset(resolved, pred_pairs)
            margins = res.diagnostics.get("pair_margins") or []
            t = res.diagnostics.get("t_yz_mm")
            extra[arm][sid] = {
                "shift": k, "shift_frac": kfrac,
                "crossing": bool(res.diagnostics.get("crossing_hypothesis",
                                                     False)),
                # signals an abstention rule may read at run time: none of
                # these needs the annotation
                "n_det_1": len(d1), "n_det_2": len(d2),
                "min_margin": float(min(margins)) if margins else None,
                "t_y_mm": float(t[0]) if t else None,
            }
            t = totals[arm]
            t[0] += correct
            t[1] += len(resolved)
            t[2] += len(pred_witnessed)
        for pair in sorted(resolved):
            claims.append({"scene": sid, "family": sid.split("/")[0],
                           "pair": pair,
                           "correct": {arm: pair in predicted[arm]
                                       for arm in ARMS}})
    return totals, per_scene, coverage, localization_errors_mm, claims, extra


IPAA_THRESHOLDS = (1.0, 0.9, 0.8, 0.5)


def ipaa_stats(per_scene_arm: dict) -> dict:
    """IPAA / IPAA-X (Cai et al. 2020, MessyTable): the unweighted mean of
    each scene's own recall, and the fraction of scenes reaching each
    threshold. A scene with zero resolved claims counts as a complete miss."""
    accs = np.array([v[3] for v in per_scene_arm.values()])
    n = len(accs)
    if n == 0:
        return {"ipaa": float("nan"), "ipaa_x": {x: float("nan") for x in IPAA_THRESHOLDS},
                "n_scenes": 0}
    return {"ipaa": float(accs.mean()),
            "ipaa_x": {x: float((accs >= x).mean()) for x in IPAA_THRESHOLDS},
            "n_scenes": n}


def ipaa_significance_rows(per_scene: dict, baseline: str, x: float = 1.0,
                           n_boot: int = N_BOOT) -> list[dict]:
    """Paired bootstrap on IPAA-X against `baseline`, treating each scene's
    threshold pass/fail as a 0/1 indicator. Scenes are summed per bundle
    before resampling (`significance.sum_by_bundle`), so the point estimate
    is still the mean over scenes while repeat captures move together."""
    scene_ids = sorted(next(iter(per_scene.values())).keys())
    ones = np.ones(len(scene_ids), dtype=np.float64)
    ind = [np.array([float(per_scene[arm][sid][3] >= x) for sid in scene_ids])
           for arm in ARMS]
    _, units, *summed = sum_by_bundle(scene_ids, ones, *ind)
    numer = dict(zip(ARMS, summed))
    rows = compare_arms(units, numer, baseline=baseline, n_boot=n_boot)
    survives = holm_reject({r["arm"]: r["p_value"] for r in rows if r["arm"] != baseline})
    for r in rows:
        r["holm"] = survives.get(r["arm"], False)
    return rows


def significance_rows(claims, baseline: str, unit: str, n_boot: int = N_BOOT):
    """Paired-bootstrap comparison of every arm's correspondence recall
    against `baseline`, resampling `unit` ('bundle', 'scene' or 'claim')."""
    if unit == "claim":
        units = np.ones(len(claims), dtype=np.float64)
        numer = {arm: np.array([float(c["correct"][arm]) for c in claims])
                 for arm in ARMS}
    elif unit in ("scene", "bundle"):
        scene_ids = sorted({c["scene"] for c in claims})
        index = {sid: k for k, sid in enumerate(scene_ids)}
        units = np.zeros(len(scene_ids), dtype=np.float64)
        numer = {arm: np.zeros(len(scene_ids)) for arm in ARMS}
        for c in claims:
            units[index[c["scene"]]] += 1.0
            for arm in ARMS:
                numer[arm][index[c["scene"]]] += float(c["correct"][arm])
        if unit == "bundle":
            _, units, *summed = sum_by_bundle(scene_ids, units,
                                              *(numer[a] for a in ARMS))
            numer = dict(zip(ARMS, summed))
    else:
        raise ValueError(f"unknown resampling unit {unit!r}")
    rows = compare_arms(units, numer, baseline=baseline, n_boot=n_boot)
    survives = holm_reject({r["arm"]: r["p_value"] for r in rows
                            if r["arm"] != baseline})
    for r in rows:
        r["unit"] = unit
        r["holm"] = survives.get(r["arm"], False)
    return rows


def significance_section(claims, baseline: str = BASELINE_ARM) -> list[str]:
    """Markdown for the paired-bootstrap section of the report."""
    lines = [
        f"### Paired-bootstrap significance vs `{baseline}`", "",
        "Every arm is re-scored on the same resampled datasets "
        f"({N_BOOT} replicates, `pairbench.significance.paired_bootstrap`).",
        "",
        "The resampling unit is the bundle: claims within a scene are "
        "correlated, and repeat captures of one bundle "
        "(`data/repeat_captures.csv`) are resampled together. The claim-level "
        "table is for comparison.",
        ""]
    for unit in ("bundle", "claim"):
        rows = significance_rows(claims, baseline=baseline, unit=unit)
        n_units = rows[0]["n_units"] if rows else 0
        lines += [f"Resampling unit: **{unit}** (n={n_units}).", "",
                  "| strategy | recall | delta vs baseline | 95% paired CI | p | "
                  "separates? | survives Holm? |", "|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -r["value"]):
            if r["arm"] == baseline:
                lines.append(f"| `{r['arm']}` (baseline) | {r['value']:.3f} | — "
                             "| — | — | — | — |")
                continue
            lines.append(
                f"| `{r['arm']}` | {r['value']:.3f} | {r['delta']:+.3f} | "
                f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_value']:.4f} | "
                f"{'yes' if r['significant'] else 'no'} | "
                f"{'yes' if r['holm'] else 'no'} |")
        lines.append("")
    lines += [f"`separates?` = the uncorrected 95% paired CI excludes zero. "
              f"`survives Holm?` applies Holm-Bonferroni over the "
              f"{len(ARMS) - 1} comparisons in each table (family-wise "
              "alpha=0.05, `pairbench.significance.holm_reject`).", ""]
    return lines


BY_SCENE_CSV = RESULTS / "correspondence_by_scene.csv"


def _error_structure_section(per_scene, extra, label):
    """Clean/partial/wrong scene counts and pure lattice shifts per arm."""
    lines = [f"## Error structure ({label})", "",
             "A scene is `clean` when every resolved claim is right, `wrong` "
             "when none is, and `partial` otherwise. `shift k` is the modal "
             "offset `j_pred - j_true` over a scene's claims; `pure shift` "
             "counts the wrong scenes with at least two claims where every "
             "claim sits at the same non-zero offset.", "",
             "| arm | clean | partial | wrong | wrong (>=2 claims) | pure shift | shift sizes |",
             "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        clean = partial = wrong = wrong2 = pure = 0
        sizes = Counter()
        for sid, (correct, resolved, _, _, _) in per_scene[arm].items():
            if resolved == 0:
                continue
            if correct == resolved:
                clean += 1
            elif correct == 0:
                wrong += 1
                if resolved >= 2:
                    wrong2 += 1
                    e = extra[arm][sid]
                    if e["shift"] not in (None, 0) and e["shift_frac"] == 1.0:
                        pure += 1
                        sizes[e["shift"]] += 1
            else:
                partial += 1
        hist = ", ".join(f"k={k}: {n}" for k, n in sorted(sizes.items())) or "—"
        lines.append(f"| `{arm}` | {clean} | {partial} | {wrong} | {wrong2} "
                     f"| {pure} | {hist} |")
    lines.append("")
    return lines


def _gate_firing_section(extra, label):
    """Number of real scenes on which the crossing gate fired."""
    crossing = sum(1 for e in extra["hybrid_aligned"].values() if e["crossing"])
    n = len(extra["hybrid_aligned"])
    return [f"## Gate firing on real scenes ({label})", "",
            f"- crossing hypothesis admitted: **{crossing} of {n}** scenes "
            "(`hybrid_aligned`)",
            ""]


ALIGNED_ARMS = ("hungarian_aligned", "monotone_aligned", "hybrid_aligned")
TRANSFER_ARM = "greedy_index_nolen"


def _arm_shape(per_scene, arm):
    """Pooled recall and the clean/partial/wrong scene counts of one arm."""
    correct = resolved = 0
    clean = partial = wrong = 0
    for c, r, _claimed, _rec, _prec in per_scene[arm].values():
        if r == 0:
            continue
        correct += c
        resolved += r
        if c == r:
            clean += 1
        elif c == 0:
            wrong += 1
        else:
            partial += 1
    return {"recall": correct / resolved if resolved else 0.0,
            "clean": clean, "partial": partial, "wrong": wrong}


def _robustness_section(this_json, this_per_scene, this_cov, this_err,
                        gt_rows):
    """Re-scores the other detector's confirmed set against the same claims."""
    other_json = OTHER_JSON if this_json == IN_JSON else IN_JSON
    if not other_json.is_file():
        print(f"{other_json.name} missing — skipping the detector-robustness "
              "section")
        return []
    scenes, _label = load_detections(False, other_json)
    _totals, other_per_scene, other_cov, other_err, _claims, _extra = score(
        scenes, gt_rows)
    other_md = (OUT_MD.name if other_json == IN_JSON
                else f"correspondence_report_{other_json.stem}.md")

    moved = max(abs(_arm_shape(this_per_scene, arm)["recall"]
                    - _arm_shape(other_per_scene, arm)["recall"])
                for arm in ALIGNED_ARMS)
    here = _arm_shape(this_per_scene, TRANSFER_ARM)
    there = _arm_shape(other_per_scene, TRANSFER_ARM)

    # The anchors were annotated from the pool detector's own output, so the
    # bias runs in one direction whichever report this is.
    pool_err, pool_cov = ((this_err, this_cov) if this_json == IN_JSON
                          else (other_err, other_cov))
    alt_err, alt_cov = ((other_err, other_cov) if this_json == IN_JSON
                        else (this_err, this_cov))

    lines = [
        "## Robustness to the detector",
        "",
        "The same arms scored on the other detector's detections "
        f"(`{other_md}`). The {len(ALIGNED_ARMS)} aligned arms move by at most "
        f"{moved:.3f} recall between the two detection sets; "
        f"`{TRANSFER_ARM}` scores {here['recall']:.3f} here and "
        f"{there['recall']:.3f} there ({here['clean']} / {here['partial']} / "
        f"{here['wrong']} clean / partial / wrong scenes here, "
        f"{there['clean']} / {there['partial']} / {there['wrong']} there).",
        "",
        "An order-anchored arm assumes both sides detected the same bars in "
        "the same order, so detector disagreement passes directly into its "
        "pairing accuracy. Rankings in this report hold at this detector's "
        "operating point.",
        "",
    ]
    if pool_err and alt_err:
        lines += [
            "The anchors were annotated from the pool detector's output: "
            "re-linking the other detector's detections has median error "
            f"{np.median(alt_err):.2f} mm against {np.median(pool_err):.2f} mm "
            f"and resolves {alt_cov[0]} claims against {pool_cov[0]}.",
            "",
        ]
    return lines


def _write_by_scene_csv(runs, path=None):
    """One row per (label, scene, arm); filter on label."""
    path = path or BY_SCENE_CSV
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "scene_id", "arm", "correct", "resolved",
                    "claimed", "recall", "precision", "shift", "shift_frac",
                    "crossing", "n_det_1", "n_det_2", "min_margin",
                    "t_y_mm"])
        for label, per_scene, extra in runs:
            for arm in ARMS:
                for sid in sorted(per_scene[arm]):
                    c, r, cl, rec, prec = per_scene[arm][sid]
                    e = extra[arm][sid]
                    w.writerow([label, sid, arm, c, r, cl, f"{rec:.4f}",
                                f"{prec:.4f}",
                                "" if e["shift"] is None else e["shift"],
                                f"{e['shift_frac']:.4f}",
                                int(e["crossing"]),
                                e["n_det_1"], e["n_det_2"],
                                "" if e["min_margin"] is None else f"{e['min_margin']:.4f}",
                                "" if e["t_y_mm"] is None else f"{e['t_y_mm']:.3f}"])
    print(f"wrote {path} ({len(runs)} scoring(s): "
          f"{', '.join(r[0] for r in runs)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("b1", "b2", "both"), default="both",
                    help="detections to score: B1 confirmed, B1+rescued, or both")
    ap.add_argument("--detections", type=Path, default=IN_JSON,
                    help="alternative detection set (same schema as the default)")
    args = ap.parse_args()
    out_md = (OUT_MD if args.detections == IN_JSON
              else RESULTS / f"correspondence_report_{args.detections.stem}.md")
    if args.detections != IN_JSON and args.arm != "b1":
        print(f"scoring {args.detections.name}: rescue merge does not apply, "
              "using the confirmed arm only")
        args.arm = "b1"

    gt_rows = {sid: r for sid, r in load_gt_correspondence(GT_CORR).items()
               if sid not in FLAGGED_SCENES}
    if not gt_rows:
        print(f"no rows in {GT_CORR} — annotate correspondence claims first")
        return
    missing_anchors = [sid for sid, r in gt_rows.items()
                       if not r.get("anchors_1") or not r.get("anchors_2")]
    if missing_anchors:
        print(f"WARNING: {len(missing_anchors)} rows have no anchors and "
              "cannot be re-linked: " + ", ".join(missing_anchors))

    wanted = {"b1": [False], "b2": [True], "both": [False, True]}[args.arm]
    runs = []
    for with_rescue in wanted:
        scenes, label = load_detections(with_rescue, args.detections)
        if scenes is None:
            print(f"{RESCUED_JSON.name} missing — run "
                  "pairbench.experiments.rescue for the post-rescue arm")
            continue
        totals, per_scene, coverage, loc_err_mm, claims, extra = score(
            scenes, gt_rows)
        runs.append((label, totals, per_scene, coverage, loc_err_mm,
                     claims, extra))

    if not runs:
        return

    n_scenes = len([sid for sid in gt_rows
                    if parse_correspondence_pairs(gt_rows[sid]["pairs"])])
    lines = ["# Correspondence-level GT — pairing strategy accuracy", "",
             f"Scored against `results/pairbench/gt_correspondence.csv` "
             f"({n_scenes} scenes with claimed pairs).", "",
             "Claims are re-linked to detections by stored YZ anchors, not by "
             "index, so the same GT scores both detection arms. `resolved` = "
             "claims whose bar the arm actually detected; recall is over "
             "those, and the unresolved remainder is a detection loss, not a "
             "pairing error.", "",
             f"**Denominators.** {len(gt_rows)} scenes are scored "
             f"(`gt.FLAGGED_SCENES` excluded); {n_scenes} claim at least one "
             f"cross-view pair. The bootstrap runs over scenes with at least "
             f"one resolved claim, grouped into bundles (repeat captures of "
             f"one bundle move together).", ""]

    for (label, totals, per_scene, coverage, loc_err_mm, claims,
         extra) in runs:
        res_n, claimed_n = coverage
        lines += [f"## {label}", "",
                  f"Resolved {res_n}/{claimed_n} claimed pairs "
                  f"({res_n / claimed_n:.3f} coverage).", "",
                  "| strategy | correct | resolved | recall | precision | IPAA "
                  "| IPAA-1.0 | IPAA-0.9 | IPAA-0.8 | IPAA-0.5 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        print(f"\n{label}: resolved {res_n}/{claimed_n} claimed pairs")
        for arm in ARMS:
            c, g, pw = totals[arm]
            recall = c / g if g else 0.0
            precision = c / pw if pw else 0.0
            ist = ipaa_stats(per_scene[arm])
            ix = ist["ipaa_x"]
            lines.append(
                f"| {arm} | {c} | {g} | {recall:.3f} | {precision:.3f} | "
                f"{ist['ipaa']:.3f} | {ix[1.0]:.3f} | {ix[0.9]:.3f} | "
                f"{ix[0.8]:.3f} | {ix[0.5]:.3f} |")
            print(f"  {arm:22s} correct {c}/{g}  recall {recall:.3f}  "
                  f"precision {precision:.3f}  IPAA {ist['ipaa']:.3f}  "
                  f"IPAA-1.0 {ix[1.0]:.3f}")
        lines += ["",
                  "IPAA (Cai et al. 2020, MessyTable) is the unweighted mean of "
                  "each scene's own recall — pooled `recall` above lets a big "
                  "scene swamp small ones, IPAA does not. IPAA-X is the fraction "
                  "of scenes reaching at least X per-scene accuracy; IPAA-1.0 is "
                  "the fraction of scenes an arm gets completely right.", ""]
        if loc_err_mm:
            e = np.array(loc_err_mm)
            lines += ["**Localization error** (annotated anchor to resolved "
                      "detection centroid, YZ mm; bounded by the anchor "
                      "resolution radius, so this reads as the noise floor "
                      "of the anchor re-link, not full-pipeline accuracy): "
                      f"n={len(e)}, median {np.median(e):.2f} mm, IQR "
                      f"[{np.percentile(e, 25):.2f}, "
                      f"{np.percentile(e, 75):.2f}] mm, max {e.max():.2f} mm.",
                      ""]
            print(f"  localization error: n={len(e)} median "
                  f"{np.median(e):.2f} mm IQR [{np.percentile(e, 25):.2f}, "
                  f"{np.percentile(e, 75):.2f}] mm")
        if claims:
            lines += significance_section(claims)
            for r in significance_rows(claims, BASELINE_ARM, "bundle"):
                if r["arm"] != BASELINE_ARM:
                    print(f"  vs {BASELINE_ARM}: {r['arm']:22s} "
                          f"delta {r['delta']:+.3f} "
                          f"CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                          f"p {r['p_value']:.4f} "
                          f"{'SEPARATES' if r['significant'] else 'ns'}")
            ipaa_rows = ipaa_significance_rows(per_scene, BASELINE_ARM, x=1.0)
            lines += [f"### IPAA-1.0 paired-bootstrap vs `{BASELINE_ARM}`", "",
                      "Same shared-resample machinery as the recall test above, "
                      "applied to the binary indicator \"this scene is completely "
                      "correct\" instead of pooled correct claims.", "",
                      "| strategy | IPAA-1.0 | delta vs baseline | 95% paired CI | "
                      "p | separates? | survives Holm? |",
                      "|---|---|---|---|---|---|---|"]
            for r in sorted(ipaa_rows, key=lambda r: -r["value"]):
                if r["arm"] == BASELINE_ARM:
                    lines.append(f"| `{r['arm']}` (baseline) | {r['value']:.3f} "
                                 "| — | — | — | — | — |")
                    continue
                lines.append(
                    f"| `{r['arm']}` | {r['value']:.3f} | {r['delta']:+.3f} | "
                    f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_value']:.4f} | "
                    f"{'yes' if r['significant'] else 'no'} | "
                    f"{'yes' if r['holm'] else 'no'} |")
            lines.append("")

    lines += _robustness_section(args.detections, runs[0][2],
                                 runs[0][3], runs[0][4], gt_rows)

    # error structure for every scoring, matching the tables above
    for lab, _t, ps, _cov, _le, _cl, ex in runs:
        lines += _error_structure_section(ps, ex, lab)
        lines += _gate_firing_section(ex, lab)

    label, _, per_scene, _, _, _, extra = runs[-1]

    shown = list(ARMS)
    lines += [f"## By scene ({label})", "",
              "correct/resolved per arm, every arm. The three aligned arms "
              "differ only in what they are allowed to express: free "
              "assignment, order constraint, order constraint + "
              "evidence-gated crossings. `greedy_index_nolen` is the "
              "order-anchored arm with its nominal-length check disabled.", "",
              "| scene | claimed | " + " | ".join(shown) + " |",
              "|---|---|" + "---|" * len(shown)]
    for sid in sorted(per_scene[shown[0]]):
        claimed = per_scene[shown[0]][sid][2]
        cells = [f"{per_scene[a][sid][0]}/{per_scene[a][sid][1]}" for a in shown]
        lines.append(f"| {sid} | {claimed} | " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_md}")
    _write_by_scene_csv(
        [(label, ps, ex) for (label, _t, ps, _cov, _le, _cl, ex) in runs],
        out_md.with_name(out_md.stem.replace("_report", "") + "_by_scene.csv"))


if __name__ == "__main__":
    main()
