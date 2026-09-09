"""Does the pairing objective rank the true correspondence first?

The aligned arms minimize one order-constrained objective (pair costs at the
hypothesis' median cross-view translation, plus half an unmatch cost per
leftover detection). Hypothesis under test: with regular bar spacing this
objective cannot tell a one-slot-shifted correspondence from the true one.
Per scene, the annotated correspondence is costed against its rivals: the
searched hypotheses (`pairing.hypotheses.enumerate_hypotheses`) and the
constructed shifts pi_k, k in SHIFTS (`pairing.hypotheses.shift_hypothesis`).

Usage: python -m pairbench.experiments.cost_gap
"""

import csv

import numpy as np

from ..detect.ccorr import dets_from_json, sort_left_to_right
from ..gt import (
    FLAGGED_SCENES,
    load_gt_correspondence,
    parse_anchors,
    parse_correspondence_pairs,
    resolve_anchor_indices,
)
from ..pairing.hungarian_aligned import ALIGNED_DIST_CAP, ALIGNED_UNMATCH_COST
from ..pairing.hypotheses import (enumerate_hypotheses, hypothesis_cost,
                                  shift_hypothesis)
from ..symmetry import period_for_scene
from .common import RESULTS
from .correspondence import BASELINE_ARM, GT_CORR, load_detections, run_arm

OUT_MD = RESULTS / "cost_gap_pool.md"
OUT_CSV = RESULTS / "cost_gap_pool.csv"

SHIFTS = (-2, -1, 1, 2)   # constructed members of the shift family


def _kind(pairs, truth) -> str:
    """What a rival hypothesis is, relative to the truth."""
    if set(pairs) < set(truth):
        return "subset of truth"
    return "other"


def scene_gap(d1, d2, period, true_pairs):
    """Per-scene comparison of the truth against its rivals, as a dict.

    Both sides are restricted to the detections the annotation witnesses, so
    every hypothesis is a matching over the same set. Returns None when no
    rival exists.
    """
    idx1 = sorted({i for i, _ in true_pairs})
    idx2 = sorted({j for _, j in true_pairs})
    sub1 = [d1[i] for i in idx1]
    sub2 = [d2[j] for j in idx2]
    pos1 = {i: k for k, i in enumerate(idx1)}
    pos2 = {j: k for k, j in enumerate(idx2)}
    truth = sorted((pos1[i], pos2[j]) for i, j in true_pairs)

    # truth and rivals costed under the aligned arms' constants
    def cost(pairs):
        return hypothesis_cost(pairs, sub1, sub2, period,
                               ALIGNED_UNMATCH_COST, ALIGNED_DIST_CAP)

    cost_true = cost(truth)
    rivals = []   # (cost, pairs, kind, k)
    hyps = enumerate_hypotheses(sub1, sub2, period,
                                ALIGNED_UNMATCH_COST, ALIGNED_DIST_CAP)
    for dp, _t, pairs in hyps:
        if sorted(pairs) != truth:
            rivals.append((dp, sorted(pairs), None, None))
    shift_cost = {}
    for k in SHIFTS:
        pairs = shift_hypothesis(sub1, sub2, k)
        if pairs and pairs != truth:
            c = cost(pairs)
            shift_cost[k] = c
            rivals.append((c, pairs, f"shift {k:+d}", k))
    if not rivals or not np.isfinite(cost_true):
        return None

    # a searched rival that coincides with a constructed shift is that shift
    shifts_by_pairs = {tuple(shift_hypothesis(sub1, sub2, k)): k for k in SHIFTS}
    c_best, p_best, kind, k_best = min(rivals, key=lambda r: r[0])
    if kind is None:
        k_best = shifts_by_pairs.get(tuple(p_best))
        kind = f"shift {k_best:+d}" if k_best is not None else _kind(p_best, truth)
    one = [shift_cost[k] for k in (-1, 1) if k in shift_cost]
    return {
        "cost_true": cost_true, "cost_best_wrong": c_best,
        "gap": c_best - cost_true, "kind_of_best_wrong": kind,
        "shift_of_best_wrong": k_best,
        "margin_shift1": (min(one) - cost_true) if one else float("nan"),
        "n_searched": len(hyps),
    }


def full_set_rows(gt_rows):
    """`BASELINE_ARM`'s wholly wrong scenes, re-costed on the full confirmed
    detection set (`full_gap = cost(arm output) - cost(annotated pairs)`),
    plus a table of scenes by (all detections witnessed, wholly wrong)."""
    scenes, label = load_detections(False)
    wrong, table = [], {}
    for sid, row in sorted(gt_rows.items()):
        if sid not in scenes or sid in FLAGGED_SCENES:
            continue
        gt_pairs = parse_correspondence_pairs(row["pairs"])
        scene = scenes[sid]
        d1 = sort_left_to_right(dets_from_json(scene["side1"]["confirmed"]))
        d2 = sort_left_to_right(dets_from_json(scene["side2"]["confirmed"]))
        map1 = resolve_anchor_indices(parse_anchors(row["anchors_1"]), d1)
        map2 = resolve_anchor_indices(parse_anchors(row["anchors_2"]), d2)
        resolved = sorted({(map1[i], map2[j]) for i, j in gt_pairs
                           if i in map1 and j in map2})
        if len(resolved) < 2:
            continue
        period = period_for_scene(sid)
        res = run_arm(BASELINE_ARM, d1, d2, period)
        out = sorted(res.pairs)
        is_wrong = not (set(out) & set(resolved))
        spare1 = sorted(set(range(len(d1))) - {i for i, _ in resolved})
        spare2 = sorted(set(range(len(d2))) - {j for _, j in resolved})
        all_witnessed = not spare1 and not spare2
        key = (all_witnessed, is_wrong)
        table[key] = table.get(key, 0) + 1
        if is_wrong:
            c_out = hypothesis_cost(out, d1, d2, period,
                                    ALIGNED_UNMATCH_COST, ALIGNED_DIST_CAP)
            c_true = hypothesis_cost(resolved, d1, d2, period,
                                     ALIGNED_UNMATCH_COST, ALIGNED_DIST_CAP)
            counting = "shift pairs more" if len(out) > len(resolved) else (
                "tie" if len(out) == len(resolved) else "truth pairs more")
            wrong.append({"scene_id": sid, "n_claims": len(resolved),
                          "n_det_1": len(d1), "n_det_2": len(d2),
                          "spare_1": spare1, "spare_2": spare2,
                          "n_pairs_out": len(out), "counting": counting,
                          "cost_out": c_out, "cost_true": c_true,
                          "full_gap": c_out - c_true})
    return label, wrong, table


def full_set_section(gt_rows) -> list[str]:
    label, wrong, table = full_set_rows(gt_rows)
    n_neg = sum(r["full_gap"] < 0 for r in wrong)
    lines = [
        f"## `{BASELINE_ARM}`'s wrong scenes on the full detection set "
        f"(`{label}`)",
        "",
        "The comparison above is restricted to witnessed detections. Here "
        "every scene `" + BASELINE_ARM + "` gets "
        "wholly wrong (at least two resolved claims, none correct) is "
        "re-costed on the full detection set: `full_gap = cost(arm output) - "
        "cost(annotated pairs)`, every unwitnessed detection left unmatched in "
        "the latter. `spare` lists the detections without an annotated "
        "partner, by index along Y on each side.",
        "",
        "| scene | claims | detections (1/2) | spare 1 | spare 2 | pairs in output | counting | cost(output) | cost(truth) | full_gap |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in wrong:
        lines.append(
            f"| {r['scene_id']} | {r['n_claims']} | {r['n_det_1']}/{r['n_det_2']} "
            f"| {r['spare_1'] or '—'} | {r['spare_2'] or '—'} "
            f"| {r['n_pairs_out']} | {r['counting']} | {r['cost_out']:.2f} "
            f"| {r['cost_true']:.2f} | {r['full_gap']:+.2f} |")
    tot = lambda w: table.get((w, True), 0) + table.get((w, False), 0)
    lines += [
        "",
        f"On **{n_neg} of {len(wrong)}** of these scenes the objective, "
        "costed on everything the arm saw, prefers the arm's wrong output to "
        "the annotated pairs.",
        "",
        "Every scene with at least two resolved claims, by whether all of "
        "its detections on both sides carry an annotated partner:",
        "",
        "| every detection witnessed | scenes | wholly wrong |",
        "|---|---|---|",
        f"| yes | {tot(True)} | {table.get((True, True), 0)} |",
        f"| no | {tot(False)} | {table.get((False, True), 0)} |",
        "",
        "A detection without an annotated partner may be a bar the other "
        "view truncates, a false detection or an unclaimed bar; this table "
        "does not tell those apart.",
        "",
    ]
    return lines


def main() -> None:
    scenes, label = load_detections(True)
    if scenes is None:
        scenes, label = load_detections(False)
    gt_rows = load_gt_correspondence(GT_CORR)

    rows, no_rival = [], []
    for sid, row in sorted(gt_rows.items()):
        if sid not in scenes or sid in FLAGGED_SCENES:
            continue
        gt_pairs = parse_correspondence_pairs(row["pairs"])
        if not gt_pairs:
            continue
        scene = scenes[sid]
        d1 = sort_left_to_right(dets_from_json(scene["side1"]["confirmed"]))
        d2 = sort_left_to_right(dets_from_json(scene["side2"]["confirmed"]))
        map1 = resolve_anchor_indices(parse_anchors(row["anchors_1"]), d1)
        map2 = resolve_anchor_indices(parse_anchors(row["anchors_2"]), d2)
        resolved = sorted({(map1[i], map2[j]) for i, j in gt_pairs
                           if i in map1 and j in map2})
        if len(resolved) < 2:
            continue
        res = scene_gap(d1, d2, period_for_scene(sid), resolved)
        if res is None:
            no_rival.append(sid)
            continue
        rows.append({"scene_id": sid, "family": sid.split("/")[0],
                     "n_claims": len(resolved), **res})

    gaps = np.array([r["gap"] for r in rows])
    m1 = np.array([r["margin_shift1"] for r in rows])
    m1 = m1[np.isfinite(m1)]
    prefers_wrong = [r for r in rows if r["gap"] < 0]
    kinds = {}
    for r in rows:
        kinds[r["kind_of_best_wrong"]] = kinds.get(r["kind_of_best_wrong"], 0) + 1

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_all = len(rows) + len(no_rival)
    lines = [
        "# Can the objective tell the true correspondence from a shifted one?",
        "",
        f"Detections: `{label}`. {n_all} scenes with at "
        f"least two resolved claims; {len(rows)} of them have at least one "
        "rival hypothesis"
        + (f" (no rival exists for: {', '.join(no_rival)})." if no_rival
           else ", i.e. all of them."),
        "",
        "For each scene the annotated correspondence is scored as a "
        "hypothesis under the order-constrained objective the aligned arms "
        "minimize, and compared against its rivals: the order-constrained "
        "solutions under every translation the scene's own detections "
        f"propose, plus the members pi_k of the shift family for k in "
        f"{list(SHIFTS)}, built level by level. "
        "`gap = cost(cheapest rival) - cost(true)`; `margin_shift1` is the "
        "same difference against the cheaper of pi_-1 and pi_+1 alone.",
        "",
        "## Result",
        "",
        f"- median gap **{np.median(gaps):+.3f}**, IQR "
        f"[{np.percentile(gaps, 25):+.3f}, {np.percentile(gaps, 75):+.3f}]",
        f"- median margin against a one-bar shift **{np.median(m1):+.3f}**, "
        f"IQR [{np.percentile(m1, 25):+.3f}, {np.percentile(m1, 75):+.3f}], "
        f"minimum {m1.min():+.3f}; negative on **{int((m1 < 0).sum())} of "
        f"{len(m1)}** scenes",
        f"- the objective ranks a **different** hypothesis first on "
        f"**{len(prefers_wrong)} of {len(rows)}** scenes",
        "- what the cheapest rival is, over all scenes: "
        + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())),
        "",
        "A negative margin against a one-bar shift is the aliasing under "
        "test. A shift rival is a wrong association; a subset of the truth "
        "only leaves a true pair unmatched because its cost exceeds the "
        "unmatch cost.",
        "",
        "## Scenes where the objective prefers a different hypothesis",
        "",
        "| scene | claims | rival | cost(true) | cost(rival) | gap |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(prefers_wrong, key=lambda r: r["gap"]):
        lines.append(f"| {r['scene_id']} | {r['n_claims']} | "
                     f"{r['kind_of_best_wrong']} | "
                     f"{r['cost_true']:.3f} | {r['cost_best_wrong']:.3f} | "
                     f"{r['gap']:+.3f} |")
    lines += ["", f"Full per-scene values: `{OUT_CSV.name}`.", ""]
    lines += full_set_section(gt_rows)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    print(f"\nwrote {OUT_MD}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
