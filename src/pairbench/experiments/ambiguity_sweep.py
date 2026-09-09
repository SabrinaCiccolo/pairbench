"""Tests whether the cross-view ambiguity is periodic in the bar pitch.

On the synthetic `ambiguity` suite (analytic bar identity) computes, per scene,
`margin(k) = cost(hypothesis shifted by k bars) - cost(truth)` under the aligned
arms' order-constrained objective, split into an assignment term (matched-pair
cost at the hypothesis' own translation) and a bookkeeping term (half an
unmatch cost per leftover detection). Writes `ambiguity_sweep.{md,csv,json}`.

Usage:
    python -m pairbench.experiments.build_synthetic_twin --suite ambiguity
    python -m pairbench.experiments.ambiguity_sweep [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from ..config import REPO_ROOT
from ..pairing.hungarian import pair_cost_matrix
from ..pairing.hungarian_aligned import (ALIGNED_DIST_CAP, ALIGNED_UNMATCH_COST,
                                         _shifted)
from ..pairing.hypotheses import enumerate_hypotheses, translation_of
from ..symmetry import period_for_dxf
from ..synth import load_truth
from ..synth.evaluate import detect_synth_scene
from .common import RESULTS

SYNTH = REPO_ROOT / "data" / "synthetic" / "ambiguity"
OUT_MD = RESULTS / "ambiguity_sweep.md"
OUT_CSV = RESULTS / "ambiguity_sweep.csv"
OUT_JSON = RESULTS / "ambiguity_sweep.json"

SHIFTS = (-2, -1, 1, 2)


def _cost_terms(pairs, dets1, dets2, period_deg):
    """(assignment, bookkeeping) summands of `hypotheses.hypothesis_cost`."""
    n1, n2 = len(dets1), len(dets2)
    if not pairs:
        return float("nan"), float("nan")
    t = translation_of(pairs, dets1, dets2)
    cm = pair_cost_matrix(dets1, _shifted(dets2, t), period_deg, ALIGNED_DIST_CAP)
    assign = sum(float(cm[i, j]) for i, j in pairs)
    book = (ALIGNED_UNMATCH_COST / 2.0) * (n1 + n2 - 2 * len(pairs))
    return assign, book


def _job(scene_dir):
    sd = detect_synth_scene(scene_dir)
    d1, d2 = sd.confirmed[1], sd.confirmed[2]
    bar1, bar2 = sd.bar_of[1], sd.bar_of[2]
    period = period_for_dxf(sd.truth["dxf"])
    params = sd.truth["params"]

    # bars detected on both sides: the only ones any hypothesis can pair
    seen1 = {b: i for i, b in enumerate(bar1) if b is not None}
    seen2 = {b: j for j, b in enumerate(bar2) if b is not None}
    pairable = sorted(set(seen1) & set(seen2))

    def hypothesis(k):
        """Pairs bar b on side 1 with bar b+k on side 2, where both exist."""
        return sorted((seen1[b], seen2[b + k])
                      for b in seen1 if (b + k) in seen2)

    truth = hypothesis(0)
    row = {
        "scene_id": sd.truth["scene_id"],
        "placement": params.get("placement"),
        "pitch_jitter_mm": params.get("pitch_jitter_mm"),
        "transport_over_pitch": params.get("transport_over_pitch"),
        "transport_y_mm": params.get("transport_y_mm"),
        "seed": params.get("seed"),
        "n_det_1": len(d1), "n_det_2": len(d2),
        "n_face_1": len(sd.truth["faces"]["side1"]),
        "n_face_2": len(sd.truth["faces"]["side2"]),
        "n_pairable": len(pairable), "n_true_pairs": len(truth),
    }
    if len(truth) < 2:
        for k in SHIFTS:
            row[f"margin_k{k:+d}"] = float("nan")
            row[f"assign_k{k:+d}"] = float("nan")
            row[f"book_k{k:+d}"] = float("nan")
        row.update(assign_true=float("nan"), book_true=float("nan"),
                   cost_true=float("nan"), best_wrong=float("nan"),
                   margin_best=float("nan"), n_hypotheses=0)
        return row

    a_t, b_t = _cost_terms(truth, d1, d2, period)
    row["assign_true"], row["book_true"] = a_t, b_t
    row["cost_true"] = a_t + b_t
    for k in SHIFTS:
        pairs = hypothesis(k)
        if len(pairs) < 1:
            row[f"margin_k{k:+d}"] = float("nan")
            row[f"assign_k{k:+d}"] = float("nan")
            row[f"book_k{k:+d}"] = float("nan")
            continue
        a, b = _cost_terms(pairs, d1, d2, period)
        row[f"assign_k{k:+d}"] = a
        row[f"book_k{k:+d}"] = b
        row[f"margin_k{k:+d}"] = (a + b) - (a_t + b_t)

    # cheapest wrong hypothesis among those the detections propose
    hyps = enumerate_hypotheses(d1, d2, period, ALIGNED_UNMATCH_COST,
                                ALIGNED_DIST_CAP)
    best = None
    for dp, _t, pairs in hyps:
        if sorted(pairs) == truth:
            continue
        if best is None or dp < best:
            best = dp
    row["n_hypotheses"] = len(hyps)
    row["best_wrong"] = best if best is not None else float("nan")
    row["margin_best"] = (best - row["cost_true"]) if best is not None else float("nan")
    return row


def _agg(rows, key, field):
    out = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is not None and np.isfinite(v):
            out[key(r)].append(v)
    return {k: (float(np.median(v)), len(v)) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    dirs = sorted(str(p.parent) for p in SYNTH.rglob("truth.json"))
    if not dirs:
        raise SystemExit(
            "no scenes under data/synthetic/ambiguity — run "
            "python -m pairbench.experiments.build_synthetic_twin --suite ambiguity")
    rows = []
    with Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap(_job, dirs)):
            rows.append(row)
            if (k + 1) % 25 == 0 or k + 1 == len(dirs):
                print(f"  [{k + 1}/{len(dirs)}]")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    jitters = sorted({r["pitch_jitter_mm"] for r in rows})
    trans = sorted({r["transport_over_pitch"] for r in rows})
    placements = ["centre", "edge"]
    n_seeds = len({r["seed"] for r in rows})
    first = load_truth(dirs[0])
    family, n_bars = first["dxf"], first["params"]["n_bars"]
    pitch_mm = first["params"]["pitch_mm"]

    def cell(pl, j, t, field):
        v = [r[field] for r in rows
             if r["placement"] == pl and r["pitch_jitter_mm"] == j
             and r["transport_over_pitch"] == t
             and r.get(field) is not None and np.isfinite(r[field])]
        return float(np.median(v)) if v else float("nan")

    lines = [
        "# Is the cross-view ambiguity periodic in the bar pitch?",
        "",
        f"Regenerate: `python -m pairbench.experiments.build_synthetic_twin "
        f"--suite ambiguity` then "
        f"`python -m pairbench.experiments.ambiguity_sweep`. "
        f"{len(rows)} synthetic scenes: {n_bars} `{family}` bars, pitch "
        f"{pitch_mm:g} mm, cross-view transport from {min(trans):g} to "
        f"{max(trans):g} pitches, both placements, {len(jitters)} pitch-jitter "
        f"levels, {n_seeds} seeds per cell.",
        "",
        "`margin(k) = cost(hypothesis shifted by k bars) - cost(truth)` under "
        "the aligned arms' order-constrained objective, with analytic bar "
        "identity. Positive = the objective prefers the truth. Periodicity "
        "would show as dips at whole values of transport / pitch.",
        "",
        "## margin(k=+1), centred (no bar leaves the window)",
        "",
        "| jitter (mm) | " + " | ".join(f"{t:g}" for t in trans) + " |",
        "|---" * (1 + len(trans)) + "|",
    ]
    for j in jitters:
        lines.append(f"| {j:g} | " + " | ".join(
            f"{cell('centre', j, t, 'margin_k+1'):.2f}" for t in trans) + " |")

    lines += [
        "",
        "## margin(k=+1), at the window edge (transport moves bars in and out)",
        "",
        "| jitter (mm) | " + " | ".join(f"{t:g}" for t in trans) + " |",
        "|---" * (1 + len(trans)) + "|",
    ]
    for j in jitters:
        lines.append(f"| {j:g} | " + " | ".join(
            f"{cell('edge', j, t, 'margin_k+1'):.2f}" for t in trans) + " |")

    lines += [
        "",
        "## Where the margin comes from",
        "",
        "Medians per placement x jitter, pooled over transport. `assignment`: "
        "matched-pair cost at the hypothesis' own translation. "
        "`bookkeeping`: half an unmatch cost per leftover detection.",
        "",
        "| placement | jitter (mm) | assignment (truth) | assignment (k=+1) | "
        "bookkeeping (truth) | bookkeeping (k=+1) | margin |",
        "|---|---|---|---|---|---|---|",
    ]
    for pl in placements:
        for j in jitters:
            sel = [r for r in rows if r["placement"] == pl
                   and r["pitch_jitter_mm"] == j
                   and np.isfinite(r.get("margin_k+1", float("nan")))]
            if not sel:
                continue
            med = lambda f: float(np.median([r[f] for r in sel]))
            lines.append(
                f"| {pl} | {j:g} | {med('assign_true'):.2f} | "
                f"{med('assign_k+1'):.2f} | {med('book_true'):.2f} | "
                f"{med('book_k+1'):.2f} | {med('margin_k+1'):.2f} |")

    lines += [
        "",
        "## margin(k=-1), at the window edge (shift towards the truncated side)",
        "",
        "When the two views hold different bars, a shift towards the cut side "
        "pairs as many detections as the truth, so only the assignment term "
        "separates them.",
        "",
        "| jitter (mm) | " + " | ".join(f"{t:g}" for t in trans) + " |",
        "|---" * (1 + len(trans)) + "|",
    ]
    for j in jitters:
        lines.append(f"| {j:g} | " + " | ".join(
            f"{cell('edge', j, t, 'margin_k-1'):.2f}" for t in trans) + " |")
    lines += [
        "",
        "Decomposition of margin(k=-1) at the edge, split by whether the two "
        "views detect the same number of bars (medians):",
        "",
        "| jitter (mm) | views agree on count | scenes | assignment (truth) | "
        "assignment (k=-1) | bookkeeping (truth) | bookkeeping (k=-1) | margin |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for j in jitters:
        for agree in (True, False):
            sel = [r for r in rows if r["placement"] == "edge"
                   and r["pitch_jitter_mm"] == j
                   and (r["n_det_1"] == r["n_det_2"]) == agree
                   and np.isfinite(r.get("margin_k-1", float("nan")))]
            if not sel:
                continue
            med = lambda f: float(np.median([r[f] for r in sel]))
            lines.append(
                f"| {j:g} | {'yes' if agree else 'no'} | {len(sel)} | "
                f"{med('assign_true'):.2f} | {med('assign_k-1'):.2f} | "
                f"{med('book_true'):.2f} | {med('book_k-1'):.2f} | "
                f"{med('margin_k-1'):.2f} |")
    n_neg = sum(1 for r in rows if r["placement"] == "edge"
                and np.isfinite(r.get("margin_k-1", float("nan")))
                and r["margin_k-1"] < 0)
    n_edge = sum(1 for r in rows if r["placement"] == "edge"
                 and np.isfinite(r.get("margin_k-1", float("nan"))))
    lines += [
        "",
        f"margin(k=-1) is negative, i.e. the objective prefers the shift, on "
        f"{n_neg} of {n_edge} edge scenes.",
    ]

    # count mismatches and truth assignment cost at the two ends of the sweep
    def _mism(pl, j, t):
        sel = [r for r in rows if r["placement"] == pl
               and r["pitch_jitter_mm"] == j and r["transport_over_pitch"] == t]
        bad = sum((r["n_det_1"] != r["n_face_1"]) + (r["n_det_2"] != r["n_face_2"])
                  for r in sel)
        return bad, 2 * len(sel)

    t_lo, t_hi = min(trans), max(trans)

    def _assign_true(pl, j, t):
        v = [r["assign_true"] for r in rows if r["placement"] == pl
             and r["pitch_jitter_mm"] == j and r["transport_over_pitch"] == t
             and np.isfinite(r.get("assign_true", float("nan")))]
        return float(np.median(v)) if v else float("nan")

    _true_note = "; ".join(
        f"jitter {j:g} mm, {_assign_true('centre', j, t_lo):.2f} at "
        f"{t_lo:g} pitches against {_assign_true('centre', j, t_hi):.2f} at "
        f"{t_hi:g}" for j in jitters)
    _mismatch_note = "; ".join(
        f"jitter {j:g} mm, {_mism('centre', j, t_lo)[0]}"
        f"/{_mism('centre', j, t_lo)[1]} at {t_lo:g} pitches and "
        f"{_mism('centre', j, t_hi)[0]}/{_mism('centre', j, t_hi)[1]} at "
        f"{t_hi:g}" for j in jitters)

    lines += [
        "",
        "## Periodicity test",
        "",
    ]
    verdict = []
    for pl in placements:
        for j in jitters:
            xs, ys = [], []
            for t in trans:
                v = cell(pl, j, t, "margin_k+1")
                if np.isfinite(v):
                    xs.append(t)
                    ys.append(v)
            if len(ys) < 4:
                continue
            ys = np.array(ys)
            xs = np.array(xs)
            whole = np.isclose(xs % 1.0, 0.0) | np.isclose(xs % 1.0, 1.0)
            at_whole = float(np.median(ys[whole])) if whole.any() else float("nan")
            off_whole = float(np.median(ys[~whole])) if (~whole).any() else float("nan")
            spread = float(ys.max() - ys.min())
            verdict.append({
                "placement": pl, "jitter_mm": j,
                "margin_at_whole_multiples": at_whole,
                "margin_between_multiples": off_whole,
                "difference": at_whole - off_whole,
                "range_over_transport": spread,
                "median_margin": float(np.median(ys)),
            })
            lines.append(
                f"- {pl}, jitter {j:g} mm: margin at whole multiples "
                f"{at_whole:.2f}, between them {off_whole:.2f}, difference "
                f"{at_whole - off_whole:+.2f}, range over transport {spread:.2f}.")

    lines += [
        "",
        "Periodicity would show as a negative `difference` repeated across "
        f"cells and large relative to the range ({n_seeds} seeds per cell).",
        "",
        "## Limits",
        "",
        "- Centred, sides whose detected count differs from the faces "
        "present: " + _mismatch_note + ".",
        "- Centred, median truth assignment cost at the smallest vs largest "
        "transport: " + _true_note + ".",
        f"- One family (`{family}`), {n_bars} bars, {n_seeds} seeds per cell.",
        "- The margin is a cost difference in the aligned arms' units, not an "
        "error rate.",
        "- Shifted hypotheses are constructed at a chosen offset; the CSV's "
        "`margin_best` is the searched counterpart (same enumeration as "
        "`experiments.cost_gap`).",
        "",
    ]

    OUT_JSON.write_text(json.dumps({
        "n_scenes": len(rows),
        "shifts": list(SHIFTS), "periodicity": verdict,
        "unmatch_cost": ALIGNED_UNMATCH_COST, "dist_cap": ALIGNED_DIST_CAP,
    }, indent=1))
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.name}, {OUT_CSV.name}, {OUT_JSON.name}")
    for v in verdict:
        print(f"  {v['placement']:7s} j={v['jitter_mm']:>4g}  "
              f"whole {v['margin_at_whole_multiples']:+7.2f}  "
              f"between {v['margin_between_multiples']:+7.2f}  "
              f"diff {v['difference']:+7.2f}")


if __name__ == "__main__":
    main()
