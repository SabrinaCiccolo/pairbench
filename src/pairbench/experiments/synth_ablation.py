"""Controlled pairing/detection experiments on the synthetic twin.

Runs the default pipeline (`load_scene` -> `project_yz` -> `detect_faces_pool`
-> pairing arms) on synthetic scenes with known 6-DOF poses and scores
correspondence against ground-truth bar identity. Sections: A pitch jitter x
transport, B elevation x yaw (crossing gate), B2 one-end-raised bar, C
localization error, D transport with and without window truncation.

Usage:
    python -m pairbench.experiments.build_synthetic_twin --suite all
    python -m pairbench.experiments.synth_ablation [--suite all] [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from ..config import REPO_ROOT
from ..significance import compare_arms
from ..synth.evaluate import detect_synth_scene, detection_report, pairing_report
from .common import RESULTS

SYNTH = REPO_ROOT / "data" / "synthetic"
OUT_MD = RESULTS / "synthetic_ablation.md"
OUT_CSV = RESULTS / "synthetic_ablation.csv"
OUT_JSON = RESULTS / "synthetic_ablation.json"

ARMS = ["greedy_index_nolen", "hungarian", "hungarian_aligned",
        "monotone_aligned", "hybrid_aligned"]

# arm reported in the per-cell tables and used as the bootstrap baseline
DEFAULT_ARM = "hybrid_aligned"


def _job(scene_dir):
    sd = detect_synth_scene(scene_dir)
    det = detection_report(sd)
    pair = pairing_report(sd, ARMS)
    err = np.asarray(det.pop("loc_errors_mm"), dtype=float)
    dy = det.pop("loc_dy_mm")
    dz = det.pop("loc_dz_mm")
    return {
        "scene_id": sd.truth["scene_id"],
        "dxf": sd.truth["dxf"],
        "params": sd.truth["params"],
        "n_det_1": len(sd.confirmed[1]), "n_det_2": len(sd.confirmed[2]),
        "n_face_1": len(sd.truth["faces"]["side1"]),
        "n_face_2": len(sd.truth["faces"]["side2"]),
        "det": det,
        "loc_med": float(np.median(err)) if len(err) else float("nan"),
        "loc_p95": float(np.percentile(err, 95)) if len(err) else float("nan"),
        "loc_errors": err.tolist(), "loc_dy": dy, "loc_dz": dz,
        "section_height_mm": sd.truth["params"].get("section_height_mm"),
        "pair": pair,
    }


def run_suite(name: str, workers: int) -> list:
    dirs = sorted(str(p.parent) for p in (SYNTH / name).rglob("truth.json"))
    if not dirs:
        raise SystemExit(f"no scenes under {SYNTH / name} — run "
                         f"python -m pairbench.experiments.build_synthetic_twin --suite {name}")
    rows = []
    with Pool(workers) as pool:
        for k, row in enumerate(pool.imap(_job, dirs)):
            rows.append(row)
            if (k + 1) % 20 == 0 or k + 1 == len(dirs):
                print(f"  [{k + 1}/{len(dirs)}] {name}")
    return rows


def _recall(rows, arm):
    c = sum(r["pair"][arm]["correct"] for r in rows)
    n = sum(r["pair"]["n_pairable"] for r in rows)
    return (c / n if n else float("nan")), c, n


def _pitch_section(rows) -> list[str]:
    by_jitter = defaultdict(list)
    by_cell = defaultdict(list)
    for r in rows:
        by_jitter[r["params"]["pitch_jitter_mm"]].append(r)
        by_cell[(r["params"]["pitch_jitter_mm"],
                 r["params"].get("transport_over_pitch"))].append(r)

    p0 = rows[0]["params"]
    lines = [
        "## A. Pitch regularity x cross-view transport",
        "",
        f"One {p0['n_bars']}-bar `{rows[0]['dxf']}` bundle against the left "
        "edge of the scan window, so transport can carry its leftmost bar in "
        "or out between captures. `pitch_jitter` is the per-bar Gaussian "
        f"sigma added to a regular {p0['pitch_mm']:g} mm spacing.",
        "",
        "| pitch jitter (mm) | scenes | count-mismatch sides | "
        + " | ".join(ARMS) + " |",
        "|---" * (3 + len(ARMS)) + "|",
    ]
    for jitter in sorted(by_jitter):
        rs = by_jitter[jitter]
        mism = sum((r["n_det_1"] != r["n_face_1"]) + (r["n_det_2"] != r["n_face_2"])
                   for r in rs)
        cells = " | ".join(f"{_recall(rs, a)[0]:.3f}" for a in ARMS)
        lines.append(f"| {jitter:g} | {len(rs)} | {mism}/{2 * len(rs)} | {cells} |")

    j_lo, j_hi = min(by_jitter), max(by_jitter)
    swing = {a: _recall(by_jitter[j_hi], a)[0] - _recall(by_jitter[j_lo], a)[0]
             for a in ARMS}
    helped = [a for a in ARMS if swing[a] > 0.02]
    hurt = [a for a in ARMS if swing[a] < -0.02]
    flat = [a for a in ARMS if abs(swing[a]) <= 0.02]

    def _listed(names):
        return ", ".join(f"`{a}` ({swing[a]:+.3f})" for a in names) or "none"

    lines += [
        "",
        f"From {j_lo:g} mm to {j_hi:g} mm of jitter: helped "
        f"{_listed(helped)}; hurt {_listed(hurt)}; within +-0.02 "
        f"{_listed(flat)}.",
        "",
        "Arms that estimate a cross-view translation need an irregular "
        "bundle for it to be identifiable: under regular spacing a whole-pitch "
        "shift fits as well as the truth. Instance-order arms estimate no "
        "translation.",
        "",
        f"{len(by_jitter[j_lo])} scenes per jitter level; the significance "
        "section tests only the two extreme rows, and the bundle sits at the "
        "window edge, so jitter and truncation are not separated here "
        "(see section D).",
    ]

    lines += [
        "",
        "Recall differences between the aligned arms, per jitter level:",
        "",
        "| pitch jitter (mm) | hybrid - monotone | "
        "monotone - hungarian_aligned |",
        "|---|---|---|",
    ]
    for jitter in sorted(by_jitter):
        rs = by_jitter[jitter]
        h = _recall(rs, "hybrid_aligned")[0]
        m = _recall(rs, "monotone_aligned")[0]
        a = _recall(rs, "hungarian_aligned")[0]
        lines.append(f"| {jitter:g} | {h - m:+.3f} | {m - a:+.3f} |")

    lines += [
        "",
        f"`{DEFAULT_ARM}` recall per (jitter, transport/pitch) cell. "
        "Periodicity in the pitch is tested in `ambiguity_sweep.md`.",
        "",
    ]
    tvals = sorted({k[1] for k in by_cell})
    lines += ["| jitter \\ transport/pitch | " + " | ".join(f"{t:g}" for t in tvals) + " |",
              "|---" * (1 + len(tvals)) + "|"]
    for jitter in sorted(by_jitter):
        cells = []
        for t in tvals:
            rs = by_cell.get((jitter, t), [])
            cells.append(f"{_recall(rs, DEFAULT_ARM)[0]:.2f}" if rs else "-")
        lines.append(f"| {jitter:g} | " + " | ".join(cells) + " |")
    return lines


def _crossing_section(rows) -> list[str]:
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(bool(r["params"]["elevated"]), r["params"]["yaw_deg"])].append(r)

    lines = [
        "## B. Elevation x yaw: the crossing gate, dose-response",
        "",
        "Three bed bars plus one probe bar. `elevated` puts the probe one "
        "section height above the bed between two bed bars; otherwise it "
        "lies flat one extra pitch beyond the row's end. `yaw` separates the "
        "probe's two end faces in transport-Y.",
        "",
        "| elevated | yaw (deg) | face offset from yaw (mm) | scenes | "
        "elevated flagged (1/2) | corroborated | rigid-motion break | "
        "crossing applied | " + " | ".join(ARMS) + " |",
        "|---" * (9 + len(ARMS)) + "|",
    ]
    for (elev, yaw) in sorted(by_cell, key=lambda k: (not k[0], k[1])):
        rs = by_cell[(elev, yaw)]
        p = rs[0]["params"]
        n = len(rs)
        e1 = np.mean([r["pair"]["hybrid_aligned"]["n_elevated_1"] or 0 for r in rs])
        e2 = np.mean([r["pair"]["hybrid_aligned"]["n_elevated_2"] or 0 for r in rs])
        corr = sum(bool(r["pair"]["hybrid_aligned"]["elevation_corroborated"]) for r in rs)
        moves = sum(bool(r["pair"]["hybrid_aligned"]["moves_with_bundle"]) for r in rs)
        fired = sum(r["pair"]["hybrid_aligned"]["crossing"] for r in rs)
        cells = " | ".join(f"{_recall(rs, a)[0]:.3f}" for a in ARMS)
        lines.append(
            f"| {'yes' if elev else 'no'} | {yaw:g} | "
            f"{p['yaw_face_offset_mm']:.0f} | {n} | {e1:.1f}/{e2:.1f} | "
            f"{corr}/{n} | {n - moves}/{n} | {fired}/{n} | {cells} |")
    lines += [
        "",
        "`rigid-motion break`: elevated bars found not to travel with the "
        "bundle. `crossing applied`: the gate's final output.",
        "",
    ]
    return lines


def _one_end_section(rows) -> list[str]:
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["params"]["lateral_mm"], r["params"]["raised_side"])].append(r)
    lines = [
        "## B2. A bar raised at one end only",
        "",
        "Three bed bars plus one probe whose raised end rests between the "
        "last two bed bars and whose other end lies on the bed `lateral` mm "
        "further along Y, so it is elevated in one view only. The hybrid "
        "arm's one-view path fires when the raised end has no partner near "
        "it under the scene translation.",
        "",
        "| lateral (mm) | raised in view | scenes | pairable | "
        "one-view crossing applied | " + " | ".join(ARMS) + " |",
        "|---" * (5 + len(ARMS)) + "|",
    ]
    for key in sorted(by_cell):
        rs = by_cell[key]
        n = len(rs)
        fired = sum(bool(r["pair"]["hybrid_aligned"]["crossing"]) for r in rs)
        cells = " | ".join(f"{_recall(rs, a)[0]:.3f}" for a in ARMS)
        lines.append(f"| {key[0]:g} | {key[1]} | {n} | "
                     f"{sum(r['pair']['n_pairable'] for r in rs)} | "
                     f"{fired}/{n} | {cells} |")
    lines.append("")
    return lines


def _localization_section(all_rows) -> list[str]:
    """Reports centroid localization error against the scenes' analytic pose truth."""
    by_dxf = defaultdict(lambda: {"e": [], "dy": [], "dz": [], "h": None})
    for r in all_rows:
        acc = by_dxf[r["dxf"]]
        acc["e"].extend(r["loc_errors"])
        acc["dy"].extend(r["loc_dy"])
        acc["dz"].extend(r["loc_dz"])
        acc["h"] = acc["h"] or r["section_height_mm"]
    lines = [
        "## C. Localization error, in millimetres",
        "",
        "Detected centroid against the analytic face centre.",
        "",
        "| profile | section height (mm) | faces | median |err| | p95 |err| | "
        "median dY | median dZ |",
        "|---|---|---|---|---|---|---|",
    ]
    for dxf in sorted(by_dxf, key=lambda k: by_dxf[k]["h"] or 0):
        acc = by_dxf[dxf]
        e = np.asarray(acc["e"], dtype=float)
        lines.append(
            f"| {dxf} | {acc['h']:.1f} | {len(e)} | {np.median(e):.2f} | "
            f"{np.percentile(e, 95):.2f} | {np.median(acc['dy']):+.2f} | "
            f"{np.median(acc['dz']):+.2f} |")
    lines += [
        "",
        "A dZ bias that grows with section height comes from the centroid "
        "estimator: the sensor samples the upper part of a tall face more "
        "densely. It is shared by both views and cancels in cross-view "
        "comparisons.",
        "",
    ]
    return lines


def _placement_section(edge_rows, centre_rows) -> list[str]:
    """Recall vs transport for an edge-placed and a centred bundle."""
    def cells(rows, key):
        out = defaultdict(list)
        for r in rows:
            out[r["params"].get(key)].append(r)
        return out

    lines = [
        "## D. Transport with and without truncation",
        "",
        f"Same {edge_rows[0]['params']['n_bars']}-bar bundle at the window "
        "edge and centred (no bar leaves the window at any swept transport). "
        f"`{DEFAULT_ARM}` recall pooled over the "
        f"{len({r['params']['pitch_jitter_mm'] for r in edge_rows})} jitter "
        "levels; `mismatch` = scene-sides whose detected count differs from "
        "the faces present.",
        "",
        "| transport/pitch | edge recall | edge mismatch | centre recall | centre mismatch |",
        "|---|---|---|---|---|",
    ]
    e_by_t, c_by_t = cells(edge_rows, "transport_over_pitch"), cells(centre_rows, "transport_over_pitch")
    for t in sorted(k for k in e_by_t if k is not None):
        er, cr = e_by_t[t], c_by_t.get(t, [])
        def mism(rs):
            m = sum((r["n_det_1"] != r["n_face_1"]) + (r["n_det_2"] != r["n_face_2"])
                    for r in rs)
            return f"{m}/{2 * len(rs)}" if rs else "—"
        cv = f"{_recall(cr, DEFAULT_ARM)[0]:.3f}" if cr else "—"
        lines.append(f"| {t:g} | {_recall(er, DEFAULT_ARM)[0]:.3f} | "
                     f"{mism(er)} | {cv} | {mism(cr)} |")

    lines += ["", "At zero transport both views truncate the same bar, so "
              "their visible sets agree; non-zero transport makes the "
              "truncation differ between views.", "",
              "Zero pitch jitter only:", "",
              "| transport/pitch | edge recall | centre recall |",
              "|---|---|---|"]
    e0 = [r for r in edge_rows if r["params"]["pitch_jitter_mm"] == 0.0]
    c0 = [r for r in centre_rows if r["params"]["pitch_jitter_mm"] == 0.0]
    e0_by_t, c0_by_t = cells(e0, "transport_over_pitch"), cells(c0, "transport_over_pitch")
    for t in sorted(k for k in e0_by_t if k is not None):
        er, cr = e0_by_t[t], c0_by_t.get(t, [])
        cv = f"{_recall(cr, DEFAULT_ARM)[0]:.3f}" if cr else "—"
        lines.append(f"| {t:g} | {_recall(er, DEFAULT_ARM)[0]:.3f} | {cv} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all",
                    choices=["all", "pitch", "crossing", "one_end", "realism",
                             "placement"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    # realism is used for section C only (it spans all four families)
    if args.suite == "all":
        suites = ["pitch", "crossing", "one_end", "realism", "pitch_edge",
                  "pitch_centre"]
    elif args.suite == "placement":
        suites = ["pitch_edge", "pitch_centre"]
    else:
        suites = [args.suite]
    suites = [s for s in suites if (SYNTH / s).is_dir()]
    rows_by_suite = {s: run_suite(s, args.workers) for s in suites}
    all_rows = [r for rs in rows_by_suite.values() for r in rs]

    lines = [
        "# Synthetic twin — controlled experiments",
        "",
        "Regenerate: `python -m pairbench.experiments.build_synthetic_twin --suite all` then "
        "`python -m pairbench.experiments.synth_ablation`. "
        "Realism check: `results/pairbench/synthetic_realism.md`.",
        "",
        "A pair is correct iff its two detections landed on the same "
        "physical bar (Hungarian match to the analytic face centres). Recall "
        "is over `n_pairable`, the bars detected on both sides.",
        "",
    ]
    if "pitch" in rows_by_suite:
        lines += _pitch_section(rows_by_suite["pitch"]) + [""]
    if "crossing" in rows_by_suite:
        lines += _crossing_section(rows_by_suite["crossing"]) + [""]
    if "one_end" in rows_by_suite:
        lines += _one_end_section(rows_by_suite["one_end"]) + [""]
    lines += _localization_section(all_rows)
    if "pitch_edge" in rows_by_suite and "pitch_centre" in rows_by_suite:
        lines += _placement_section(rows_by_suite["pitch_edge"],
                                    rows_by_suite["pitch_centre"]) + [""]

    # arm comparison within the two extreme jitter levels
    if "pitch" in rows_by_suite:
        lines += [
            "## Significance at the ends of the jitter sweep",
            "",
            f"Paired scene bootstrap against `{DEFAULT_ARM}`, within each of "
            "the two extreme jitter levels. The jitter x arm interaction is "
            "not tested.",
            "",
            "| jitter | arm | recall | delta | 95% CI | p |",
            "|---|---|---|---|---|---|",
        ]
        for jitter in (0.0, max(r["params"]["pitch_jitter_mm"]
                                for r in rows_by_suite["pitch"])):
            rs = [r for r in rows_by_suite["pitch"]
                  if r["params"]["pitch_jitter_mm"] == jitter]
            if not rs:
                continue
            units = np.array([r["pair"]["n_pairable"] for r in rs], dtype=float)
            nums = {a: np.array([r["pair"][a]["correct"] for r in rs], dtype=float)
                    for a in ARMS}
            for res in compare_arms(units, nums, baseline=DEFAULT_ARM):
                if res["arm"] == DEFAULT_ARM:
                    continue
                lines.append(
                    f"| {jitter:g} | {res['arm']} | {res['value']:.3f} | "
                    f"{res['delta']:+.3f} | [{res['ci_lo']:+.3f}, "
                    f"{res['ci_hi']:+.3f}] | {res['p_value']:.3f} |")
        lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    with open(OUT_CSV, "w", newline="") as f:
        cols = ["suite", "scene_id", "dxf", "n_det_1", "n_det_2", "n_face_1",
                "n_face_2", "tp", "fp", "fn", "loc_med", "loc_p95", "n_pairable"] + \
               [f"correct_{a}" for a in ARMS] + \
               ["min_margin", "t_y_mm"] + \
               ["pitch_jitter_mm", "transport_over_pitch", "yaw_deg", "elevated",
                "crossing_applied"]
        w = csv.DictWriter(f, fieldnames=cols, restval="", extrasaction="ignore")
        w.writeheader()
        for suite, rs in rows_by_suite.items():
            for r in rs:
                w.writerow({
                    "suite": suite, "scene_id": r["scene_id"], "dxf": r["dxf"],
                    "n_det_1": r["n_det_1"], "n_det_2": r["n_det_2"],
                    "n_face_1": r["n_face_1"], "n_face_2": r["n_face_2"],
                    **r["det"], "loc_med": r["loc_med"], "loc_p95": r["loc_p95"],
                    "n_pairable": r["pair"]["n_pairable"],
                    **{f"correct_{a}": r["pair"][a]["correct"] for a in ARMS},
                    "min_margin": r["pair"]["hungarian_aligned"]["min_margin"],
                    # both scene-level signals come from hungarian_aligned:
                    # it is the only arm that solves a cost matrix, so the
                    # order-constrained arms produce no pair margins
                    "t_y_mm": r["pair"]["hungarian_aligned"]["t_y_mm"],
                    "crossing_applied": int(r["pair"]["hybrid_aligned"]["crossing"]),
                    **{k: v for k, v in r["params"].items() if k in
                       ("pitch_jitter_mm", "transport_over_pitch", "yaw_deg", "elevated")},
                })
    OUT_JSON.write_text(json.dumps(
        {"rows": [{k: v for k, v in r.items() if k != "loc_errors"}
                  for rs in rows_by_suite.values() for r in rs]}, indent=1))
    print(f"wrote {OUT_MD.name}, {OUT_CSV.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
