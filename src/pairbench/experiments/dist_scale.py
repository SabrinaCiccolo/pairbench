"""Sensitivity of the results to the shared distance scale.

`DIST_SCALE_MM` (YZ centroid distance costing one unit in
`pairing.hungarian.pair_cost_matrix`) is 45 mm, selected by leave-one-family-out
+ k-fold on the real scenes. This module re-checks it on both detection sets:
recall, IPAA-1.0 and wholly wrong scenes over a grid of values, the two
invariant tests at each value, and a leave-one-profile-out selection.

Usage: python -m pairbench.experiments.dist_scale
"""

from __future__ import annotations

import csv
import importlib.util
import json

from ..gt import FLAGGED_SCENES, load_gt_correspondence
from ..pairing import hungarian, hybrid
from . import correspondence as C
from .common import RESULTS

OUT_MD = RESULTS / "dist_scale_sensitivity.md"
OUT_CSV = RESULTS / "dist_scale_sensitivity.csv"
OUT_JSON = RESULTS / "dist_scale_sensitivity.json"

GRID_MM = (10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 90.0)
DEFAULT_MM = 45.0
SCALE_ARMS = ("hungarian", "hungarian_aligned", "monotone_aligned",
              "hybrid_aligned")
REFERENCE_ARM = "greedy_index_order"   # reads no distance: one value per detector
SELECT_ARM = "hybrid_aligned"
DETECTIONS = (("pool", C.IN_JSON),
              ("ccorr", RESULTS / "b1_detections_ccorr.json"))
TEST_FILE = RESULTS.parents[1] / "tests" / "pairbench" / "test_pairing.py"
INVARIANT_TESTS = ("test_aligned_impostor_does_not_beat_noisy_true_pair",
                   "test_hybrid_keeps_refusing_the_impostor_crossing")


def _set_scale(mm: float) -> None:
    # hybrid imports the constant by name, so both modules are patched
    hungarian.DIST_SCALE_MM = mm
    hybrid.DIST_SCALE_MM = mm


def _profile(scene_id: str) -> str:
    fam = scene_id.split("/")[0]
    return "l-profile" if fam.startswith("l-profile") else fam


def _invariants() -> dict[str, bool]:
    spec = importlib.util.spec_from_file_location("_pairing_tests", TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {}
    for name in INVARIANT_TESTS:
        try:
            getattr(mod, name)()
            out[name] = True
        except AssertionError:
            out[name] = False
    return out


def _summary(per_scene_arm: dict, scenes=None) -> tuple[float, float, int, int]:
    """(pooled recall, IPAA-1.0, wholly wrong scenes with >=2 claims,
    resolved claims) over `scenes` (all scored scenes when None)."""
    rows = [v for s, v in per_scene_arm.items()
            if (scenes is None or s in scenes) and v[1] > 0]
    correct = sum(v[0] for v in rows)
    resolved = sum(v[1] for v in rows)
    ipaa = sum(v[3] >= 1.0 for v in rows) / len(rows) if rows else float("nan")
    wrong = sum(1 for v in rows if v[1] >= 2 and v[0] == 0)
    return (correct / resolved if resolved else float("nan"), ipaa, wrong,
            resolved)


def main() -> None:
    gt = {k: v for k, v in load_gt_correspondence(C.GT_CORR).items()
          if k not in FLAGGED_SCENES}
    original = hungarian.DIST_SCALE_MM
    runs, invariants = {}, {}
    try:
        for mm in GRID_MM:
            _set_scale(mm)
            invariants[mm] = _invariants()
            for tag, path in DETECTIONS:
                scenes, _label = C.load_detections(False, path)
                _tot, per_scene, *_ = C.score(scenes, gt)
                runs[(tag, mm)] = per_scene
    finally:
        _set_scale(original)

    rows = []
    for (tag, mm), per_scene in runs.items():
        for arm in SCALE_ARMS + (REFERENCE_ARM,):
            rec, ipaa, wrong, res = _summary(per_scene[arm])
            rows.append({"detections": tag, "dist_scale_mm": mm, "arm": arm,
                         "recall": rec, "ipaa_1": ipaa, "wholly_wrong": wrong,
                         "resolved": res,
                         "invariants_pass": all(invariants[mm].values())})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # leave-one-profile-out selection, on the pool detections
    per_scene_sel = {mm: runs[("pool", mm)][SELECT_ARM] for mm in GRID_MM}
    all_scenes = set(per_scene_sel[GRID_MM[0]])
    profiles = sorted({_profile(s) for s in all_scenes})
    admissible = [mm for mm in GRID_MM if all(invariants[mm].values())]
    folds = []
    for held in profiles:
        train = {s for s in all_scenes if _profile(s) != held}
        test = all_scenes - train
        best = max(admissible, key=lambda mm: (_summary(per_scene_sel[mm], train)[0], -abs(mm - DEFAULT_MM)))
        folds.append({
            "held_out": held, "n_scenes": len(test), "chosen_mm": best,
            "train_recall": _summary(per_scene_sel[best], train)[0],
            "held_out_recall": _summary(per_scene_sel[best], test)[0],
            "held_out_recall_at_default": _summary(per_scene_sel[DEFAULT_MM], test)[0],
        })

    OUT_JSON.write_text(json.dumps({
        "grid_mm": GRID_MM, "default_mm": DEFAULT_MM,
        "invariants": {str(k): v for k, v in invariants.items()},
        "rows": rows, "folds": folds}, indent=1))

    def cell(tag, mm, arm, key, fmt):
        r = next(r for r in rows if r["detections"] == tag
                 and r["dist_scale_mm"] == mm and r["arm"] == arm)
        return format(r[key], fmt)

    lines = [
        "# Sensitivity to the shared distance scale",
        "",
        "Regenerate: `python -m pairbench.experiments.dist_scale`. Confirmed "
        f"detections only, every scene with a claimed pair ({len(all_scenes)} "
        "scenes), recall pooled over resolved claims as in "
        "correspondence_report.md.",
        "",
        f"`DIST_SCALE_MM` is {DEFAULT_MM:g} mm, selected by "
        "leave-one-family-out + k-fold on these scenes. This report shows how "
        "the results move with it and re-runs a leave-one-profile-out "
        "selection.",
        "",
        "## Invariants",
        "",
        "| scale (mm) | " + " | ".join(f"`{t}`" for t in INVARIANT_TESTS) + " |",
        "|---" * (1 + len(INVARIANT_TESTS)) + "|",
    ]
    for mm in GRID_MM:
        lines.append(f"| {mm:g} | " + " | ".join(
            "pass" if invariants[mm][t] else "FAIL" for t in INVARIANT_TESTS)
            + " |")
    for tag, _p in DETECTIONS:
        lines += [
            "",
            f"## Correspondence recall by distance scale (`{tag}` detections)",
            "",
            "| arm | " + " | ".join(f"{mm:g}" for mm in GRID_MM) + " |",
            "|---" * (1 + len(GRID_MM)) + "|",
        ]
        for arm in SCALE_ARMS:
            lines.append(f"| `{arm}` | " + " | ".join(
                cell(tag, mm, arm, "recall", ".3f") for mm in GRID_MM) + " |")
        lines.append(
            f"| `{REFERENCE_ARM}` (reads no distance) | "
            + cell(tag, GRID_MM[0], REFERENCE_ARM, "recall", ".3f")
            + " |" + " |" * (len(GRID_MM) - 1))
        lines += [
            "",
            f"IPAA-1.0 and wholly wrong scenes (at least two claims, none "
            f"correct) of `{SELECT_ARM}`:",
            "",
            "| | " + " | ".join(f"{mm:g}" for mm in GRID_MM) + " |",
            "|---" * (1 + len(GRID_MM)) + "|",
            "| IPAA-1.0 | " + " | ".join(
                cell(tag, mm, SELECT_ARM, "ipaa_1", ".3f") for mm in GRID_MM) + " |",
            "| wholly wrong | " + " | ".join(
                cell(tag, mm, SELECT_ARM, "wholly_wrong", "d") for mm in GRID_MM) + " |",
        ]
    lines += [
        "",
        "## Leave-one-profile-out selection (`pool` detections)",
        "",
        f"For each held-out profile, the grid value with the highest pooled "
        f"`{SELECT_ARM}` recall on the other profiles, among values that pass "
        f"both invariants (ties broken towards {DEFAULT_MM:g} mm), scored on "
        "the held-out profile. The L profile's two acquisition campaigns are "
        "one cross-section and are held out together.",
        "",
        "| held-out profile | scenes | chosen scale (mm) | train recall | held-out recall | held-out recall at "
        f"{DEFAULT_MM:g} mm |",
        "|---|---|---|---|---|---|",
    ]
    for f_ in folds:
        lines.append(
            f"| {f_['held_out']} | {f_['n_scenes']} | {f_['chosen_mm']:g} | "
            f"{f_['train_recall']:.3f} | {f_['held_out_recall']:.3f} | "
            f"{f_['held_out_recall_at_default']:.3f} |")
    lines += [
        "",
        "## Limits",
        "",
        "- The selection uses one criterion (pooled recall of one arm) and "
        "no k-fold stage.",
        "- Every value here is scored on the scenes that also informed the "
        "cost model's other constants, so no row is an out-of-sample "
        "estimate for the model as a whole.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
