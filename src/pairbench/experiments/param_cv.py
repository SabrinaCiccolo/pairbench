"""Cross-validated selection of the hybrid arm's constants.

Random search over nine cost-model constants and the two hybrid extensions
(level matching by height, one-view crossings): 3000 configurations including
the default one and every one-at-a-time move from it, scored per scene on the
pool detector's confirmed detections. Selection is cross-validated
leave-one-profile-out and by grouped 5-fold over bundles (50 repeats); tied
candidates are reported both with a tie-break towards the default constants
and as the expected recall under a uniform choice among them.

Usage: python -m pairbench.experiments.param_cv
"""

from __future__ import annotations

import json
import random
from multiprocessing import Pool

import numpy as np

from ..gt import (FLAGGED_SCENES, bundle_of, load_bundles,
                  load_gt_correspondence, parse_anchors,
                  parse_correspondence_pairs, resolve_anchor_indices)
from ..pairing import STRATEGIES, common, hungarian, hybrid
from ..symmetry import period_for_scene
from . import correspondence as C
from .common import RESULTS

OUT_MD = RESULTS / "param_cv.md"
OUT_JSON = RESULTS / "param_cv.json"

GRID = {
    "DIST_SCALE_MM": [20.0, 30.0, 40.0, 45.0, 50.0, 60.0],
    "TWIST_SCALE_DEG": [1.5, 2.5, 4.0],
    "LEVEL_PENALTY": [1.0, 2.0, 3.0],
    "Z_OUTLIER_PENALTY": [1.0, 2.0, 3.0],
    "ALIGNED_UNMATCH_COST": [5.5, 7.5, 9.5],
    "ALIGNED_DIST_CAP": [3.5, 4.5, 5.5],
    "CONSENSUS_GATE_MM": [20.0, 30.0, 40.0],
    "Z_OUTLIER_MM": [5.0, 8.0, 12.0],
    "LEVEL_MATCH_MM": [10.0, 20.0, 30.0],
    "ONE_VIEW": [True, False],
    "LEVEL_BY_HEIGHT": [True, False],
}
DEFAULT = {
    "DIST_SCALE_MM": hungarian.DIST_SCALE_MM,
    "TWIST_SCALE_DEG": hungarian.TWIST_SCALE_DEG,
    "LEVEL_PENALTY": hungarian.LEVEL_PENALTY,
    "Z_OUTLIER_PENALTY": hungarian.Z_OUTLIER_PENALTY,
    "ALIGNED_UNMATCH_COST": hybrid.ALIGNED_UNMATCH_COST,
    "ALIGNED_DIST_CAP": hybrid.ALIGNED_DIST_CAP,
    "CONSENSUS_GATE_MM": hybrid.CONSENSUS_GATE_MM,
    "Z_OUTLIER_MM": common.Z_OUTLIER_MM,
    "LEVEL_MATCH_MM": hybrid.LEVEL_MATCH_MM,
    "ONE_VIEW": True,
    "LEVEL_BY_HEIGHT": True,
}
N_CONFIGS = 3000
N_REPEATS = 50
N_FOLDS = 5
SEED = 0
REFERENCE_ARM = "greedy_index_order"   # reads none of these constants

_BREAKS_RIGID_MOTION = hybrid.breaks_rigid_motion
_DATA = None


def _load() -> list:
    gt = {k: v for k, v in load_gt_correspondence(C.GT_CORR).items()
          if k not in FLAGGED_SCENES}
    scenes, _label = C.load_detections(False, C.IN_JSON)
    out = []
    for sid, row in sorted(gt.items()):
        if sid not in scenes:
            continue
        gt_pairs = parse_correspondence_pairs(row["pairs"])
        if not gt_pairs:
            continue
        d1 = C.sort_left_to_right(C.dets_from_json(scenes[sid]["side1"]["confirmed"]))
        d2 = C.sort_left_to_right(C.dets_from_json(scenes[sid]["side2"]["confirmed"]))
        m1 = resolve_anchor_indices(parse_anchors(row["anchors_1"]), d1)
        m2 = resolve_anchor_indices(parse_anchors(row["anchors_2"]), d2)
        resolved = {(m1[i], m2[j]) for i, j in gt_pairs if i in m1 and j in m2}
        if resolved:
            out.append((sid, d1, d2, period_for_scene(sid), resolved))
    return out


def _apply(p: dict) -> None:
    # hybrid imports some constants by name and some are bound as default
    # arguments at definition time, so every copy is patched
    hungarian.DIST_SCALE_MM = hybrid.DIST_SCALE_MM = p["DIST_SCALE_MM"]
    hungarian.TWIST_SCALE_DEG = hybrid.TWIST_SCALE_DEG = p["TWIST_SCALE_DEG"]
    hungarian.LEVEL_PENALTY = p["LEVEL_PENALTY"]
    hungarian.Z_OUTLIER_PENALTY = p["Z_OUTLIER_PENALTY"]
    # no level ever matches by height: levels matched by index
    hybrid.LEVEL_MATCH_MM = p["LEVEL_MATCH_MM"] if p["LEVEL_BY_HEIGHT"] else -1.0
    common.z_outlier_flags.__defaults__ = (p["Z_OUTLIER_MM"],)
    hybrid.elevated_outliers.__defaults__ = (p["Z_OUTLIER_MM"],)
    hybrid.breaks_rigid_motion = (_BREAKS_RIGID_MOTION if p["ONE_VIEW"]
                                  else (lambda *a, **k: None))


def _init() -> None:
    global _DATA
    _DATA = _load()


def _score(p: dict) -> list[int]:
    _apply(p)
    try:
        return [len(res & set(hybrid.pair_hybrid_aligned(
                    d1, d2, period_deg=per,
                    unmatch_cost=p["ALIGNED_UNMATCH_COST"],
                    dist_cap=p["ALIGNED_DIST_CAP"],
                    gate_mm=p["CONSENSUS_GATE_MM"]).pairs))
                for _sid, d1, d2, per, res in _DATA]
    finally:
        _apply(DEFAULT)


def _configs() -> list[dict]:
    configs = [dict(DEFAULT)]
    for k, values in GRID.items():
        for v in values:
            c = dict(DEFAULT, **{k: v})
            if c not in configs:
                configs.append(c)
    rng = random.Random(SEED)
    while len(configs) < N_CONFIGS:
        c = {k: rng.choice(v) for k, v in GRID.items()}
        if c not in configs:
            configs.append(c)
    return configs


def _profile(scene_id: str) -> str:
    fam = scene_id.split("/")[0]
    return "l-profile" if fam.startswith("l-profile") else fam


def main() -> None:
    configs = _configs()
    with Pool(initializer=_init) as pool:
        correct = np.array(pool.map(_score, configs, chunksize=20))
    data = _load()
    scenes = [d[0] for d in data]
    resolved = np.array([len(d[4]) for d in data])
    reference = np.array([len(res & set(STRATEGIES[REFERENCE_ARM](
        d1, d2, period_deg=per).pairs)) for _s, d1, d2, per, res in data])
    n = int(resolved.sum())
    default = 0
    all_idx = np.arange(len(scenes))

    def tied(train):
        tot = correct[:, train].sum(axis=1)
        return np.flatnonzero(tot == tot.max())

    def fold_row(test):
        cand = tied(np.setdiff1d(all_idx, test))
        held = correct[cand][:, test].sum(axis=1)
        # tie-break towards the default constants: fewest values changed
        near = cand[np.argmin([sum(configs[k][a] != DEFAULT[a] for a in DEFAULT)
                               for k in cand])]
        return {"n_tied": len(cand), "default_tied": bool(default in cand),
                "held_tiebreak": int(correct[near, test].sum()),
                "held_min": int(held.min()), "held_mean": float(held.mean()),
                "held_max": int(held.max()),
                "held_default": int(correct[default, test].sum()),
                "resolved": int(resolved[test].sum())}

    profiles = np.array([_profile(s) for s in scenes])
    lopo = {p: fold_row(np.flatnonzero(profiles == p))
            for p in sorted(set(profiles))}

    bundles = load_bundles()
    labels = np.array([bundle_of(s, bundles) for s in scenes])
    unique = sorted(set(labels))
    kfold = []
    for rep in range(N_REPEATS):
        order = unique[:]
        random.Random(rep).shuffle(order)
        kfold.append([fold_row(np.flatnonzero(np.isin(labels, order[i::N_FOLDS])))
                      for i in range(N_FOLDS)])

    def cv(rows, key):
        return sum(r[key] for r in rows) / n

    full_best = int(correct.sum(axis=1).max())
    best_cfgs = [configs[k] for k in np.flatnonzero(correct.sum(axis=1) == full_best)]
    k_default = [cv(rows, "held_tiebreak") for rows in kfold]
    k_unif = [cv(rows, "held_mean") for rows in kfold]
    k_tied = sum(r["default_tied"] for rows in kfold for r in rows)
    summary = {
        "in_sample_default": int(correct[default].sum()) / n,
        "in_sample_best": full_best / n,
        "n_best_configs": len(best_cfgs),
        "best_configs_keep_both_extensions": all(
            c["ONE_VIEW"] and c["LEVEL_BY_HEIGHT"] for c in best_cfgs),
        "reference_recall": int(reference.sum()) / n,
        "lopo_cv_default_tiebreak": cv(lopo.values(), "held_tiebreak"),
        "lopo_cv_uniform_among_tied": cv(lopo.values(), "held_mean"),
        "kfold_cv_default_tiebreak_mean": float(np.mean(k_default)),
        "kfold_cv_uniform_among_tied_mean": float(np.mean(k_unif)),
        "kfold_cv_uniform_among_tied_range": [float(min(k_unif)), float(max(k_unif))],
        "kfold_default_tied": f"{k_tied}/{N_REPEATS * N_FOLDS}",
    }

    OUT_JSON.write_text(json.dumps({
        "grid": GRID, "default": DEFAULT,
        "n_configs": len(configs), "n_scenes": len(scenes),
        "resolved_claims": n, "summary": summary, "lopo": lopo}, indent=1))

    lines = [
        "# Cross-validated selection of the hybrid arm's constants",
        "",
        "Regenerate: `python -m pairbench.experiments.param_cv`. "
        f"`pool` confirmed detections, {len(scenes)} scenes, {n} resolved claims, "
        f"{len(configs)} configurations of nine constants and the two extensions "
        "(grid in `param_cv.json`).",
        "",
        "## In sample",
        "",
        f"- default constants: recall {summary['in_sample_default']:.3f}; best of the "
        f"search: {summary['in_sample_best']:.3f}, reached by {len(best_cfgs)} configurations",
        f"- every best configuration keeps both extensions: "
        f"{'yes' if summary['best_configs_keep_both_extensions'] else 'no'}",
        f"- `{REFERENCE_ARM}` (reads none of these constants): {summary['reference_recall']:.3f}",
        "",
        "## Leave-one-profile-out",
        "",
        "`tied` counts the configurations with the highest recall on the other "
        "profiles; the held-out columns give their correct claims on the held-out one.",
        "",
        "| held-out profile | resolved | tied | default among tied | held-out min | mean | max | default |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, r in lopo.items():
        lines.append(f"| {p} | {r['resolved']} | {r['n_tied']} | "
                     f"{'yes' if r['default_tied'] else 'no'} | {r['held_min']} | "
                     f"{r['held_mean']:.1f} | {r['held_max']} | {r['held_default']} |")
    lines += [
        "",
        f"Cross-validated recall: {summary['lopo_cv_default_tiebreak']:.3f} with ties broken "
        f"towards the default constants, {summary['lopo_cv_uniform_among_tied']:.3f} expected "
        "under a uniform choice among the tied configurations.",
        "",
        f"## Grouped {N_FOLDS}-fold by bundle, {N_REPEATS} repeats",
        "",
        f"- default constants among the tied configurations in "
        f"{summary['kfold_default_tied']} folds",
        f"- cross-validated recall, ties towards default: mean "
        f"{summary['kfold_cv_default_tiebreak_mean']:.3f}",
        f"- cross-validated recall, uniform among tied: mean "
        f"{summary['kfold_cv_uniform_among_tied_mean']:.3f}, range "
        f"{summary['kfold_cv_uniform_among_tied_range'][0]:.3f}-"
        f"{summary['kfold_cv_uniform_among_tied_range'][1]:.3f}",
        "",
        "## Reading",
        "",
        "Many configurations tie on the training scenes and score differently "
        "on the held-out scenes, so the training scenes do not single out the "
        "default constants. No row here is an out-of-sample estimate of the "
        "default arm.",
        "",
        "## Limits",
        "",
        "- Three values per constant (six for the distance scale) and a random "
        "sample of the grid; a finer grid can only add tied configurations.",
        "- The unit-test invariants that gate `dist_scale` are not applied.",
        "- The two extensions were designed on these scenes; switching them on "
        "or off in the search does not make them out-of-sample.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.relative_to(RESULTS.parents[1])}")


if __name__ == "__main__":
    main()
