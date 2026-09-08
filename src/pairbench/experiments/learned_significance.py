"""Bootstrap intervals for the learned detection baseline, without retraining.

Reads the seed-0 detections written by `experiments.learned_baseline`
(`learned_detections_{split}.json`) and the classical detector's output, and
compares per-scene counts with a paired bootstrap over bundles
(`significance.sum_by_bundle`). `learned_baseline` uses the same function.

Usage: python -m pairbench.experiments.learned_significance
"""

from __future__ import annotations

import json

import numpy as np

from ..metrics import bootstrap_f1_ci
from ..significance import paired_bootstrap_f1, sum_by_bundle
from .common import RESULTS

CLASSICAL_JSON = RESULTS / "b1_detections_pool.json"
OUT_MD = RESULTS / "learned_significance.md"
BASELINE = "classical (pool)"


def bundle_bootstrap(per_scene_arms: dict[str, np.ndarray],
                     scene_ids: list[str]) -> tuple[list[dict], tuple, int]:
    """(paired bootstrap rows vs the classical arm, classical pooled-F1 CI,
    number of bundles), resampling bundles."""
    arms = list(per_scene_arms)
    bundles, *summed = sum_by_bundle(scene_ids,
                                     *(per_scene_arms[a] for a in arms))
    by_bundle = dict(zip(arms, summed))
    rows = paired_bootstrap_f1(by_bundle, baseline=BASELINE)
    ci = bootstrap_f1_ci([tuple(r) for r in by_bundle[BASELINE]])
    return rows, ci, len(bundles)


def _counts(path) -> dict[tuple[str, int], int]:
    d = json.loads(path.read_text())
    return {(sid, side): len(d["scenes"][sid][f"side{side}"]["confirmed"])
            for sid in d["scenes"] for side in (1, 2)}


def main() -> None:
    from .learned_baseline import DATASET, load_dataset, score_arm
    samples = load_dataset(DATASET)
    scene_ids = sorted({s.scene_id for s in samples})
    sources = {BASELINE: CLASSICAL_JSON}
    for split in ("scene", "family"):
        path = RESULTS / f"learned_detections_{split}.json"
        if path.is_file():
            sources[f"learned ({split}-disjoint)"] = path
    per_scene = {a: score_arm(_counts(p), samples, scene_ids)[1]["per_scene"]
                 for a, p in sources.items()}
    rows, (lo, hi), n_bundles = bundle_bootstrap(per_scene, scene_ids)

    lines = [
        "# Learned baseline: bootstrap intervals",
        "",
        "Regenerate: `python -m pairbench.experiments.learned_significance`. "
        "Seed-0 detections from `learned_baseline` and the classical "
        f"detector's output. {len(scene_ids)} scenes in {n_bundles} bundles; "
        "the bootstrap resamples bundles.",
        "",
        f"Classical pooled F1 95% CI: [{lo:.3f}, {hi:.3f}].",
        "",
        "| arm | F1 | delta vs classical | 95% CI | p | separated |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["arm"] == r["baseline"]:
            continue
        lines.append(f"| {r['arm']} | {r['f1']:.3f} | {r['delta']:+.3f} | "
                     f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                     f"{r['p_value']:.3f} | "
                     f"{'yes' if r['significant'] else 'no'} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
