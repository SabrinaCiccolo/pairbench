"""Does a pairing arm's accuracy survive a change of upstream detector?

Reads the per-scene correspondence scorings of the `pool` and `ccorr` detectors
(same ground truth), restricts to scenes both resolved, and reports per arm the
transfer delta `recall(pool) - recall(ccorr)` with a paired bundle bootstrap,
plus the difference of differences against the order-anchored arm (Holm).

Usage: python -m pairbench.experiments.detector_transfer
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from ..significance import (N_BOOT, SEED, bootstrap_p_value, holm_reject,
                            sum_by_bundle,
                            percentile_ci)
from .common import RESULTS

POOL_CSV = RESULTS / "correspondence_by_scene.csv"
CCORR_CSV = RESULTS / "correspondence_b1_detections_ccorr_by_scene.csv"
OUT_MD = RESULTS / "detector_transfer.md"
OUT_CSV = RESULTS / "detector_transfer.csv"
OUT_JSON = RESULTS / "detector_transfer.json"

# `greedy_index` is excluded: the offline calibration's world-X bias puts every
# candidate outside its length gate, so it scores 0 under either detector.
ARMS = ["greedy_index_nolen", "greedy_index_debiased", "greedy_index_order",
        "hungarian", "hungarian_aligned", "monotone_aligned", "hybrid_aligned"]

# The order-anchored arm (instance order plus its twist and tilt checks) and
# the same pairing with every check off.
REFERENCE = "greedy_index_nolen"
ORDER_ONLY = "greedy_index_order"

# confirmed detections only, on both sides
POOL_LABEL = "B1 confirmed"


def _load(path, want_label=None):
    """({arm: {scene: (correct, resolved)}}, label) for one scoring `label`
    of a per-scene correspondence CSV; None requires the file to hold one."""
    rows = list(csv.DictReader(open(path)))
    labels = sorted({r["label"] for r in rows})
    if want_label is None:
        if len(labels) != 1:
            raise SystemExit(f"{path} holds several scorings {labels}; "
                             "name the one to use")
        want_label = labels[0]
    elif want_label not in labels:
        raise SystemExit(f"{path} has no scoring {want_label!r}; has {labels}")
    out = defaultdict(dict)
    for r in rows:
        if r["label"] != want_label:
            continue
        out[r["arm"]][r["scene_id"]] = (int(r["correct"]), int(r["resolved"]))
    return out, want_label


def _pooled(num, den):
    d = den.sum()
    return float(num.sum() / d) if d > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    pool, pool_label = _load(POOL_CSV, POOL_LABEL)
    ccorr, ccorr_label = _load(CCORR_CSV)
    missing = [a for a in ARMS if a not in pool or a not in ccorr]
    if missing:
        raise SystemExit(f"arms missing from one of the scorings: {missing}")

    # scenes with resolved > 0 under both detectors (arm-independent)
    scenes = sorted(
        s for s in set(pool[REFERENCE]) & set(ccorr[REFERENCE])
        if pool[REFERENCE][s][1] > 0 and ccorr[REFERENCE][s][1] > 0)
    n = len(scenes)
    if n == 0:
        raise SystemExit("no scenes resolved under both detectors")

    num = {d: {} for d in ("pool", "ccorr")}
    den = {}
    for tag, src in (("pool", pool), ("ccorr", ccorr)):
        den[tag] = np.array([src[REFERENCE][s][1] for s in scenes], dtype=float)
        for a in ARMS:
            num[tag][a] = np.array([src[a][s][0] for s in scenes], dtype=float)

    point = {tag: {a: _pooled(num[tag][a], den[tag]) for a in ARMS}
             for tag in ("pool", "ccorr")}
    delta = {a: point["pool"][a] - point["ccorr"][a] for a in ARMS}

    # resample bundles, not scenes: repeat captures of one unchanged bundle
    # move together (significance.sum_by_bundle); pooled ratios are unchanged
    tags = ("pool", "ccorr")
    bundles, *summed = sum_by_bundle(
        scenes, *(den[t] for t in tags), *(num[t][a] for t in tags for a in ARMS))
    bden = dict(zip(tags, summed[:2]))
    bnum = {t: {} for t in tags}
    k = 2
    for t in tags:
        for a in ARMS:
            bnum[t][a] = summed[k]
            k += 1
    nb = len(bundles)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, nb, size=(args.n_boot, nb))
    den_b = {tag: bden[tag][idx].sum(axis=1) for tag in bden}
    boot_delta = {}
    for a in ARMS:
        r_pool = bnum["pool"][a][idx].sum(axis=1) / np.maximum(den_b["pool"], 1e-12)
        r_cc = bnum["ccorr"][a][idx].sum(axis=1) / np.maximum(den_b["ccorr"], 1e-12)
        boot_delta[a] = r_pool - r_cc

    rows = []
    for a in ARMS:
        lo, hi = percentile_ci(boot_delta[a])
        rows.append({
            "arm": a, "recall_pool": point["pool"][a],
            "recall_ccorr": point["ccorr"][a], "delta": delta[a],
            "ci_lo": lo, "ci_hi": hi,
            "p_value": bootstrap_p_value(boot_delta[a]),
            "moves": bool(lo > 0.0 or hi < 0.0),
            "dod_vs_reference": float("nan"), "dod_ci_lo": float("nan"),
            "dod_ci_hi": float("nan"), "dod_p": float("nan"),
            "dod_moves": False, "dod_holm": False,
        })

    # difference of differences; positive = more detector-stable than REFERENCE
    dod_p = {}
    for row in rows:
        a = row["arm"]
        if a == REFERENCE:
            continue
        d = np.abs(boot_delta[REFERENCE]) - np.abs(boot_delta[a])
        lo, hi = percentile_ci(d)
        row["dod_vs_reference"] = abs(delta[REFERENCE]) - abs(delta[a])
        row["dod_ci_lo"], row["dod_ci_hi"] = lo, hi
        row["dod_p"] = bootstrap_p_value(d)
        row["dod_moves"] = bool(lo > 0.0 or hi < 0.0)
        dod_p[a] = row["dod_p"]
    holm = holm_reject(dod_p)
    for row in rows:
        row["dod_holm"] = bool(holm.get(row["arm"], False))

    rows.sort(key=lambda r: abs(r["delta"]))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    OUT_JSON.write_text(json.dumps({
        "n_scenes": n, "n_bundles": nb, "n_boot": args.n_boot,
        "reference": REFERENCE, "rows": rows,
        "pool_scoring": pool_label, "ccorr_scoring": ccorr_label,
        "claims_pool": int(den["pool"].sum()),
        "claims_ccorr": int(den["ccorr"].sum()),
    }, indent=1))

    lines = [
        "# Detector transfer — which pairing arms survive a change of detector",
        "",
        "Regenerate: `python -m pairbench.experiments.detector_transfer`.",
        "",
        f"Scorings `{pool_label}` and `{ccorr_label}` (confirmed detections "
        f"only), on the {n} scenes where both detectors resolved at least one "
        f"claim ({int(den['pool'].sum())} resolved claims under `pool`, "
        f"{int(den['ccorr'].sum())} under `ccorr`). Recall is pooled over "
        f"each detector's own resolved claims. `greedy_index` is excluded: "
        f"the calibration's world-X bias puts every candidate outside its "
        f"length gate under either detector.",
        "",
        "## Transfer delta per arm",
        "",
        "`delta` = recall(pool) - recall(ccorr). Paired bootstrap over "
        f"{nb} bundles ({args.n_boot} replicates, one draw shared by both "
        "detectors and every arm); `moves?` = interval excludes zero.",
        "",
        "| arm | recall (pool) | recall (ccorr) | delta | 95% paired CI | p | moves? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['arm']}` | {r['recall_pool']:.3f} | {r['recall_ccorr']:.3f} "
            f"| {r['delta']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
            f"| {r['p_value']:.4f} | {'yes' if r['moves'] else 'no'} |")

    ref = next(r for r in rows if r["arm"] == REFERENCE)
    order = next(r for r in rows if r["arm"] == ORDER_ONLY)
    stable = [r["arm"] for r in rows
              if r["arm"] not in (REFERENCE, ORDER_ONLY) and not r["moves"]]
    lines += [
        "",
        "## Difference of differences vs the order anchor",
        "",
        f"|delta({REFERENCE})| - |delta(arm)| on the same paired draw; "
        f"positive = the arm is more detector-stable. Holm-Bonferroni over "
        f"{len(dod_p)} comparisons, alpha 0.05.",
        "",
        "| arm | |delta| | advantage over the order anchor | 95% paired CI | p | separates? | survives Holm? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["arm"] == REFERENCE:
            continue
        lines.append(
            f"| `{r['arm']}` | {abs(r['delta']):.3f} | "
            f"{r['dod_vs_reference']:+.3f} | "
            f"[{r['dod_ci_lo']:+.3f}, {r['dod_ci_hi']:+.3f}] | {r['dod_p']:.4f} "
            f"| {'yes' if r['dod_moves'] else 'no'} "
            f"| {'yes' if r['dod_holm'] else 'no'} |")

    lines += [
        "",
        "## Reading",
        "",
        f"`{REFERENCE}` (the order-anchored arm: i-th face to i-th face, then "
        f"its twist and tilt checks) has delta {ref['delta']:+.3f}. "
        f"`{ORDER_ONLY}` (same ordering, no checks) has delta "
        f"{order['delta']:+.3f} ({order['recall_pool']:.3f} -> "
        f"{order['recall_ccorr']:.3f}; interval [{order['ci_lo']:+.3f}, "
        f"{order['ci_hi']:+.3f}]), so the difference between them is due to "
        f"the checks.",
        "",
        "Arms whose transfer interval includes zero: "
        + (", ".join(f"`{a}`" for a in stable) if stable else "none")
        + ". Per-detector rankings are in `correspondence_report.md` and "
        "`correspondence_report_b1_detections_ccorr.md`.",
        "",
        "The correspondence anchors were annotated from `pool` output, so "
        "re-linking `ccorr` faces to them is less exact and resolves fewer "
        "claims; the comparison is within-column stability, not a "
        "cross-detector ranking.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.name}, {OUT_CSV.name}, {OUT_JSON.name}")
    for r in rows:
        print(f"  {r['arm']:24s} pool {r['recall_pool']:.3f} "
              f"ccorr {r['recall_ccorr']:.3f} delta {r['delta']:+.3f}")


if __name__ == "__main__":
    main()
