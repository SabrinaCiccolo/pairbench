"""Learned detection baseline vs the classical `pool` detector.

Trains the CPU centre-point CNN of `pairbench.detect.learned` under a 5-fold
scene-disjoint split (`--splits scene`) and a leave-one-family-out split
(`--splits family`; l-profile and l-profile-reshoot form one group), with the
threshold picked on an inner validation split, and scores both arms with the
ground truth and metrics of `experiments.detect`.

Usage: python -m pairbench.experiments.learned_baseline [--steps N] [--seeds N]
"""

import argparse
import dataclasses
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT
from ..detect.ccorr import dets_from_json
from ..gt import (
    ANCHOR_GATE_MM,
    load_gt_correspondence,
    load_gt,
    parse_anchors,
)
from ..detect.learned import (
    THRESHOLD_GRID,
    SceneSide,
    count_counts,
    group_folds,
    load_dataset,
    peaks_to_detections,
    pick_threshold,
    positional_counts,
    predict_peaks,
    scene_folds,
    train_model,
    write_csv,
)
from ..metrics import precision_recall_f1 as prf
from .learned_significance import bundle_bootstrap
from .common import RESULTS

DATASET = REPO_ROOT / "data" / "learned" / "dataset.npz"


# --------------------------------------------------------------------------
# scoring, shared by both arms
# --------------------------------------------------------------------------

def score_arm(per_side_counts: dict[tuple[str, int], int],
              samples: list[SceneSide], scene_ids: list[str],
              ) -> tuple[dict, dict[str, np.ndarray]]:
    """Count P/R/F1 pooled and per family, plus per-scene (tp, fp, fn)
    triples for the paired bootstrap."""
    by_scene = {sid: [0, 0, 0] for sid in scene_ids}
    by_family = defaultdict(lambda: [0, 0, 0])
    for s in samples:
        if s.scene_id not in by_scene:
            continue
        tp, fp, fn = count_counts(per_side_counts[(s.scene_id, s.side)], s.gt_count)
        for k, v in enumerate((tp, fp, fn)):
            by_scene[s.scene_id][k] += v
            by_family[s.family][k] += v
    tot = np.array([by_scene[sid] for sid in scene_ids], dtype=np.int64)
    p, r, f1 = prf(*tot.sum(axis=0))
    fam = {}
    for f, c in sorted(by_family.items()):
        fam[f] = prf(*c) + (c[0] + c[2],)
    return ({"P": p, "R": r, "F1": f1, "tp": int(tot[:, 0].sum()),
             "fp": int(tot[:, 1].sum()), "fn": int(tot[:, 2].sum()),
             "by_family": fam}, {"per_scene": tot})


def positional_score(dets_by_side: dict[tuple[str, int], list],
                     corr: dict[str, dict], scene_ids: list[str]) -> dict:
    """Fraction of hand-confirmed anchor positions that still have an
    unambiguous detection nearby, run identically on both arms."""
    linked = total = 0
    per_family = defaultdict(lambda: [0, 0])
    for sid in scene_ids:
        c = corr.get(sid)
        if not c:
            continue
        fam = sid.split("/")[0]
        for side in (1, 2):
            anchors = parse_anchors(c[f"anchors_{side}"])
            got, n = positional_counts(dets_by_side.get((sid, side), []),
                                       anchors, ANCHOR_GATE_MM)
            linked += got
            total += n
            per_family[fam][0] += got
            per_family[fam][1] += n
    return {"linked": linked, "total": total,
            "recall": linked / total if total else 0.0,
            "by_family": {f: (v[0], v[1], v[0] / v[1] if v[1] else 0.0)
                          for f, v in sorted(per_family.items())}}


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------

def inner_val_split(train_ids: list[str], samples: list[SceneSide],
                    split: str, rng: np.random.Generator
                    ) -> tuple[list[str], list[str]]:
    """Returns (fit ids, val ids) from a fold's training scenes. For the
    family split, validation is a whole remaining group, so the threshold is
    tuned under the same unseen-cross-section transfer the test fold uses."""
    groups = defaultdict(list)
    for s in samples:
        if s.scene_id in set(train_ids):
            groups[s.group].append(s.scene_id)
    groups = {g: sorted(set(v)) for g, v in groups.items()}
    if split == "family" and len(groups) >= 2:
        val_group = min(groups, key=lambda g: (len(groups[g]), g))
        val = groups[val_group]
        fit = sorted(set(train_ids) - set(val))
        return fit, val
    ids = sorted(train_ids)
    n_val = max(1, int(round(0.15 * len(ids))))
    perm = rng.permutation(len(ids))
    val = sorted(ids[i] for i in perm[:n_val])
    return sorted(set(ids) - set(val)), val


def run_split(samples: list[SceneSide], split: str, steps: int, seed: int,
              threads: int, verbose: bool = True) -> dict:
    """Trains and predicts every fold of one split. Returns per-scene-side
    peaks, the per-fold thresholds, and timing."""
    scene_ids = sorted({s.scene_id for s in samples})
    if split == "family":
        folds = group_folds(samples)
    else:
        folds = [(f"fold{k}", ids) for k, ids in
                 enumerate(scene_folds(scene_ids, n_folds=5, seed=seed))]

    rng = np.random.default_rng(seed)
    peaks: dict[tuple[str, int], np.ndarray] = {}
    in_sample: list[tuple[int, int, int]] = []
    thresholds: dict[str, float] = {}
    fold_rows = []
    train_seconds = 0.0
    infer_seconds = 0.0
    n_infer = 0

    for name, test_ids in folds:
        test_set = set(test_ids)
        train_ids = [s for s in scene_ids if s not in test_set]
        fit_ids, val_ids = inner_val_split(train_ids, samples, split, rng)
        # A scene-side with no labels is dropped from fitting, kept in scoring.
        fit = [s for s in samples if s.scene_id in set(fit_ids) and len(s.centers_px)]
        val = [s for s in samples if s.scene_id in set(val_ids)]
        test = [s for s in samples if s.scene_id in test_set]

        if verbose:
            print(f"[{split}/{name}] fit {len(fit)} sides "
                  f"({len(set(s.scene_id for s in fit))} scenes), "
                  f"val {len(val)}, test {len(test)}", flush=True)
        t0 = time.perf_counter()
        model, losses = train_model(fit, steps=steps, seed=seed, threads=threads)
        train_seconds += time.perf_counter() - t0

        val_peaks = {(s.scene_id, s.side): predict_peaks(model, s.image) for s in val}
        thr, val_f1 = pick_threshold(val_peaks, val)
        thresholds[name] = thr

        t0 = time.perf_counter()
        for s in test:
            peaks[(s.scene_id, s.side)] = predict_peaks(model, s.image)
        infer_seconds += time.perf_counter() - t0
        n_infer += len(test)
        # In-sample: this fold's own model on this fold's own fit scenes.
        # Accumulated per fold, since a scene appears in several folds' fit sets.
        for s in fit:
            n = int((predict_peaks(model, s.image)[:, 2] >= thr).sum())
            in_sample.append(count_counts(n, s.gt_count))

        fold_rows.append({
            "split": split, "seed": seed, "fold": name,
            "n_fit_scenes": len(set(s.scene_id for s in fit)),
            "n_fit_faces": int(sum(len(s.centers_px) for s in fit)),
            "n_val_scenes": len(val_ids), "n_test_scenes": len(test_ids),
            "threshold": thr, "val_f1": round(val_f1, 4),
            "final_loss": round(float(np.mean(losses[-50:])), 4),
        })
        if verbose:
            print(f"    thr {thr:.3f} (inner-val count-F1 {val_f1:.3f}), "
                  f"final loss {fold_rows[-1]['final_loss']:.3f}", flush=True)

    in_arr = np.array(in_sample, dtype=np.int64).reshape(-1, 3).sum(axis=0)
    return {"peaks": peaks, "in_sample_f1": prf(*in_arr)[2],
            "thresholds": thresholds, "folds": fold_rows,
            "fold_of": {sid: name for name, ids in folds for sid in ids},
            "train_seconds": train_seconds,
            "infer_seconds_per_side": infer_seconds / max(n_infer, 1)}


def counts_at(peaks: dict, samples: list[SceneSide], thr_of_scene) -> dict:
    out = {}
    for s in samples:
        p = peaks.get((s.scene_id, s.side))
        if p is None:
            continue
        out[(s.scene_id, s.side)] = int((p[:, 2] >= thr_of_scene(s.scene_id)).sum())
    return out


def convergence_curve(samples: list[SceneSide], budgets: list[int], seed: int,
                      threads: int) -> list[dict]:
    """Re-trains one scene-disjoint fold at several step budgets, to check
    whether held-out F1 is still rising at the step budget the main run uses.
    Each budget is a fresh model."""
    scene_ids = sorted({s.scene_id for s in samples})
    test_ids = set(scene_folds(scene_ids, n_folds=5, seed=seed)[0])
    train_ids = [s for s in scene_ids if s not in test_ids]
    fit_ids, val_ids = inner_val_split(train_ids, samples, "scene",
                                       np.random.default_rng(seed))
    fit = [s for s in samples if s.scene_id in set(fit_ids) and len(s.centers_px)]
    val = [s for s in samples if s.scene_id in set(val_ids)]
    test = [s for s in samples if s.scene_id in test_ids]

    rows = []
    for steps in budgets:
        t0 = time.perf_counter()
        model, _ = train_model(fit, steps=steps, seed=seed, threads=threads)
        thr, val_f1 = pick_threshold(
            {(s.scene_id, s.side): predict_peaks(model, s.image) for s in val}, val)
        counts = {(s.scene_id, s.side):
                  int((predict_peaks(model, s.image)[:, 2] >= thr).sum())
                  for s in test}
        stats, _ = score_arm(counts, test, sorted(test_ids))
        rows.append({"steps": steps, "threshold": thr,
                     "val_f1": round(val_f1, 4), "P": round(stats["P"], 4),
                     "R": round(stats["R"], 4), "F1": round(stats["F1"], 4),
                     "train_seconds": round(time.perf_counter() - t0, 1)})
        print(f"  convergence steps={steps}: held-out F1 {stats['F1']:.3f} "
              f"(thr {thr:.3f}, {rows[-1]['train_seconds']:.0f}s)", flush=True)
    return rows


def oracle_f1(peaks: dict, samples: list[SceneSide], scene_ids: list[str]) -> float:
    """Best pooled count-F1 over a single global threshold chosen on the test
    predictions — an upper bound, not an achievable protocol."""
    best = 0.0
    for thr in THRESHOLD_GRID:
        counts = counts_at(peaks, samples, lambda _sid, t=thr: t)
        stats, _ = score_arm(counts, samples, scene_ids)
        best = max(best, stats["F1"])
    return best


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def verdict_lines(cls_stats: dict, cls_pos: dict, summary: dict, boot: list,
                  conv_rows: list, splits: list[str], seeds: int,
                  fold_rows: list[dict], scenes_per_fam: dict) -> list[str]:
    """Builds the report's summary section from the measured numbers."""
    out = ["## Summary", ""]
    boot_by_arm = {r["arm"]: r for r in boot}
    cls_f1 = cls_stats["F1"]

    for split in splits:
        s = summary[split]
        f1 = float(np.mean(s["f1_seeds"]))
        row = boot_by_arm.get(f"learned ({split}-disjoint)")
        label = ("Scene-disjoint" if split == "scene"
                 else "Family-disjoint (held-out family)")
        sep = ""
        if row is not None:
            sep = (f" Paired bundle bootstrap: delta {row['delta']:+.3f}, "
                   f"95% CI [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}], "
                   f"p = {row['p_value']:.3f} "
                   f"({'separated' if row['significant'] else 'not separated'}).")
        out += [f"**{label}:** held-out F1 {f1:.3f} (mean over seeds) vs "
                f"classical {cls_f1:.3f}.{sep}", ""]
    out += ["F1 here is the mean over seeds; the Headline table reports the "
            "seed-0 pooled F1.", ""]

    if "scene" in summary and "family" in summary:
        sc = float(np.mean(summary["scene"]["f1_seeds"]))
        fa = float(np.mean(summary["family"]["f1_seeds"]))
        ins = float(np.mean(summary["scene"]["in_sample_f1_seeds"]))
        out += [f"In-sample F1 {ins:.3f}, scene-disjoint {sc:.3f}, "
                f"family-disjoint {fa:.3f} (difference {sc - fa:+.3f}). The "
                "classical arm needs only the family's DXF template.", ""]

    pos_lines = []
    for split in splits:
        pos_lines.append(f"{split}-disjoint {summary[split]['pos']['recall']:.3f}")
    out += [f"Position-anchored recall: classical {cls_pos['recall']:.3f}, "
            + ", ".join(pos_lines) + ".", ""]

    if conv_rows and len(conv_rows) >= 2:
        first, last = conv_rows[0], conv_rows[-1]
        out += [f"Held-out F1 moves {last['F1'] - first['F1']:+.3f} between "
                f"{first['steps']} and {last['steps']} training steps (see "
                "'Was it trained long enough?').", ""]

    seed0_folds = [r for r in fold_rows if r["seed"] == 0]
    fold_size_parts = []
    for split in splits:
        srows = [r for r in seed0_folds if r["split"] == split]
        if not srows:
            continue
        smin = min(r["n_fit_scenes"] for r in srows)
        smax = max(r["n_fit_scenes"] for r in srows)
        fmin = min(r["n_fit_faces"] for r in srows)
        fmax = max(r["n_fit_faces"] for r in srows)
        scenes_str = f"{smin} scenes" if smin == smax else f"{smin}-{smax} scenes"
        faces_str = f"{fmin} faces" if fmin == fmax else f"{fmin}-{fmax} faces"
        fold_size_parts.append(f"{scenes_str} / {faces_str} in the {split} split")
    fold_size_desc = ("; ".join(fold_size_parts) if fold_size_parts
                       else "an unmeasured number of scenes/faces")
    fam_sizes = sorted(scenes_per_fam.values())
    fam_desc = (f"{len(fam_sizes)} profile families, the smallest "
                f"{fam_sizes[0]} scenes" if fam_sizes
                else "an unmeasured number of profile families")
    out += [f"Training folds (seed 0): {fold_size_desc}; {fam_desc}; "
            f"{seeds} seed(s).", ""]
    return out


def classical_infer_s_per_side(report_path: Path) -> float | None:
    """Seconds per scene-side of the `pool` detector, read from the
    `detect_total` row of its timing table; None if not found."""
    if not report_path.exists():
        return None
    text = report_path.read_text()
    m = re.search(r"\|\s*detect_total\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|",
                  text)
    if not m:
        return None
    return float(m.group(1)) / 1000.0


def fmt_family_table(stats: dict, scenes_per_fam: dict) -> list[str]:
    lines = ["| family | scenes | P | R | F1 | faces |", "|---|---|---|---|---|---|"]
    for f, (p, r, f1, faces) in stats["by_family"].items():
        lines.append(f"| {f} | {scenes_per_fam[f]} | {p:.3f} | {r:.3f} "
                     f"| {f1:.3f} | {faces} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["scene", "family"],
                    choices=["scene", "family"])
    ap.add_argument("--convergence-steps", nargs="*", type=int,
                    default=[750, 1500, 3000, 6000],
                    help="step budgets for the one-fold convergence probe "
                         "(empty to skip)")
    ap.add_argument("--classical", type=Path,
                    default=RESULTS / "b1_detections_pool.json")
    ap.add_argument("--out", type=Path, default=RESULTS / "learned_baseline.md")
    args = ap.parse_args()

    samples = load_dataset(args.dataset)
    scene_ids = sorted({s.scene_id for s in samples})
    scenes_per_fam = defaultdict(int)
    for sid in scene_ids:
        scenes_per_fam[sid.split("/")[0]] += 1
    corr = load_gt_correspondence(RESULTS / "gt_correspondence.csv")
    gt2 = load_gt(RESULTS / "gt.csv")

    # ---- classical arm, read from its detection output ---------------------
    classical = json.loads(args.classical.read_text())
    cls_counts, cls_dets = {}, {}
    for sid, r in classical["scenes"].items():
        for side in (1, 2):
            dets = dets_from_json(r[f"side{side}"]["confirmed"])
            cls_dets[(sid, side)] = dets
            cls_counts[(sid, side)] = len(dets)
    cls_stats, cls_scene = score_arm(cls_counts, samples, scene_ids)
    cls_pos = positional_score(cls_dets, corr, scene_ids)

    # ---- learned arm, one run per split per seed ---------------------------
    runs = {}
    for split in args.splits:
        for seed in range(args.seeds):
            print(f"\n=== split={split} seed={seed} ===", flush=True)
            runs[(split, seed)] = run_split(samples, split, args.steps, seed,
                                            args.threads)

    conv_rows = []
    if args.convergence_steps:
        print("\n=== convergence probe (one scene-disjoint fold) ===", flush=True)
        conv_rows = convergence_curve(samples, args.convergence_steps, 0,
                                      args.threads)
        write_csv(RESULTS / "learned_convergence.csv", conv_rows)

    fold_rows = [r for v in runs.values() for r in v["folds"]]
    write_csv(RESULTS / "learned_folds.csv", fold_rows)

    summary = {}
    counts_by_split: dict[str, dict[tuple[str, int], int]] = {}
    per_scene_arms = {"classical (pool)": cls_scene["per_scene"]}
    for split in args.splits:
        seed_stats, seed_pos, seed_oracle, seed_insample = [], [], [], []
        for seed in range(args.seeds):
            run = runs[(split, seed)]
            thr_of = lambda sid, run=run: run["thresholds"][run["fold_of"][sid]]
            counts = counts_at(run["peaks"], samples, thr_of)
            stats, scene_arr = score_arm(counts, samples, scene_ids)
            dets = {}
            for s in samples:
                p = run["peaks"].get((s.scene_id, s.side))
                if p is not None:
                    dets[(s.scene_id, s.side)] = peaks_to_detections(
                        p, s, thr_of(s.scene_id))
            pos = positional_score(dets, corr, scene_ids)
            seed_stats.append((stats, scene_arr["per_scene"]))
            seed_pos.append(pos)
            seed_oracle.append(oracle_f1(run["peaks"], samples, scene_ids))
            seed_insample.append(run["in_sample_f1"])
            if seed == 0:
                per_scene_arms[f"learned ({split}-disjoint)"] = scene_arr["per_scene"]
                counts_by_split[split] = counts
                # detections in the classical runners' schema, for rescoring
                dump = {"split": split, "seed": seed,
                        "steps": args.steps,
                        "thresholds": run["thresholds"], "scenes": {}}
                for sid in scene_ids:
                    dump["scenes"][sid] = {
                        f"side{side}": {
                            "confirmed": [dataclasses.asdict(d)
                                          for d in dets.get((sid, side), [])],
                            "lost": [],
                        } for side in (1, 2)}
                (RESULTS / f"learned_detections_{split}.json").write_text(
                    json.dumps(dump, indent=1))
        summary[split] = {
            "stats": seed_stats[0][0],
            "f1_seeds": [s["F1"] for s, _ in seed_stats],
            "pos": seed_pos[0],
            "pos_seeds": [p["recall"] for p in seed_pos],
            "oracle_f1_seeds": seed_oracle,
            "in_sample_f1_seeds": seed_insample,
            "train_seconds": np.mean([runs[(split, s)]["train_seconds"]
                                      for s in range(args.seeds)]),
            "infer_s_per_side": np.mean([runs[(split, s)]["infer_seconds_per_side"]
                                         for s in range(args.seeds)]),
            "thresholds": runs[(split, 0)]["thresholds"],
        }

    # bundle resampling, shared with experiments.learned_significance so the
    # two agree; that module recomputes these intervals without retraining
    boot, (lo_c, hi_c), _ = bundle_bootstrap(per_scene_arms, scene_ids)

    # ---- report -----------------------------------------------------------
    n_lab = sum(len(s.centers_px) for s in samples)
    n_gt = sum(s.gt_count for s in samples)
    L = []
    a = L.append
    a("# Learned detection baseline vs the classical detector")
    a("")
    a(f"Generated by `pairbench.experiments.learned_baseline`, "
      f"{len(scene_ids)} scenes / {len(samples)} scene-sides / {n_gt} GT faces. "
      f"{args.seeds} seed(s) x {args.steps} training steps per fold, CPU only "
      f"({args.threads} threads).")
    a("")
    L.extend(verdict_lines(cls_stats, cls_pos, summary, boot, conv_rows,
                           args.splits, args.seeds, fold_rows,
                           scenes_per_fam))
    a("")
    a("## What is being compared")
    a("")
    a("Both arms are scored on `gt.csv`'s per-side counts (TP per side = "
      "min(pred, gt), as in `experiments.detect`) and on the position-anchored "
      "re-link against `gt_correspondence.csv`. Classical numbers are read "
      f"from `{args.classical.name}`.")
    a("")
    a(f"Training labels are the {n_lab} anchors in `gt_correspondence.csv` "
      f"({n_lab / n_gt:.1%} of the {n_gt} GT faces). The other "
      f"{n_gt - n_lab} ({1 - n_lab / n_gt:.1%}) were never produced by the "
      "classical detector, so they are unlabelled (background) in training; "
      "scoring uses the complete `gt.csv` counts.")
    a("")
    a("## Headline")
    a("")
    a("| arm | split | P | R | F1 |")
    a("|---|---|---|---|---|")
    a(f"| classical `pool` (default) | none — no training set | "
      f"{cls_stats['P']:.3f} | {cls_stats['R']:.3f} | **{cls_stats['F1']:.3f}** |")
    for split in args.splits:
        s = summary[split]["stats"]
        f1s = summary[split]["f1_seeds"]
        spread = (f" (seeds {min(f1s):.3f}-{max(f1s):.3f})"
                  if len(f1s) > 1 else "")
        label = ("scene-disjoint 5-fold" if split == "scene"
                 else "family-disjoint LOGO (held-out family)")
        a(f"| learned CenterNet | {label} | {s['P']:.3f} | {s['R']:.3f} "
          f"| **{s['F1']:.3f}**{spread} |")
    a("")
    a(f"Classical pooled F1 95% CI (bundle bootstrap): "
      f"[{lo_c:.3f}, {hi_c:.3f}].")
    a("")
    a("### Paired bootstrap vs the classical arm")
    a("")
    a("Bundles resampled (repeat captures of one unchanged bundle move "
      "together), every arm scored on the same draw (seed 0 run).")
    a("")
    a("| arm | F1 | delta | 95% CI | p | separated |")
    a("|---|---|---|---|---|---|")
    for row in boot:
        if row["arm"] == row["baseline"]:
            continue
        a(f"| {row['arm']} | {row['f1']:.3f} | {row['delta']:+.3f} "
          f"| [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] | {row['p_value']:.3f} "
          f"| {'yes' if row['significant'] else 'no'} |")
    a("")
    a("## Generalization: in-sample vs held-out vs oracle threshold")
    a("")
    a("| split | in-sample F1 (fit scenes) | held-out F1 | held-out F1, "
      "oracle threshold |")
    a("|---|---|---|---|")
    for split in args.splits:
        s = summary[split]
        a(f"| {split}-disjoint | {np.mean(s['in_sample_f1_seeds']):.3f} "
          f"| {np.mean(s['f1_seeds']):.3f} "
          f"| {np.mean(s['oracle_f1_seeds']):.3f} |")
    a("")
    a("The oracle column tunes one global threshold on the test fold itself "
      "(an upper bound, not an achievable protocol).")
    a("")
    a("## Position-anchored recall")
    a("")
    a("Fraction of `gt_correspondence.csv` anchors that re-link to a "
      "detection under `gt.resolve_anchor_indices` with the default "
      f"{ANCHOR_GATE_MM:.0f} mm gate, as in the correspondence scorer.")
    a("")
    a("| arm | linked | anchors | recall |")
    a("|---|---|---|---|")
    a(f"| classical `pool` | {cls_pos['linked']} | {cls_pos['total']} "
      f"| **{cls_pos['recall']:.3f}** |")
    for split in args.splits:
        p = summary[split]["pos"]
        a(f"| learned ({split}-disjoint) | {p['linked']} | {p['total']} "
          f"| **{p['recall']:.3f}** |")
    a("")
    a("Anchor coordinates are centroids of classical detections, so this "
      "metric favours the classical arm; the count metric has no such bias.")
    a("")
    a("## Where the learned arm's errors are")
    a("")
    a("Per scene-side count error against `gt.csv`, seed 0, worst first. "
      "`labels` is how many of that side's faces were available to train on "
      "anywhere in the dataset.")
    a("")
    a("| scene | side | cat | gt | labels | classical | learned "
      f"({args.splits[0]}) | error |")
    a("|---|---|---|---|---|---|---|---|")
    worst = sorted(
        ((s, counts_by_split[args.splits[0]].get((s.scene_id, s.side), 0))
         for s in samples),
        key=lambda t: -abs(t[1] - t[0].gt_count))[:15]
    for s, n in worst:
        a(f"| {s.scene_id} | {s.side} | {gt2[s.scene_id]['category']} "
          f"| {s.gt_count} | {len(s.centers_px)} "
          f"| {cls_counts[(s.scene_id, s.side)]} | {n} "
          f"| {n - s.gt_count:+d} |")
    a("")
    a("## By profile family")
    a("")
    a("l-profile and l-profile-reshoot share a profile and form one group in "
      "the family-disjoint split.")
    a("")
    a("### classical `pool`")
    a("")
    L.extend(fmt_family_table(cls_stats, scenes_per_fam))
    for split in args.splits:
        a("")
        a(f"### learned, {split}-disjoint")
        a("")
        L.extend(fmt_family_table(summary[split]["stats"], scenes_per_fam))
    a("")
    a("## Folds and thresholds")
    a("")
    a("| split | fold | fit scenes | fit faces | test scenes | threshold "
      "| inner-val F1 |")
    a("|---|---|---|---|---|---|---|")
    for r in fold_rows:
        if r["seed"] != 0:
            continue
        a(f"| {r['split']} | {r['fold']} | {r['n_fit_scenes']} "
          f"| {r['n_fit_faces']} | {r['n_test_scenes']} | {r['threshold']:.3f} "
          f"| {r['val_f1']:.3f} |")
    if conv_rows:
        a("")
        a("## Was it trained long enough?")
        a("")
        a("One scene-disjoint fold, re-trained from scratch at several step "
          "budgets (`learned_convergence.csv`). The main run uses "
          f"{args.steps} steps per fold.")
        a("")
        a("| steps | inner-val threshold | held-out P | R | F1 | train time |")
        a("|---|---|---|---|---|---|")
        for r in conv_rows:
            a(f"| {r['steps']} | {r['threshold']:.3f} | {r['P']:.3f} "
              f"| {r['R']:.3f} | {r['F1']:.3f} | {r['train_seconds'] / 60:.1f} min |")
        first, last = conv_rows[0], conv_rows[-1]
        a("")
        a(f"From {first['steps']} to {last['steps']} steps the held-out F1 "
          f"moves {first['F1']:.3f} -> {last['F1']:.3f} at "
          f"{last['train_seconds'] / first['train_seconds']:.1f}x the training "
          "cost.")
    a("")
    a("## Cost")
    a("")
    a("| arm | train | inference |")
    a("|---|---|---|")
    cls_infer_s = classical_infer_s_per_side(RESULTS / "b1_report_pool.md")
    cls_infer_str = (f"{cls_infer_s:.2f} s/side" if cls_infer_s is not None
                      else "unknown (`b1_report_pool.md` timing table not found)")
    a(f"| classical `pool` | none (the DXF template is given) | {cls_infer_str} "
      "(`b1_report_pool.md`) |")
    for split in args.splits:
        s = summary[split]
        a(f"| learned ({split}) | {s['train_seconds'] / 60:.1f} min for all "
          f"folds, CPU | {s['infer_s_per_side'] * 1000:.0f} ms/side |")
    a("")
    args.out.write_text("\n".join(L) + "\n")

    rows = []
    for split in args.splits:
        run = runs[(split, 0)]
        thr_of = lambda sid, run=run: run["thresholds"][run["fold_of"][sid]]
        for s in samples:
            p = run["peaks"].get((s.scene_id, s.side))
            if p is None:
                continue
            rows.append({
                "split": split, "scene_id": s.scene_id, "side": s.side,
                "family": s.family, "category": gt2[s.scene_id]["category"],
                "gt": s.gt_count, "n_labels": len(s.centers_px),
                "n_learned": int((p[:, 2] >= thr_of(s.scene_id)).sum()),
                "n_classical": cls_counts[(s.scene_id, s.side)],
                "threshold": thr_of(s.scene_id),
            })
    write_csv(RESULTS / "learned_counts.csv", rows)

    print(f"\nwrote {args.out.name}, learned_counts.csv, learned_folds.csv")
    print(f"classical F1 {cls_stats['F1']:.3f}")
    for split in args.splits:
        print(f"learned {split}-disjoint F1 {summary[split]['stats']['F1']:.3f} "
              f"(seeds {summary[split]['f1_seeds']})")


if __name__ == "__main__":
    main()
