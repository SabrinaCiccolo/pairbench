"""Compares the synthetic twin's clouds against real scans.

Matches real `single`-category scenes with the twin's mirrored `realism` suite
on sampling statistics, per-pixel depth-channel statistics and detector
response (`ccorr_frac`, pool score), compares detection P/R/F1, and writes
side-by-side projection panels per family.

Usage: python -m pairbench.experiments.synth_realism [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import ks_2samp

from ..config import REPO_ROOT
from ..io.calib import load_calibration
from ..detect.pool import detect_faces_pool
from ..io.dxf_template import render_template
from ..io.loader import DATA_ROOT, FAMILY_DXF, load_scene
from ..io.projection import depth_channels, project_yz
from ..synth import load_truth
from .common import RESULTS

GT = RESULTS / "gt.csv"
SYNTH = REPO_ROOT / "data" / "synthetic" / "realism"
OUT_MD = RESULTS / "synthetic_realism.md"
OUT_CSV = RESULTS / "synthetic_realism.csv"
OUT_JSON = RESULTS / "synthetic_realism.json"
PANEL_DIR = RESULTS / "synthetic_realism"

CELL_MM = 0.4

CLOUD_METRICS = ["n_kept", "density_pts_mm2", "area_mm2", "radial_mm", "x_spread_mm"]
PIXEL_METRICS = ["px_density", "px_x_spread_mm", "n_observed_px"]
DET_METRICS = ["ccorr_frac", "score", "n_points"]


def _cloud_stats(world, scanner_pts, side, template):
    keys = set(zip(np.floor(world[:, 1] / CELL_MM).astype(int),
                   np.floor(world[:, 2] / CELL_MM).astype(int)))
    area = len(keys) * CELL_MM * CELL_MM
    proj = project_yz(world, padding_px=template.default_padding)
    ch = depth_channels(world, proj, side)
    obs = ch.observed
    dets, _, _ = detect_faces_pool(world, template, side)
    return {
        "n_kept": float(len(world)),
        "density_pts_mm2": float(len(world) / area) if area else np.nan,
        "area_mm2": float(area),
        "radial_mm": float(np.median(np.linalg.norm(scanner_pts, axis=1))),
        "x_spread_mm": float(world[:, 0].max() - world[:, 0].min()),
        "n_observed_px": float(obs.sum()),
        "_px_density": ch.density[obs].astype(float),
        "_px_x_spread": ch.x_spread_mm[obs].astype(float),
        "_ccorr": np.array([d.ccorr_frac for d in dets], dtype=float),
        "_score": np.array([d.score for d in dets], dtype=float),
        "_npts": np.array([float(d.n_points) for d in dets]),
        "_proj": proj,
        "n_det": float(len(dets)),
    }


def _real_job(job):
    scene_id, dxf = job
    calib = load_calibration()
    template = render_template(dxf)
    clouds = load_scene(DATA_ROOT / scene_id, calib=calib)
    out = []
    for side, world in ((1, clouds.side1_world), (2, clouds.side2_world)):
        scanner = (world - calib.for_side(side).t) @ calib.for_side(side).R
        s = _cloud_stats(world, scanner, side, template)
        s.update(scene_id=scene_id, side=side, source="real", dxf=dxf)
        out.append(s)
    return out


def _synth_job(job):
    scene_dir = Path(job)
    truth = load_truth(scene_dir)
    calib = load_calibration()
    template = render_template(truth["dxf"])
    clouds = load_scene(scene_dir, calib=calib)
    out = []
    for side, world in ((1, clouds.side1_world), (2, clouds.side2_world)):
        scanner = (world - calib.for_side(side).t) @ calib.for_side(side).R
        s = _cloud_stats(world, scanner, side, template)
        s.update(scene_id=truth["scene_id"], side=side, source="synth",
                 dxf=truth["dxf"])
        out.append(s)
    return out


def _pool(rows, key):
    return np.concatenate([r[key] for r in rows]) if rows else np.zeros(0)


def _q(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (np.nan, np.nan, np.nan)
    return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))


def _row(name, real, synth, unit=""):
    rm, rlo, rhi = _q(real)
    sm, slo, shi = _q(synth)
    ratio = sm / rm if rm else np.nan
    ks = ks_2samp(np.asarray(real, float), np.asarray(synth, float))
    return {
        "metric": name, "unit": unit,
        "real_med": rm, "real_iqr": [rlo, rhi],
        "synth_med": sm, "synth_iqr": [slo, shi],
        "ratio": float(ratio), "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
    }


def _panel(real_rows, synth_rows, out_dir: Path):
    """Writes one side-by-side projection PNG per family: real above, twin below."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_dxf_real, by_dxf_synth = {}, {}
    for r in real_rows:
        by_dxf_real.setdefault(r["dxf"], r)
    for r in synth_rows:
        by_dxf_synth.setdefault(r["dxf"], r)
    index = {}
    for dxf in sorted(set(by_dxf_real) & set(by_dxf_synth)):
        panels = []
        for row in (by_dxf_real[dxf], by_dxf_synth[dxf]):
            panels.append(cv2.cvtColor(row["_proj"].image, cv2.COLOR_GRAY2BGR))
        w = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 0, 0, w - p.shape[1],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
                  for p in panels]
        # No caption; ids in panels.json.
        cv2.imwrite(str(out_dir / f"{dxf}.png"), np.vstack(panels))
        index[dxf] = {
            "real_scene_id": by_dxf_real[dxf]["scene_id"],
            "twin_scene_id": by_dxf_synth[dxf]["scene_id"],
            "panel_height_px": int(panels[0].shape[0]),
            "panel_width_px": int(w),
            "order": ["real", "twin"],
        }
    (out_dir / "panels.json").write_text(json.dumps(index, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    from multiprocessing import Pool

    real_jobs = []
    for row in csv.DictReader(open(GT)):
        if row["category"] == "single":
            family = row["scene_id"].split("/")[0]
            real_jobs.append((row["scene_id"], FAMILY_DXF[family]))
    synth_jobs = sorted(str(p.parent) for p in SYNTH.rglob("truth.json"))
    if not synth_jobs:
        raise SystemExit(f"no synthetic scenes under {SYNTH} — run "
                         "python -m pairbench.experiments.build_synthetic_twin --suite realism first")

    real_rows, synth_rows = [], []
    with Pool(args.workers) as pool:
        for out in pool.imap(_real_job, real_jobs):
            real_rows.extend(out)
        print(f"real: {len(real_rows)} clouds")
        for out in pool.imap(_synth_job, synth_jobs):
            synth_rows.extend(out)
        print(f"synth: {len(synth_rows)} clouds")

    table = []
    for m in CLOUD_METRICS + ["n_observed_px", "n_det"]:
        table.append(_row(m, [r[m] for r in real_rows], [r[m] for r in synth_rows]))
    table.append(_row("px_density", _pool(real_rows, "_px_density"),
                      _pool(synth_rows, "_px_density")))
    table.append(_row("px_x_spread_mm", _pool(real_rows, "_px_x_spread"),
                      _pool(synth_rows, "_px_x_spread")))
    for key, name in (("_ccorr", "det ccorr_frac"), ("_score", "det score"),
                      ("_npts", "det n_points")):
        table.append(_row(name, _pool(real_rows, key), _pool(synth_rows, key)))

    _panel(real_rows, synth_rows, PANEL_DIR)

    gt = {r["scene_id"]: r for r in csv.DictReader(open(GT))}
    real_counts = [(int(gt[r["scene_id"]]["gt_A" if r["side"] == 1 else "gt_B"]),
                    int(r["n_det"])) for r in real_rows]
    synth_counts = []
    for d in synth_jobs:
        truth = load_truth(d)
        for side in (1, 2):
            row = next(r for r in synth_rows
                       if r["scene_id"] == truth["scene_id"] and r["side"] == side)
            synth_counts.append((len(truth["faces"][f"side{side}"]), int(row["n_det"])))

    def prf(counts):
        tp = sum(min(g, p) for g, p in counts)
        fp = sum(max(p - g, 0) for g, p in counts)
        fn = sum(max(g - p, 0) for g, p in counts)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0), tp, fp, fn

    det_cmp = {"real": prf(real_counts), "synth": prf(synth_counts)}

    with open(OUT_CSV, "w", newline="") as f:
        cols = ["source", "scene_id", "side", "dxf"] + CLOUD_METRICS + \
               ["n_observed_px", "n_det"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(real_rows + synth_rows)
    OUT_JSON.write_text(json.dumps({"table": table}, indent=1))

    lines = [
        "# Synthetic twin — realism check",
        "",
        "Regenerate: `python -m pairbench.experiments.build_synthetic_twin --suite realism` then "
        "`python -m pairbench.experiments.synth_realism`.",
        f"Real reference: {len(real_rows)} clouds from the "
        f"{len(real_jobs)} `single`-category scenes. Twin: {len(synth_rows)} "
        f"clouds from the {len(synth_jobs)} mirrored scenes (same family, same "
        "bar count).",
        "",
        "The twin's constants are measured in "
        "`results/pairbench/synthetic_sensor_model.md`.",
        "",
        "| statistic | real median [IQR] | twin median [IQR] | twin/real | KS D |",
        "|---|---|---|---|---|",
    ]
    for t in table:
        lines.append(
            f"| {t['metric']} | {t['real_med']:.3g} "
            f"[{t['real_iqr'][0]:.3g}, {t['real_iqr'][1]:.3g}] "
            f"| {t['synth_med']:.3g} [{t['synth_iqr'][0]:.3g}, "
            f"{t['synth_iqr'][1]:.3g}] | {t['ratio']:.2f} | {t['ks_D']:.2f} |")
    lines += [
        "",
        "KS D: two-sample Kolmogorov-Smirnov statistic (0 = identical, "
        "1 = disjoint).",
        "",
        "## Detection precision/recall on real vs. synthetic",
        "",
        "`pool` detector, count metric of `experiments.detect`.",
        "",
        "| set | P | R | F1 | tp | fp | fn |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, (p_, r_, f1_, tp, fp, fn) in det_cmp.items():
        lines.append(f"| {name} | {p_:.3f} | {r_:.3f} | {f1_:.3f} | "
                     f"{tp} | {fp} | {fn} |")
    lines += [
        "",
        f"Side-by-side projections, one per family: `{PANEL_DIR.name}/`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.name}, {OUT_CSV.name}, {PANEL_DIR.name}/")


if __name__ == "__main__":
    main()
