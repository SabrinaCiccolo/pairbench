"""Generate the synthetic twin dataset.

Extrudes each family's DXF cross-section, poses bundles of bars with known
6-DOF poses, and raycasts two virtual scanners at the calibrated scanner
positions. Output is one directory per scene (`scanner-1.ply`, `scanner-2.ply`,
`truth.json`), readable by `loader.load_scene`.

Suites (`pairbench.synth.suites`):

    realism   mirrors the real `single`-category scenes
    pitch     bar-pitch regularity x cross-view transport
    crossing  elevation x yaw
    one_end   a bar raised at one end only, elevated in one view
    pitch_edge    four bars against the window edge, transport can truncate
    pitch_centre  four bars centred, no bar leaves the window at any swept
                  transport (control for pitch_edge)
    ambiguity     transport swept past three whole bar pitches at both
                  placements, for the hypothesis-cost margin sweep

`realism` reads scene ids and bar counts from `results/pairbench/gt.csv`.

Usage:
    python -m pairbench.experiments.build_synthetic_twin
        [--suite all] [--workers 6] [--out data/synthetic]
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

from ..config import REPO_ROOT
from ..io.loader import FAMILY_DXF
from ..synth import render_scene
from ..synth.suites import (ambiguity_suite, crossing_suite, one_end_suite,
                             pitch_suite,
                            realism_suite)
from .common import GT

DEFAULT_OUT = REPO_ROOT / "data" / "synthetic"


def realism_specs(gt_path: Path = GT):
    """Real `single`-category scenes -> (scene_id, dxf, n_bars) to mirror."""
    specs = []
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["category"] != "single":
                continue
            family = row["scene_id"].split("/")[0]
            specs.append((row["scene_id"], FAMILY_DXF[family], int(row["gt_A"])))
    return specs


def _render(job):
    scene, out_root = job
    path = Path(out_root) / scene.scene_id
    render_scene(scene, path)
    return scene.scene_id, len(scene.bars)


def build(suite: str, out_root: Path, workers: int) -> list:
    if suite == "realism":
        scenes = realism_suite(realism_specs())
    elif suite == "pitch":
        scenes = pitch_suite()
    elif suite == "crossing":
        scenes = crossing_suite()
    elif suite == "one_end":
        scenes = one_end_suite()
    elif suite == "pitch_edge":
        scenes = pitch_suite(n_bars=4, placement="edge", arm="pitch_edge")
    elif suite == "pitch_centre":
        scenes = pitch_suite(n_bars=4, placement="centre", arm="pitch_centre")
    elif suite == "ambiguity":
        scenes = ambiguity_suite()
    else:
        raise ValueError(suite)
    jobs = [(s, out_root) for s in scenes]
    done = []
    with Pool(workers) as pool:
        for k, (sid, n) in enumerate(pool.imap_unordered(_render, jobs)):
            done.append(sid)
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                print(f"  [{k + 1}/{len(jobs)}] {sid} ({n} bars)")
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all",
                    choices=["all", "realism", "pitch", "crossing", "one_end",
                             "pitch_edge", "pitch_centre", "placement",
                             "ambiguity"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.suite == "all":
        suites = ["realism", "pitch", "crossing", "one_end", "pitch_edge",
                  "pitch_centre", "ambiguity"]
    elif args.suite == "placement":
        suites = ["pitch_edge", "pitch_centre"]
    else:
        suites = [args.suite]
    manifest = []
    for suite in suites:
        print(f"suite {suite}:")
        for scene in build(suite, args.out, args.workers):
            manifest.append({
                "scene_id": scene.scene_id, "suite": suite, "dxf": scene.dxf_stem,
                "n_bars": len(scene.bars), "seed": scene.seed,
                "transport_y_mm": scene.transport_y_mm,
                **{k: v for k, v in scene.params.items()},
            })

    args.out.mkdir(parents=True, exist_ok=True)
    # a partial build must not silently replace the full build's manifest
    name = "manifest.csv" if args.suite == "all" else f"manifest_{args.suite}.csv"
    cols = sorted({k for row in manifest for k in row})
    cols = ["scene_id", "suite", "dxf", "n_bars", "seed"] + \
           [c for c in cols if c not in ("scene_id", "suite", "dxf", "n_bars", "seed")]
    with open(args.out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(manifest)
    print(f"\n{len(manifest)} scenes under {args.out}")


if __name__ == "__main__":
    main()
