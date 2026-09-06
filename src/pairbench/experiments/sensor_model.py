"""Measures the synthetic twin's sensor and scene constants from real scans.

Field of view, point density, range noise, bundle geometry from the
correspondence GT anchors, scan window with and without the radial gate.
Writes `results/pairbench/synthetic_sensor_model.{md,json,csv}`; `synth/sensor.py`
and `synth/generator.py` carry the measured values as constants.

Usage: python -m pairbench.experiments.sensor_model [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from ..gt import FLAGGED_SCENES
from ..io.calib import load_calibration, radial_filter
from ..io.loader import DATA_ROOT, load_scene
from ..io.ply_io import read_ply_points
from ..synth.generator import APPARENT_LENGTH_MM
from .common import RESULTS

GT = RESULTS / "gt.csv"
GT_CORR = RESULTS / "gt_correspondence.csv"
OUT_MD = RESULTS / "synthetic_sensor_model.md"
OUT_JSON = RESULTS / "synthetic_sensor_model.json"
OUT_CSV = RESULTS / "synthetic_sensor_model.csv"

CELL_MM = 0.4   # YZ cell for the density statistic
N_U_PX = 1544   # sensor pixel count along u


def _scan(job):
    scene_id, side = job
    calib = load_calibration()
    pts = read_ply_points(DATA_ROOT / scene_id / f"scanner-{side}.ply")
    u = pts[:, 0] / pts[:, 2]
    v = pts[:, 1] / pts[:, 2]
    kept = radial_filter(pts, calib.max_distance_mm)
    world = calib.for_side(side).apply(kept)
    world_raw = calib.for_side(side).apply(pts)
    rad = np.linalg.norm(kept, axis=1)
    # scan window: world-Y span (0.5-99.5 percentile), with and without the radial gate
    win_gated = float(np.diff(np.percentile(world[:, 1], [0.5, 99.5]))[0])
    win_raw = float(np.diff(np.percentile(world_raw[:, 1], [0.5, 99.5]))[0])
    keys = set(zip(np.floor(world[:, 1] / CELL_MM).astype(int),
                   np.floor(world[:, 2] / CELL_MM).astype(int)))
    area = len(keys) * CELL_MM * CELL_MM
    return {
        "scene_id": scene_id, "side": side,
        "n_raw": int(len(pts)), "n_kept": int(len(kept)),
        "u_min": float(u.min()), "u_max": float(u.max()),
        "v_min": float(v.min()), "v_max": float(v.max()),
        "z_max": float(pts[:, 2].max()),
        "rad_med": float(np.median(rad)),
        "area_mm2": float(area),
        "density": float(len(world) / area) if area else float("nan"),
        "x_spread": float(world[:, 0].max() - world[:, 0].min()),
        "window_y_gated_mm": win_gated,
        "window_y_raw_mm": win_raw,
    }


def _plane_residual(scene_id: str, side: int) -> tuple[float, int]:
    """Std of a plane fit to the dominant end-face plane of one real cloud."""
    calib = load_calibration()
    pts = radial_filter(read_ply_points(DATA_ROOT / scene_id / f"scanner-{side}.ply"),
                        calib.max_distance_mm)
    world = calib.for_side(side).apply(pts)
    sel = np.abs(world[:, 0] - np.median(world[:, 0])) < 3.0
    p = world[sel]
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return float(((p - c) @ vt[-1]).std()), int(sel.sum())


def _point_noise(specs) -> dict:
    """Per-point range noise from the spread of points sharing one projection pixel."""
    from ..io.dxf_template import render_template
    from ..io.projection import depth_channels, project_yz

    out = {}
    for scene_id, dxf in specs:
        template = render_template(dxf)
        clouds = load_scene(DATA_ROOT / scene_id)
        proj = project_yz(clouds.side1_world, padding_px=template.default_padding)
        ch = depth_channels(clouds.side1_world, proj, 1)
        pair = ch.density == 2
        if pair.sum() < 50:
            continue
        out[scene_id] = float(ch.x_spread_mm[pair].mean() / 1.128)
    return out


def _anchor_geometry():
    """Bed height, bundle centre, per-family pitch, cross-view transport."""
    def parse(a):
        if not a:
            return np.zeros((0, 2))
        return np.array([[float(x) for x in p.split("|")] for p in a.split(";")])

    pitch_by_family = defaultdict(list)
    z_by_family = defaultdict(list)
    dy, dz, ys, zs = [], [], [], []
    for row in csv.DictReader(open(GT_CORR)):
        if row["scene_id"] in FLAGGED_SCENES:
            continue
        family = row["scene_id"].split("/")[0]
        a1, a2 = parse(row["anchors_1"]), parse(row["anchors_2"])
        for a in (a1, a2):
            if len(a) >= 2:
                pitch_by_family[family].extend(np.diff(np.sort(a[:, 0])).tolist())
            if len(a):
                ys.extend(a[:, 0].tolist())
                zs.extend(a[:, 1].tolist())
                z_by_family[family].extend(a[:, 1].tolist())
        offs, offz = [], []
        for pair in filter(None, row["pairs"].split(",")):
            i, j = (int(x) for x in pair.split("-"))
            if i < len(a1) and j < len(a2):
                offs.append(a2[j, 0] - a1[i, 0])
                offz.append(a2[j, 1] - a1[i, 1])
        if offs:
            dy.append(float(np.median(offs)))
            dz.append(float(np.median(offz)))
    return {
        "pitch_by_family": {k: [float(np.median(v)), len(v)]
                            for k, v in pitch_by_family.items()},
        # end-face centroid height, not bed height (varies with section height)
        "face_centroid_z_mm": [float(np.median(zs)),
                               *np.percentile(zs, [25, 75]).tolist(),
                               len(zs)],
        "face_centroid_z_by_family": {
            k: [float(np.median(v)), len(v)] for k, v in sorted(z_by_family.items())},
        "bundle_y_mm": float(np.median(ys)),
        "transport_y_mm": [float(np.median(dy)), *np.percentile(dy, [25, 75]).tolist(),
                           float(min(dy)), float(max(dy))],
        "transport_z_mm": float(np.median(dz)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    scene_ids = [r["scene_id"] for r in csv.DictReader(open(GT))
                 if r["scene_id"] not in FLAGGED_SCENES]
    if args.limit:
        scene_ids = scene_ids[: args.limit]
    jobs = [(sid, side) for sid in scene_ids for side in (1, 2)]

    rows = []
    with Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap(_scan, jobs)):
            rows.append(row)
            if (k + 1) % 20 == 0:
                print(f"[{k + 1}/{len(jobs)}]")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["scene_id"], r["side"])))

    by_side = {s: [r for r in rows if r["side"] == s] for s in (1, 2)}
    fov = {}
    for side, rs in by_side.items():
        # u hits a hard limit (median is the estimate); v's envelope over
        # scenes is a lower bound on its limit
        u_lo = float(np.median([r["u_min"] for r in rs]))
        u_hi = float(np.median([r["u_max"] for r in rs]))
        fov[side] = {
            "u_lo": u_lo, "u_hi": u_hi,
            "u_lo_on_limit": int(sum(abs(r["u_min"] - u_lo) < 1e-3 for r in rs)),
            "u_hi_on_limit": int(sum(abs(r["u_max"] - u_hi) < 1e-3 for r in rs)),
            "n": len(rs),
            "v_lo": min(r["v_min"] for r in rs), "v_hi": max(r["v_max"] for r in rs),
            "v_lo_med": float(np.median([r["v_min"] for r in rs])),
            "v_hi_med": float(np.median([r["v_max"] for r in rs])),
        }

    noise_sigma, noise_n = _plane_residual("complex-profile/single-04", 1)
    point_noise = _point_noise([("l-profile/single-01", "l-profile"),
                                ("complex-profile/single-04", "complex-profile"),
                                ("square-profile/single-01", "square-profile")])
    geom = _anchor_geometry()
    dens = np.array([r["density"] for r in rows])
    rad = np.array([r["rad_med"] for r in rows])
    win_g = np.array([r["window_y_gated_mm"] for r in rows])
    win_r = np.array([r["window_y_raw_mm"] for r in rows])
    window = {
        "gated": [float(np.median(win_g)), *np.percentile(win_g, [25, 75, 90]).tolist(),
                  float(win_g.max())],
        "ungated": [float(np.median(win_r)), *np.percentile(win_r, [25, 75, 90]).tolist(),
                    float(win_r.max())],
        "radial_gate_mm": float(load_calibration().max_distance_mm),
        "n_clouds": int(len(win_g)),
    }

    payload = {"n_scenes": len(scene_ids), "fov": fov,
               "noise_sigma_mm": noise_sigma, "noise_n": noise_n,
               "point_noise_mm": point_noise,
               "density_pts_mm2": [float(np.median(dens)),
                                   *np.percentile(dens, [25, 75]).tolist()],
               "radial_mm": [float(np.median(rad)),
                             *np.percentile(rad, [25, 75]).tolist()],
               "scan_window_mm": window,
               **geom}
    OUT_JSON.write_text(json.dumps(payload, indent=1))

    fz = geom["face_centroid_z_by_family"]
    pitch_lines = "\n".join(
        f"| {fam} | {fz[fam][0]:.1f} | {fz[fam][1]} | {med:.1f} | {n} |"
        for fam, (med, n) in sorted(geom["pitch_by_family"].items()))
    t = geom["transport_y_mm"]
    lines = [
        "# Synthetic twin — sensor and scene constants, measured from real scans",
        "",
        f"Regenerate: `python -m pairbench.experiments.sensor_model`; "
        f"{len(scene_ids)} scenes, {len(jobs)} clouds.",
        "",
        "## Field of view (scanner-frame tangent coordinates u = x/z, v = y/z)",
        "",
        "| side | u range (modal device limit) | scenes on the limit | "
        "v envelope (lower bound on the limit) | v median extent |",
        "|---|---|---|---|---|",
    ]
    for side in (1, 2):
        f = fov[side]
        lines.append(
            f"| {side} | [{f['u_lo']:+.6f}, {f['u_hi']:+.6f}] "
            f"= [{np.degrees(np.arctan(f['u_lo'])):+.3f}, "
            f"{np.degrees(np.arctan(f['u_hi'])):+.3f}] deg "
            f"| {f['u_lo_on_limit']}/{f['n']} lo, {f['u_hi_on_limit']}/{f['n']} hi "
            f"| [{f['v_lo']:+.6f}, {f['v_hi']:+.6f}] "
            f"= [{np.degrees(np.arctan(f['v_lo'])):+.3f}, "
            f"{np.degrees(np.arctan(f['v_hi'])):+.3f}] deg "
            f"| [{np.degrees(np.arctan(f['v_lo_med'])):+.2f}, "
            f"{np.degrees(np.arctan(f['v_hi_med'])):+.2f}] deg |")
    lines += [
        "",
        "`u` reaches the same two values (to 1e-3) in most scenes: a hard "
        "limit. `v` never repeats, so its envelope is used as a lower bound.",
        "",
        "## Sampling, noise, standoff",
        "",
        f"- point density on the YZ projection ({CELL_MM} mm cells): "
        f"median {np.median(dens):.2f} pts/mm2, IQR "
        f"[{np.percentile(dens, 25):.2f}, {np.percentile(dens, 75):.2f}]",
        f"- median radial standoff: {np.median(rad):.0f} mm, IQR "
        f"[{np.percentile(rad, 25):.0f}, {np.percentile(rad, 75):.0f}]",
        f"- range noise, plane fit over a whole real end face (n={noise_n}): "
        f"sigma = {noise_sigma:.3f} mm",
        "- range noise, spread within one projection pixel: "
        + ", ".join(f"{k} {v:.3f} mm" for k, v in sorted(point_noise.items()))
        + f" (median {np.median(list(point_noise.values())):.3f} mm)",
        "",
        f"The plane fit is {noise_sigma / np.median(list(point_noise.values())):.1f}x "
        "the per-pixel figure because it also absorbs the face's form error; "
        "the twin uses the per-pixel figure.",
        "",
        f"With {N_U_PX} pixels along u, the tangent pitch is "
        f"{(fov[1]['u_hi'] - fov[1]['u_lo']) / (N_U_PX - 1):.3e} "
        f"(~{(fov[1]['u_hi'] - fov[1]['u_lo']) / (N_U_PX - 1) * np.median(rad):.2f} mm "
        f"ground sampling, {1.0 / ((fov[1]['u_hi'] - fov[1]['u_lo']) / (N_U_PX - 1) * np.median(rad))**2:.1f} pts/mm2).",
        "",
        "## Bundle geometry (from the correspondence GT anchors)",
        "",
        f"- end-face centroid height (median anchor Z): "
        f"{geom['face_centroid_z_mm'][0]:.1f} mm, IQR "
        f"[{geom['face_centroid_z_mm'][1]:.1f}, {geom['face_centroid_z_mm'][2]:.1f}], "
        f"n={geom['face_centroid_z_mm'][3]} anchors",
        f"- bundle centre (median anchor Y): {geom['bundle_y_mm']:.1f} mm",
        f"- cross-view transport offset dY: median {t[0]:+.1f} mm, "
        f"IQR [{t[1]:+.1f}, {t[2]:+.1f}], range [{t[3]:+.1f}, {t[4]:+.1f}]",
        f"- cross-view dZ: median {geom['transport_z_mm']:+.2f} mm (negligible — "
        "only Y is treated as transport)",
        "",
        "This is the height of an end-face centroid, not of the bed: it sits "
        "about half a section height above the bed, so it varies by family. "
        "`synth.generator.BED_Z_MM` is the twin's placement parameter, not a "
        "measurement of this quantity.",
        "",
        "| family | median end-face centroid Z (mm) | n anchors | median bar pitch (mm) | n gaps |",
        "|---|---|---|---|---|",
        pitch_lines,
        "",
        "## Scan window",
        "",
        f"World-Y span of each cloud (0.5-99.5 percentile), over "
        f"{window['n_clouds']} clouds, after and before the "
        f"{window['radial_gate_mm']:.0f} mm radial gate "
        f"(`io/calib.py::radial_filter`).",
        "",
        "| | median | IQR | p90 | max |",
        "|---|---|---|---|---|",
        f"| after the radial gate | {window['gated'][0]:.0f} mm | "
        f"[{window['gated'][1]:.0f}, {window['gated'][2]:.0f}] | "
        f"{window['gated'][3]:.0f} mm | **{window['gated'][4]:.0f} mm** |",
        f"| before the radial gate | {window['ungated'][0]:.0f} mm | "
        f"[{window['ungated'][1]:.0f}, {window['ungated'][2]:.0f}] | "
        f"{window['ungated'][3]:.0f} mm | **{window['ungated'][4]:.0f} mm** |",
        "",
        "## Reconstructed bar length",
        "",
        "In the offline-calibrated world a 6005 mm bar reconstructs as two "
        f"faces {APPARENT_LENGTH_MM:.0f} mm apart "
        "(`synth.generator.APPARENT_LENGTH_MM`).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.name}, {OUT_JSON.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
