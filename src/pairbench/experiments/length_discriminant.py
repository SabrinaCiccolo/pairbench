"""How much of a one-bar association error the bar-length check can see.

`greedy_index`'s length gate accepts a pair when the end-face distance is within
a tolerance of the nominal length. Pairing bar `i` with bar `i + k` changes the
reconstructed length by `dl(k) = sqrt(L^2 + (k*p)^2) - L ~= (k*p)^2 / (2L)`,
with L = 6005 mm and `p` the bar pitch. Reports `dl` per family against the
gate and the self-calibration residual, the lateral offset needed to reach
them, and `dl` against the measured cost gap of `experiments.cost_gap`.

Usage: python -m pairbench.experiments.length_discriminant
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from ..selfcalib import PROFILE_LENGTH_MM
from .common import RESULTS

SENSOR_JSON = RESULTS / "synthetic_sensor_model.json"
SELFCAL_JSON = RESULTS / "a2_calibration.json"
COST_GAP_CSV = RESULTS / "cost_gap_pool.csv"
OUT_MD = RESULTS / "length_discriminant.md"
OUT_CSV = RESULTS / "length_discriminant.csv"
OUT_JSON = RESULTS / "length_discriminant.json"

GATE_MM = 10.0     # pairing.greedy_index's default length tolerance
IQR_TO_SIGMA = 1.349   # IQR of a normal in units of its sigma


def delta_length_mm(offset_mm: float, length_mm: float = PROFILE_LENGTH_MM) -> float:
    """Exact change in reconstructed length for a lateral end offset."""
    return float(np.hypot(length_mm, offset_mm) - length_mm)


def offset_for_delta_mm(delta_mm: float,
                        length_mm: float = PROFILE_LENGTH_MM) -> float:
    """Inverse of `delta_length_mm`."""
    return float(np.sqrt((length_mm + delta_mm) ** 2 - length_mm ** 2))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without a scipy dependency; ties get average ranks."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks within tied groups
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(np.arange(i, j + 1))
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-mm", type=float, default=GATE_MM)
    args = ap.parse_args()

    sensor = json.loads(SENSOR_JSON.read_text())
    pitch = {fam: v[0] for fam, v in sensor["pitch_by_family"].items()}
    n_gaps = {fam: v[1] for fam, v in sensor["pitch_by_family"].items()}

    selfcal = json.loads(SELFCAL_JSON.read_text())
    q1, q3 = selfcal["iqr_mm"]
    resid_iqr = float(q3 - q1)
    resid_sigma = resid_iqr / IQR_TO_SIGMA

    # -- 1 and 2: per family, analytic ------------------------------------
    rows = []
    for fam in sorted(pitch):
        p = pitch[fam]
        d1, d2 = delta_length_mm(p), delta_length_mm(2 * p)
        need = offset_for_delta_mm(args.gate_mm)
        rows.append({
            "family": fam, "pitch_mm": p, "n_gaps": n_gaps[fam],
            "dl_1_pitch_mm": d1, "dl_2_pitch_mm": d2,
            "dl_1_over_gate": d1 / args.gate_mm,
            "dl_1_over_resid_sigma": d1 / resid_sigma,
            "pitches_to_reach_gate": need / p,
        })
    need_mm = offset_for_delta_mm(args.gate_mm)
    win = sensor["scan_window_mm"]["gated"]   # median, q1, q3, p90, max
    need_sigma_mm = offset_for_delta_mm(3.0 * resid_sigma)

    # -- 3: empirical, against the measured cost gap ----------------------
    cg = [r for r in csv.DictReader(open(COST_GAP_CSV))]
    emp = []
    for r in cg:
        fam = r["family"]
        if fam not in pitch or not r["shift_of_best_wrong"]:
            continue
        k = int(r["shift_of_best_wrong"])
        emp.append({
            "scene_id": r["scene_id"], "family": fam, "shift": k,
            "pitch_mm": pitch[fam],
            "dl_mm": delta_length_mm(abs(k) * pitch[fam]),
            "gap": float(r["gap"]),
        })
    rho = _spearman(np.array([e["dl_mm"] for e in emp]),
                    np.array([e["gap"] for e in emp])) if emp else float("nan")
    wrong = [e for e in emp if e["gap"] < 0]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    OUT_JSON.write_text(json.dumps({
        "length_mm": PROFILE_LENGTH_MM,
        "gate_mm": args.gate_mm,
        "residual_iqr_mm": resid_iqr, "residual_sigma_mm": resid_sigma,
        "offset_to_reach_gate_mm": need_mm,
        "offset_to_reach_3sigma_mm": need_sigma_mm,
        "by_family": rows, "empirical": emp, "spearman_dl_vs_gap": rho,
    }, indent=1))

    lines = [
        "# What the bar-length check can resolve",
        "",
        "Regenerate: `python -m pairbench.experiments.length_discriminant`.",
        "",
        f"Pairing bar `i` with bar `i+k` changes the reconstructed length by "
        f"`dl(k) = sqrt(L^2 + (k*p)^2) - L ~= (k*p)^2 / (2L)`, "
        f"L = {PROFILE_LENGTH_MM:.0f} mm, `p` the per-family median pitch "
        f"(`synthetic_sensor_model.json`). Residual: post-self-calibration "
        f"length residual (`a2_calibration.json`), IQR {resid_iqr:.2f} mm, "
        f"sigma ~= {resid_sigma:.2f} mm under a normal.",
        "",
        "## The signal, per family",
        "",
        f"| family | pitch (mm) | n gaps | dl, 1 pitch | dl, 2 pitches | "
        f"as a fraction of the ±{args.gate_mm:.0f} mm gate | in residual sigmas |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['pitch_mm']:.1f} | {r['n_gaps']} | "
            f"{r['dl_1_pitch_mm']:.2f} mm | {r['dl_2_pitch_mm']:.2f} mm | "
            f"{r['dl_1_over_gate']:.3f} | {r['dl_1_over_resid_sigma']:.2f} |")

    lines += [
        "",
        f"A one-bar error changes the length by "
        f"{min(r['dl_1_pitch_mm'] for r in rows):.2f} to "
        f"{max(r['dl_1_pitch_mm'] for r in rows):.2f} mm, against a "
        f"±{args.gate_mm:.0f} mm gate and a residual sigma of about "
        f"{resid_sigma:.1f} mm.",
        "",
        "## Lateral offset needed to be detectable",
        "",
        f"- to reach the ±{args.gate_mm:.0f} mm gate: **{need_mm:.0f} mm** of "
        f"lateral offset — "
        + ", ".join(f"{r['pitches_to_reach_gate']:.0f} pitches for "
                    f"`{r['family']}`" for r in rows) + ";",
        f"- to reach three residual sigmas ({3 * resid_sigma:.1f} mm): "
        f"**{need_sigma_mm:.0f} mm**.",
        "",
        f"Scan window after the radial gate (`synthetic_sensor_model.md`): "
        f"median {win[0]:.0f} mm, 90th percentile {win[3]:.0f} mm, max "
        f"{win[4]:.0f} mm.",
        "",
        "## Against the measured cost gap",
        "",
        f"`gap = cost(cheapest rival) - cost(true)` from "
        f"`experiments.cost_gap`. Over the {len(emp)} scenes whose cheapest "
        f"rival is a shift, Spearman rho between `dl` and `gap` is "
        f"{rho:+.3f}.",
        "",
        "The objective (`pairing.hungarian.pair_cost_matrix`) has no length "
        "term; `dl` and `gap` both increase with the pitch.",
        "",
        (f"Of those, {len(wrong)} are scenes the objective gets wrong. Their "
         f"implied length changes are "
         + ", ".join(f"{e['dl_mm']:.2f} mm"
                     for e in sorted(wrong, key=lambda e: e["dl_mm"]))
         + f"; {sum(e['dl_mm'] < args.gate_mm for e in wrong)} of them inside "
         f"the ±{args.gate_mm:.0f} mm gate."
         if wrong else
         "On none of those scenes does the objective prefer the shift to the "
         "truth."),
        "",
        "| scene | family | shift k | pitch (mm) | implied dl (mm) | cost gap |",
        "|---|---|---|---|---|---|",
    ]
    for e in sorted(emp, key=lambda e: e["gap"]):
        lines.append(
            f"| {e['scene_id']} | {e['family']} | {e['shift']:+d} | "
            f"{e['pitch_mm']:.1f} | {e['dl_mm']:.2f} | {e['gap']:+.3f} |")

    lines += [
        "",
        "## Limits",
        "",
        "- `dl` covers a pure lateral shift only (the length channel "
        "`greedy_index` gates on); a real mis-pair also changes tilt and "
        "twist.",
        "- The pitch is a per-family median; per-scene spacing varies.",
        f"- The correlation is over {len(emp)} scenes in "
        f"{len({e['family'] for e in emp})} families, descriptive only.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.name}, {OUT_CSV.name}, {OUT_JSON.name}")
    print(f"  offset to reach the gate: {need_mm:.0f} mm; spearman {rho:+.3f}")


if __name__ == "__main__":
    main()
