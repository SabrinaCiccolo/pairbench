"""Pairing-strategy scaling bench on synthetic bar counts.

Times every strategy in `pairbench.pairing.STRATEGIES` on synthetic scenes
with more bars per side than the real dataset. The aligned arms
(`hungarian_aligned`, `monotone_aligned`, `hybrid_aligned`) each call
`pairing.common.estimate_translation_ransac` once (O(n^4 log n) by step
count); the others scale as their own assignment step.

Usage: python -m pairbench.experiments.scaling
"""

from __future__ import annotations

import time

import numpy as np

from ..detect.ccorr import Detection
from ..pairing import STRATEGIES
from .common import RESULTS

OUT_MD = RESULTS / "scaling_report.md"
OUT_PNG = RESULTS / "scaling_report.png"

REPS = 3
SIZES_CHEAP = [4, 8, 16, 32, 64, 128, 256, 512]
SIZES_ALIGNED = [4, 8, 16, 24, 32, 48, 64]
ALIGNED_ARMS = {"hungarian_aligned", "monotone_aligned",
                "hybrid_aligned"}


def _det(side: int, y: float, z: float, angle: float = 0.0,
        x: float | None = None, y_ext: float = 20.0, z_ext: float = 12.0) -> Detection:
    """One synthetic detection at world X = +-3002.5 mm (pair distance
    6005 mm, the length check's nominal)."""
    if x is None:
        x = 3002.5 if side == 1 else -3002.5
    return Detection(
        side=side, angle_deg=angle, score=0.9, ccorr_frac=0.9, kind="confirmed",
        obb=((y, z), (40.0, 12.0), -angle),
        centroid_yz_mm=(y, z), centroid_world_mm=(x, y, z),
        normal_world=(1.0, 0.0, 0.0), n_points=1000,
        z_extent_mm=(z - z_ext / 2, z + z_ext / 2),
        y_extent_mm=(y - y_ext / 2, y + y_ext / 2),
    )


def synthetic_scene(n: int, pitch_mm: float = 30.0,
                    cross_view_offset_mm: float = 5.0
                    ) -> tuple[list[Detection], list[Detection]]:
    """Builds n evenly-spaced bars on one flat stacking level, with a
    constant cross-view Y offset."""
    d1 = [_det(1, i * pitch_mm, 0.0) for i in range(n)]
    d2 = [_det(2, i * pitch_mm + cross_view_offset_mm, 0.0) for i in range(n)]
    return d1, d2


def time_arm(arm: str, n: int, reps: int = REPS) -> float:
    """Median wall time in ms over `reps` fresh scenes."""
    times = []
    for _ in range(reps):
        d1, d2 = synthetic_scene(n)
        t0 = time.perf_counter()
        STRATEGIES[arm](d1, d2)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def fit_log_log_slope(ns: list[int], ms: list[float]) -> float:
    """Empirical complexity exponent: slope of log(time) vs log(n), fit on
    n>=16 only."""
    ns_a = np.array(ns, dtype=np.float64)
    ms_a = np.array(ms, dtype=np.float64)
    keep = ns_a >= 16
    if keep.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(ns_a[keep]), np.log(ms_a[keep]), 1)
    return float(slope)


def main() -> None:
    results: dict[str, dict[int, float]] = {}
    for arm in STRATEGIES:
        sizes = SIZES_ALIGNED if arm in ALIGNED_ARMS else SIZES_CHEAP
        results[arm] = {}
        for n in sizes:
            ms = time_arm(arm, n)
            results[arm][n] = ms
            print(f"{arm:20s} n={n:4d} {ms:9.2f} ms")

    lines = ["# Pairing-strategy scaling bench (synthetic bar counts)", "",
             "Synthetic scenes (`synthetic_scene`, one flat stacking level, "
             "constant 5 mm cross-view Y offset). Wall-clock, one machine, "
             f"CPU only, median of {REPS} reps per (strategy, n).", "",
             "`hungarian_aligned`/`monotone_aligned`/`hybrid_aligned` call "
             "the RANSAC translation estimate "
             "(`pairing.common.estimate_translation_ransac`) once; the other "
             "arms do not, and are swept to larger n. Blank cells are outside "
             "an arm's swept range. The fitted exponent is a regression over "
             "the swept range, not an asymptotic bound.", ""]

    all_sizes = sorted(set(SIZES_CHEAP) | set(SIZES_ALIGNED))
    lines += ["| strategy | " + " | ".join(str(n) for n in all_sizes)
              + " | fit exponent (n>=16) |",
              "|---|" + "---|" * (len(all_sizes) + 1)]
    for arm in STRATEGIES:
        row = []
        for n in all_sizes:
            v = results[arm].get(n)
            row.append(f"{v:.2f}" if v is not None else "")
        sizes = sorted(results[arm])
        exponent = fit_log_log_slope(sizes, [results[arm][n] for n in sizes])
        lines.append(f"| {arm} | " + " | ".join(row) + f" | {exponent:.2f} |")

    lines += ["", "Reference exponents: O(n^2) fits ~2.0, O(n^3) ~3.0, "
              "O(n^4) ~4.0.", "",
              "At the bar counts of the real dataset every arm costs well "
              "under the per-scene detection time; see docs/method.md."]
    OUT_MD.write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm in STRATEGIES:
        sizes = sorted(results[arm])
        ax.plot(sizes, [results[arm][n] for n in sizes], marker="o", label=arm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bars per side (n)")
    ax.set_ylabel("median wall time (ms)")
    ax.set_title("Pairing strategy cost vs bar count (synthetic, log-log)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nwrote {OUT_MD.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
