"""Paired bootstrap tests for comparing pairing arms.

Every arm is scored on the same resampling indices. The resampling unit is the
bundle (`gt.load_bundles`); `sum_by_bundle` aggregates per-scene arrays.
"""

from __future__ import annotations

import numpy as np

from .metrics import bootstrap_resample_f1

N_BOOT = 10000
SEED = 0


def sum_by_bundle(scene_ids, *arrays, bundles: dict[str, str] | None = None):
    """Sum per-scene rows into per-bundle rows.

    `arrays` are indexed like `scene_ids` along axis 0. Returns
    (bundle labels, *summed arrays), bundles in first-seen order."""
    from .gt import bundle_of, load_bundles
    if bundles is None:
        bundles = load_bundles()
    labels = [bundle_of(s, bundles) for s in scene_ids]
    order = list(dict.fromkeys(labels))
    index = {b: k for k, b in enumerate(order)}
    out = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        acc = np.zeros((len(order),) + arr.shape[1:], dtype=np.float64)
        for k, lab in enumerate(labels):
            acc[index[lab]] += arr[k]
        out.append(acc)
    return (order, *out)


# --------------------------------------------------------------------------
# paired bootstrap over a shared resampling
# --------------------------------------------------------------------------

def _pooled(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Ratio of summed numerators to summed denominators, 0 where empty."""
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def paired_bootstrap(units: np.ndarray, arm_numerators: dict[str, np.ndarray],
                     n_boot: int = N_BOOT, seed: int = SEED,
                     ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Bootstrap every arm's pooled ratio over one shared set of resamples.

    `units` is the (n_units,) denominator per resampling unit; `arm_numerators`
    maps an arm name to its (n_units,) numerator. Returns ({arm: (n_boot,)
    pooled ratios}, point estimates' denominators), with the same draw applied
    to every arm so the per-replicate differences are paired.
    """
    n = len(units)
    if n == 0:
        return ({a: np.zeros(0) for a in arm_numerators}, np.zeros(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    den = units[idx].sum(axis=1)
    return ({a: _pooled(v[idx].sum(axis=1), den) for a, v in arm_numerators.items()},
            den)


def bootstrap_p_value(diff: np.ndarray) -> float:
    """Two-sided bootstrap p: twice the smaller tail crossing zero, capped at 1."""
    if diff.size == 0:
        return float("nan")
    return float(min(1.0, 2.0 * min(float(np.mean(diff <= 0.0)),
                                    float(np.mean(diff >= 0.0)))))


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if samples.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def holm_reject(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down: which of a family of tests reject at
    family-wise error rate `alpha`. Sort ascending, reject while
    p_(k) <= alpha / (m - k + 1), stop at the first failure."""
    items = sorted(p_values.items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    m = len(items)
    out = {}
    still = True
    for k, (name, p) in enumerate(items):
        still = still and not np.isnan(p) and p <= alpha / (m - k)
        out[name] = bool(still)
    return out


def compare_arms(units: np.ndarray, arm_numerators: dict[str, np.ndarray],
                 baseline: str, n_boot: int = N_BOOT, seed: int = SEED) -> list[dict]:
    """One row per arm: its pooled ratio, the paired delta vs `baseline`,
    that delta's 95% percentile CI and two-sided bootstrap p-value."""
    if baseline not in arm_numerators:
        raise KeyError(f"baseline {baseline!r} not among arms")
    boots, _ = paired_bootstrap(units, arm_numerators, n_boot=n_boot, seed=seed)
    total = float(units.sum())
    point = {a: (float(v.sum()) / total if total else 0.0)
             for a, v in arm_numerators.items()}
    rows = []
    for arm in arm_numerators:
        diff = boots[arm] - boots[baseline]
        lo, hi = percentile_ci(diff)
        rows.append({
            "arm": arm, "baseline": baseline,
            "value": point[arm], "value_baseline": point[baseline],
            "delta": point[arm] - point[baseline],
            "ci_lo": lo, "ci_hi": hi,
            "p_value": bootstrap_p_value(diff),
            "significant": bool(lo > 0.0 or hi < 0.0),
            "n_units": int(len(units)), "n_boot": int(n_boot),
        })
    return rows


def paired_bootstrap_f1(per_scene: dict[str, np.ndarray], baseline: str,
                        n_boot: int = N_BOOT, seed: int = SEED) -> list[dict]:
    """Paired bootstrap on pooled count-F1, one row per arm vs `baseline`.
    `per_scene` maps an arm name to an (n_scenes, 3) array of (tp, fp, fn)."""
    arms = list(per_scene)
    if baseline not in per_scene:
        raise KeyError(f"baseline {baseline!r} not among arms")
    n = len(per_scene[baseline])
    if any(len(per_scene[a]) != n for a in arms):
        raise ValueError("every arm must have the same number of scenes")
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    identity_idx = np.arange(n)[None, :]

    boots, point = {}, {}
    for a in arms:
        c = np.asarray(per_scene[a], dtype=np.float64)
        boots[a] = bootstrap_resample_f1(c, idx)
        point[a] = float(bootstrap_resample_f1(c, identity_idx)[0])

    rows = []
    for a in arms:
        diff = boots[a] - boots[baseline]
        lo, hi = percentile_ci(diff)
        rows.append({
            "arm": a, "baseline": baseline, "f1": point[a],
            "f1_baseline": point[baseline], "delta": point[a] - point[baseline],
            "ci_lo": lo, "ci_hi": hi, "p_value": bootstrap_p_value(diff),
            "significant": bool(lo > 0.0 or hi < 0.0),
            "n_units": int(n), "n_boot": int(n_boot),
        })
    return rows
