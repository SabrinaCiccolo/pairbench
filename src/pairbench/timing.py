"""Named-stage wall-clock accumulator: `with stage("projection"): ...`.

State is per process; `reset()` / `snapshot()` / `merge()` collect timings
from multiprocessing workers.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


def reset() -> None:
    _totals.clear()
    _counts.clear()


def snapshot() -> dict[str, dict[str, float]]:
    """{stage -> {total_s, calls}} for everything timed since the last
    reset() (or process start, if reset() was never called)."""
    return {name: {"total_s": _totals[name], "calls": _counts[name]}
            for name in _totals}


def merge(*snapshots: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Sum several snapshot() results (e.g. one per scene) into one."""
    out: dict[str, dict[str, float]] = {}
    for snap in snapshots:
        for name, v in snap.items():
            acc = out.setdefault(name, {"total_s": 0.0, "calls": 0})
            acc["total_s"] += v["total_s"]
            acc["calls"] += v["calls"]
    return out


def mean_ms(entry: dict[str, float]) -> float:
    return 1000.0 * entry["total_s"] / entry["calls"] if entry["calls"] else 0.0


@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _totals[name] += dt
        _counts[name] += 1


def report_lines(snap: dict[str, dict[str, float]], title: str = "Timing") -> list[str]:
    """Markdown table lines, ready to append to a results/*.md report."""
    if not snap:
        return []
    lines = [f"## {title}", "",
             "Wall-clock, single machine, CPU only. Mean is per call, not "
             "per scene, where a stage runs once per scene-side.",
             "",
             "| stage | calls | total s | mean ms/call |",
             "|---|---|---|---|"]
    for name in sorted(snap):
        v = snap[name]
        lines.append(f"| {name} | {v['calls']} | {v['total_s']:.3f} "
                     f"| {mean_ms(v):.3f} |")
    return lines
