"""Shared paths for the experiment driver modules."""
from __future__ import annotations

from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "results" / "pairbench"
GT = RESULTS / "gt.csv"
