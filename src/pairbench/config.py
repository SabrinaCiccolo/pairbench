"""Repo-root path resolution and optional data_root override.

Scan data lives under `<repo_root>/data` by default. `PAIRBENCH_DATA_ROOT`
(env var) or config.yaml's `data_root` key overrides it; the env var wins.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def data_root() -> Path:
    env = os.environ.get("PAIRBENCH_DATA_ROOT")
    if env:
        return Path(env)
    cfg_root = load_config().get("data_root")
    if cfg_root:
        p = Path(cfg_root)
        return p if p.is_absolute() else REPO_ROOT / p
    return REPO_ROOT / "data"
