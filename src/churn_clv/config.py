"""Configuration loading."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("CHURN_CLV_ROOT", Path(__file__).resolve().parents[2]))


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
