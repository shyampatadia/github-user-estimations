"""Utility functions."""

import json
import logging
import os
import sys
from datetime import datetime, timezone


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("github_estimator.log", mode="a"),
        ],
    )


def format_large_number(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def save_results_to_json(results: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _serialize(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, float) and (obj != obj):  # NaN check
            return None
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)


def load_results_from_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)
