"""Sampling logic for stratified random sampling."""

import numpy as np

from . import config


def generate_random_ids(start: int, end: int, n: int, rng: np.random.Generator = None) -> list[int]:
    if rng is None:
        rng = np.random.default_rng()
    population_size = end - start + 1
    n = min(n, population_size)
    offsets = rng.choice(population_size, size=n, replace=False)
    return (offsets + start).tolist()


def proportional_allocation(strata: dict, total_n: int) -> dict[str, int]:
    """Allocate samples proportional to stratum size."""
    total_size = sum(s["size"] for s in strata.values())
    allocation = {}
    allocated = 0

    items = list(strata.items())
    for i, (sid, s) in enumerate(items):
        if i == len(items) - 1:
            allocation[sid] = total_n - allocated
        else:
            n_h = max(1, round(total_n * s["size"] / total_size))
            allocation[sid] = n_h
            allocated += n_h

    return allocation


def stratified_sample(
    strata: dict,
    total_n: int,
    rng: np.random.Generator = None,
) -> dict[str, list[int]]:
    """Generate stratified random sample IDs."""
    if rng is None:
        rng = np.random.default_rng()

    allocation = proportional_allocation(strata, total_n)
    samples = {}
    for sid, s in strata.items():
        n_h = allocation[sid]
        ids = generate_random_ids(s["start"], s["end"], n_h, rng)
        samples[sid] = ids

    return samples


def sample_from_ground_truth(
    ground_truth: list[dict],
    n: int,
    rng: np.random.Generator = None,
) -> list[dict]:
    """Sample n records from ground truth data (no API calls)."""
    if rng is None:
        rng = np.random.default_rng()
    n = min(n, len(ground_truth))
    indices = rng.choice(len(ground_truth), size=n, replace=False)
    return [ground_truth[i] for i in indices]
