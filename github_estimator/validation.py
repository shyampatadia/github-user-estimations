"""Validation experiments using ground truth data."""

import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from . import config, database
from .estimator import (
    bootstrap_ci_simple,
    calculate_all_metrics,
    calculate_relative_error,
    estimate_total,
)
from .sampler import sample_from_ground_truth

logger = logging.getLogger(__name__)


def run_validation_experiment(
    db_conn,
    stratum_id: str,
    stratum_config: dict,
    sampling_rate: float,
    n_reps: int = 50,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    Run validation experiment on a single stratum at a given sampling rate.
    Resamples from ground truth — no API calls needed.
    """
    if rng is None:
        rng = np.random.default_rng()

    ground_truth = database.get_ground_truth_by_stratum(db_conn, stratum_id)
    if not ground_truth:
        logger.warning(f"No ground truth data for stratum {stratum_id}")
        return []

    population_size = stratum_config["size"]
    true_valid = sum(1 for r in ground_truth if r["is_valid"])
    sample_size = max(1, int(len(ground_truth) * sampling_rate))

    results = []
    for rep in range(n_reps):
        sample = sample_from_ground_truth(ground_truth, sample_size, rng)
        valid_in_sample = sum(1 for s in sample if s["is_valid"])
        estimated = estimate_total(valid_in_sample, sample_size, population_size)
        rel_error = calculate_relative_error(estimated, true_valid)

        run_id = f"val_{stratum_id}_{sampling_rate:.3f}_{rep:03d}"
        run = {
            "id": run_id,
            "run_type": "validation",
            "stratum_id": stratum_id,
            "sample_size": sample_size,
            "sampling_rate": sampling_rate,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "valid_count": valid_in_sample,
            "total_sampled": sample_size,
            "estimated_total": estimated,
            "estimated_rate": valid_in_sample / sample_size if sample_size > 0 else 0,
            "ci_lower": None,
            "ci_upper": None,
            "bootstrap_samples": None,
            "ground_truth_total": true_valid,
            "relative_error": rel_error,
            "notes": f"Validation rep {rep+1}/{n_reps} at {sampling_rate:.1%}",
        }
        database.insert_estimation_run(db_conn, run)
        results.append(run)

    return results


def run_all_validation_experiments(
    db_conn,
    sampling_rates: list[float] = None,
    n_reps: int = 50,
    rng: np.random.Generator = None,
) -> dict:
    """Run validation experiments across all strata and sampling rates."""
    if sampling_rates is None:
        sampling_rates = config.SAMPLING_RATES
    if rng is None:
        rng = np.random.default_rng(42)

    all_results = {}
    for stratum_id, stratum_config in config.VALIDATION_STRATA.items():
        gt_count = database.get_ground_truth_count(db_conn, stratum_id)
        if gt_count == 0:
            logger.warning(f"Skipping {stratum_id}: no ground truth data")
            continue

        stratum_results = {}
        for rate in sampling_rates:
            logger.info(f"Running validation: {stratum_id} at {rate:.1%}")
            results = run_validation_experiment(
                db_conn, stratum_id, stratum_config, rate, n_reps, rng
            )
            stratum_results[rate] = results

        all_results[stratum_id] = stratum_results

    total_runs = sum(
        len(runs) for sr in all_results.values() for runs in sr.values()
    )
    logger.info(f"Completed {total_runs} validation experiments")
    return all_results


def test_unbiasedness(estimates: list[float]) -> dict:
    """Test if estimates are unbiased (roughly 50% above and below mean)."""
    arr = np.array(estimates)
    mean = float(arr.mean())
    above = int(np.sum(arr > mean))
    below = int(np.sum(arr < mean))
    equal = int(np.sum(arr == mean))
    total = len(estimates)

    return {
        "mean": mean,
        "std": float(arr.std()),
        "above_count": above,
        "below_count": below,
        "equal_count": equal,
        "above_pct": above / total * 100 if total > 0 else 0,
        "below_pct": below / total * 100 if total > 0 else 0,
        "is_unbiased": abs(above / total - 0.5) < 0.1 if total > 0 else False,
    }


def test_correctness(estimates_by_budget: dict[int, list[float]]) -> dict:
    """Test if mean estimate stays constant across budgets."""
    means = {}
    stds = {}
    for budget, ests in estimates_by_budget.items():
        arr = np.array(ests)
        means[budget] = float(arr.mean())
        stds[budget] = float(arr.std())

    mean_values = list(means.values())
    overall_mean = np.mean(mean_values)
    max_deviation = max(abs(m - overall_mean) / overall_mean for m in mean_values) if overall_mean > 0 else 0

    return {
        "means": means,
        "stds": stds,
        "overall_mean": float(overall_mean),
        "max_relative_deviation": float(max_deviation),
        "is_correct": max_deviation < 0.02,  # Less than 2% deviation
    }
