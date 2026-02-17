"""Statistical estimation: stratified estimator, bootstrap CI, multi-metric."""

import numpy as np
from typing import Optional


def estimate_total(valid_count: int, sample_size: int, population_size: int) -> float:
    """Simple expansion estimator: N_hat = M * (valid / n)."""
    if sample_size == 0:
        return 0.0
    return population_size * (valid_count / sample_size)


def estimate_stratified(stratum_results: dict[str, dict], strata_sizes: dict[str, int]) -> float:
    """
    Stratified estimator: N_hat = sum(M_h * p_hat_h).
    stratum_results: {stratum_id: {"valid": int, "total": int}}
    strata_sizes: {stratum_id: int (M_h)}
    """
    total = 0.0
    for sid, res in stratum_results.items():
        M_h = strata_sizes[sid]
        p_h = res["valid"] / res["total"] if res["total"] > 0 else 0
        total += M_h * p_h
    return total


def stratified_variance(stratum_results: dict[str, dict], strata_sizes: dict[str, int]) -> float:
    """Variance of stratified estimator."""
    var = 0.0
    for sid, res in stratum_results.items():
        M_h = strata_sizes[sid]
        n_h = res["total"]
        p_h = res["valid"] / n_h if n_h > 0 else 0
        if n_h > 1:
            var += (M_h ** 2) * p_h * (1 - p_h) / (n_h - 1)
    return var


def bootstrap_ci_simple(
    valid_count: int,
    sample_size: int,
    population_size: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple[float, float, list[float]]:
    """Bootstrap CI for simple random sampling estimate."""
    if rng is None:
        rng = np.random.default_rng()

    # Create binary array: 1=valid, 0=invalid
    sample = np.zeros(sample_size)
    sample[:valid_count] = 1

    estimates = []
    for _ in range(n_bootstrap):
        boot_sample = rng.choice(sample, size=sample_size, replace=True)
        boot_valid = boot_sample.sum()
        est = population_size * (boot_valid / sample_size)
        estimates.append(est)

    estimates = np.array(estimates)
    ci_lower = float(np.percentile(estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper, estimates.tolist()


def bootstrap_ci_stratified(
    stratum_samples: dict[str, np.ndarray],
    strata_sizes: dict[str, int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple[float, float, list[float]]:
    """
    Bootstrap CI for stratified estimator.
    stratum_samples: {stratum_id: np.array of 0/1 validity indicators}
    strata_sizes: {stratum_id: M_h}
    """
    if rng is None:
        rng = np.random.default_rng()

    estimates = []
    for _ in range(n_bootstrap):
        total = 0.0
        for sid, sample in stratum_samples.items():
            M_h = strata_sizes[sid]
            n_h = len(sample)
            boot = rng.choice(sample, size=n_h, replace=True)
            p_h = boot.sum() / n_h
            total += M_h * p_h
        estimates.append(total)

    estimates = np.array(estimates)
    ci_lower = float(np.percentile(estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper, estimates.tolist()


def calculate_relative_error(estimate: float, ground_truth: float) -> float:
    if ground_truth == 0:
        return float("inf")
    return (estimate - ground_truth) / ground_truth


def calculate_all_metrics(samples: list[dict]) -> dict:
    """Calculate all secondary metrics from a list of sample records."""
    valid = [s for s in samples if s.get("is_valid")]
    total = len(samples)
    valid_count = len(valid)

    metrics = {
        "validity_rate": valid_count / total if total > 0 else 0,
        "valid_count": valid_count,
        "total_sampled": total,
    }

    if not valid:
        return metrics

    # User vs Organization
    users = [s for s in valid if s.get("type") == "User"]
    orgs = [s for s in valid if s.get("type") == "Organization"]
    metrics["user_count"] = len(users)
    metrics["org_count"] = len(orgs)
    metrics["user_rate"] = len(users) / valid_count if valid_count > 0 else 0
    metrics["org_rate"] = len(orgs) / valid_count if valid_count > 0 else 0

    # Repository stats
    repos = [s["public_repos"] for s in valid if s.get("public_repos") is not None]
    if repos:
        metrics["mean_public_repos"] = float(np.mean(repos))
        metrics["median_public_repos"] = float(np.median(repos))
        metrics["empty_account_rate"] = sum(1 for r in repos if r == 0) / len(repos)

    # Gist stats
    gists = [s["public_gists"] for s in valid if s.get("public_gists") is not None]
    if gists:
        metrics["mean_public_gists"] = float(np.mean(gists))

    # Social stats
    followers = [s["followers"] for s in valid if s.get("followers") is not None]
    if followers:
        metrics["mean_followers"] = float(np.mean(followers))
        metrics["median_followers"] = float(np.median(followers))

    following = [s["following"] for s in valid if s.get("following") is not None]
    if following:
        metrics["mean_following"] = float(np.mean(following))
        metrics["median_following"] = float(np.median(following))

    # Activity / dormancy
    activity = [s for s in valid if s.get("created_at") and s.get("updated_at")]
    if activity:
        dormant = sum(1 for s in activity if s["created_at"] == s["updated_at"])
        metrics["dormant_rate"] = dormant / len(activity)
        metrics["active_rate"] = 1 - metrics["dormant_rate"]

    return metrics


def bootstrap_metric(
    samples: list[dict],
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple[float, float, float]:
    """Bootstrap a single metric function. Returns (estimate, ci_lower, ci_upper)."""
    if rng is None:
        rng = np.random.default_rng()

    point_estimate = metric_fn(samples)
    boot_estimates = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(len(samples), size=len(samples), replace=True)
        boot_sample = [samples[i] for i in boot_idx]
        boot_estimates.append(metric_fn(boot_sample))

    boot_estimates = np.array(boot_estimates)
    ci_lower = float(np.percentile(boot_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_estimates, 100 * (1 - alpha / 2)))
    return float(point_estimate), ci_lower, ci_upper
