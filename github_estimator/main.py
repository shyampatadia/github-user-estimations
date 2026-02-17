"""Main orchestration: runs all phases of the GitHub User Estimation project.

Phase 3 uses a POOL-AND-PARTITION strategy:
  1. Collect ONE large stratified pool of 92K unique IDs (single API pass)
  2. Shuffle and partition into independent subsamples at various budget levels
  3. Each partition is a proper independent stratified estimate — zero extra API calls
  4. This gives 92 independent runs at 1K, 46 at 2K, 18 at 5K, 9 at 10K, etc.
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

from . import config, database
from .crawler import collect_ground_truth, collect_random_sample, find_frontier
from .estimator import (
    bootstrap_ci_stratified,
    calculate_all_metrics,
    calculate_relative_error,
    estimate_stratified,
    stratified_variance,
    bootstrap_metric,
)
from .sampler import proportional_allocation, stratified_sample
from .token_manager import TokenRotator
from .utils import format_large_number, save_results_to_json, setup_logging
from .validation import (
    run_all_validation_experiments,
    test_correctness,
    test_unbiasedness,
)
from .visualization import generate_all_figures

logger = logging.getLogger(__name__)


# ─── Phase 1: Ground Truth Collection ────────────────────────────────────────

async def phase1_ground_truth(db_conn, rotator: TokenRotator):
    """Collect exhaustive ground truth for all validation strata (V1-V8)."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Ground Truth Collection (8K IDs)")
    logger.info("=" * 60)

    total_ids = sum(s["size"] for s in config.VALIDATION_STRATA.values())
    pbar = tqdm(total=total_ids, desc="Ground Truth", unit="ids")

    def _progress(n):
        pbar.update(n)

    for i, (stratum_id, stratum_config) in enumerate(config.VALIDATION_STRATA.items(), 1):
        already = database.get_ground_truth_count(db_conn, stratum_id)
        if already >= stratum_config["size"]:
            logger.info(f"[{i}/8] Stratum {stratum_id}: Already complete ({already} records)")
            pbar.update(stratum_config["size"])
            continue

        remaining = stratum_config["size"] - already
        logger.info(f"[{i}/8] Stratum {stratum_id} ({stratum_config['description']}): "
                     f"collecting {remaining} IDs (API calls used so far: {rotator.total_used:,})")
        pbar.update(already)
        await collect_ground_truth(
            stratum_id, stratum_config, db_conn, rotator, progress_callback=_progress
        )
        valid = database.get_ground_truth_valid_count(db_conn, stratum_id)
        total = database.get_ground_truth_count(db_conn, stratum_id)
        logger.info(f"[{i}/8] Stratum {stratum_id} done: {valid}/{total} valid ({valid/total:.1%})")

    pbar.close()

    # Summary
    logger.info("\nGround Truth Summary:")
    logger.info(f"{'Stratum':<10} {'Total':>8} {'Valid':>8} {'Rate':>8}")
    logger.info("-" * 40)
    for sid in config.VALIDATION_STRATA:
        total = database.get_ground_truth_count(db_conn, sid)
        valid = database.get_ground_truth_valid_count(db_conn, sid)
        rate = valid / total if total > 0 else 0
        logger.info(f"{sid:<10} {total:>8} {valid:>8} {rate:>8.1%}")


# ─── Phase 2: Validation Experiments ─────────────────────────────────────────

def phase2_validation(db_conn) -> dict:
    """Run validation experiments (no API calls, uses ground truth)."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Validation Experiments (0 API calls)")
    logger.info("=" * 60)

    rng = np.random.default_rng(42)
    all_results = run_all_validation_experiments(
        db_conn,
        sampling_rates=config.SAMPLING_RATES,
        n_reps=config.DEFAULT_REPETITIONS,
        rng=rng,
    )

    # Aggregate error data for plotting
    error_by_rate = {}
    for rate in config.SAMPLING_RATES:
        errors = []
        for stratum_id, stratum_results in all_results.items():
            if rate in stratum_results:
                for run in stratum_results[rate]:
                    if run["relative_error"] is not None:
                        errors.append(run["relative_error"])
        error_by_rate[rate] = errors

    # Validation scatter data
    ground_truths = {}
    mean_estimates = {}
    for sid in config.VALIDATION_STRATA:
        valid = database.get_ground_truth_valid_count(db_conn, sid)
        ground_truths[sid] = valid
        if sid in all_results and 0.10 in all_results[sid]:
            ests = [r["estimated_total"] for r in all_results[sid][0.10]]
            mean_estimates[sid] = float(np.mean(ests))

    logger.info("Phase 2 complete.")
    return {
        "error_by_rate": error_by_rate,
        "validation_scatter": {
            "ground_truths": ground_truths,
            "estimates": mean_estimates,
        },
        "all_results": all_results,
    }


# ─── Phase 3: Pool-and-Partition Full Space Estimation ────────────────────────

async def phase3_full_space(db_conn, rotator: TokenRotator) -> dict:
    """
    Collect ONE large stratified pool, then partition into independent subsamples.

    Strategy:
      1. Draw 92K unique stratified random IDs across the full ID space
      2. Fetch all of them (single API pass, ~18.4 hrs)
      3. Shuffle within each stratum, then partition into groups of size B
      4. Each group → one independent stratified estimate
      5. Repeat for B = 1K, 2K, 5K, 10K to prove correctness & unbiasedness
    """
    logger.info("=" * 60)
    logger.info(f"PHASE 3: Full Space Pool Collection ({format_large_number(config.FULL_SPACE_POOL_SIZE)} IDs)")
    logger.info("=" * 60)

    rng = np.random.default_rng(123)
    strata_sizes = {sid: s["size"] for sid, s in config.FULL_SPACE_STRATA.items()}
    pool_size = config.FULL_SPACE_POOL_SIZE
    run_id = "pool_main"

    # Step 1: Generate stratified sample IDs for the full pool
    sample_ids = stratified_sample(config.FULL_SPACE_STRATA, pool_size, rng)
    allocation = {sid: len(ids) for sid, ids in sample_ids.items()}
    logger.info("Pool allocation per stratum:")
    for sid, n in allocation.items():
        logger.info(f"  {sid}: {n:,} IDs")

    # Step 2: Collect pool samples via API (with resume support)
    already_collected = database.get_collected_sample_ids_for_run(db_conn, run_id)
    already_count = len(already_collected)

    if already_count >= pool_size:
        logger.info(f"Pool already fully collected ({already_count:,} samples). Loading from DB...")
    elif already_count > 0:
        logger.info(f"Resuming pool collection: {already_count:,}/{pool_size:,} already done")
    else:
        logger.info(f"Starting fresh pool collection: {pool_size:,} IDs")

    # Fetch only the IDs we haven't collected yet
    remaining_by_stratum = {}
    total_remaining = 0
    for sid, ids in sample_ids.items():
        remaining = [uid for uid in ids if uid not in already_collected]
        remaining_by_stratum[sid] = remaining
        total_remaining += len(remaining)

    if total_remaining > 0:
        pbar = tqdm(total=total_remaining, desc="Pool Collection", unit="ids")
        collected_in_session = 0

        def _progress(n):
            nonlocal collected_in_session
            collected_in_session += n
            pbar.update(n)

        for sid, ids in remaining_by_stratum.items():
            if not ids:
                logger.info(f"  Stratum {sid}: already complete")
                continue
            logger.info(f"  Stratum {sid}: {len(ids):,} IDs to fetch")
            for chunk_start in range(0, len(ids), 1000):
                chunk_ids = ids[chunk_start:chunk_start + 1000]
                await collect_random_sample(
                    chunk_ids, run_id, sid, db_conn, rotator, progress_callback=_progress
                )
                # Log progress every 1000 IDs
                total_done = already_count + collected_in_session
                logger.info(f"  Progress: {total_done:,}/{pool_size:,} "
                            f"({total_done/pool_size:.0%}) | API calls: {rotator.total_used:,}")

        pbar.close()

    # Load all pool samples from DB (whether freshly collected or resumed)
    all_pool_samples = database.get_samples_by_run(db_conn, run_id)
    pool_by_stratum: dict[str, list[dict]] = {sid: [] for sid in config.FULL_SPACE_STRATA}
    for sample in all_pool_samples:
        sid = sample.get("stratum_id")
        if sid in pool_by_stratum:
            pool_by_stratum[sid].append(sample)

    logger.info(f"Pool ready: {len(all_pool_samples):,} samples")

    # Step 3: Shuffle within each stratum (for random partitioning)
    for sid in pool_by_stratum:
        rng.shuffle(pool_by_stratum[sid])

    # Step 4: Partition and estimate at each budget level
    logger.info("\n" + "=" * 60)
    logger.info("PARTITION ANALYSIS")
    logger.info("=" * 60)

    estimates_by_budget = {}

    for budget in config.PARTITION_BUDGETS:
        # Proportional allocation for this budget
        budget_allocation = proportional_allocation(config.FULL_SPACE_STRATA, budget)

        # How many complete independent partitions can we make?
        n_partitions = min(
            len(pool_by_stratum[sid]) // budget_allocation[sid]
            for sid in config.FULL_SPACE_STRATA
            if budget_allocation[sid] > 0
        )
        if n_partitions < 1:
            logger.warning(f"Budget {format_large_number(budget)}: not enough data for even 1 partition, skipping")
            continue

        logger.info(f"\nBudget {format_large_number(budget)}: {n_partitions} independent partitions")
        budget_estimates = []

        for p in range(n_partitions):
            stratum_results = {}
            stratum_validity = {}

            for sid in config.FULL_SPACE_STRATA:
                n_h = budget_allocation[sid]
                start_idx = p * n_h
                end_idx = start_idx + n_h
                partition = pool_by_stratum[sid][start_idx:end_idx]

                valid = sum(1 for r in partition if r["is_valid"])
                stratum_results[sid] = {"valid": valid, "total": len(partition)}
                stratum_validity[sid] = np.array([1 if r["is_valid"] else 0 for r in partition])

            est = estimate_stratified(stratum_results, strata_sizes)
            budget_estimates.append(est)

            # Store each partition as an estimation run
            part_run_id = f"part_{budget}_{p:03d}"
            run = {
                "id": part_run_id,
                "run_type": "full_space_partition",
                "stratum_id": None,
                "sample_size": budget,
                "sampling_rate": budget / config.FRONTIER_M,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "valid_count": sum(r["valid"] for r in stratum_results.values()),
                "total_sampled": sum(r["total"] for r in stratum_results.values()),
                "estimated_total": est,
                "estimated_rate": est / config.FRONTIER_M,
                "ci_lower": None,
                "ci_upper": None,
                "bootstrap_samples": None,
                "ground_truth_total": None,
                "relative_error": None,
                "notes": f"Partition {p+1}/{n_partitions} at budget={budget}",
            }
            database.insert_estimation_run(db_conn, run)

        estimates_by_budget[budget] = budget_estimates

        # Summary for this budget
        arr = np.array(budget_estimates)
        unbias = test_unbiasedness(budget_estimates)
        logger.info(f"  Mean: {format_large_number(arr.mean())}  "
                     f"Std: {format_large_number(arr.std())}  "
                     f"Above/Below mean: {unbias['above_pct']:.0f}%/{unbias['below_pct']:.0f}%")

    # Step 5: Full-pool estimate with bootstrap CI
    logger.info("\n" + "=" * 60)
    logger.info("FULL POOL ESTIMATE (all 92K samples)")
    logger.info("=" * 60)

    full_stratum_results = {}
    full_stratum_validity = {}
    for sid in config.FULL_SPACE_STRATA:
        samples = pool_by_stratum[sid]
        valid = sum(1 for r in samples if r["is_valid"])
        full_stratum_results[sid] = {"valid": valid, "total": len(samples)}
        full_stratum_validity[sid] = np.array([1 if r["is_valid"] else 0 for r in samples])

    point_estimate = estimate_stratified(full_stratum_results, strata_sizes)
    ci_lower, ci_upper, boot_ests = bootstrap_ci_stratified(
        full_stratum_validity, strata_sizes,
        n_bootstrap=config.BOOTSTRAP_ITERATIONS, rng=rng,
    )

    logger.info(f"Point Estimate:   {format_large_number(point_estimate)}")
    logger.info(f"95% Bootstrap CI: [{format_large_number(ci_lower)}, {format_large_number(ci_upper)}]")
    logger.info(f"Validity Rate:    {point_estimate / config.FRONTIER_M:.2%}")

    # Stratum-level rates and CIs
    stratum_rates = {}
    stratum_cis = {}
    stratum_labels = {}
    for sid, s in config.FULL_SPACE_STRATA.items():
        arr = full_stratum_validity[sid]
        rate = float(arr.mean())
        stratum_rates[sid] = rate
        boot_rates = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(1000)]
        stratum_cis[sid] = (
            float(np.percentile(boot_rates, 2.5)),
            float(np.percentile(boot_rates, 97.5)),
        )
        stratum_labels[sid] = f"{sid}\n({format_large_number(s['start'])}-{format_large_number(s['end'])})"

    # Convergence data (running estimate as pool grows)
    convergence_estimates = []
    running_valid = 0
    running_total = 0
    for sample in all_pool_samples:
        running_total += 1
        if sample["is_valid"]:
            running_valid += 1
        if running_total % 100 == 0:
            convergence_estimates.append((running_valid / running_total) * config.FRONTIER_M)

    # Multi-metric bootstrap CIs
    multi_metrics = {}
    metric_fns = {
        "Validity Rate (%)": lambda s: sum(1 for x in s if x.get("is_valid")) / len(s) * 100 if s else 0,
        "User Rate (%)": lambda s: sum(1 for x in s if x.get("type") == "User") / max(1, sum(1 for x in s if x.get("is_valid"))) * 100,
        "Org Rate (%)": lambda s: sum(1 for x in s if x.get("type") == "Organization") / max(1, sum(1 for x in s if x.get("is_valid"))) * 100,
        "Mean Public Repos": lambda s: float(np.mean([x.get("public_repos", 0) or 0 for x in s if x.get("is_valid")])) if any(x.get("is_valid") for x in s) else 0,
        "Empty Account Rate (%)": lambda s: sum(1 for x in s if x.get("is_valid") and (x.get("public_repos") or 0) == 0) / max(1, sum(1 for x in s if x.get("is_valid"))) * 100,
        "Mean Followers": lambda s: float(np.mean([x.get("followers", 0) or 0 for x in s if x.get("is_valid")])) if any(x.get("is_valid") for x in s) else 0,
    }
    for mname, mfn in metric_fns.items():
        est, ci_l, ci_u = bootstrap_metric(all_pool_samples, mfn, n_bootstrap=500, rng=rng)
        multi_metrics[mname] = (est, ci_l, ci_u)

    # Store metrics
    metric_records = []
    for name, (est, ci_l, ci_u) in multi_metrics.items():
        metric_records.append({
            "run_id": run_id,
            "metric_name": name,
            "metric_value": est,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
        })
    database.insert_metrics_batch(db_conn, metric_records)

    # Stratum metric data for heatmap
    stratum_metric_data = {}
    for sid in config.FULL_SPACE_STRATA:
        samples = pool_by_stratum[sid]
        valid_samples = [s for s in samples if s.get("is_valid")]
        stratum_metric_data[sid] = {
            "validity_rate": float(full_stratum_validity[sid].mean()),
            "empty_rate": sum(1 for s in valid_samples if (s.get("public_repos") or 0) == 0) / max(1, len(valid_samples)),
            "mean_repos": float(np.mean([s.get("public_repos", 0) or 0 for s in valid_samples])) if valid_samples else 0,
            "mean_followers": float(np.mean([s.get("followers", 0) or 0 for s in valid_samples])) if valid_samples else 0,
        }

    # Correctness test
    correctness = test_correctness({
        b: ests for b, ests in estimates_by_budget.items() if len(ests) >= 2
    })
    logger.info(f"\nCorrectness test: max deviation = {correctness['max_relative_deviation']:.3%} "
                f"({'PASS' if correctness['is_correct'] else 'FAIL'})")

    return {
        "estimates_by_budget": {b: ests for b, ests in estimates_by_budget.items()},
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_estimates": boot_ests,
        "stratum_rates": stratum_rates,
        "stratum_cis": stratum_cis,
        "stratum_labels": stratum_labels,
        "multi_metrics": multi_metrics,
        "stratum_metric_data": stratum_metric_data,
        "time_accuracy": None,
        "convergence": {"estimates": convergence_estimates},
    }


# ─── Load Phase 3 Results from DB ─────────────────────────────────────────────

def load_phase3_from_db(db_conn) -> dict:
    """Reconstruct Phase 3 results from the database (no API calls)."""
    rng = np.random.default_rng(123)
    strata_sizes = {sid: s["size"] for sid, s in config.FULL_SPACE_STRATA.items()}

    # Load all pool samples
    all_pool_samples = database.get_samples_by_run(db_conn, "pool_main")
    pool_by_stratum: dict[str, list[dict]] = {sid: [] for sid in config.FULL_SPACE_STRATA}
    for sample in all_pool_samples:
        sid = sample.get("stratum_id")
        if sid in pool_by_stratum:
            pool_by_stratum[sid].append(sample)

    logger.info(f"Loaded {len(all_pool_samples):,} pool samples from DB")

    # Shuffle for partitioning
    for sid in pool_by_stratum:
        rng.shuffle(pool_by_stratum[sid])

    # Reconstruct estimates_by_budget from estimation_runs table
    runs = database.get_estimation_runs(db_conn, "full_space_partition")
    estimates_by_budget = {}
    for run in runs:
        budget = run["sample_size"]
        if budget not in estimates_by_budget:
            estimates_by_budget[budget] = []
        estimates_by_budget[budget].append(run["estimated_total"])

    # Full-pool validity arrays
    full_stratum_validity = {}
    for sid in config.FULL_SPACE_STRATA:
        samples = pool_by_stratum[sid]
        full_stratum_validity[sid] = np.array([1 if r["is_valid"] else 0 for r in samples])

    # Full-pool stratified estimate
    full_stratum_results = {}
    for sid in config.FULL_SPACE_STRATA:
        samples = pool_by_stratum[sid]
        valid = sum(1 for r in samples if r["is_valid"])
        full_stratum_results[sid] = {"valid": valid, "total": len(samples)}

    point_estimate = estimate_stratified(full_stratum_results, strata_sizes)

    # Bootstrap CI
    ci_lower, ci_upper, boot_ests = bootstrap_ci_stratified(
        full_stratum_validity, strata_sizes,
        n_bootstrap=config.BOOTSTRAP_ITERATIONS, rng=rng,
    )

    # Stratum rates + CIs
    stratum_rates = {}
    stratum_cis = {}
    stratum_labels = {}
    for sid, s in config.FULL_SPACE_STRATA.items():
        arr = full_stratum_validity[sid]
        if len(arr) == 0:
            continue
        rate = float(arr.mean())
        stratum_rates[sid] = rate
        boot_rates = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(1000)]
        stratum_cis[sid] = (float(np.percentile(boot_rates, 2.5)), float(np.percentile(boot_rates, 97.5)))
        stratum_labels[sid] = f"{sid}\n({format_large_number(s['start'])}-{format_large_number(s['end'])})"

    # Convergence
    convergence_estimates = []
    running_valid = 0
    running_total = 0
    for sample in all_pool_samples:
        running_total += 1
        if sample["is_valid"]:
            running_valid += 1
        if running_total % 100 == 0:
            convergence_estimates.append((running_valid / running_total) * config.FRONTIER_M)

    # Multi-metrics from DB
    metrics_rows = db_conn.execute("SELECT * FROM metrics WHERE run_id = 'pool_main'").fetchall()
    multi_metrics = {}
    for m in metrics_rows:
        multi_metrics[m["metric_name"]] = (m["metric_value"], m["ci_lower"], m["ci_upper"])

    # Stratum metric data for heatmap
    stratum_metric_data = {}
    for sid in config.FULL_SPACE_STRATA:
        samples = pool_by_stratum[sid]
        valid_samples = [s for s in samples if s.get("is_valid")]
        if not valid_samples:
            continue
        stratum_metric_data[sid] = {
            "validity_rate": float(full_stratum_validity[sid].mean()),
            "empty_rate": sum(1 for s in valid_samples if (s.get("public_repos") or 0) == 0) / len(valid_samples),
            "mean_repos": float(np.mean([s.get("public_repos", 0) or 0 for s in valid_samples])),
            "mean_followers": float(np.mean([s.get("followers", 0) or 0 for s in valid_samples])),
        }

    logger.info(f"Point estimate: {format_large_number(point_estimate)} "
                f"[{format_large_number(ci_lower)}, {format_large_number(ci_upper)}]")

    return {
        "estimates_by_budget": estimates_by_budget,
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_estimates": boot_ests,
        "stratum_rates": stratum_rates,
        "stratum_cis": stratum_cis,
        "stratum_labels": stratum_labels,
        "multi_metrics": multi_metrics,
        "stratum_metric_data": stratum_metric_data,
        "time_accuracy": None,
        "convergence": {"estimates": convergence_estimates},
    }


# ─── Phase 4: Generate Figures ───────────────────────────────────────────────

def phase4_figures(results: dict):
    """Generate all figures from collected results."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Figure Generation")
    logger.info("=" * 60)

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    paths = generate_all_figures(results, config.FIGURES_DIR)
    for p in paths:
        logger.info(f"  Generated: {p}")

    save_results_to_json(results, os.path.join(config.DATA_OUTPUT_DIR, "all_results.json"))
    logger.info(f"  Saved results to {config.DATA_OUTPUT_DIR}/all_results.json")

    return paths


# ─── Full Pipeline ────────────────────────────────────────────────────────────

async def run_full_pipeline(skip_phase1=False, skip_phase3=False):
    """Run the complete estimation pipeline."""
    setup_logging()
    logger.info("Starting GitHub User Estimation Pipeline")
    logger.info(f"Rate limit: {config.RATE_LIMIT_PER_HOUR} req/hr (shared account)")

    db_conn = database.get_connection()
    database.create_tables(db_conn)

    need_api = not skip_phase1 or not skip_phase3
    rotator = TokenRotator() if need_api else None

    results = {}

    # Phase 1: Ground Truth
    if not skip_phase1:
        await phase1_ground_truth(db_conn, rotator)
    else:
        logger.info("Skipping Phase 1 (ground truth already collected)")

    # Phase 2: Validation (no API calls)
    phase2_results = phase2_validation(db_conn)
    results.update(phase2_results)

    # Phase 3: Full Space (pool-and-partition)
    if not skip_phase3:
        phase3_results = await phase3_full_space(db_conn, rotator)
        results.update(phase3_results)
    else:
        logger.info("Skipping Phase 3 — loading results from DB")
        phase3_results = load_phase3_from_db(db_conn)
        results.update(phase3_results)

    # Phase 4: Figures
    phase4_figures(results)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total API calls: {rotator.total_used if rotator else 0}")
    logger.info("=" * 60)

    db_conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="GitHub User Estimation Pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="Run only a specific phase")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip ground truth collection")
    parser.add_argument("--skip-phase3", action="store_true",
                        help="Skip full-space estimation (API intensive)")
    args = parser.parse_args()

    if args.phase:
        setup_logging()
        db_conn = database.get_connection()
        database.create_tables(db_conn)

        if args.phase == 1:
            rotator = TokenRotator()
            asyncio.run(phase1_ground_truth(db_conn, rotator))
        elif args.phase == 2:
            phase2_validation(db_conn)
        elif args.phase == 3:
            rotator = TokenRotator()
            asyncio.run(phase3_full_space(db_conn, rotator))
        elif args.phase == 4:
            results_path = os.path.join(config.DATA_OUTPUT_DIR, "all_results.json")
            if os.path.exists(results_path):
                from .utils import load_results_from_json
                results = load_results_from_json(results_path)
                phase4_figures(results)
            else:
                logger.error(f"No results file found at {results_path}. Run phases 2-3 first.")

        db_conn.close()
    else:
        asyncio.run(run_full_pipeline(
            skip_phase1=args.skip_phase1,
            skip_phase3=args.skip_phase3,
        ))


if __name__ == "__main__":
    main()
