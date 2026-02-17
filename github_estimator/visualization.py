"""Figure generation — clean, publication-ready plots."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from . import config

# ─── Global Style ─────────────────────────────────────────────────────────────

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#d1d5db",
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _fmt(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.0f}K"
    return f"{n:.0f}"


# ─── Figure 1: Unbiasedness ──────────────────────────────────────────────────

def plot_unbiasedness_proof(estimates_by_budget, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig1_unbiasedness.png")
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    budgets = sorted(estimates_by_budget.keys())

    all_ests = []
    for b in budgets:
        all_ests.extend(estimates_by_budget[b])
    overall_mean = np.mean(all_ests)

    for i, budget in enumerate(budgets):
        ests = np.array(estimates_by_budget[budget])
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(ests))
        x = np.full(len(ests), i) + jitter

        above = np.sum(ests > overall_mean)
        below = np.sum(ests <= overall_mean)
        total = len(ests)

        ax.scatter(x, ests / 1e6, color=PALETTE[i % len(PALETTE)],
                   alpha=0.6, s=50, edgecolor="white", linewidth=0.5, zorder=3)

        ax.annotate(f"{above}/{total} above\n{below}/{total} below",
                    xy=(i, np.median(ests) / 1e6),
                    xytext=(0.35, 0), textcoords="offset fontsize",
                    fontsize=9, color="#6b7280",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e5e7eb", alpha=0.9))

    ax.axhline(y=overall_mean / 1e6, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Overall mean: {_fmt(overall_mean)}")

    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([_fmt(b) for b in budgets])
    ax.set_xlabel("Sampling Budget")
    ax.set_ylabel("Estimated Valid Users (millions)")
    ax.set_title("Unbiasedness: Estimates Scatter Evenly Around the Mean")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 2: Correctness ──────────────────────────────────────────────────

def plot_correctness_proof(estimates_by_budget, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig2_correctness.png")
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    budgets = sorted(estimates_by_budget.keys())

    means, ses = [], []
    for b in budgets:
        arr = np.array(estimates_by_budget[b])
        means.append(arr.mean())
        ses.append(arr.std() / np.sqrt(len(arr)))

    means = np.array(means)
    ses = np.array(ses)
    overall_mean = means.mean()
    x = range(len(budgets))

    ax.errorbar(x, means / 1e6, yerr=1.96 * ses / 1e6,
                fmt="o-", capsize=8, capthick=2, linewidth=2.5, markersize=10,
                color=PALETTE[0], markerfacecolor="white", markeredgewidth=2,
                label="Mean +/- 95% CI", zorder=3)

    ax.axhline(y=overall_mean / 1e6, color="#ef4444", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Grand mean: {_fmt(overall_mean)}")

    for i, b in enumerate(budgets):
        ax.annotate(f"{_fmt(means[i])}\n+/-{_fmt(1.96*ses[i])}",
                    xy=(i, means[i] / 1e6), xytext=(0, 12),
                    textcoords="offset points", fontsize=9, ha="center", color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels([_fmt(b) for b in budgets])
    ax.set_xlabel("Sampling Budget")
    ax.set_ylabel("Mean Estimate (millions)")
    ax.set_title("Correctness: Mean is Stable, Variance Decreases with Budget")
    ax.legend(framealpha=0.9)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 3: Relative Error Boxplot ────────────────────────────────────────

def plot_relative_error_boxplot(error_data, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig3_relative_error.png")
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(11, 6))
    rates = sorted(error_data.keys())
    data = [error_data[r] for r in rates]
    labels = [f"{r*100:.1f}%" for r in rates]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    widths=0.6,
                    flierprops=dict(marker=".", markerfacecolor="#9ca3af", markersize=4, alpha=0.4),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#6b7280"),
                    capprops=dict(color="#6b7280"))

    cmap = plt.cm.get_cmap("Blues")
    colors = [cmap(0.3 + 0.6 * i / len(rates)) for i in range(len(rates))]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#374151")
        patch.set_linewidth(1)

    ax.axhline(y=0, color="#ef4444", linestyle="-", linewidth=1.5, alpha=0.7, label="Zero error")
    ax.axhspan(-0.02, 0.02, alpha=0.08, color="#10b981", label="+/-2% band")

    ax.set_xlabel("Sampling Rate")
    ax.set_ylabel("Relative Error")
    ax.set_title("Estimation Error Shrinks with Higher Sampling Rates")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 4: Validity by Stratum ──────────────────────────────────────────

def plot_density_by_stratum(stratum_rates, stratum_cis, stratum_labels=None, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig4_density_stratum.png")
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(11, 6))
    strata = list(stratum_rates.keys())
    rates = [stratum_rates[s] * 100 for s in strata]
    labels = [stratum_labels.get(s, s).replace("\n", " ") if stratum_labels else s for s in strata]

    if stratum_cis:
        lower = [stratum_cis[s][0] * 100 for s in strata]
        upper = [stratum_cis[s][1] * 100 for s in strata]
        yerr = [[r - l for r, l in zip(rates, lower)],
                [u - r for r, u in zip(rates, upper)]]
    else:
        yerr = None

    cmap = plt.cm.get_cmap("RdYlGn")
    colors = [cmap(0.25 + 0.6 * r / 100) for r in rates]

    bars = ax.bar(range(len(strata)), rates, yerr=yerr, capsize=5,
                  color=colors, edgecolor="#374151", linewidth=0.8, width=0.65)

    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1f2937")

    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Validity Rate (%)")
    ax.set_title("Account Validity Rate by ID Range")
    ax.set_ylim(min(rates) - 10, 105)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 5: Bootstrap Distribution ────────────────────────────────────────

def plot_bootstrap_distribution(bootstrap_estimates, point_estimate, ci_lower, ci_upper, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig5_bootstrap_dist.png")
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    arr = np.array(bootstrap_estimates) / 1e6

    ax.hist(arr, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85, linewidth=0.5)

    ax.axvline(point_estimate / 1e6, color="#ef4444", linewidth=2.5,
               label=f"Point estimate: {_fmt(point_estimate)}")
    ax.axvline(ci_lower / 1e6, color="#f59e0b", linewidth=2, linestyle="--",
               label=f"95% CI: [{_fmt(ci_lower)}, {_fmt(ci_upper)}]")
    ax.axvline(ci_upper / 1e6, color="#f59e0b", linewidth=2, linestyle="--")

    ax.set_xlabel("Estimated Valid Users (millions)")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap Distribution of Total Estimate (B=1,000)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 6: Multi-Metric Summary ─────────────────────────────────────────

def plot_multi_metric_summary(metrics, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig6_multi_metric.png")
    _ensure_dir(output_path)

    names = list(metrics.keys())
    estimates = [metrics[n][0] for n in names]
    ci_lowers = [metrics[n][1] for n in names]
    ci_uppers = [metrics[n][2] for n in names]
    errors_lower = [e - l for e, l in zip(estimates, ci_lowers)]
    errors_upper = [u - e for e, u in zip(estimates, ci_uppers)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8)))
    y_pos = range(len(names))

    ax.barh(y_pos, estimates, xerr=[errors_lower, errors_upper],
            capsize=4, color=PALETTE[0], edgecolor="white", alpha=0.85, height=0.55)

    for i, (est, cl, cu) in enumerate(zip(estimates, ci_lowers, ci_uppers)):
        ax.text(max(est, cu) + (cu - cl) * 0.3, i,
                f"  {est:.2f}  [{cl:.2f}, {cu:.2f}]",
                va="center", fontsize=9, color="#374151")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Value")
    ax.set_title("Multi-Metric Summary with 95% Bootstrap CIs")
    ax.invert_yaxis()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 7: Stratum Heatmap ──────────────────────────────────────────────

def plot_stratum_heatmap(data, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig7_stratum_heatmap.png")
    _ensure_dir(output_path)

    import pandas as pd
    df = pd.DataFrame(data).T
    df.columns = [c.replace("_", " ").title() for c in df.columns]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.9)))
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-10)

    sns.heatmap(df_norm, annot=df.round(3).values, fmt="",
                cmap="YlOrRd", ax=ax, linewidths=1, linecolor="white",
                cbar_kws={"label": "Normalized", "shrink": 0.8},
                annot_kws={"fontsize": 10})
    ax.set_title("Metric Comparison Across Strata")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 8: Time-Accuracy Tradeoff ───────────────────────────────────────

def plot_time_accuracy_tradeoff(sampling_rates, mean_errors, mean_times, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig8_time_accuracy.png")
    _ensure_dir(output_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    rates_pct = [r * 100 for r in sampling_rates]

    ax1.plot(rates_pct, [abs(e) * 100 for e in mean_errors], "o-",
             color=PALETTE[0], linewidth=2.5, markersize=8, label="Mean |Error| (%)")
    ax1.set_xlabel("Sampling Rate (%)")
    ax1.set_ylabel("Mean Absolute Error (%)", color=PALETTE[0])

    ax2.plot(rates_pct, mean_times, "s-",
             color=PALETTE[1], linewidth=2.5, markersize=8, label="Time (sec)")
    ax2.set_ylabel("Processing Time (sec)", color=PALETTE[1])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)
    ax1.set_title("Accuracy vs Processing Time Trade-off")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 9: Convergence ──────────────────────────────────────────────────

def plot_convergence(running_estimates, running_cis=None, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig9_convergence.png")
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(1, len(running_estimates) + 1) * 100  # Each point = 100 samples
    y = np.array(running_estimates) / 1e6

    ax.plot(x, y, color=PALETTE[0], linewidth=1.2, alpha=0.8)

    if running_cis:
        lower = np.array([ci[0] for ci in running_cis]) / 1e6
        upper = np.array([ci[1] for ci in running_cis]) / 1e6
        ax.fill_between(x, lower, upper, alpha=0.15, color=PALETTE[0])

    final = y[-1]
    ax.axhline(y=final, color="#ef4444", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Final estimate: {final:.1f}M")

    ax.set_xlabel("Cumulative Samples Collected")
    ax.set_ylabel("Running Estimate (millions)")
    ax.set_title("Estimation Convergence Over Sample Collection")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt(v)))
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Figure 10: Validation Scatter ──────────────────────────────────────────

def plot_validation_scatter(ground_truths, estimates, output_path=None):
    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, "fig10_validation_scatter.png")
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    strata = [s for s in ground_truths if s in estimates]
    gt = [ground_truths[s] for s in strata]
    est = [estimates[s] for s in strata]

    ax.scatter(gt, est, s=120, color=PALETTE[0], edgecolor="white", linewidth=1.5, zorder=3)

    for i, s in enumerate(strata):
        offset_x = 5
        offset_y = -12 if i % 2 == 0 else 10
        ax.annotate(s, (gt[i], est[i]), xytext=(offset_x, offset_y),
                    textcoords="offset points", fontsize=10, color="#374151")

    all_vals = gt + est
    mn, mx = min(all_vals) * 0.95, max(all_vals) * 1.05
    ax.plot([mn, mx], [mn, mx], color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.6, label="y = x (perfect)")

    gt_arr = np.array(gt, dtype=float)
    est_arr = np.array(est, dtype=float)
    ss_res = np.sum((est_arr - gt_arr) ** 2)
    ss_tot = np.sum((gt_arr - gt_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    ax.text(0.05, 0.93, f"R$^2$ = {r2:.4f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d1d5db"))

    ax.set_xlabel("Ground Truth Valid Count")
    ax.set_ylabel("Estimated Valid Count")
    ax.set_title("Validation: Estimated vs Ground Truth")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ─── Generate All ────────────────────────────────────────────────────────────

def generate_all_figures(results, output_dir=None):
    if output_dir is None:
        output_dir = config.FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    paths = []

    if "estimates_by_budget" in results:
        paths.append(plot_unbiasedness_proof(
            results["estimates_by_budget"],
            os.path.join(output_dir, "fig1_unbiasedness.png")))
        paths.append(plot_correctness_proof(
            results["estimates_by_budget"],
            os.path.join(output_dir, "fig2_correctness.png")))

    if "error_by_rate" in results:
        paths.append(plot_relative_error_boxplot(
            results["error_by_rate"],
            os.path.join(output_dir, "fig3_relative_error.png")))

    if "stratum_rates" in results:
        paths.append(plot_density_by_stratum(
            results["stratum_rates"],
            results.get("stratum_cis", {}),
            results.get("stratum_labels", {}),
            os.path.join(output_dir, "fig4_density_stratum.png")))

    if "bootstrap_estimates" in results:
        paths.append(plot_bootstrap_distribution(
            results["bootstrap_estimates"],
            results["point_estimate"],
            results["ci_lower"],
            results["ci_upper"],
            os.path.join(output_dir, "fig5_bootstrap_dist.png")))

    if "multi_metrics" in results:
        paths.append(plot_multi_metric_summary(
            results["multi_metrics"],
            os.path.join(output_dir, "fig6_multi_metric.png")))

    if "stratum_metric_data" in results:
        paths.append(plot_stratum_heatmap(
            results["stratum_metric_data"],
            os.path.join(output_dir, "fig7_stratum_heatmap.png")))

    if "time_accuracy" in results and results["time_accuracy"]:
        ta = results["time_accuracy"]
        paths.append(plot_time_accuracy_tradeoff(
            ta["rates"], ta["errors"], ta["times"],
            os.path.join(output_dir, "fig8_time_accuracy.png")))

    if "convergence" in results and results["convergence"].get("estimates"):
        paths.append(plot_convergence(
            results["convergence"]["estimates"],
            results["convergence"].get("cis"),
            os.path.join(output_dir, "fig9_convergence.png")))

    if "validation_scatter" in results:
        vs = results["validation_scatter"]
        if vs.get("ground_truths") and vs.get("estimates"):
            paths.append(plot_validation_scatter(
                vs["ground_truths"],
                vs["estimates"],
                os.path.join(output_dir, "fig10_validation_scatter.png")))

    return [p for p in paths if p]
