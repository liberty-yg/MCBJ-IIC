"""Plotting script for Bayesian optimisation results.

Reads the JSON files produced by bayesian_opt.py and generates:
  1. Best loss over trials per sampler (convergence curves)
  2. Convergence curve fits (log / linear / quadratic)
  3. Loss vs AMI scatter per trial
  4. AMI mean vs AMI std (stability analysis)
  5. GPU and RAM usage over trials
  6. Collapse rate per sampler
  7. Parameter importance (correlation with loss)

Run with:
    python scripts/plot_bo_results.py --results-dir runs/bayesian_opt_results
    python scripts/plot_bo_results.py --results-dir runs/bayesian_opt_results --output-dir runs/bo_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

SAMPLER_COLOURS = {
    "TPE":    "#2196F3",   # blue
    "CMA-ES": "#F44336",   # red
    "Random": "#4CAF50",   # green
    "GP":     "#FF9800",   # orange
}

def _colour(name: str) -> str:
    return SAMPLER_COLOURS.get(name, "#9C27B0")


def _finalise(fig, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all per-sampler result JSON files from the results directory."""
    results: dict[str, dict] = {}

    # Try combined file first
    combined = results_dir / "comparison_all_samplers.json"
    if combined.exists():
        data = json.loads(combined.read_text())
        for sampler_name, sampler_data in data.items():
            results[sampler_name] = sampler_data
        print(f"Loaded combined results for samplers: {list(results.keys())}")
        return results

    # Fall back to individual files
    for f in sorted(results_dir.glob("results_*.json")):
        sampler_name = f.stem.replace("results_", "")
        results[sampler_name] = json.loads(f.read_text())
        print(f"Loaded {sampler_name} from {f.name}")

    if not results:
        raise FileNotFoundError(
            f"No result JSON files found in {results_dir}. "
            "Run bayesian_opt.py first."
        )
    return results


def extract_trials(sampler_data: dict) -> list[dict]:
    """Return only completed trials with non-None loss values."""
    return [
        t for t in sampler_data.get("all_trials", [])
        if t.get("loss") is not None
    ]


def best_so_far(losses: list[float]) -> list[float]:
    """Return running minimum of losses."""
    current = float("inf")
    result = []
    for v in losses:
        current = min(current, v)
        result.append(current)
    return result


# --------------------------------------------------------------------------
# Plot 1 — Convergence curves (best loss over trials)
# this function plots loss found so far vs trial number per sampler, with individual trial losses as faint scatter points. 
# Shows how fast each sampler improves
# --------------------------------------------------------------------------
def plot_convergence(results: dict[str, dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for sampler_name, data in results.items():
        trials = extract_trials(data)
        if not trials:
            continue
        losses = [t["loss"] for t in trials]
        best = best_so_far(losses)
        x = list(range(len(best)))
        colour = _colour(sampler_name)
        ax.plot(x, best, label=sampler_name, colour=colour, linewidth=2)
        ax.scatter(x, losses, alpha=0.15, s=10, colour=colour)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Loss (lower = better clustering)")
    ax.set_title("Convergence: best loss found over trials")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalise(fig, output_dir / "1_convergence_curves.png")


# --------------------------------------------------------------------------
# Plot 2 — Convergence curve fits
# One panel per sampler, fitting log and lin curves to the best-value trajectory. 
# Shows the predicted improvement over the next 20 trials and R^2 of the fit
# --------------------------------------------------------------------------
def plot_convergence_fits(results: dict[str, dict], output_dir: Path) -> None:
    n_samplers = len(results)
    fig, axes = plt.subplots(1, n_samplers, figsize=(7 * n_samplers, 5), squeeze=False)

    for ax, (sampler_name, data) in zip(axes[0], results.items()):
        trials = extract_trials(data)
        if not trials:
            ax.set_title(f"{sampler_name} — no data")
            continue

        losses = [t["loss"] for t in trials]
        best = np.array(best_so_far(losses))
        x = np.arange(len(best), dtype=float)
        colour = _colour(sampler_name)

        ax.scatter(x, best, s=15, alpha=0.6, colour=colour, label="observed")

        # Logarithmic fit
        fits = data.get("convergence_fits", {})
        if fits.get("log"):
            log = fits["log"]
            a, b = log["a"], log["b"]
            x_pred = np.linspace(0, len(x) * 1.3, 200)
            y_pred = a * np.log(x_pred + 1) + b
            ax.plot(x_pred, y_pred, "--", colour="black",
                    label=f"log fit R²={log['r2']:.3f}")
            pred_imp = log.get("predicted_improvement_next_20_trials", 0)
            ax.annotate(
                f"Δ next 20 trials ≈ {pred_imp:.3f}",
                xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

        # Linear fit
        if fits.get("linear"):
            lin = fits["linear"]
            slope, intercept = lin["slope"], lin["intercept"]
            y_lin = slope * x + intercept
            ax.plot(x, y_lin, ":", colour="grey",
                    label=f"linear fit R²={lin['r2']:.3f}")

        ax.set_xlabel("Trial number")
        ax.set_ylabel("Best loss")
        ax.set_title(f"{sampler_name} convergence fits")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence curve fits per sampler", fontsize=14)
    fig.tight_layout()
    _finalise(fig, output_dir / "2_convergence_fits.png")


# --------------------------------------------------------------------------
# Plot 3 — Loss vs AMI scatter
# Every trial plotted as loss vs AMI. Should show a clear negative correlation, 
# validating thay loss is a good approximator for AMI for unlabelled data
# --------------------------------------------------------------------------
def plot_loss_vs_ami(results: dict[str, dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for sampler_name, data in results.items():
        trials = extract_trials(data)
        losses = [t["loss"] for t in trials if t.get("ami_mean") is not None]
        amis   = [t["ami_mean"] for t in trials if t.get("ami_mean") is not None]
        if not losses:
            continue
        ax.scatter(losses, amis, alpha=0.5, s=20,
                   colour=_colour(sampler_name), label=sampler_name)

    ax.set_xlabel("Final training loss")
    ax.set_ylabel("AMI (mean over repeats)")
    ax.set_title("Loss vs AMI — validating loss as optimisation proxy\n"
                 "(negative correlation confirms loss is a good proxy for AMI)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalise(fig, output_dir / "3_loss_vs_ami.png")


# --------------------------------------------------------------------------
# Plot 4 — AMI mean vs AMI std (stability analysis)
# AMI mean vs AMI std per trial. Ideal configurations appear in the bottom right (high mean, low std). Collapsed trials cluster at (0,0)
# --------------------------------------------------------------------------
def plot_ami_stability(results: dict[str, dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for sampler_name, data in results.items():
        trials = extract_trials(data)
        means = [t["ami_mean"] for t in trials
                 if t.get("ami_mean") is not None and t.get("ami_std") is not None]
        stds  = [t["ami_std"]  for t in trials
                 if t.get("ami_mean") is not None and t.get("ami_std") is not None]
        if not means:
            continue
        ax.scatter(means, stds, alpha=0.5, s=20,
                   colour=_colour(sampler_name), label=sampler_name)

    ax.set_xlabel("AMI mean (across repeats)")
    ax.set_ylabel("AMI std (across repeats)")
    ax.set_title("AMI stability: mean vs std per trial\n"
                 "(ideal configs have high mean AND low std — bottom-right)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate ideal region
    ax.annotate(
        "← ideal region\n(high AMI, stable)",
        xy=(0.75, 0.05), xycoords="axes fraction",
        fontsize=9, color="green",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.4),
    )
    _finalise(fig, output_dir / "4_ami_stability.png")


# --------------------------------------------------------------------------
# Plot 5 — GPU and RAM usage over trials
# Plots memory usage: GPU peak MB and RAM GB over trials for each sampler.
# --------------------------------------------------------------------------
def plot_memory_usage(results: dict[str, dict], output_dir: Path) -> None:
    fig, (ax_gpu, ax_ram) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for sampler_name, data in results.items():
        trials = extract_trials(data)
        colour = _colour(sampler_name)

        gpu_vals = [(t["number"], t["gpu_peak_mb"])
                    for t in trials if t.get("gpu_peak_mb") is not None]
        if gpu_vals:
            x, y = zip(*gpu_vals)
            ax_gpu.plot(x, y, marker="o", ms=3, colour=colour,
                        label=sampler_name, linewidth=1.2)

        ram_vals = [(t["number"], t["ram_used_gb"])
                    for t in trials if t.get("ram_used_gb") is not None]
        if ram_vals:
            x, y = zip(*ram_vals)
            ax_ram.plot(x, y, marker="o", ms=3, colour=colour,
                        label=sampler_name, linewidth=1.2)

    ax_gpu.set_ylabel("GPU peak memory (MB)")
    ax_gpu.set_title("GPU memory usage over trials")
    ax_gpu.legend()
    ax_gpu.grid(True, alpha=0.3)

    ax_ram.set_xlabel("Trial number")
    ax_ram.set_ylabel("RAM used (GB)")
    ax_ram.set_title("System RAM usage over trials")
    ax_ram.legend()
    ax_ram.grid(True, alpha=0.3)

    fig.suptitle("Memory usage across samplers", fontsize=13)
    fig.tight_layout()
    _finalise(fig, output_dir / "5_memory_usage.png")


# --------------------------------------------------------------------------
# Plot 6 — Collapse rate per sampler
# Bar shart showing what percentage of trials collapsed per sampler with counts annotated
# --------------------------------------------------------------------------
def plot_collapse_rate(results: dict[str, dict], output_dir: Path) -> None:
    sampler_names: list[str] = []
    collapse_rates: list[float] = []
    total_trials: list[int] = []

    for sampler_name, data in results.items():
        trials = extract_trials(data)
        if not trials:
            continue
        n_total = len(trials)
        # Collapsed = loss very close to zero
        n_collapsed = sum(1 for t in trials if abs(t["loss"]) < 1e-4)
        sampler_names.append(sampler_name)
        collapse_rates.append(100 * n_collapsed / n_total)
        total_trials.append(n_total)

    if not sampler_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = [_colour(n) for n in sampler_names]
    bars = ax.bar(sampler_names, collapse_rates, colour=colours, alpha=0.8)

    # Annotate bars with counts
    for bar, name, rate, total in zip(bars, sampler_names, collapse_rates, total_trials):
        n_collapsed = int(round(rate * total / 100))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{n_collapsed}/{total}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Collapse rate (%)")
    ax.set_title("Trial collapse rate per sampler\n"
                 "(collapsed = loss ≈ 0, model assigned all traces to one cluster)")
    ax.set_ylim(0, max(collapse_rates) * 1.25 + 5)
    ax.grid(True, alpha=0.3, axis="y")
    _finalise(fig, output_dir / "6_collapse_rate.png")


# --------------------------------------------------------------------------
# Plot 7 — Parameter importance (Spearman correlation with loss)
# Spearman correlation between each hyperparameter value and the final loss across all non-collapsed trials. SHows which parameters matter most. 
# --------------------------------------------------------------------------
def plot_parameter_importance(results: dict[str, dict], output_dir: Path) -> None:
    from scipy.stats import spearmanr

    # Collect all non-collapsed trials across all samplers
    all_trials: list[dict] = []
    for data in results.values():
        all_trials.extend([
            t for t in extract_trials(data)
            if abs(t.get("loss", 0)) > 1e-4  # exclude collapsed
        ])

    if len(all_trials) < 10:
        print("  Skipping parameter importance — not enough non-collapsed trials.")
        return

    param_names = list(all_trials[0].get("params", {}).keys())
    losses = np.array([t["loss"] for t in all_trials])

    correlations: dict[str, float] = {}
    pvalues: dict[str, float] = {}

    for param in param_names:
        values = np.array([t["params"].get(param, np.nan) for t in all_trials])
        mask = ~np.isnan(values)
        if mask.sum() < 5:
            continue
        corr, pval = spearmanr(values[mask], losses[mask])
        correlations[param] = float(corr)
        pvalues[param] = float(pval)

    if not correlations:
        return

    # Sort by absolute correlation
    sorted_params = sorted(correlations, key=lambda k: abs(correlations[k]), reverse=True)
    corr_vals = [correlations[p] for p in sorted_params]
    colours = ["#F44336" if c < 0 else "#2196F3" for c in corr_vals]
    sig_markers = ["*" if pvalues[p] < 0.05 else "" for p in sorted_params]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(sorted_params, corr_vals, colour=colours, alpha=0.8)

    # Mark significant correlations
    for bar, marker in zip(bars, sig_markers):
        if marker:
            ax.text(
                bar.get_width() + 0.01 if bar.get_width() >= 0 else bar.get_width() - 0.01,
                bar.get_y() + bar.get_height() / 2,
                marker, va="center", ha="left" if bar.get_width() >= 0 else "right",
                fontsize=12, color="black",
            )

    ax.axvline(0, colour="black", linewidth=0.8)
    ax.set_xlabel("Spearman correlation with training loss\n"
                  "(negative = lower loss / better, * = p < 0.05)")
    ax.set_title("Parameter importance\n"
                 "(non-collapsed trials only, all samplers combined)")
    ax.grid(True, alpha=0.3, axis="x")

    # Legend for colour meaning
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#F44336", alpha=0.8, label="negative corr (higher value → better loss)"),
        Patch(facecolor="#2196F3", alpha=0.8, label="positive corr (lower value → better loss)"),
    ], fontsize=8, loc="lower right")

    fig.tight_layout()
    _finalise(fig, output_dir / "7_parameter_importance.png")


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------
def print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Sampler':<10} {'Trials':>7} {'Collapsed':>10} {'Best loss':>11} {'Best AMI':>10} {'Time (h)':>10}"
    print(header)
    print("-" * 70)

    for sampler_name, data in results.items():
        trials = extract_trials(data)
        n_total = len(trials)
        n_collapsed = sum(1 for t in trials if abs(t.get("loss", 0)) < 1e-4)
        best_loss = data.get("best_loss", float("inf"))
        best_ami  = data.get("best_ami_mean")
        elapsed   = data.get("elapsed_hours")

        ami_str     = f"{best_ami:.4f}" if best_ami is not None else "N/A"
        elapsed_str = f"{elapsed:.1f}" if elapsed is not None else "N/A"

        print(
            f"{sampler_name:<10} {n_total:>7} {n_collapsed:>10} "
            f"{best_loss:>11.4f} {ami_str:>10} {elapsed_str:>10}"
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Bayesian optimisation results from bayesian_opt.py"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="runs/bayesian_opt_results",
        help="Directory containing the JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save plots (default: <results-dir>/plots)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Saving plots to:      {output_dir}\n")

    results = load_results(results_dir)
    print_summary(results)

    print("\nGenerating plots...")
    plot_convergence(results, output_dir)
    plot_convergence_fits(results, output_dir)
    plot_loss_vs_ami(results, output_dir)
    plot_ami_stability(results, output_dir)
    plot_memory_usage(results, output_dir)
    plot_collapse_rate(results, output_dir)
    plot_parameter_importance(results, output_dir)

    print(f"\nAll plots saved to {output_dir}")