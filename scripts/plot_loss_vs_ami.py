#!/usr/bin/env python3
"""Plot training loss vs AMI for TPE trials to validate loss as proxy."""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path


def plot_loss_vs_ami_tpe(
    results_dir: str,
    output_path: str,
    title: str = "Loss vs AMI — TPE trials",
) -> None:

    summary_path = f"{results_dir}/results_TPE.json"
    data = json.loads(open(summary_path).read())
    trials = data.get("all_trials", [])

    losses = []
    amis   = []
    ks     = []

    for t in trials:
        loss     = t.get("loss")
        ami_mean = t.get("ami_mean")
        k        = t.get("params", {}).get("num_clusters", 1)

        if loss is None or ami_mean is None:
            continue
        if abs(loss) < 1e-4:
            continue

        losses.append(loss)
        amis.append(ami_mean)
        ks.append(k)

    losses = np.array(losses)
    amis   = np.array(amis)
    ks     = np.array(ks)

    rho, p = spearmanr(losses, amis)
    print(f"n={len(losses)}, Spearman rho={rho:.3f}, p={p:.3e}")

    # Colour palette per k value
    k_values = sorted(np.unique(ks))
    cmap     = plt.cm.get_cmap("tab10", len(k_values))
    colours  = {k: cmap(i) for i, k in enumerate(k_values)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for k_val in k_values:
        mask = ks == k_val
        ax.scatter(
            losses[mask], amis[mask],
            label=f"k={k_val}",
            color=colours[k_val],
            alpha=0.7, s=45, edgecolors="none",
        )

    # Overall trendline
    coeffs = np.polyfit(losses, amis, deg=1)
    x_line = np.linspace(losses.min(), losses.max(), 200)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, color="black", linewidth=1.5,
            linestyle="--", alpha=0.7,
            label=f"Overall fit ($\\rho={rho:.2f}$, $p={p:.2e}$)")

    ax.set_xlabel("Final training loss", fontsize=12)
    ax.set_ylabel("AMI (mean over 3 repeats)", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")




if __name__ == "__main__":
    plot_loss_vs_ami_tpe(
        results_dir="runs/bayesian_opt_tpe_200_run1",
        output_path="runs/loss_vs_ami_run1.png",
        title="Loss vs AMI — TPE, 74 trials (new architecture)",
    )

    plot_loss_vs_ami_tpe(
        results_dir="runs/bayesian_opt_tpe_200_run2",
        output_path="runs/loss_vs_ami_run2.png",
        title="Loss vs AMI — TPE, 113 trials (new architecture)",
    )
