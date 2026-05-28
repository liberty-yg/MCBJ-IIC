"""Cluster Validation System (CVS) for determining the optimal number of clusters.

Based on Chapter 6 of the thesis. Trains the model for k=2...k_max clusters, 
computed pairwise correlations between cluster 2D histograms, and identitifies the optimal k 
as the point where the maximum correlation jumps - indicating the model started subdividing 
a real cluster into two.

Run with:
python scripts/cvs.py --data Data.mat --output-dir runs/cvs"""


from __future__ import annotations

import dataclasses
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from mcbj_iic.data import load_mat_dataset
from mcbj_iic.model import ModelConfig, TrainingConfig, train_iic_model
from mcbj_iic.utils import ensure_dir, predict_in_batches, save_json


def compute_cluster_histogram(
    raw_traces: np.ndarray,
    predicted_labels: np.ndarray,
    cluster_id: int,
    d_edges: np.ndarray,
    g_edges: np.ndarray,
) -> np.ndarray:
    """Compute aggregated 2D histogram for one cluster."""
    idx = np.where(predicted_labels == cluster_id)[0]
    aggregate = np.zeros((len(d_edges) - 1, len(g_edges) - 1))
    for i in idx:
        trace = np.asarray(raw_traces[i])
        displacement = trace[:, 0]
        conductance = 10 ** trace[:, 1]
        h, _, _ = np.histogram2d(displacement, conductance, bins=[d_edges, g_edges])
        aggregate += h
    return aggregate


def compute_correlation(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Compute normalised correlation between two 2D histograms.
    
    Follows Algorithm 3 from the thesis exactly.
    """
    a = hist_a.flatten().astype(float)
    b = hist_b.flatten().astype(float)
    
    # Standardise
    a = a - a.mean()
    b = b - b.mean()
    
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def run_cvs(
    dataset,
    model_config_base: ModelConfig,
    training_config: TrainingConfig,
    k_min: int = 2,
    k_max: int = 9,
    n_repeats: int = 3,
    output_dir: Path = Path("runs/cvs"),
) -> dict:
    """Run the full CVS over a range of cluster numbers."""
    
    output_dir = ensure_dir(output_dir)
    
    # Histogram bin edges — same as plot_cluster_histograms in plotting.py
    gmin, gmax = 3e-6, 10
    dmin, dmax = -0.5, 2.5
    mean_step = abs(float(np.mean(np.diff(np.asarray(dataset.raw_traces[0])[:, 0]))))
    mean_step = max(mean_step, 1e-6)
    d_edges = np.linspace(dmin, dmax, int((dmax - dmin) / (2 * mean_step)))
    g_edges = np.logspace(np.log10(gmin), np.log10(gmax), 100)
    
    max_correlations = {}
    
    for k in range(k_min, k_max + 1):
        print(f"\nTraining with k={k} clusters...")
        
        # Update num_clusters for this run
        model_config = dataclasses.replace(model_config_base, num_clusters=k,num_clusters_overclustering=k * 3)
        
        # Run n_repeats times, take the best (lowest loss) result
        best_loss = float("inf")
        best_labels = None
        
        for repeat in range(n_repeats):
            tf.keras.backend.clear_session()
            
            model, history = train_iic_model(
                dataset.x,
                model_config=model_config,
                training_config=training_config,
                output_dir=None,
            )
            
            final_loss = history[-1]["loss"]
            if final_loss < best_loss and abs(final_loss) > 1e-4:
                best_loss = final_loss
                probabilities = predict_in_batches(
                    dataset.x, model,
                    num_clusters=k, batch_size=64,
                )
                best_labels = np.argmax(probabilities, axis=1)
        
        if best_labels is None:
            print(f" k={k} all repeats collapsed - skipping")
            max_correlations[k] = 0.0
            continue

        print(f"  k={k} best loss: {best_loss:.4f}")
        
        # Compute pairwise correlations between cluster histograms
        unique_clusters = np.unique(best_labels)
        histograms = {
            c: compute_cluster_histogram(
                dataset.raw_traces, best_labels, c, d_edges, g_edges
            )
            for c in unique_clusters
        }
        
        # Build correlation matrix (upper triangle only, diagonal=1)
        correlations = []
        for i, ci in enumerate(unique_clusters):
            for j, cj in enumerate(unique_clusters):
                if j > i:
                    corr = compute_correlation(histograms[ci], histograms[cj])
                    correlations.append(corr)
        
        max_corr = float(max(correlations)) if correlations else 0.0
        max_correlations[k] = max_corr
        print(f"  k={k} max correlation: {max_corr:.4f}")
    
    return max_correlations


def plot_cvs(max_correlations: dict, output_path: Path) -> None:
    """Plot maximum correlation vs number of clusters, looking for the jump."""
    ks = sorted(max_correlations.keys())
    corrs = [max_correlations[k] for k in ks]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, corrs, marker="o", linewidth=2, markersize=8)
    
    # Annotate the jump — largest increase between consecutive k values
    diffs = [corrs[i+1] - corrs[i] for i in range(len(corrs)-1)]
    if diffs:
        jump_idx = int(np.argmax(diffs))
        optimal_k = ks[jump_idx + 1]
        ax.axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7,
                   label=f"Optimal k={optimal_k} (largest jump)")
        ax.legend()
    
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Maximum pairwise correlation")
    ax.set_title("Cluster Validation System — finding optimal k")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"CVS plot saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/cvs")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=9)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    # Architecture — fill these from your BO results
    parser.add_argument("--numfilters", type=int, default=32)
    parser.add_argument("--filter-size", type=int, default=9)
    parser.add_argument("--numlayers", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--xshift", type=float, default=0.05)
    parser.add_argument("--xscale", type=float, default=0.05)
    parser.add_argument("--yshift", type=float, default=0.05)
    parser.add_argument("--yscale", type=float, default=0.05)
    args = parser.parse_args()
    
    output_dir = ensure_dir(args.output_dir)
    
    dataset = load_mat_dataset(args.data, crop_size=340, conductance_floor=-5.5)
    
    model_config_base = ModelConfig(
        input_shape=(340, 1),
        numfilters=args.numfilters,
        filter_size=args.filter_size,
        numlayers=args.numlayers,
        num_clusters=2,  # placeholder — gets overridden per k
        stride=args.stride,
        padding="same",
        dilation=1,
        use_batch_norm=True,
        num_clusters_overclustering=0, #disable for CVS - simpler and faster
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=64,
        xshift=args.xshift,
        xscale=args.xscale,
        yshift=args.yshift,
        yscale=args.yscale,
        order=3,
        verbose=0,
        early_stop_epoch=999,
    )
    
    max_correlations = run_cvs(
        dataset,
        model_config_base=model_config_base,
        training_config=training_config,
        k_min=args.k_min,
        k_max=args.k_max,
        n_repeats=args.n_repeats,
        output_dir=output_dir,
    )
    
    # Find optimal k
    ks = sorted(max_correlations.keys())
    corrs = [max_correlations[k] for k in ks]
    diffs = [corrs[i+1] - corrs[i] for i in range(len(corrs)-1)]
    optimal_k = ks[int(np.argmax(diffs)) + 1] if diffs else ks[0]
    
    print(f"\nMax correlations: {max_correlations}")
    print(f"Optimal number of clusters: {optimal_k}")
    
    plot_cvs(max_correlations, output_dir / "cvs_plot.png")
    save_json(output_dir / "cvs_results.json", {
        "max_correlations": {str(k): v for k, v in max_correlations.items()},
        "optimal_k": optimal_k,
    })
    
    print(f"\nTo run the final benchmark with optimal k:")
    print(f"python scripts/bench7.py --num-clusters {optimal_k} --output-dir runs/cvs_final")

