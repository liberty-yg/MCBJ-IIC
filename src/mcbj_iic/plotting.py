from __future__ import annotations

"""Plot helpers for inspecting and visualising data, training, model outputs, and learned features."""

from pathlib import Path
from typing import Iterable

import matplotlib                       # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # type: ignore
import numpy as np                      # type: ignore

from .utils import ensure_dir, require_tensorflow


def _finalise(fig, output_path: str | Path, *, dpi: int = 200) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_training_history(history: list[dict[str, float]], output_path: str | Path) -> None:
    """Plot the training loss across epochs."""

    fig, ax = plt.subplots(figsize=(7, 4))
    if history:
        ax.plot([row["epoch"] for row in history], [row["loss"] for row in history], marker="o", ms=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    _finalise(fig, output_path)


def plot_trace_examples(raw_traces: Iterable[np.ndarray], output_path: str | Path, *, max_examples: int = 12) -> None:
    """Plot a small gallery of raw displacement vs conductance traces."""

    traces = list(raw_traces)[:max_examples]
    ncols = 3
    nrows = int(np.ceil(len(traces) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False, constrained_layout=True)

    for ax, trace in zip(axes.flat, traces):
        trace = np.asarray(trace)
        ax.plot(trace[:, 0], trace[:, 1], linewidth=1.0)
        ax.set_title("Raw trace", pad=8)
        ax.set_xlabel("Displacement", labelpad=6)
        ax.set_ylabel("log(G)", labelpad=6)
        ax.grid(True, alpha=0.2)

    for ax in axes.flat[len(traces):]:
        ax.axis("off")

    fig.suptitle("Example raw MCBJ traces", fontsize=16)
    _finalise(fig, output_path)


def plot_trace_length_distribution(raw_traces: Iterable[np.ndarray], output_path: str | Path) -> None:
    """Plot a histogram of raw trace lengths before interpolation.

    To check the variation in the original dataset before every 
    trace is projected onto the common crop size.
    """

    lengths = [len(np.asarray(trace)) for trace in raw_traces]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(lengths, bins=min(30, max(5, len(lengths))))
    ax.set_xlabel("Raw trace length")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of raw trace lengths")
    ax.grid(True, alpha=0.2, axis="y")
    _finalise(fig, output_path)


def plot_augmented_examples(
    original: np.ndarray,
    augmented: np.ndarray,
    output_path: str | Path,
    *,
    max_examples: int = 6,
) -> None:
    """Visualise how the augmentation pipeline modifies input traces."""

    n = min(max_examples, len(original), len(augmented))
    fig, axes = plt.subplots(n, 1, figsize=(8, max(2.5 * n, 3.2)), squeeze=False, constrained_layout=True)

    for idx in range(n):
        ax = axes[idx, 0]
        ax.plot(original[idx].reshape(-1), label="original", linewidth=1.2)
        ax.plot(augmented[idx].reshape(-1), label="augmented", linewidth=1.2)
        ax.set_xlabel("Interpolated point", labelpad=6)
        ax.set_ylabel("log(G)")
        ax.set_title(f"Augmentation example {idx + 1}", pad=8)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")

    fig.suptitle("Physically constrained trace augmentations", fontsize=16)
    _finalise(fig, output_path)


def plot_label_distribution(labels: np.ndarray, output_path: str | Path, *, title: str) -> None:
    """Plot a simple cluster/label count bar chart."""

    labels = np.asarray(labels).reshape(-1)
    values, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(values.astype(str), counts)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")
    _finalise(fig, output_path)


def plot_prediction_probabilities(probabilities: np.ndarray, output_path: str | Path, *, max_samples: int = 100) -> None:
    """Show model output probabilities as a heat map for the first samples."""

    display = probabilities[:max_samples]
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(display, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Cluster index")
    ax.set_ylabel("Sample index")
    ax.set_title("Predicted cluster probabilities")
    fig.colorbar(image, ax=ax, shrink=0.85)
    _finalise(fig, output_path)


def plot_confusion_matrix(
    confusion: np.ndarray,
    output_path: str | Path,
    *,
    normalize: bool = False,
) -> None:
    """Plot the true-label vs predicted-cluster confusion matrix."""

    confusion = np.asarray(confusion, dtype=float)
    if normalize and confusion.size:
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        confusion = confusion / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, interpolation="nearest", aspect="auto")
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("True label")
    ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    fig.colorbar(image, ax=ax, shrink=0.85)
    _finalise(fig, output_path)


def plot_cluster_mean_traces(
    processed_traces: np.ndarray,
    predicted_labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot the mean processed trace for each predicted cluster.

    This provides a compact view of what the network groups together in the 1D input
    space, which complements the aggregated 2D histogram plots.
    """

    flattened = np.asarray(processed_traces)
    if flattened.ndim == 3:
        flattened = flattened[..., 0]

    predicted_labels = np.asarray(predicted_labels).astype(int)
    unique_clusters = np.unique(predicted_labels)
    ncols = 3
    nrows = int(np.ceil(len(unique_clusters) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False, constrained_layout=True)

    for ax, cluster in zip(axes.flat, unique_clusters):
        idx = np.where(predicted_labels == cluster)[0]
        cluster_traces = flattened[idx]
        mean_trace = cluster_traces.mean(axis=0)
        std_trace = cluster_traces.std(axis=0)
        x_axis = np.arange(mean_trace.shape[0])

        ax.plot(x_axis, mean_trace, linewidth=1.4)
        ax.fill_between(x_axis, mean_trace - std_trace, mean_trace + std_trace, alpha=0.2)
        ax.set_title(f"Cluster {cluster} (n={len(idx)})", pad=8)
        ax.set_xlabel("Interpolated point", labelpad=6)
        ax.set_ylabel("log(G)", labelpad=6)
        ax.grid(True, alpha=0.2)

    for ax in axes.flat[len(unique_clusters):]:
        ax.axis("off")

    fig.suptitle("Mean processed traces by predicted cluster", fontsize=16)
    _finalise(fig, output_path)

def plot_cluster_histograms(
    raw_traces: np.ndarray,
    predicted_labels: np.ndarray,
    output_path: str | Path,
    *,
    gmin: float = 3e-6,
    gmax: float = 10,
    bins_per_g: int = 20,
    dmin: float = -0.5,
    dmax: float = 2.5,
) -> None:
    """Plot aggregated 2D displacement-vs-conductance histograms by cluster."""

    from matplotlib.colors import LogNorm           # type: ignore

    predicted_labels = np.asarray(predicted_labels).astype(int)
    unique_clusters = np.unique(predicted_labels)

    if len(raw_traces) == 0:
        raise ValueError("`raw_traces` must not be empty.")

    # Estimate displacement step for x-binning
    mean_step = abs(float(np.mean(np.diff(np.asarray(raw_traces[0])[:, 0]))))
    mean_step = max(mean_step, 1e-6)

    # Bin edges:
    # x-axis is displacement in nm
    d_edges = np.linspace(dmin, dmax, int((dmax - dmin) / (2 * mean_step)))

    # y-axis is conductance, logarithmically spaced
    n_gbins = int((np.log10(gmax) - np.log10(gmin)) * bins_per_g)
    g_edges = np.logspace(np.log10(gmin), np.log10(gmax), n_gbins)

    # Build per-trace histograms in physical coordinates:
    # trace[:, 0] = displacement
    # trace[:, 1] = log10(G/G0), so convert back to conductance
    images = np.zeros((len(raw_traces), len(d_edges) - 1, len(g_edges) - 1), dtype=float)

    for idx, trace in enumerate(raw_traces):
        trace = np.asarray(trace)
        displacement = trace[:, 0]
        conductance = 10 ** trace[:, 1]

        histogram, _, _ = np.histogram2d(
            displacement,
            conductance,
            bins=[d_edges, g_edges],
        )
        images[idx] = histogram

    ncols = 3
    nrows = int(np.ceil(len(unique_clusters) / ncols)) or 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    # Compute aggregates and global color scaling
    aggregates = []
    vmax = 0.0
    positive_min = None

    for cluster in unique_clusters:
        idx = np.where(predicted_labels == cluster)[0]
        aggregate = images[idx].sum(axis=0)
        aggregates.append((cluster, aggregate))

        local_max = np.max(aggregate)
        if local_max > vmax:
            vmax = float(local_max)

        positive_vals = aggregate[aggregate > 0]
        if positive_vals.size:
            local_min = float(positive_vals.min())
            if positive_min is None or local_min < positive_min:
                positive_min = local_min

    if positive_min is None:
        positive_min = 1.0
    if vmax <= 0:
        vmax = 1.0

    mesh = None
    for ax, (cluster, aggregate) in zip(axes.flat, aggregates):
        mesh = ax.pcolormesh(
            d_edges,
            g_edges,
            aggregate.T,
            shading="auto",
            cmap="jet",
            norm=LogNorm(vmin=positive_min, vmax=vmax),
        )

        ax.set_title(f"Cluster {cluster}", pad=8)
        ax.set_xlabel("Displacement (nm)")
        ax.set_ylabel(r"Conductance ($G/G_0$)")
        ax.set_yscale("log")
        ax.set_xlim(dmin, dmax)
        ax.set_ylim(gmin, gmax)

    for ax in axes.flat[len(unique_clusters):]:
        ax.axis("off")

    if mesh is not None:
        fig.colorbar(mesh, ax=axes, location="right", shrink=0.95, pad=0.02, label="Counts")

    fig.suptitle("2D histograms aggregated by predicted cluster", fontsize=16)
    _finalise(fig, output_path)


def plot_conv_filter_weights(model, output_path: str | Path, *, max_filters_per_layer: int = 8) -> None:
    """Plot learned 1D convolution filters for each Conv1D layer."""

    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == "Conv1D"]
    nrows = max(len(conv_layers), 1)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 3.4 * nrows), squeeze=False, constrained_layout=True)

    for row_idx, layer in enumerate(conv_layers):
        ax = axes[row_idx, 0]
        kernels = layer.get_weights()[0]
        num_filters = min(kernels.shape[-1], max_filters_per_layer)
        for filter_idx in range(num_filters):
            ax.plot(kernels[:, 0, filter_idx], alpha=0.8)
        ax.set_title(f"{layer.name} kernels", pad=8)
        ax.set_xlabel("Kernel position", labelpad=6)
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Learned convolution filter weights", fontsize=16)
    _finalise(fig, output_path)


def plot_layer_activations(
    model,
    sample_batch: np.ndarray,
    output_prefix: str | Path,
    *,
    max_layers: int | None = None,
    max_samples: int = 3,
) -> None:
    """Save activation maps for the model's Conv1D layers."""

    tf = require_tensorflow()
    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == "Conv1D"]
    if max_layers is not None:
        conv_layers = conv_layers[:max_layers]

    activation_model = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])
    activations = activation_model(sample_batch[:max_samples], training=False)

    output_prefix = Path(output_prefix)
    ensure_dir(output_prefix.parent)

    for layer, layer_activation in zip(conv_layers, activations):
        layer_activation = np.asarray(layer_activation)
        averaged = layer_activation.mean(axis=-1)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        image = ax.imshow(averaged, aspect="auto", interpolation="nearest")
        ax.set_title(f"Average activation - {layer.name}")
        ax.set_xlabel("Sequence position")
        ax.set_ylabel("Sample index")
        fig.colorbar(image, ax=ax, shrink=0.85)
        _finalise(fig, output_prefix.parent / f"{output_prefix.name}_{layer.name}.png")
