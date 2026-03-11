from __future__ import annotations

from pathlib import Path

import numpy as np

from mcbj_iic.plotting import (
    plot_augmented_examples,
    plot_cluster_histograms,
    plot_cluster_mean_traces,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_prediction_probabilities,
    plot_trace_examples,
    plot_trace_length_distribution,
    plot_training_history,
)


def _make_trace(length: int, offset: float = 0.0):
    displacement = np.linspace(-0.2, 2.2, length)
    conductance = np.linspace(0.0 + offset, -5.0 + offset, length)
    return np.column_stack([displacement, conductance])


def test_plot_functions_create_files(tmp_path: Path):
    traces = np.array([_make_trace(30, 0.0), _make_trace(28, 0.1), _make_trace(32, -0.2)], dtype=object)
    plot_trace_examples(traces, tmp_path / "traces.png")
    plot_trace_length_distribution(traces, tmp_path / "lengths.png")
    plot_label_distribution(np.array([0, 1, 1]), tmp_path / "labels.png", title="labels")
    plot_prediction_probabilities(np.random.rand(5, 3), tmp_path / "probabilities.png")
    plot_confusion_matrix(np.array([[3, 1], [0, 4]]), tmp_path / "confusion.png")
    plot_training_history([{"epoch": 1.0, "loss": -1.2}, {"epoch": 2.0, "loss": -1.4}], tmp_path / "history.png")

    original = np.random.rand(3, 20, 1)
    augmented = np.random.rand(3, 20, 1)
    plot_augmented_examples(original, augmented, tmp_path / "augmented.png")
    plot_cluster_mean_traces(original, np.array([0, 1, 1]), tmp_path / "mean_traces.png")
    plot_cluster_histograms(traces, np.array([0, 1, 1]), tmp_path / "clusters.png")

    expected = [
        "traces.png",
        "lengths.png",
        "labels.png",
        "probabilities.png",
        "confusion.png",
        "history.png",
        "augmented.png",
        "mean_traces.png",
        "clusters.png",
    ]
    for name in expected:
        assert (tmp_path / name).exists()
