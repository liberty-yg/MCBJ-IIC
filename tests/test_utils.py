from __future__ import annotations

import numpy as np

from mcbj_iic.utils import evaluate_clustering, majority_vote_accuracy, to_one_hot, transform_traces


def test_transform_traces_preserves_shape():
    data = np.tile(np.linspace(0, 1, 32), (5, 1))[..., np.newaxis]
    transformed = transform_traces(data, crop_size=32, rng=np.random.default_rng(0))
    assert transformed.shape == data.shape
    assert np.isfinite(transformed).all()


def test_majority_vote_accuracy_matches_expected_mapping():
    true_labels = np.array([0, 0, 1, 1])
    predicted = np.array([5, 5, 3, 3])
    assert majority_vote_accuracy(true_labels, predicted) == 1.0


def test_to_one_hot_builds_expected_matrix():
    labels = np.array([0, 2, 1])
    encoded = to_one_hot(labels)
    assert encoded.shape == (3, 3)
    assert np.all(encoded.sum(axis=1) == 1)


def test_evaluate_clustering_returns_expected_keys():
    true_labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    metrics = evaluate_clustering(true_labels, probabilities)
    assert {"accuracy", "fowlkes_mallows", "num_predicted_clusters", "predicted_labels", "confusion_matrix", "cluster_to_label_map"} <= set(metrics)
