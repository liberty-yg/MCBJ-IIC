from __future__ import annotations

"""Utility helpers for augmentation, metrics, batching, and experiment outputs."""

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score


def require_tensorflow():
    """Import TensorFlow lazily.

    Lazy imports keep preprocessing, plotting, and tests usable on machines where
    only the scientific Python stack is installed.
    """

    try:
        import tensorflow as tf  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when missing
        raise ModuleNotFoundError(
            "TensorFlow is required for model building/training. Install with: pip install -e .[training]"
        ) from exc
    return tf


def configure_gpu_memory_growth() -> None:
    """Enable TensorFlow GPU memory growth when GPUs are available."""

    tf = require_tensorflow()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON file with small conveniences for dataclasses and NumPy values."""

    def _normalise(value: Any) -> Any:
        if is_dataclass(value):
            return {k: _normalise(v) for k, v in asdict(value).items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: _normalise(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_normalise(v) for v in value]
        return value

    Path(path).write_text(json.dumps(_normalise(payload), indent=2), encoding="utf-8")


def interpolation_function(x: np.ndarray, y: np.ndarray, order: int):
    """Return an interpolation function matching the original spline choice."""

    interpolation_names = ["zero", "slinear", "quadratic", "cubic"]
    return interp1d(x, y, kind=interpolation_names[order], fill_value="extrapolate")


def transform_traces(
    data: np.ndarray,
    crop_size: int,
    *,
    order: int = 3,
    xshift: float = 0.05,
    xscale: float = 0.05,
    yshift: float = 0.05,
    yscale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply physically meaningful augmentations to univariate traces.

    The augmentation choices: modest shifts and scalings that preserve 
    the interpretation of a breaking trace instead of arbitrary signal warps.
    """

    rng = rng or np.random.default_rng()
    flattened = np.asarray(data).reshape(data.shape[0], crop_size)
    transformed = np.zeros_like(flattened)

    for row_idx, series in enumerate(flattened):
        y = series.copy()

        # 1) Horizontal shift: only shift to the right so we do not reintroduce the
        # metallic regime that was already cropped away during preprocessing.
        max_shift_points = int(np.ceil(len(y) * xshift))
        source_grid = np.linspace(1, len(y), len(y))
        extended_grid = np.linspace(1, len(y) + max_shift_points, len(y) + max_shift_points)
        shift_interp = interpolation_function(source_grid, y, 0)
        start = int(rng.integers(0, max_shift_points)) if max_shift_points > 0 else 0
        y = shift_interp(extended_grid)[start : start + len(y)]

        # 2) Horizontal scaling changes the apparent plateau length but the trace is
        # resampled back to the same crop size so the CNN input shape stays fixed.
        x_scale_factor = 1 + rng.uniform(-xscale, xscale)
        source_grid = np.linspace(1, len(y), len(y))
        scaled_grid = np.linspace(1, len(y), len(y)) * x_scale_factor
        scale_interp = interpolation_function(source_grid, y, order)
        y = scale_interp(scaled_grid)

        # 3) Vertical scaling adjusts the conductance range multiplicatively.
        y = (1 + rng.uniform(-yscale, yscale)) * y

        # 4) Vertical shift offsets the conductance trace additively.
        amplitude = max(y) - min(y)
        y = y + rng.uniform(-amplitude * yshift, amplitude * yshift)

        transformed[row_idx] = y

    return transformed[..., np.newaxis]


def iic_loss(pi_x, pi_gx, num_clusters: int):
    """Return the negative mutual information loss used by IIC."""

    tf = require_tensorflow()
    k = num_clusters
    joint = tf.transpose(pi_x) @ pi_gx
    joint = (joint + tf.transpose(joint)) / 2.0
    joint = tf.clip_by_value(joint, clip_value_min=1e-6, clip_value_max=tf.float32.max)
    joint /= tf.reduce_sum(joint)

    pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(joint, axis=0), (k, 1)), (k, k))
    pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(joint, axis=1), (1, k)), (k, k))
    return tf.reduce_sum(joint * (tf.math.log(pi) + tf.math.log(pj) - tf.math.log(joint)))


def predict_in_batches(x: np.ndarray, model, *, num_clusters: int, batch_size: int = 64,) -> np.ndarray:
    
    probabilities = np.zeros((x.shape[0], num_clusters), dtype=float)
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        output = model(x[start:stop], training=False)

        if isinstance(output, (list, tuple)):
            output = output[0]
        probabilities[start:stop] = np.asarray(output)
    return probabilities
    # """Run model inference in batches without dropping the last partial batch."""

    # probabilities = np.zeros((x.shape[0], num_clusters), dtype=float)
    # for start in range(0, x.shape[0], batch_size):
    #     stop = min(start + batch_size, x.shape[0])
    #     probabilities[start:stop] = np.asarray(model(x[start:stop], training=False))
    # return probabilities


def to_one_hot(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Convert integer labels to one-hot form."""

    labels = np.asarray(labels, dtype=int).reshape(-1)
    num_classes = int(num_classes or (labels.max() + 1))
    one_hot = np.zeros((len(labels), num_classes), dtype=float)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def map_clusters_to_ground_truth(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[int, int]:
    """Map each predicted cluster to the most frequent true label inside it."""

    mapping: dict[int, int] = {}
    for cluster in np.unique(predicted_labels):
        mask = predicted_labels == cluster
        cluster_truth = true_labels[mask]
        if len(cluster_truth) == 0:
            continue
        counts = np.bincount(cluster_truth.astype(int))
        mapping[int(cluster)] = int(np.argmax(counts))
    return mapping


def majority_vote_accuracy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute clustering accuracy after majority-vote label alignment."""

    true_labels = np.asarray(true_labels).reshape(-1).astype(int)
    predicted_labels = np.asarray(predicted_labels).reshape(-1).astype(int)
    mapping = map_clusters_to_ground_truth(true_labels, predicted_labels)
    aligned = np.vectorize(mapping.get)(predicted_labels)
    return float(np.mean(aligned == true_labels))


def evaluate_clustering(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute the main benchmark metrics from model output probabilities."""

    predicted_labels = np.argmax(probabilities, axis=1).astype(int)
    accuracy = majority_vote_accuracy(true_labels, predicted_labels)
    fm_index = float(fowlkes_mallows_score(true_labels, predicted_labels))
    ami_index = float(adjusted_mutual_info_score(true_labels, predicted_labels))
    ari_index = float(adjusted_rand_score(true_labels, predicted_labels))
    confusion = confusion_matrix(true_labels, predicted_labels)

    return {
        "accuracy": float(accuracy),
        "fowlkes_mallows": fm_index,
        "ami_index": ami_index,
        "ari_index": ari_index,
        "num_predicted_clusters": int(len(np.unique(predicted_labels))),
        "predicted_labels": predicted_labels,
        "confusion_matrix": confusion,
        "cluster_to_label_map": map_clusters_to_ground_truth(true_labels, predicted_labels),
    }
