from __future__ import annotations

"""Model creation and training for the univariate IIC workflow."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import csv
import numpy as np

from .plotting import plot_training_history
from .utils import ensure_dir, iic_loss, require_tensorflow, save_json, transform_traces


@dataclass(slots=True)
class ModelConfig:
    """CNN architecture parameters used by the 1D clustering model."""

    input_shape: tuple[int, int] = (340, 1)
    numfilters: int = 8
    filter_size: int = 9
    numlayers: int = 4
    num_clusters: int = 7
    stride: int = 1
    padding: str = "same"
    dilation: int = 1


@dataclass(slots=True)
class TrainingConfig:
    """Training and augmentation parameters for the univariate workflow.

    Augmentation is intentionally stochastic and each run generates 
    fresh augmented views of the traces.
    """

    learning_rate: float = 1e-4
    epochs: int = 100
    batch_size: int = 32
    xshift: float = 0.05
    xscale: float = 0.05
    yshift: float = 0.05
    yscale: float = 0.05
    order: int = 3
    verbose: int = 1
    early_stop_epoch: int = 9
    early_stop_abs_loss: float = 1e-3
    save_history_csv: bool = True
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def build_iic_model(config: ModelConfig):
    """Build the 1D CNN used for univariate clustering.

    Each convolutional block doubles the number of filters.
    """

    tf = require_tensorflow()
    keras = tf.keras

    inputs = keras.layers.Input(shape=config.input_shape, name="trace_input")
    x = inputs

    for layer_idx in range(config.numlayers):
        filters = config.numfilters * (2 ** layer_idx)
        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=config.filter_size,
            strides=config.stride,
            padding=config.padding,
            dilation_rate=config.dilation,
            activation="elu",
            kernel_initializer="glorot_normal",
            name=f"conv_{layer_idx + 1}",
        )(x)

    x = keras.layers.Flatten(name="flatten_features")(x)
    outputs = keras.layers.Dense(config.num_clusters, activation="softmax", name="cluster_probabilities")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="iic_univariate")


def _write_training_history_csv(history: list[dict[str, float]], path: str | Path) -> None:
    """Save the epoch history so runs are easy to compare across experiments."""

    if not history:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

def train_iic_model(
    data: np.ndarray,
    *,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: str | Path | None = None,
):
    """Train the univariate IIC model with a manual TensorFlow loop.

    Every epoch sees a new augmented view, and separate runs are also
    stochastic by default.
    """

    tf = require_tensorflow()
    output_path = ensure_dir(output_dir) if output_dir is not None else None

    rng = np.random.default_rng()

    model = build_iic_model(model_config)

    # Build the model once so weight shapes exist before the first gradient step.
    model(rng.uniform(size=(1, *model_config.input_shape)).astype(np.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_config.learning_rate)

    history: list[dict[str, float]] = []
    n_samples = data.shape[0]

    for epoch in range(1, training_config.epochs + 1):
        permutation = rng.permutation(n_samples)
        shuffled = data[permutation]

        # A new augmented view is created every epoch from fresh random draws.
        augmented = transform_traces(
            shuffled,
            model_config.input_shape[0],
            order=training_config.order,
            xshift=training_config.xshift,
            xscale=training_config.xscale,
            yshift=training_config.yshift,
            yscale=training_config.yscale,
            rng=rng,
        ).astype(np.float32)

        batch_losses: list[float] = []
        for start in range(0, n_samples, training_config.batch_size):
            stop = min(start + training_config.batch_size, n_samples)
            batch_x = shuffled[start:stop].astype(np.float32)
            batch_aug = augmented[start:stop]

            with tf.GradientTape() as tape:
                predictions_x = model(batch_x, training=True)
                predictions_aug = model(batch_aug, training=True)
                current_loss = iic_loss(predictions_x, predictions_aug, model_config.num_clusters)

            gradients = tape.gradient(current_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            batch_losses.append(float(current_loss.numpy()))

        epoch_entry = {
            "epoch": float(epoch),
            "loss": float(np.mean(batch_losses)),
        }
        history.append(epoch_entry)

        if training_config.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == training_config.epochs):
            print(f"Epoch {epoch}/{training_config.epochs} - loss: {epoch_entry['loss']:.6f}")

        if epoch == training_config.early_stop_epoch and abs(epoch_entry["loss"]) < training_config.early_stop_abs_loss:
            if training_config.verbose:
                print("Stopping early because the training loss stayed too close to zero.")
            break

    if output_path is not None:
        if training_config.save_history_csv:
            _write_training_history_csv(history, output_path / "training_history.csv")
        plot_training_history(history, output_path / "training_loss.png")
        save_json(
            output_path / "training_config.json",
            {
                "model_config": model_config,
                "training_config": training_config,
            },
        )

    return model, history
