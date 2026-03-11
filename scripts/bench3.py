from __future__ import annotations

"""Run the benchmark workflow only on original clusters 5, 6, and 7.

The MATLAB dataset uses 1-based labels. This script keeps only labels (5, 6, 7)
from the original dataset, then lets the preprocessing pipeline remap them to
(0, 1, 2) internally so the clustering model works with 3 clusters.

In other words:
    original labels: 5, 6, 7
    internal labels: 0, 1, 2
"""

from pathlib import Path
import sys
import argparse

import numpy as np                      # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcbj_iic.data import load_mat_dataset, resolve_mat_dataset_path            # type: ignore
from mcbj_iic.model import ModelConfig, TrainingConfig, train_iic_model         # type: ignore
from mcbj_iic.plotting import (                                                 # type: ignore
    plot_augmented_examples,
    plot_cluster_histograms,
    plot_cluster_mean_traces,
    plot_confusion_matrix,
    plot_conv_filter_weights,
    plot_label_distribution,
    plot_layer_activations,
    plot_prediction_probabilities,
    plot_trace_examples,
    plot_trace_length_distribution,
)
from mcbj_iic.utils import (                                                    # type: ignore
    configure_gpu_memory_growth,
    ensure_dir,
    evaluate_clustering,
    predict_in_batches,
    save_json,
    transform_traces,
)


ORIGINAL_LABELS_TO_KEEP = (5, 6, 7)
INTERNAL_NUM_CLUSTERS = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MCBJ IIC workflow only on original labels 5, 6, and 7."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the local MATLAB dataset. If omitted, the script searches for Data.mat in the current folder or repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/benchmark_clusters_5_6_7",
        help="Directory where results are saved.",
    )
    parser.add_argument("--crop-size", type=int, default=340)
    parser.add_argument("--conductance-floor", type=float, default=-5.5)

    # Model hyperparameters.
    parser.add_argument("--numfilters", type=int, default=32)
    parser.add_argument("--filter-size", type=int, default=7)
    parser.add_argument("--numlayers", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--padding", type=str, default="same")
    parser.add_argument("--dilation", type=int, default=1)

    # Training hyperparameters.
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--xshift", type=float, default=0.05)
    parser.add_argument("--xscale", type=float, default=0.05)
    parser.add_argument("--yshift", type=float, default=0.05)
    parser.add_argument("--yscale", type=float, default=0.05)
    parser.add_argument("--order", type=int, default=3)

    parser.add_argument("--no-gpu-memory-growth", action="store_true")
    parser.add_argument("--save-activation-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = ensure_dir(args.output_dir)
    if not args.no_gpu_memory_growth:
        try:
            configure_gpu_memory_growth()
        except ModuleNotFoundError:
            pass

    resolved_data_path = resolve_mat_dataset_path(args.data)

    # Keep only original labels 5, 6, and 7.
    # After filtering, the preprocessing code remaps them internally to 0, 1, and 2.
    dataset = load_mat_dataset(
        resolved_data_path,
        allowed_labels=ORIGINAL_LABELS_TO_KEEP,
        crop_size=args.crop_size,
        conductance_floor=args.conductance_floor,
    )

    model_config = ModelConfig(
        input_shape=(dataset.crop_size, 1),
        numfilters=args.numfilters,
        filter_size=args.filter_size,
        numlayers=args.numlayers,
        num_clusters=INTERNAL_NUM_CLUSTERS,
        stride=args.stride,
        padding=args.padding,
        dilation=args.dilation,
    )

    # No seed is provided here. Each run and each epoch uses fresh random augmentation.
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        xshift=args.xshift,
        xscale=args.xscale,
        yshift=args.yshift,
        yscale=args.yscale,
        order=args.order,
        verbose=1,
    )

    # Quick-look data plots before training.
    plot_trace_examples(
        dataset.raw_traces,
        output_dir / "trace_examples_subset_5_6_7.png",
    )
    plot_trace_length_distribution(
        dataset.raw_traces,
        output_dir / "trace_length_distribution_subset_5_6_7.png",
    )
    plot_label_distribution(
        dataset.y,
        output_dir / "ground_truth_distribution_subset_5_6_7.png",
        title="Ground-truth distribution for original labels 5, 6, 7 (internally remapped to 0, 1, 2)",
    )

    augmented_preview = transform_traces(
        dataset.x[:6],
        dataset.crop_size,
        order=training_config.order,
        xshift=training_config.xshift,
        xscale=training_config.xscale,
        yshift=training_config.yshift,
        yscale=training_config.yscale,
        rng=np.random.default_rng(),
    )
    plot_augmented_examples(
        dataset.x[:6],
        augmented_preview,
        output_dir / "augmentation_examples_subset_5_6_7.png",
    )

    model, history = train_iic_model(
        dataset.x,
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
    )

    probabilities = predict_in_batches(
        dataset.x,
        model,
        num_clusters=INTERNAL_NUM_CLUSTERS,
        batch_size=args.batch_size,
    )
    metrics = evaluate_clustering(dataset.y, probabilities)

    predicted_labels = metrics.pop("predicted_labels")
    confusion = np.asarray(metrics["confusion_matrix"])

    plot_label_distribution(
        predicted_labels,
        output_dir / "predicted_cluster_distribution_subset_5_6_7.png",
        title="Predicted cluster distribution for subset 5, 6, 7",
    )
    plot_prediction_probabilities(
        probabilities,
        output_dir / "prediction_probabilities_subset_5_6_7.png",
    )
    plot_confusion_matrix(
        confusion,
        output_dir / "confusion_matrix_subset_5_6_7.png",
    )
    plot_confusion_matrix(
        confusion,
        output_dir / "confusion_matrix_normalized_subset_5_6_7.png",
        normalize=True,
    )
    plot_cluster_mean_traces(
        dataset.x,
        predicted_labels,
        output_dir / "cluster_mean_traces_subset_5_6_7.png",
    )
    plot_cluster_histograms(
        dataset.raw_traces,
        predicted_labels,
        output_dir / "cluster_histograms_subset_5_6_7.png",
    )

    try:
        plot_conv_filter_weights(model, output_dir / "conv_filter_weights_subset_5_6_7.png")
        if args.save_activation_plots:
            plot_layer_activations(
                model,
                dataset.x[: min(16, len(dataset.x))],
                output_dir / "activations_subset_5_6_7",
            )
    except Exception as exc:
        print(f"Warning: optional model-interpretation plots could not be generated: {exc}")

    try:
        model.save(output_dir / "iic_model_subset_5_6_7.keras")
    except Exception as exc:
        print(f"Warning: model could not be saved: {exc}")

    save_json(
        output_dir / "metrics_subset_5_6_7.json",
        {
            "dataset": {
                **dataset.metadata,
                "source_path": str(Path(resolved_data_path).resolve()),
                "original_labels_selected": list(ORIGINAL_LABELS_TO_KEEP),
                "internal_labels_used_for_training": [0, 1, 2],
            },
            "model_config": model_config,
            "training_config": training_config,
            "history": history,
            "metrics": metrics,
        },
    )

    print(f"Dataset: {resolved_data_path}")
    print(f"Selected original labels: {ORIGINAL_LABELS_TO_KEEP}")
    print("Internal labels used for training/evaluation: (0, 1, 2)")
    print(
        "Accuracy: {accuracy:.4f} | FM index: {fm:.4f} | AMI: {ami:.4f} | ARI: {ari:.4f} | predicted clusters: {clusters}".format(
            accuracy=metrics["accuracy"],
            fm=metrics["fowlkes_mallows"],
            ami=metrics["ami_index"],
            ari=metrics["ari_index"],
            clusters=metrics["num_predicted_clusters"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())