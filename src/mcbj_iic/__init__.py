"""MCBJ IIC clustering package.

The package is split into small modules so the research workflow is easier to reuse
in scripts, notebooks, and tests.
"""

from .cli import main
from .data import infer_crop_size, load_mat_dataset, preprocess_matlab_dataset, resolve_mat_dataset_path
from .model import ModelConfig, TrainingConfig, build_iic_model, train_iic_model
from .utils import configure_gpu_memory_growth, evaluate_clustering, iic_loss, predict_in_batches, transform_traces

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "build_iic_model",
    "configure_gpu_memory_growth",
    "evaluate_clustering",
    "iic_loss",
    "infer_crop_size",
    "load_mat_dataset",
    "main",
    "predict_in_batches",
    "preprocess_matlab_dataset",
    "resolve_mat_dataset_path",
    "train_iic_model",
    "transform_traces",
]
