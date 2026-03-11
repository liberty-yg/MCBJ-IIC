from __future__ import annotations

"""Dataset loading, and preprocessing for MCBJ traces.

This module includes .mat file loading, label remapping, cropping and interpolation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


@dataclass(slots=True)
class DatasetBundle:
    """Container holding both the processed and raw benchmark dataset.

    Attributes
    ----------
    x:
        Preprocessed conductance traces with shape ``(n_samples, crop_size, 1)``.
    y:
        Zero-based integer labels.
    raw_traces:
        Original .mat MATLAB trace objects, each typically shaped ``(n_points, 2)`` 
        with displacement in column 0 and log-conductance in column 1.
    crop_size:
        Length used for the univariate representation.
    metadata:
        Extra information that can be saved into ``metrics.json``.
    """

    x: np.ndarray
    y: np.ndarray
    raw_traces: np.ndarray
    crop_size: int
    metadata: dict


def infer_crop_size(raw_traces: np.ndarray) -> int:
    """Return the maximum positive-displacement trace length.

    The metallic regime corresponds to negative displacement and is removed before
    training. The crop size is then chosen as the longest remaining trace length.
    """

    crop_size = 0
    for trace in raw_traces:
        displacement = np.asarray(trace)[:, 0]
        start_idx = int(np.sum(displacement < 0))
        cropped = displacement[start_idx:-1]
        crop_size = max(crop_size, len(cropped))
    return int(crop_size)


def resolve_mat_dataset_path(
    mat_path: str | Path | None = None,
    *,
    search_roots: Iterable[str | Path] | None = None,
) -> Path:
    """Resolve the local benchmark MATLAB file.

    Parameters
    ----------
    mat_path:
        Explicit path provided by the user.
    search_roots:
        Directories to search when ``mat_path`` is omitted. The default search is
        intentionally conservative and looks only in the current working directory
        and the repository root.

    Notes
    -----
    The dataset is not part of the repository because it is not public. In normal
    use it is expected to be placed next to the repository or the script and then
    discovered automatically.
    """

    if mat_path is not None:
        resolved = Path(mat_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset file not found: {resolved}")
        return resolved

    if search_roots is None:
        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [Path.cwd(), repo_root]

    roots: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        path = Path(root).expanduser().resolve()
        if path not in seen and path.exists():
            roots.append(path)
            seen.add(path)

    preferred_names = ["Data.mat", "data.mat", "Dataset.mat", "dataset.mat"]
    for root in roots:
        for name in preferred_names:
            candidate = root / name
            if candidate.exists():
                return candidate

    discovered: list[Path] = []
    for root in roots:
        discovered.extend(sorted(path.resolve() for path in root.glob("*.mat") if path.is_file()))

    unique_discovered: list[Path] = []
    seen_paths: set[Path] = set()
    for path in discovered:
        if path not in seen_paths:
            unique_discovered.append(path)
            seen_paths.add(path)

    if len(unique_discovered) == 1:
        return unique_discovered[0]

    if len(unique_discovered) > 1:
        matches = "\n".join(f"  - {path}" for path in unique_discovered)
        raise FileNotFoundError(
            "Multiple local .mat files were found. Pass --data to select the dataset explicitly:\n"
            f"{matches}"
        )

    searched = "\n".join(f"  - {root}" for root in roots)
    raise FileNotFoundError(
        "Could not find a local MATLAB dataset file. Place Data.mat in the repository "
        "root (or current working directory), or pass --data explicitly.\n"
        f"Searched:\n{searched}"
    )


def _filter_and_zero_base_labels(labels: np.ndarray, allowed_labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Keep only requested labels and convert them to ``0..n-1``."""

    allowed_labels = np.asarray(list(allowed_labels), dtype=int)
    labels = np.asarray(labels).reshape(-1)
    mask = np.isin(labels, allowed_labels)
    filtered = labels[mask]

    mapping = {label: idx for idx, label in enumerate(sorted(np.unique(allowed_labels)))}
    zero_based = np.vectorize(mapping.get)(filtered).astype(int)
    return mask, zero_based


def preprocess_raw_traces(
    raw_traces: np.ndarray,
    *,
    crop_size: int,
    conductance_floor: float = -5.5,
) -> np.ndarray:
    """Convert raw ``(distance, logG)`` traces into fixed-length univariate inputs.

    The workflow keeps the conductance signal directly and interpolates each trace
    onto a shared length. The negative displacement region is removed because it
    belongs to the metallic regime and would otherwise dominate the learning signal.
    """

    processed = np.zeros((len(raw_traces), crop_size), dtype=float)

    for row_idx, trace in enumerate(raw_traces):
        trace = np.asarray(trace)
        displacement = trace[:, 0]
        conductance = trace[:, 1]

        # Remove the metallic regime 
        start_idx = int(np.sum(displacement < 0))
        conductance = conductance[start_idx:-1]

        # Some traces can become empty after cropping. In that case we create a flat
        # floor-valued trace so the batch still has a valid fixed-size representation.
        if len(conductance) == 0:
            processed[row_idx] = np.full(crop_size, conductance_floor, dtype=float)
            continue

        # Interpolate onto a common grid so every trace has the same input length.
        source_grid = np.linspace(0, len(conductance), len(conductance))
        target_grid = np.linspace(0, crop_size, crop_size)
        processed[row_idx] = np.interp(target_grid, source_grid, conductance, right=conductance_floor)

    return processed


def preprocess_matlab_dataset(
    matlab_dict: dict,
    *,
    allowed_labels: Iterable[int] = range(1, 8),
    crop_size: int | None = 340,
    conductance_floor: float = -5.5,
    shuffle: bool = True,
) -> DatasetBundle:
    """Create the benchmark training tensor from a MATLAB dictionary.

    The dataset order is shuffled with a fresh random generator so each run starts
    from a new random ordering. The Framework is stochastic during training.
    """

    data = matlab_dict["Data"]
    labels = matlab_dict["Labels"]

    mask, zero_based_labels = _filter_and_zero_base_labels(labels, allowed_labels)
    filtered_raw = data[mask].reshape(-1)

    if shuffle:
        order = np.random.default_rng().permutation(len(filtered_raw))
        filtered_raw = filtered_raw[order]
        zero_based_labels = zero_based_labels[order]

    inferred_crop_size = infer_crop_size(filtered_raw)
    final_crop_size = int(crop_size or inferred_crop_size)
    processed = preprocess_raw_traces(
        filtered_raw,
        crop_size=final_crop_size,
        conductance_floor=conductance_floor,
    )

    x = processed[..., np.newaxis]
    metadata = {
        "num_samples": int(len(filtered_raw)),
        "num_clusters": int(len(np.unique(zero_based_labels))),
        "inferred_crop_size": int(inferred_crop_size),
        "crop_size": int(final_crop_size),
        "conductance_floor": float(conductance_floor),
    }

    return DatasetBundle(
        x=x,
        y=zero_based_labels.astype(int),
        raw_traces=filtered_raw,
        crop_size=final_crop_size,
        metadata=metadata,
    )

def load_mat_dataset(
    mat_path: str | Path | None = None,
    **kwargs,
) -> DatasetBundle:
    """Load the benchmark dataset from a MATLAB ``.mat`` file.

    When ``mat_path`` is omitted the function tries to discover the dataset locally.
    """

    resolved_path = resolve_mat_dataset_path(mat_path)
    matlab_dict = loadmat(resolved_path)
    bundle = preprocess_matlab_dataset(matlab_dict, **kwargs)
    bundle.metadata["source_path"] = str(resolved_path)
    return bundle
