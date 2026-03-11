# MCBJ IIC Clustering

A repository for clustering mechanically controlled break junction (MCBJ) traces with an **Invariant Information Clustering (IIC)** style 1D CNN.

![](https://user-images.githubusercontent.com/46565095/160436611-996a1791-3f15-42b5-84b1-3035359c45c3.png)

## What this repository contains

-  `src/mcbj_iic/data.py` loads the local MATLAB dataset, filters/remaps labels, removes the negative-displacement metallic regime, and interpolates every trace onto a common length.

-  `src/mcbj_iic/model.py` defines the configurable 1D CNN and the manual IIC training loop.

-  `src/mcbj_iic/utils.py` contains augmentation, batching, metric, JSON, and TensorFlow helper functions.

-  `src/mcbj_iic/plotting.py` provides plots for raw traces, augmentation previews, training loss, label distributions, confusion matrices, predicted clusters, aggregated histograms, mean cluster traces, learned filters, and optional activations.

-  `src/mcbj_iic/cli.py` is the main command-line entry point.

-  `scripts/benchmark_clustering.py` is a thin runner for running directly from the repository checkout.

-  `tests/` contains unit tests for the non-TensorFlow parts of the workflow.

## Repository layout

```text
mcbj-iic-clustering/
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│ └── benchmark_clustering.py
├── src/
│ └── mcbj_iic/
│ ├── __init__.py
│ ├── cli.py
│ ├── data.py
│ ├── model.py
│ ├── plotting.py
│ └── utils.py
└── tests/
├── test_data.py
├── test_plotting.py
└── test_utils.py
``` 

## Dataset

The benchmark dataset is **not public** and is therefore **not included** in the repository. The expected setup is:

1. keep the `.mat` dataset file locally next to the repository, or in the repository root
2. do **not** commit it to GitHub
3. let the CLI discover it automatically, or pass `--data /path/to/Data.mat`

The CLI searches first for `Data.mat` / `data.mat` in the current folder and repository root. If it does not find one clear match, it asks for `--data` explicitly.

## Installation

```bash
pip  install  -e  .
```
For training:
```bash
pip  install  -e  .[training]
```
For development and tests:
```bash
pip  install  -e  .[dev]
pytest
```

## Quick start

If the dataset file is already in the repository root as `Data.mat`:

```bash
python  scripts/benchmark_clustering.py
```
Or after installation:
```bash
mcbj-benchmark
```
With an explicit dataset path:
```bash
python  scripts/benchmark_clustering.py  --data  /path/to/Data.mat  --output-dir  runs/run_1 
```
With preprocessing, network and training hyperparameters:
```bash
python  scripts/bench3.py \
  --data  ./Data.mat \
  --output-dir  runs/run_1 \
  --num cluster  7 \
  --crop-size  340 \
  --conductance-floor -5.5 \
  --numfilters  32 \
  --filter-size  9 \
  --numlayers  3 \
  --stride  5 \
  --padding  "same" \
  --dilation  1 \
  --learning-rate  5e-4 \
  --epochs  100 \
  --batch-size  64 \
  --xshift  0.05 \
  --xscale  0.05 \
  --yshift  0.05 \
  --yscale  0.05 \
  --order  3
```


## Main outputs

The benchmark run writes a folder containing, among others:

-  `metrics.json`
-  `training_history.csv`
-  `training_loss.png`
-  `trace_examples.png`
-  `trace_length_distribution.png`
-  `augmentation_examples.png`
-  `ground_truth_distribution.png`
-  `predicted_cluster_distribution.png`
-  `prediction_probabilities.png`
-  `confusion_matrix.png`
-  `confusion_matrix_normalized.png`
-  `cluster_mean_traces.png`
-  `cluster_histograms.png`
-  `conv_filter_weights.png`

- optional activation maps

## Notes

- The benchmark script expects a MATLAB file with the structure: `Data` and `Labels` arrays.
- The default hyperparameters are chosen after Bayesian Optimisation.