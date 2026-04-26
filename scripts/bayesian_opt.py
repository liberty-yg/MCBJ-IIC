"""Bayesian hyperparameter optimisation for the MCBJ IIC clustering pipeline.

Uses Optuna to search over model and training hyperparameters, optimising for AMI.
Run with:
    python scripts/bayesian_opt.py

Install Optuna first if needed:
    pip install optuna
"""

from __future__ import annotations

import optuna

from mcbj_iic.data import load_mat_dataset
from mcbj_iic.model import ModelConfig, TrainingConfig, train_iic_model
from mcbj_iic.utils import evaluate_clustering, predict_in_batches, save_json, ensure_dir

# --------------------------------------------------------------------------
# Load the dataset once — shared across all trials to save time
# --------------------------------------------------------------------------
print("Loading dataset...")
dataset = load_mat_dataset(crop_size=340, conductance_floor=-5.5)
print(f"Dataset loaded: {dataset.metadata['num_samples']} samples, {dataset.metadata['num_clusters']} ground-truth clusters.")


# --------------------------------------------------------------------------
# Objective function — called once per trial by Optuna
# --------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    model_config = ModelConfig(
        input_shape=(340, 1),
        numfilters=trial.suggest_int("numfilters", 16, 64, step=16),
        filter_size=trial.suggest_int("filter_size", 5, 15, step=2),
        numlayers=trial.suggest_int("numlayers", 2, 4),
        num_clusters=trial.suggest_int("num_clusters", 5, 9),
        stride=trial.suggest_int("stride", 1, 7, step=2),
        padding="same",
        dilation=1,
    )
    training_config = TrainingConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        epochs=50,  # shorter than benchmark for speed during search
        batch_size=64,
        xshift=trial.suggest_float("xshift", 0.02, 0.1),
        xscale=trial.suggest_float("xscale", 0.02, 0.1),
        yshift=trial.suggest_float("yshift", 0.02, 0.1),
        yscale=trial.suggest_float("yscale", 0.02, 0.1),
        order=3,
        verbose=0,  # silent during search
    )

    model, _ = train_iic_model(
        dataset.x,
        model_config=model_config,
        training_config=training_config,
        output_dir=None,  # don't save plots for every trial
    )

    probabilities = predict_in_batches(
        dataset.x,
        model,
        num_clusters=model_config.num_clusters,
        batch_size=64,
    )
    metrics = evaluate_clustering(dataset.y, probabilities)

    ami = metrics["ami_index"]

    print(
        f"  Trial {trial.number} | "
        f"clusters={model_config.num_clusters} lr={training_config.learning_rate:.2e} "
        f"filters={model_config.numfilters} filter_size={model_config.filter_size} "
        f"numlayers={model_config.numlayers} stride={model_config.stride} | "
        f"AMI={ami:.4f} ACC={metrics['accuracy']:.4f}"
    )

    return ami


# --------------------------------------------------------------------------
# Run the optimisation study
# --------------------------------------------------------------------------
if __name__ == "__main__":
    N_TRIALS = 50  # increase for a more thorough search, decrease for a quick test
    OUTPUT_DIR = ensure_dir("runs/bayesian_opt")

    print(f"\nStarting Bayesian optimisation — {N_TRIALS} trials")
    print(f"Results will be saved to {OUTPUT_DIR}\n")

    # Optuna study — maximise AMI
    study = optuna.create_study(
        direction="maximize",
        study_name="mcbj_iic_opt",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # --------------------------------------------------------------------------
    # Print and save results
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Optimisation complete.")
    print(f"Best AMI:          {study.best_value:.4f}")
    print(f"Best parameters:   {study.best_params}")
    print("=" * 60)

    # Save full results to JSON
    results = {
        "best_ami": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "n_trials": N_TRIALS,
        "all_trials": [
            {
                "number": t.number,
                "ami": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    save_json(OUTPUT_DIR / "optuna_results.json", results)
    print(f"\nFull results saved to {OUTPUT_DIR / 'optuna_results.json'}")

    # Print the CLI command to run the benchmark with the best parameters
    p = study.best_params
    print("\nRun the full benchmark with the best parameters:")
    print(
        f"python scripts/bench7.py "
        f"--epochs 100 "
        f"--num-clusters {p['num_clusters']} "
        f"--numfilters {p['numfilters']} "
        f"--filter-size {p['filter_size']} "
        f"--numlayers {p['numlayers']} "
        f"--stride {p['stride']} "
        f"--learning-rate {p['learning_rate']:.2e} "
        f"--xshift {p['xshift']:.3f} "
        f"--xscale {p['xscale']:.3f} "
        f"--yshift {p['yshift']:.3f} "
        f"--yscale {p['yscale']:.3f} "
        f"--output-dir runs/best_params"
    )