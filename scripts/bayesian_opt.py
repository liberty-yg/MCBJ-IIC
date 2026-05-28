"""Bayesian hyperparameter optimisation for the MCBJ IIC clustering pipeline.

Optimises using training loss as the objective (works for both labelled and
unlabelled data). AMI is logged per trial for reference but not optimised.

Key improvements over the previous version:
- Loss-based objective (no ground truth labels required)
- GPU memory growth configured to prevent OOM
- TF session cleared between trials to prevent memory accumulation
- Explicit garbage collection after every trial to prevent RAM accumulation
- 3 repeats per trial for stable estimates
- Multiple sampler comparison (TPE, CMA-ES, Random, GP)
- Convergence detection with automatic early stopping
- Logarithmic curve fitting to characterise convergence rate
- GPU and RAM memory monitoring per trial
- Per-trial incremental JSON saving (crash-safe)
- Timing tracking (start time, end time, elapsed hours per sampler)
- Top-3 validation at full 100 epochs
- Exception catching to prevent single trial crashes killing the study
- --test flag for quick sanity checks without editing the file

Run with:
    python scripts/bayesian_opt.py           # full run
    python scripts/bayesian_opt.py --test    # 10-trial sanity check

Install dependencies first if needed:
    pip install optuna scipy psutil
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime
import numpy as np
import optuna
import psutil
import tensorflow as tf
from scipy.optimize import curve_fit
from scipy.stats import linregress

from mcbj_iic.data import load_mat_dataset
from mcbj_iic.model import ModelConfig, TrainingConfig, train_iic_model
from mcbj_iic.utils import (
    configure_gpu_memory_growth,
    ensure_dir,
    evaluate_clustering,
    predict_in_batches,
    save_json,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run quick sanity check")
args = parser.parse_args()

if args.test:
    N_TRIALS = 10
    N_REPEATS_PER_TRIAL = 1
    EPOCHS_SEARCH = 30
    EPOCHS_VALIDATION = 100
    CONVERGENCE_PATIENCE = 5
    CONVERGENCE_THRESHOLD = 1e-3
    OUTPUT_DIR = ensure_dir("runs/bayesian_opt_test")
    SAMPLERS = {"TPE": optuna.samplers.TPESampler(seed=42)}
    print("*** RUNNING IN TEST MODE ***")
else:
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    N_TRIALS = 200
    N_REPEATS_PER_TRIAL = 3
    EPOCHS_SEARCH = 50
    EPOCHS_VALIDATION = 100
    CONVERGENCE_PATIENCE = 200
    CONVERGENCE_THRESHOLD = 1e-3
    OUTPUT_DIR = ensure_dir("runs/bayesian_opt_tpe_200_run2")
    SAMPLERS = {
        "TPE":    optuna.samplers.TPESampler(seed=42),
        # "CMA-ES": optuna.samplers.CmaEsSampler(seed=42),
        # "Random": optuna.samplers.RandomSampler(seed=42),
    }
    # GP sampler requires optuna >= 3.6
    # try:
    #     SAMPLERS["GP"] = optuna.samplers.GPSampler(seed=42)
    # except AttributeError:
    #     print("GP sampler not available in this Optuna version — skipping.")

# --------------------------------------------------------------------------
# GPU setup — must happen before any TF operations
# --------------------------------------------------------------------------
try:
    configure_gpu_memory_growth()
    print("GPU memory growth enabled.")
except Exception as e:
    print(f"GPU memory growth not configured: {e}")

# --------------------------------------------------------------------------
# Load dataset once — shared across all trials
# --------------------------------------------------------------------------
print("Loading dataset...")
dataset = load_mat_dataset(crop_size=340, conductance_floor=-5.5)
print(
    f"Dataset loaded: {dataset.metadata['num_samples']} samples, "
    f"{dataset.metadata['num_clusters']} ground-truth clusters."
)


# --------------------------------------------------------------------------
# Memory monitoring helper
# --------------------------------------------------------------------------
def get_memory_usage() -> dict:
    """Return GPU and system RAM usage. Returns empty dict if unavailable."""
    result: dict = {}

    # GPU memory
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        result["gpu_peak_mb"]    = round(info["peak"]    / 1024 ** 2, 1)
        result["gpu_current_mb"] = round(info["current"] / 1024 ** 2, 1)
    except Exception:
        pass

    # System RAM
    try:
        ram = psutil.virtual_memory()
        result["ram_used_gb"]  = round(ram.used  / 1024 ** 3, 2)
        result["ram_percent"]  = ram.percent
    except Exception:
        pass

    return result


# --------------------------------------------------------------------------
# Objective function
# --------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Train the IIC model N_REPEATS times and return mean final training loss.

    Loss is the optimisation target because it requires no ground-truth labels
    and therefore generalises to unlabelled datasets. AMI is logged as a user
    attribute for post-hoc analysis on the benchmark data.
    """
    # Clear accumulated TF graphs and weights from previous trials
    tf.keras.backend.clear_session()

    num_clusters = trial.suggest_int("num_clusters", 5, 9)

    model_config = ModelConfig(
        input_shape=(340, 1),
        numfilters=trial.suggest_int("numfilters", 16, 64, step=16),
        filter_size=trial.suggest_int("filter_size", 5, 15, step=2),
        numlayers=trial.suggest_int("numlayers", 2, 4),
        num_clusters=num_clusters,
        stride=trial.suggest_int("stride", 1, 7, step=2),
        padding="same",
        dilation=1,
        use_batch_norm=True,
        num_clusters_overclustering=num_clusters * 3,
    )
    training_config = TrainingConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        epochs=EPOCHS_SEARCH,
        batch_size=64,
        xshift=trial.suggest_float("xshift", 0.02, 0.20),
        xscale=trial.suggest_float("xscale", 0.02, 0.15),
        yshift=trial.suggest_float("yshift", 0.02, 0.20),
        yscale=trial.suggest_float("yscale", 0.02, 0.15),
        order=3,
        verbose=0,
        early_stop_epoch=999,
    )

    losses: list[float] = []
    amis:   list[float] = []

    for _ in range(N_REPEATS_PER_TRIAL):
        tf.keras.backend.clear_session()

        model, history = train_iic_model(
            dataset.x,
            model_config=model_config,
            training_config=training_config,
            output_dir=None,
        )

        # Final epoch loss — the optimisation target
        losses.append(float(history[-1]["loss"]))

        # AMI — logged for reference only, not optimised
        probabilities = predict_in_batches(
            dataset.x,
            model,
            num_clusters=model_config.num_clusters,
            batch_size=64,
        )
        metrics = evaluate_clustering(dataset.y, probabilities)
        amis.append(float(metrics["ami_index"]))

    mean_loss = float(np.mean(losses))
    mean_ami  = float(np.mean(amis))
    std_ami   = float(np.std(amis))
    std_loss  = float(np.std(losses))

    # Monitor GPU and RAM memory
    mem = get_memory_usage()
    if mem.get("gpu_peak_mb"):
        trial.set_user_attr("gpu_peak_mb", mem["gpu_peak_mb"])
    if mem.get("ram_used_gb"):
        trial.set_user_attr("ram_used_gb", mem["ram_used_gb"])
        trial.set_user_attr("ram_percent", mem["ram_percent"])

    # Store AMI and stability metrics for post-hoc analysis
    trial.set_user_attr("ami_mean", mean_ami)
    trial.set_user_attr("ami_std",  std_ami)
    trial.set_user_attr("loss_std", std_loss)
    trial.set_user_attr(
        "num_predicted_clusters",
        int(metrics["num_predicted_clusters"])
    )

    print(
        f"  Trial {trial.number:>3d} | "
        f"k={model_config.num_clusters} "
        f"lr={training_config.learning_rate:.2e} "
        f"filters={model_config.numfilters} "
        f"fsize={model_config.filter_size} "
        f"layers={model_config.numlayers} "
        f"stride={model_config.stride} | "
        f"loss={mean_loss:.4f}±{std_loss:.4f}  "
        f"AMI={mean_ami:.4f}±{std_ami:.4f}"
        + (f"  GPU={mem.get('gpu_peak_mb', '?')}MB"
           f"  RAM={mem.get('ram_used_gb', '?')}GB({mem.get('ram_percent', '?')}%)"
           if mem else "")
    )

    # Incremental save — written after every trial so crashes don't lose results
    save_json(
        OUTPUT_DIR / f"trial_{trial.number:04d}.json",
        {
            "number":                 trial.number,
            "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loss":                   mean_loss,
            "ami_mean":               mean_ami,
            "ami_std":                std_ami,
            "loss_std":               std_loss,
            "gpu_peak_mb":            mem.get("gpu_peak_mb"),
            "ram_used_gb":            mem.get("ram_used_gb"),
            "ram_percent":            mem.get("ram_percent"),
            "num_predicted_clusters": int(metrics["num_predicted_clusters"]),
            "params":                 trial.params,
        },
    )

    # Explicit garbage collection to prevent RAM accumulation across trials
    gc.collect()
    tf.keras.backend.clear_session()

    return mean_loss


# --------------------------------------------------------------------------
# Convergence callback
# --------------------------------------------------------------------------
def make_convergence_callback(patience: int, threshold: float):
    """Return an Optuna callback that stops the study when converged."""

    def convergence_callback(
        study: optuna.Study, trial: optuna.Trial
    ) -> None:
        completed = [t for t in study.trials if t.value is not None]
        if len(completed) <= patience:
            return

        best_values: list[float] = []
        current_best = float("inf")
        for t in completed:
            current_best = min(current_best, t.value)
            best_values.append(current_best)

        recent_improvement = abs(best_values[-1] - best_values[-(patience + 1)])
        if recent_improvement < threshold:
            print(
                f"\nEarly stopping at trial {trial.number}: "
                f"no improvement > {threshold} in last {patience} trials."
            )
            study.stop()

    return convergence_callback


# --------------------------------------------------------------------------
# Convergence curve fitting
# --------------------------------------------------------------------------
def fit_convergence_curves(best_values: list[float]) -> dict:
    """Fit logarithmic, linear, and quadratic curves to the best-value trajectory.

    This characterises the convergence rate and lets you predict how much
    improvement remains if you ran more trials.
    """
    x = np.arange(len(best_values), dtype=float)
    y = np.array(best_values, dtype=float)
    results: dict = {}

    # Logarithmic fit: f(x) = a * log(x+1) + b
    try:
        def log_curve(x, a, b):
            return a * np.log(x + 1) + b

        popt, _ = curve_fit(log_curve, x, y, maxfev=5000)
        y_pred = log_curve(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        future_improvement = abs(
            log_curve(len(x) + 20, *popt) - log_curve(len(x), *popt)
        )
        results["log"] = {
            "a": float(popt[0]),
            "b": float(popt[1]),
            "r2": r2,
            "predicted_improvement_next_20_trials": float(future_improvement),
        }
    except Exception:
        results["log"] = None

    # Linear fit: f(x) = slope * x + intercept
    try:
        slope, intercept, r, _, _ = linregress(x, y)
        results["linear"] = {
            "slope":     float(slope),
            "intercept": float(intercept),
            "r2":        float(r ** 2),
        }
    except Exception:
        results["linear"] = None

    # Quadratic fit: f(x) = ax² + bx + c
    try:
        coeffs = np.polyfit(x, y, deg=2)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        results["quadratic"] = {
            "coefficients": [float(c) for c in coeffs],
            "r2": r2,
        }
    except Exception:
        results["quadratic"] = None

    return results


# --------------------------------------------------------------------------
# Top-3 validation at full epochs
# --------------------------------------------------------------------------
def validate_top_configs(
    study: optuna.Study,
    n_top: int = 3,
    full_epochs: int = EPOCHS_VALIDATION,
) -> list[dict]:
    """Re-run the top N configs at full epochs to confirm search rankings hold."""

    completed = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
    )[:n_top]

    validation_results = []

    for rank, trial in enumerate(completed):
        print(f"\nValidating rank {rank + 1} config (trial {trial.number})...")
        tf.keras.backend.clear_session()

        num_clusters_val = trial.params["num_clusters"]
        model_config = ModelConfig(
            input_shape=(340, 1),
            numfilters=trial.params["numfilters"],
            filter_size=trial.params["filter_size"],
            numlayers=trial.params["numlayers"],
            num_clusters=num_clusters_val,
            stride=trial.params["stride"],
            padding="same",
            dilation=1,
            use_batch_norm=True,
            num_clusters_overclustering=num_clusters_val * 3,
        )
        training_config = TrainingConfig(
            learning_rate=trial.params["learning_rate"],
            epochs=full_epochs,
            batch_size=64,
            xshift=trial.params["xshift"],
            xscale=trial.params["xscale"],
            yshift=trial.params["yshift"],
            yscale=trial.params["yscale"],
            order=3,
            verbose=1,
            early_stop_epoch=999,
        )

        val_losses: list[float] = []
        val_amis:   list[float] = []

        for _ in range(N_REPEATS_PER_TRIAL):
            tf.keras.backend.clear_session()
            model, history = train_iic_model(
                dataset.x,
                model_config=model_config,
                training_config=training_config,
            )
            probabilities = predict_in_batches(
                dataset.x,
                model,
                num_clusters=model_config.num_clusters,
                batch_size=64,
            )
            metrics = evaluate_clustering(dataset.y, probabilities)
            val_losses.append(float(history[-1]["loss"]))
            val_amis.append(float(metrics["ami_index"]))

        result = {
            "rank":                    rank + 1,
            "trial_number":            trial.number,
            "search_loss":             float(trial.value),
            "search_ami_mean":         trial.user_attrs.get("ami_mean"),
            "validation_loss_mean":    float(np.mean(val_losses)),
            "validation_loss_std":     float(np.std(val_losses)),
            "validation_ami_mean":     float(np.mean(val_amis)),
            "validation_ami_std":      float(np.std(val_amis)),
            "params":                  trial.params,
        }
        validation_results.append(result)

        print(
            f"  Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}  "
            f"AMI:  {np.mean(val_amis):.4f} ± {np.std(val_amis):.4f}"
        )

    return validation_results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_sampler_results: dict = {}

    for sampler_name, sampler in SAMPLERS.items():
        print(f"\n{'='*60}")
        print(f"Sampler: {sampler_name}  ({N_TRIALS} trials, "
              f"{N_REPEATS_PER_TRIAL} repeats each, "
              f"{EPOCHS_SEARCH} epochs per repeat)")
        print(f"{'='*60}\n")

        tf.keras.backend.clear_session()

        sampler_start_time = time.time()
        sampler_start_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Started at: {sampler_start_str}")

        study = optuna.create_study(
            direction="minimize",    # minimise loss — more negative = better clustering
            study_name=f"mcbj_iic_{sampler_name}",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            callbacks=[
                make_convergence_callback(CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD)
            ],
            catch=(Exception,),      # don't let one bad trial kill the whole study
            show_progress_bar=False,
        )

        # Build best-value trajectory for curve fitting
        sampler_end_time  = time.time()
        sampler_end_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_seconds   = sampler_end_time - sampler_start_time
        elapsed_hours     = elapsed_seconds / 3600
        print(f"Finished at: {sampler_end_str}")
        print(f"Total time:  {elapsed_hours:.2f} hours ({elapsed_seconds / 60:.1f} minutes)")

        completed = [t for t in study.trials if t.value is not None]
        best_values: list[float] = []
        current_best = float("inf")
        for t in completed:
            current_best = min(current_best, t.value)
            best_values.append(current_best)

        curve_fits = fit_convergence_curves(best_values)

        # Validate top 3 configs at full epochs
        print(f"\nValidating top 3 configs at {EPOCHS_VALIDATION} epochs...")
        validation = validate_top_configs(study, n_top=3)

        # Collect all trial data
        trial_data = [
            {
                "number":                 t.number,
                "loss":                   t.value,
                "ami_mean":               t.user_attrs.get("ami_mean"),
                "ami_std":                t.user_attrs.get("ami_std"),
                "loss_std":               t.user_attrs.get("loss_std"),
                "gpu_peak_mb":            t.user_attrs.get("gpu_peak_mb"),
                "ram_used_gb":            t.user_attrs.get("ram_used_gb"),
                "ram_percent":            t.user_attrs.get("ram_percent"),
                "num_predicted_clusters": t.user_attrs.get("num_predicted_clusters"),
                "params":                 t.params,
                "state":                  str(t.state),
            }
            for t in study.trials
        ]

        sampler_result = {
            "sampler":                  sampler_name,
            "start_time":               sampler_start_str,
            "end_time":                 sampler_end_str,
            "elapsed_hours":            round(elapsed_hours, 3),
            "n_trials_run":             len(completed),
            "best_loss":                float(study.best_value),
            "best_params":              study.best_params,
            "best_trial_number":        study.best_trial.number,
            "best_ami_mean":            study.best_trial.user_attrs.get("ami_mean"),
            "convergence_fits":         curve_fits,
            "best_values_over_trials":  best_values,
            "validation":               validation,
            "all_trials":               trial_data,
        }
        all_sampler_results[sampler_name] = sampler_result

        # Save per-sampler results immediately in case of later crash
        save_json(OUTPUT_DIR / f"results_{sampler_name}.json", sampler_result)

        print(f"\n{sampler_name} best loss:   {study.best_value:.4f}")
        ami = study.best_trial.user_attrs.get("ami_mean")
        if ami is not None:
            print(f"{sampler_name} best AMI:    {ami:.4f}")
        print(f"{sampler_name} best params: {study.best_params}")

        if curve_fits.get("log"):
            log = curve_fits["log"]
            print(
                f"{sampler_name} log-fit R²={log['r2']:.3f}  "
                f"predicted improvement next 20 trials: "
                f"{log['predicted_improvement_next_20_trials']:.4f}"
            )

    # Save combined comparison across all samplers
    save_json(OUTPUT_DIR / "comparison_all_samplers.json", all_sampler_results)

    # --------------------------------------------------------------------------
    # Final summary table
    # --------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL COMPARISON ACROSS SAMPLERS")
    print(f"{'='*60}")
    print(f"{'Sampler':<10} {'Best loss':>12} {'Best AMI':>10} {'Trials run':>12}")
    print("-" * 48)
    for name, res in all_sampler_results.items():
        ami_str = f"{res['best_ami_mean']:.4f}" if res.get("best_ami_mean") else "N/A"
        print(
            f"{name:<10} {res['best_loss']:>12.4f} "
            f"{ami_str:>10} {res['n_trials_run']:>12}"
        )

    # Print best overall config as a ready-to-run bench command
    best_sampler = min(
        all_sampler_results,
        key=lambda k: all_sampler_results[k]["best_loss"],
    )
    best = all_sampler_results[best_sampler]
    p = best["best_params"]

    print(f"\nBest overall config found by {best_sampler}.")
    print("Run the full benchmark with:")
    print(
        f"\npython scripts/bench7.py"
        f" --epochs {EPOCHS_VALIDATION}"
        f" --num-clusters {p['num_clusters']}"
        f" --num-clusters-overclustering {p['num_clusters'] * 3}"
        f" --numfilters {p['numfilters']}"
        f" --filter-size {p['filter_size']}"
        f" --numlayers {p['numlayers']}"
        f" --stride {p['stride']}"
        f" --learning-rate {p['learning_rate']:.2e}"
        f" --xshift {p['xshift']:.3f}"
        f" --xscale {p['xscale']:.3f}"
        f" --yshift {p['yshift']:.3f}"
        f" --yscale {p['yscale']:.3f}"
        f" --padding same"
        f" --dilation 1"
        f" --output-dir runs/best_params_final"
    )

    print(f"\nAll results saved to {OUTPUT_DIR}")