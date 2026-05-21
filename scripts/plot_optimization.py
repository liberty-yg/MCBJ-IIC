import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss_over_trials(results_path, output_path):
    """Plot best loss found so far over Bayesian optimisation trials."""
    
    data = json.loads(Path(results_path).read_text())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for sampler_name, results in data.items():
        trials = [t for t in results["trials"] if t["loss"] is not None]
        losses = [t["loss"] for t in trials]
        amis = [t["ami"] for t in trials if t["ami"] is not None]
        
        # Best so far curve
        best_so_far = np.minimum.accumulate(losses)
        axes[0].plot(best_so_far, label=sampler_name)
        
        # AMI over trials if available
        if amis:
            best_ami_so_far = np.maximum.accumulate(amis)
            axes[1].plot(best_ami_so_far, label=sampler_name)
    
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("Best loss (lower is better)")
    axes[0].set_title("Convergence: Best loss over trials")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Trial number")
    axes[1].set_ylabel("Best AMI (higher is better)")
    axes[1].set_title("Best AMI over trials (for reference)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_fit(results_path, output_path):
    """Fit log/linear/quadratic curves to convergence trajectory."""
    from scipy.optimize import curve_fit
    
    data = json.loads(Path(results_path).read_text())
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for sampler_name, results in data.items():
        losses = [t["loss"] for t in results["trials"] if t["loss"] is not None]
        best_so_far = np.minimum.accumulate(losses)
        x = np.arange(len(best_so_far))
        
        ax.scatter(x, best_so_far, s=10, alpha=0.5, label=f"{sampler_name} (observed)")
        
        # Fit logarithmic curve
        try:
            def log_curve(x, a, b):
                return a * np.log(x + 1) + b
            popt, _ = curve_fit(log_curve, x, best_so_far)
            x_pred = np.linspace(0, len(x) * 1.5, 200)
            ax.plot(x_pred, log_curve(x_pred, *popt), 
                   linestyle="--", label=f"{sampler_name} (log fit)")
        except Exception:
            pass
    
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best loss")
    ax.set_title("Convergence curves with logarithmic fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)