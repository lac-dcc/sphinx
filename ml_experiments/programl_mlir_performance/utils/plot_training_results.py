import torch
import os
import sys
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
import logging
import glob

from config.params import params

# Test Accuracy
# MEAN_BASELINE_RMSE = 0.05825
# MEAN_BASELINE_MAE  = 0.02458

# Training Time
MEAN_BASELINE_RMSE = 919.52343
MEAN_BASELINE_MAE  = 763.86714

LINEAR_REGRESSION_RMSE = 374.55331
LINEAR_REGRESSION_MAE = 301.66871

FOLDER_NAME = "run_9"

# FIGURE_TITLE = "Training History and Validation Metrics - Original StableHLO  - Training Time"
FIGURE_TITLE = "Training History and Validation Metrics - Legalized Linalg  - Training Time - Fine Tuning Experiment 9"
# FIGURE_TITLE = "Training History and Validation Metrics - Legalized Linalg  - Training Time - Trained on Linalg"

def highlight_max(ax, x, y, line_color, dot_color, edge_color, label=None, mode="max"):
    i = 0
    if mode == "max":
        i = numpy.argmax(y)
    elif mode == "min":
        i = numpy.argmin(y)

    # Capture the handle (lines[0])
    lines = ax.plot(
        x[i], y[i],
        marker='o',
        markersize=8,
        color=line_color,
        markerfacecolor=dot_color,
        markeredgecolor=edge_color,
        zorder=5,
        label=label + f" ({y[i]:.3f})"
    )
    return lines[0]


def highlight_at_index(ax, x, y, idx, color, label):
    point, = ax.plot(
        x[idx], y[idx],
        marker='o',
        markersize=8,
        color=color,
        markerfacecolor="gold",
        markeredgecolor="black",
        zorder=5,
        label=f"{label} ({y[idx]:.3f})"
    )
    return point


def plot_training_history_from_checkpoints(run_dir):
    logging.info("Loading training history from checkpoints...")

    epoch_files = glob.glob(os.path.join(run_dir, "epoch_*.pt"))
    if not epoch_files:
        logging.warning("No epoch checkpoints found. Skipping history plot.")
        return

    history = []
    for f in tqdm(epoch_files, desc="Reading Checkpoints"):
        try:
            cp = torch.load(f, map_location='cpu', weights_only=False)
            history.append({
                'epoch': cp['epoch'],
                'train_mse': cp['train_mse'],
                'val_rmse': cp['val_rmse'],
                'val_mae': cp['val_mae'],
                'val_r2': cp.get('val_r2', 0),
                'val_tau': cp['val_tau'],
                'val_spearman': cp['val_spearman']
            })
        except Exception as e:
            logging.warning(f"Failed to load {f}: {e}")

    history.sort(key=lambda x: x['epoch'])

    epochs = [x['epoch'] for x in history]
    taus = [x['val_tau'] for x in history]
    spearmans = [x['val_spearman'] for x in history]
    rmses = [x['val_rmse'] for x in history]
    maes = [x['val_mae'] for x in history]
    r2s = [x['val_r2'] for x in history]
    train_mses = [x['train_mse'] for x in history]

    best_i = numpy.argmax(taus)

    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        FIGURE_TITLE,
        fontsize=22,
        fontweight="bold"
    )

    # --- Panel 1: Ranking Metrics ---
    ax = axes[0]
    l1, = ax.plot(epochs, spearmans, label="Spearman's Rho", color='green', linewidth=2)
    # p1 = highlight_max(ax, epochs, spearmans, "green", "gold", "black", "Best Spearman")
    p1 = highlight_at_index(ax, epochs, spearmans, best_i, "green", "Spearman")

    l2, = ax.plot(epochs, taus, label="Kendall's Tau", color='blue', linewidth=2)
    # p2 = highlight_max(ax, epochs, taus, "blue", "gold", "black", "Best Tau")
    p2 = highlight_at_index(ax, epochs, taus, best_i, "blue", "Best Tau")

    ax.set_title("Ranking Reliability (Correlation)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3)

    # Legend 1: Lines (Inside)
    leg1 = ax.legend(handles=[l1, l2], loc='best')
    ax.add_artist(leg1)  # Keep this legend visible
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    # Legend 2: Best Points (Below)
    ax.legend(handles=[p1, p2], loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fancybox=True, frameon=True, edgecolor="black")

    # --- Panel 2: Error Metrics ---
    ax = axes[1]
    l1, = ax.plot(epochs, rmses, label="RMSE (Root Mean Squared Error)", color='red', linewidth=2)
    # p1 = highlight_max(ax, epochs, rmses, "red", "gold", "black", "Best RMSE", mode="min")
    p1 = highlight_at_index(ax, epochs, rmses, best_i, "red", "RMSE")

    l2, = ax.plot(epochs, maes, label="MAE (Mean Absolute Error)", color='orange', linewidth=2)
    # p2 = highlight_max(ax, epochs, maes, "orange", "gold", "black", "Best MAE", mode="min")
    p2 = highlight_at_index(ax, epochs, maes, best_i, "orange", "MAE")

    ax.set_title("Prediction Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)

    mean_baseline_rmse_line = ax.axhline(
        MEAN_BASELINE_RMSE,
        color='crimson',
        linestyle=':',
        linewidth=2,
        alpha=0.7,
        label='Baseline RMSE - Mean'
    )

    lr_baseline_rmse_line = ax.axhline(
        LINEAR_REGRESSION_RMSE,
        color='crimson',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='Baseline RMSE - Linear Regression'
    )

    mean_baseline_mae_line = ax.axhline(
        MEAN_BASELINE_MAE,
        color='orange',
        linestyle=':',
        linewidth=2,
        alpha=0.7,
        label='Baseline MAE - Mean'
    )

    lr_baseline_mae_line = ax.axhline(
        LINEAR_REGRESSION_MAE,
        color='orange',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='Baseline MAE - Linear Regression'
    )

    # Legend 1: Lines (Inside)
    leg1 = ax.legend(handles=[mean_baseline_rmse_line, mean_baseline_mae_line,
                              lr_baseline_rmse_line, lr_baseline_mae_line,
                              l1, l2],
                     loc='upper right', bbox_to_anchor=(0.96, 0.79))
    ax.add_artist(leg1)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    # Legend 2: Best Points (Below)
    ax.legend(handles=[p1, p2], loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fancybox=True, frameon=True, edgecolor="black")

    # --- Panel 3: Convergence ---
    ax = axes[2]
    # Left Axis (R2)
    l1, = ax.plot(epochs, r2s, label="R² (Explained Var)", color='purple', linewidth=2)
    # p1 = highlight_max(ax, epochs, r2s, "purple", "gold", "black", "Best R²")
    p1 = highlight_at_index(ax, epochs, r2s, best_i, "purple", "R²")
    ax.set_ylabel("R² Score", color='purple')
    ax.tick_params(axis='y', labelcolor='purple')

    # Right Axis (Train Loss)
    ax2 = ax.twinx()
    l2, = ax2.plot(epochs, train_mses, label="Train Loss (Norm MSE)", color='brown', linewidth=2)
    # p2 = highlight_max(ax2, epochs, train_mses, "brown", "gold", "black", "Best Train Loss", mode="min")
    p2 = highlight_at_index(ax2, epochs, train_mses, best_i, "brown", "Train Loss")
    ax2.set_ylabel("Train Loss", color='brown')
    ax2.tick_params(axis='y', labelcolor='brown')

    ax.set_title("Model Convergence")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # Legend 1: Lines (Inside - Combined from ax and ax2)
    leg1 = ax.legend(handles=[l1, l2], loc='center right')
    ax.add_artist(leg1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    # Legend 2: Best Points (Below - Combined from ax and ax2)
    # We must add this legend to 'ax', but pass handles from both
    ax.legend(handles=[p1, p2], loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fancybox=True, frameon=True, edgecolor="black")

    save_path = os.path.join(run_dir, "training_history.png")
    # Adjust layout to make room for bottom legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(save_path, dpi=600)
    logging.info(f"Training history saved to {save_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    run_folder_name = sys.argv[1] if len(sys.argv) > 1 else FOLDER_NAME

    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    run_dir = os.path.join(base_checkpoint_dir, run_folder_name)
    model_path = os.path.join(run_dir, "best_model.pt")

    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}")
        return

    plot_training_history_from_checkpoints(run_dir)


if __name__ == '__main__':
    main()