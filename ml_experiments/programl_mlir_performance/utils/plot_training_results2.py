import torch
import os
import sys
import numpy
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
LINEAR_REGRESSION_MAE  = 301.66871

FOLDER_NAME = "TrainingTime_40Epochs_StableHLO"
# FOLDER_NAME = "FT1"
# FOLDER_NAME = "TrainingTime_100Epochs_Linalg"

FIGURE_TITLE = "Training History and Validation Metrics - Original StableHLO  - Training Time"
# FIGURE_TITLE = "Training History and Validation Metrics - Legalized Linalg  - Training Time - Fine Tuning Experiment 1"
# FIGURE_TITLE = "Training History and Validation Metrics - Legalized Linalg  - Training Time - Trained on Linalg"

# ── Seaborn theme ─────────────────────
sns.set_theme(
    style="whitegrid",
    palette="deep",
    font_scale=1.2,
    rc={
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# Coordinated palette
_PALETTE  = sns.color_palette("deep")
C_BLUE    = _PALETTE[0]
C_ORANGE  = _PALETTE[1]
C_GREEN   = _PALETTE[2]
C_RED     = _PALETTE[3]
C_PURPLE  = _PALETTE[4]
C_BROWN   = _PALETTE[5]


def highlight_at_index(ax, x, y, idx, color, label):
    """Highlight a single point using sns.scatterplot for consistent styling."""
    return sns.scatterplot(
        x=[x[idx]], y=[y[idx]],
        ax=ax,
        color=color,
        s=120,
        zorder=6,
        edgecolor="white",
        linewidth=1.5,
        label=f"{label} ({y[idx]:.3f})",
        legend=False,
    )


def plot_training_history_from_checkpoints(run_dir):
    logging.info("Loading training history from checkpoints...")

    epoch_files = glob.glob(os.path.join(run_dir, "epoch_*.pt"))
    if not epoch_files:
        logging.warning("No epoch checkpoints found. Skipping history plot.")
        return

    history = []
    for f in tqdm(epoch_files, desc="Reading Checkpoints"):
        try:
            cp = torch.load(f, map_location="cpu", weights_only=False)
            history.append({
                "epoch":       cp["epoch"],
                "train_mse":   cp["train_mse"],
                "val_rmse":    cp["val_rmse"],
                "val_mae":     cp["val_mae"],
                "val_r2":      cp.get("val_r2", 0),
                "val_tau":     cp["val_tau"],
                "val_spearman": cp["val_spearman"],
            })
        except Exception as e:
            logging.warning(f"Failed to load {f}: {e}")

    history.sort(key=lambda x: x["epoch"])

    epochs     = [x["epoch"]       for x in history]
    taus       = [x["val_tau"]     for x in history]
    spearmans  = [x["val_spearman"] for x in history]
    rmses      = [x["val_rmse"]    for x in history]
    maes       = [x["val_mae"]     for x in history]
    r2s        = [x["val_r2"]      for x in history]
    train_mses = [x["train_mse"]   for x in history]

    best_i = int(numpy.argmax(taus))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(FIGURE_TITLE, fontsize=22, fontweight="bold")

    # ── Panel 1: Ranking Metrics ─────────────────────────────────────────────
    ax = axes[0]

    l1, = ax.plot(epochs, spearmans, color=C_GREEN, linewidth=2.5, label="Spearman's Rho")
    l2, = ax.plot(epochs, taus, color=C_BLUE,  linewidth=2.5, label="Kendall's Tau")

    highlight_at_index(ax, epochs, spearmans, best_i, C_GREEN, f"Spearman @ best Tau")
    highlight_at_index(ax, epochs, taus,      best_i, C_BLUE, f"Best Tau")

    ax.set_title("Ranking reliability (correlation)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    leg1 = ax.legend(handles=[l1, l2], loc="best", framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(
        handles=[h for h in ax.get_legend_handles_labels()[0] if h not in [l1, l2]],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2, fancybox=True, frameon=True, edgecolor="black",
    )

    # ── Panel 2: Error Metrics ───────────────────────────────────────────────
    ax = axes[1]

    ax.plot(epochs, rmses, color=C_RED,    linewidth=2, alpha=0.30)
    ax.plot(epochs, maes,  color=C_ORANGE, linewidth=2, alpha=0.30)
    l1, = ax.plot(epochs, rmses, color=C_RED,    linewidth=2.5, label="RMSE")
    l2, = ax.plot(epochs, maes, color=C_ORANGE, linewidth=2.5, label="MAE")

    highlight_at_index(ax, epochs, rmses, best_i, C_RED, f"RMSE @ best tau")
    highlight_at_index(ax, epochs, maes,  best_i, C_ORANGE, f"MAE @ best tau")

    mean_rmse_line = ax.axhline(
        MEAN_BASELINE_RMSE, color=C_RED,    linestyle=":", linewidth=1.8, alpha=0.7,
        label=f"Mean RMSE ({MEAN_BASELINE_RMSE:.0f})",
    )
    mean_mae_line = ax.axhline(
        MEAN_BASELINE_MAE,  color=C_ORANGE, linestyle=":", linewidth=1.8, alpha=0.7,
        label=f"Mean MAE ({MEAN_BASELINE_MAE:.0f})",
    )
    lr_rmse_line = ax.axhline(
        LINEAR_REGRESSION_RMSE, color=C_RED,    linestyle="--", linewidth=1.8, alpha=0.7,
        label=f"LR RMSE ({LINEAR_REGRESSION_RMSE:.0f})",
    )
    lr_mae_line = ax.axhline(
        LINEAR_REGRESSION_MAE,  color=C_ORANGE, linestyle="--", linewidth=1.8, alpha=0.7,
        label=f"LR MAE ({LINEAR_REGRESSION_MAE:.0f})",
    )

    ax.set_title("Prediction error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    leg1 = ax.legend(
        handles=[mean_rmse_line, mean_mae_line, lr_rmse_line, lr_mae_line, l1, l2],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.74),
        framealpha=0.85,
        fontsize=11,
    )
    ax.add_artist(leg1)
    point_handles = [h for h in ax.get_legend_handles_labels()[0]
                     if h not in [mean_rmse_line, mean_mae_line,
                                  lr_rmse_line, lr_mae_line, l1, l2]]
    ax.legend(
        handles=point_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2, fancybox=True, frameon=True, edgecolor="black",
    )

    # # ── Panel 3: Convergence ─────────────────────────────────────────────────
    ax  = axes[2]
    ax2 = ax.twinx()
    ax2.grid(False)

    ax.plot(epochs, r2s, color=C_PURPLE, linewidth=2, alpha=0.30)
    l1, = ax.plot(epochs, r2s, color=C_PURPLE, linewidth=2.5, label="R² (explained var)")
    ax.set_ylabel("R² score", color=C_PURPLE)
    ax.tick_params(axis="y", labelcolor=C_PURPLE)

    ax2.plot(epochs, train_mses, color=C_BROWN, linewidth=2, alpha=0.30)
    l2, = ax2.plot(epochs, train_mses, color=C_BROWN, linewidth=2.5, label="Train loss (norm MSE)")
    ax2.set_ylabel("Train loss", color=C_BROWN)
    ax2.tick_params(axis="y", labelcolor=C_BROWN)

    highlight_at_index(ax,  epochs, r2s,        best_i, C_PURPLE, "R² @ best tau")
    highlight_at_index(ax2, epochs, train_mses, best_i, C_BROWN,  "Loss @ best tau")

    ax.set_title("Model convergence")
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    leg1 = ax.legend(handles=[l1, l2], loc="center right", framealpha=0.85)
    ax.add_artist(leg1)

    point_handles_ax  = [h for h in ax.get_legend_handles_labels()[0]  if h not in [l1]]
    point_handles_ax2 = [h for h in ax2.get_legend_handles_labels()[0] if h not in [l2]]
    ax.legend(
        handles=point_handles_ax + point_handles_ax2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2, fancybox=True, frameon=True, edgecolor="black",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = os.path.join(run_dir, "training_history.png")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(save_path, dpi=600)
    logging.info(f"Training history saved to {save_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_folder_name = sys.argv[1] if len(sys.argv) > 1 else FOLDER_NAME

    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    run_dir   = os.path.join(base_checkpoint_dir, run_folder_name)
    model_path = os.path.join(run_dir, "best_tau_model.pt")

    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}")
        return

    plot_training_history_from_checkpoints(run_dir)


if __name__ == "__main__":
    main()