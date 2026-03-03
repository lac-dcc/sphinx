import torch
import os
import sys
import glob
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from config.params import params

# --- CONFIGURATION ---
# Map folder names to Legend Labels
RUNS_CONFIG = {
    # "TrainingTime_20Epochs_StableHLO": {"label": "Training on Linalg", "color": "pink", "style": "-"},
    "run_1": {"label": "FT: Exp 1", "color": "green", "style": "-"},
    "run_2": {"label": "FT: Exp 2", "color": "red", "style": "-"},
    "run_3": {"label": "FT: Exp 3", "color": "orange", "style": "-"},
    "run_4": {"label": "FT: Exp 4", "color": "yellow", "style": "-"},
    "run_5": {"label": "FT: Exp 5", "color": "aqua", "style": "-"},
    "run_6": {"label": "FT: Exp 6", "color": "blue", "style": "-"},
    "run_7": {"label": "FT: Exp 7", "color": "violet", "style": "-"},
    "run_8": {"label": "FT: Exp 8", "color": "grey", "style": "-"},
    "TrainingTime_20Epochs_Linalg": {"label": "Training on Linalg", "color": "black", "style": "-"},
}

# Define which metrics to plot
METRICS_TO_PLOT = [
    {"key": "val_rmse", "title": "RMSE (Lower is Better)", "ylabel": "RMSE"},
    {"key": "val_mae", "title": "MAE (Lower is Better)", "ylabel": "MAE"},
    {"key": "val_tau", "title": "Kendall's Tau (Higher is Better)", "ylabel": "Correlation"},
]


# ---------------------

def load_run_history(run_dir):
    """Reads all epoch_*.pt files in a directory and returns sorted history."""
    epoch_files = glob.glob(os.path.join(run_dir, "epoch_*.pt"))
    if not epoch_files:
        logging.warning(f"No checkpoints found in {run_dir}")
        return []

    history = []
    for f in epoch_files:
        try:
            cp = torch.load(f, map_location='cpu', weights_only=False)
            history.append({
                'epoch': cp['epoch'],
                'val_rmse': cp['val_rmse'],
                'val_mae': cp['val_mae'],
                'val_tau': cp['val_tau'],
            })
        except Exception as e:
            pass

    history.sort(key=lambda x: x['epoch'])
    return history


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)

    if not os.path.exists(base_checkpoint_dir):
        base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)

    logging.info(f"Looking for runs in: {base_checkpoint_dir}")

    data = {}
    for run_name, config in RUNS_CONFIG.items():
        run_path = os.path.join(base_checkpoint_dir, run_name)
        logging.info(f"Loading {run_name}...")
        history = load_run_history(run_path)

        if history:
            data[run_name] = {
                "epochs": [h['epoch'] for h in history],
                "val_rmse": [h['val_rmse'] for h in history],
                "val_mae": [h['val_mae'] for h in history],
                "val_tau": [h['val_tau'] for h in history],
                "config": config
            }
        else:
            logging.warning(f"Skipping {run_name} (No data)")

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Comparative Training Progression: StableHLO to Linalg Fine Tuning vs Linalg Training", fontsize=20, fontweight='bold')

    for i, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        key = metric["key"]

        for run_name, run_data in data.items():
            cfg = run_data["config"]
            ax.plot(
                run_data["epochs"],
                run_data[key],
                label=cfg["label"],
                color=cfg["color"],
                linestyle=cfg["style"],
                linewidth=2.5 if "Target" in cfg["label"] else 2,
                marker='o',
                markersize=4
            )

        ax.set_title(metric["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

        if i == 0:
            ax.legend(loc='best', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make space for title

    save_path = base_checkpoint_dir + "/comparative_training_results.png"
    plt.savefig(save_path, dpi=300)
    logging.info(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()