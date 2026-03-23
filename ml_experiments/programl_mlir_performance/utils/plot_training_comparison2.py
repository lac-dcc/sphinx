import torch
import os
import glob
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from config.params import params

# ── Seaborn theme ────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    font_scale=1.2,
    rc={
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# ── Configuration ────────────────────────────────────────────────────────────
# Use seaborn's palette for consistent, harmonious colors
_PAL = sns.color_palette("tab10", 10)

RUNS_CONFIG = {
    # "TrainingTime_20Epochs_StableHLO": {"label": "Training on Linalg", "color": _PAL[9], "style": "-"},
    "FT1":                          {"label": "FT: Exp 1",          "color": _PAL[0], "style": "-"},
    "FT2":                          {"label": "FT: Exp 2",          "color": _PAL[1], "style": "-"},
    "FT3":                          {"label": "FT: Exp 3",          "color": _PAL[2], "style": "-"},
    "FT4":                          {"label": "FT: Exp 4",          "color": _PAL[3], "style": "-"},
    "FT5":                          {"label": "FT: Exp 5",          "color": _PAL[4], "style": "-"},
    "FT6":                          {"label": "FT: Exp 6",          "color": _PAL[5], "style": "-"},
    "FT7":                          {"label": "FT: Exp 7",          "color": _PAL[6], "style": "-"},
    "FT8":                          {"label": "FT: Exp 8",          "color": _PAL[7], "style": "-"},
    "TrainingTime_100Epochs_Linalg": {"label": "Training on Linalg", "color": _PAL[8], "style": "-"},
}

METRICS_TO_PLOT = [
    {"key": "val_rmse", "title": "RMSE (lower is better)",        "ylabel": "RMSE"},
    {"key": "val_mae",  "title": "MAE (lower is better)",         "ylabel": "MAE"},
    {"key": "val_tau",  "title": "Kendall's Tau (higher is better)", "ylabel": "Correlation"},
]


# ── Data loading ─────────────────────────────────────────────────────────────
def load_run_history(run_dir):
    """Reads all epoch_*.pt files in a directory and returns sorted history."""
    epoch_files = glob.glob(os.path.join(run_dir, "epoch_*.pt"))
    if not epoch_files:
        logging.warning(f"No checkpoints found in {run_dir}")
        return []

    history = []
    for f in epoch_files:
        try:
            cp = torch.load(f, map_location="cpu", weights_only=False)
            history.append({
                "epoch":    cp["epoch"],
                "val_rmse": cp["val_rmse"],
                "val_mae":  cp["val_mae"],
                "val_tau":  cp["val_tau"],
            })
        except (RuntimeError, KeyError, FileNotFoundError) as e:
            print(f"Skipping {f}: {e}")

    history.sort(key=lambda x: x["epoch"])
    return history


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    logging.info(f"Looking for runs in: {base_checkpoint_dir}")

    data = {}
    for run_name, config in RUNS_CONFIG.items():
        run_path = os.path.join(base_checkpoint_dir, run_name)
        logging.info(f"Loading {run_name}...")
        history = load_run_history(run_path)

        if history:
            data[run_name] = {
                "epochs":   [h["epoch"]    for h in history],
                "val_rmse": [h["val_rmse"] for h in history],
                "val_mae":  [h["val_mae"]  for h in history],
                "val_tau":  [h["val_tau"]  for h in history],
                "config":   config,
            }
        else:
            logging.warning(f"Skipping {run_name} (no data)")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        "Comparative Training Progression: StableHLO to Linalg Fine Tuning vs Linalg Training",
        fontsize=20,
        fontweight="bold",
    )

    for i, metric in enumerate(METRICS_TO_PLOT):
        ax  = axes[i]
        key = metric["key"]

        for run_name, run_data in data.items():
            cfg = run_data["config"]
            is_baseline = "Linalg" in cfg["label"] and "FT" not in cfg["label"]
            ax.plot(
                run_data["epochs"],
                run_data[key],
                label=cfg["label"],
                color=cfg["color"],
                linestyle=cfg["style"],
                linewidth=2.5 if is_baseline else 1.8,
                marker="o",
                markersize=3,
                alpha=1.0 if is_baseline else 0.85,
            )

        ax.set_title(metric["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric["ylabel"])
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
        ax.tick_params(axis="both", labelcolor="black")

    # Single shared legend below all panels
    # Build handles manually so we control exactly what appears
    legend_handles = [
        Line2D([0], [0],
               color=run_data["config"]["color"],
               linestyle=run_data["config"]["style"],
               linewidth=2,
               marker="o",
               markersize=5,
               label=run_data["config"]["label"])
        for run_data in data.values()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(data),
        frameon=True,
        edgecolor="black",
        framealpha=0.85,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)

    save_path = os.path.join(base_checkpoint_dir, "comparative_training_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()