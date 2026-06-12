import torch
import os
import glob
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from config.params import params
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("fontTools").setLevel(logging.WARNING)

plt.rcParams["pdf.fonttype"] = 42   # TrueType — fully embedded
plt.rcParams["ps.fonttype"] = 42

# ── Seaborn theme ────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#882255",  # wine
    "#332288",  # indigo
]

# ── Configuration ────────────────────────────────────────────────────────────
# _PAL = sns.color_palette("tab10", 10)
# _PAL = sns.color_palette("colorblind", 10)
# _PAL = sns.color_palette("Set2", 8)
_PAL = sns.color_palette(OKABE_ITO)

# RUNS_CONFIG = {
#     "FT2":                              {"label": "FT: Exp 1",                      "color": _PAL[2], "style": "-"},
#     "FT1":                              {"label": "FT: Exp 4",                      "color": _PAL[8], "style": "-"},
#     "FT8":                              {"label": "FT: Exp 7",                      "color": _PAL[7], "style": "-"},
#     "TrainingTime_40Epochs_StableHLO":  {"label": "StableHLO (90%)",                "color": _PAL[0], "style": "-"},
#     "FT3":                              {"label": "FT: Exp 2",                      "color": _PAL[4], "style": "-"},
#     "FT6":                              {"label": "FT: Exp 5",                      "color": _PAL[3], "style": "-"},
#     "FT5":                              {"label": "FT: Exp 8",                      "color": _PAL[9], "style": "-"},
#     "FT4":                              {"label": "FT: Exp 3",                      "color": _PAL[6], "style": "-"},
#     "FT7":                              {"label": "FT: Exp 6",                      "color": _PAL[5], "style": "-"},
#     "TrainingTime_100Epochs_Linalg":    {"label": "Linalg (10%)",                   "color": _PAL[1], "style": "-"},
# }

# RUNS_CONFIG = {
#     "FT2":                              {"label": "FT: Exp 1",                      "color": _PAL[2], "style": "-"},
#     "FT1":                              {"label": "FT: Exp 4",                      "color": _PAL[8], "style": "-"},
#     "FT8":                              {"label": "FT: Exp 7",                      "color": _PAL[7], "style": "-"},
#     "TestAccuracy_60Epochs_StableHLO":  {"label": "StableHLO (90%)",                "color": _PAL[0], "style": "-"},
#     "FT3":                              {"label": "FT: Exp 2",                      "color": _PAL[4], "style": "-"},
#     "FT6":                              {"label": "FT: Exp 5",                      "color": _PAL[3], "style": "-"},
#     "FT5":                              {"label": "FT: Exp 8",                      "color": _PAL[9], "style": "-"},
#     "FT4":                              {"label": "FT: Exp 3",                      "color": _PAL[6], "style": "-"},
#     "FT7":                              {"label": "FT: Exp 6",                      "color": _PAL[5], "style": "-"},
#     "TestAccuracy_100Epochs_Linalg":    {"label": "Linalg (10%)",                   "color": _PAL[1], "style": "-"},
# }

# RUNS_CONFIG = {
#     "1%":                           {"label": "1%",           "color": _PAL[0], "style": "-"},
#     "30%":                          {"label": "30%",          "color": _PAL[4], "style": "-"},
#     "5%":                           {"label": "5%",           "color": _PAL[1], "style": "-"},
#     "40%":                          {"label": "40%",          "color": _PAL[5], "style": "-"},
#     "10%":                          {"label": "10%",          "color": _PAL[2], "style": "-"},
#     "50%":                          {"label": "50%",          "color": _PAL[6], "style": "-"},
#     "20%":                          {"label": "20%",          "color": _PAL[3], "style": "-"},
#     "60%":                          {"label": "60%",          "color": _PAL[7], "style": "-"},
# }

# RUNS_CONFIG = {
#     "1%":                           {"label": "1%",             "color": _PAL[0], "style": "-"},
#     "1%_FT":                        {"label": "1% Fine Tuned",  "color": _PAL[1], "style": "-"},
#     "5%":                           {"label": "5%",             "color": _PAL[2], "style": "-"},
#     "5%_FT":                        {"label": "5% Fine Tuned",  "color": _PAL[4], "style": "-"},
#     "10%":                          {"label": "10%",            "color": _PAL[3], "style": "-"},
#     "10%_FT":                       {"label": "10% Fine Tuned","color": _PAL[7], "style": "-"},
# }

# RUNS_CONFIG = {
#     "TrainingTime_40Epochs_StableHLO":  {"label": "StableHLO (90%)",                "color": _PAL[0], "style": "-"},
#     "TrainingTime_T10_FTE2":            {"label": "FT: Exp 2",                      "color": _PAL[4], "style": "-"},
#     "TrainingTime_T10_FTE6":            {"label": "FT: Exp 6",                      "color": _PAL[3], "style": "-"},
#     "TrainingTime_T10_FTE4":            {"label": "FT: Exp 4",                      "color": _PAL[6], "style": "-"},
#     "TrainingTime_T10_Scratch":         {"label": "LLVM (10%)",                   "color": _PAL[1], "style": "-"},
# }

# RUNS_CONFIG = {
#     "TrainingTime_40Epochs_StableHLO":  {"label": "StableHLO (90%)",                "color": _PAL[0], "style": "-"},
#     "TrainingTime_T20_FTE2":            {"label": "FT: Exp 2",                      "color": _PAL[4], "style": "-"},
#     "TrainingTime_T20_FTE4":            {"label": "FT: Exp 4",                      "color": _PAL[6], "style": "-"},
#     "TrainingTime_T20_Scratch":         {"label": "LLVM (10%)",                   "color": _PAL[1], "style": "-"},
# }

# RUNS_CONFIG = {
#     "FTE1":                              {"label": "FT: Exp 1",                      "color": _PAL[2], "style": "-"},
#     "FTE4":                              {"label": "FT: Exp 4",                      "color": _PAL[8], "style": "-"},
#     "FTE7":                              {"label": "FT: Exp 7",                      "color": _PAL[7], "style": "-"},
#     "TrainingTime_40Epochs_StableHLO":   {"label": "StableHLO (90%)",                "color": _PAL[0], "style": "-"},
#     "FTE2":                              {"label": "FT: Exp 2",                      "color": _PAL[4], "style": "-"},
#     "FTE5":                              {"label": "FT: Exp 5",                      "color": _PAL[3], "style": "-"},
#     "FTE8":                              {"label": "FT: Exp 8",                      "color": _PAL[9], "style": "-"},
#     "FTE3":                              {"label": "FT: Exp 3",                      "color": _PAL[6], "style": "-"},
#     "FTE6":                              {"label": "FT: Exp 6",                      "color": _PAL[5], "style": "-"},
#     "TrainingTime_100Epochs_Arith":      {"label": "SCF/MemRef/Arith (10%)",         "color": _PAL[1], "style": "-"},
# }

RUNS_CONFIG = {
    "FTE1":                              {"label": "FT: Exp 1",                      "color": _PAL[2], "style": "-"},
    "FTE4":                              {"label": "FT: Exp 4",                      "color": _PAL[8], "style": "-"},
    "FTE7":                              {"label": "FT: Exp 7",                      "color": _PAL[7], "style": "-"},
    "TestAccuracy_60Epochs_StableHLO":   {"label": "StableHLO (90%)",                "color": _PAL[0], "style": "-"},
    "FTE2":                              {"label": "FT: Exp 2",                      "color": _PAL[4], "style": "-"},
    "FTE5":                              {"label": "FT: Exp 5",                      "color": _PAL[3], "style": "-"},
    "FTE8":                              {"label": "FT: Exp 8",                      "color": _PAL[9], "style": "-"},
    "FTE3":                              {"label": "FT: Exp 3",                      "color": _PAL[6], "style": "-"},
    "FTE6":                              {"label": "FT: Exp 6",                      "color": _PAL[5], "style": "-"},
    "TestAccuracy_100Epochs_Arith":      {"label": "SCF/MemRef/Arith (10%)",         "color": _PAL[1], "style": "-"},
}

# ── Metric selection — comment/uncomment to choose which one to display ──────
# ACTIVE_METRIC = {"key": "val_rmse", "title": "RMSE (lower is better)",          "ylabel": "RMSE"}
ACTIVE_METRIC = {"key": "val_tau",  "title": "Kendall's Tau (higher is better)", "ylabel": "Correlation"}
# ACTIVE_METRIC = {"key": "val_r2",   "title": "R² (higher is better)",            "ylabel": "R²"}


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
                "val_tau":  cp["val_tau"],
                "val_r2":   cp["val_r2"],
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
                "val_tau":  [h["val_tau"]  for h in history],
                "val_r2":   [h["val_r2"]   for h in history],
                "config":   config,
            }
        else:
            logging.warning(f"Skipping {run_name} (no data)")

    fig, ax = plt.subplots(figsize=(10, 7))
    # fig.suptitle(
    #     "Training Progression\nStableHLO Pre-training + Linalg Fine Tuning vs Linalg Training",
    #     fontsize=20,
    #     fontweight="bold",
    # )

    key = ACTIVE_METRIC["key"]

    for run_name, run_data in data.items():
        cfg = run_data["config"]
        ax.plot(
            run_data["epochs"],
            run_data[key],
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["style"],
            linewidth=2.5,
            # marker="o",
            markersize=3,
            alpha=0.85,
        )

    # ax.set_title(ACTIVE_METRIC["title"])
    ax.set_xlabel("Epoch", labelpad=8.0)
    ax.set_ylabel(ACTIVE_METRIC["ylabel"], labelpad=8.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.tick_params(axis="both", labelcolor="black", labelsize=14)

    legend_handles = [
        Line2D([0], [0],
               color=run_data["config"]["color"],
               linestyle=run_data["config"]["style"],
               linewidth=2.5,
               # marker="o",
               markersize=5,
               label=run_data["config"]["label"])
        for run_data in data.values()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.57, 0.15),
        ncol=3,
        # ncol=len(data),
        frameon=True,
        edgecolor="black",
        framealpha=1,
        fontsize=15,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.8, bottom=0.15)
    # plt.subplots_adjust(top=0.92, bottom=0.22)

    save_path = os.path.join(base_checkpoint_dir, f"training_comparison_{key}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    # plt.savefig(save_path)
    logging.info(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()