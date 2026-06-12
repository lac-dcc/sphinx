import torch
import os
import glob
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from config.params import params

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("fontTools").setLevel(logging.WARNING)

plt.rcParams["pdf.fonttype"] = 42  # TrueType — fully embedded
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

_PAL = sns.color_palette(OKABE_ITO)

# ── Configuration ────────────────────────────────────────────────────────────
# The percentages you ran experiments for
PERCENTAGES = [1, 5, 10, 20, 30, 40, 50, 60]

CONFIG = {
    "StableHLO": {"prefix": "HLO_", "color": _PAL[0], "style": "-"},
    "Linalg": {"prefix": "Linalg_", "color": _PAL[1], "style": "-"},
}


# ── Data loading ─────────────────────────────────────────────────────────────
def load_epoch_time(run_dir):
    """Reads the epoch_*.pt file and returns the elapsed epoch time."""
    epoch_files = glob.glob(os.path.join(run_dir, "epoch_*.pt"))
    if not epoch_files:
        logging.warning(f"No checkpoints found in {run_dir}")
        return None

    # We just need the first epoch file
    target_file = epoch_files[0]
    try:
        cp = torch.load(target_file, map_location="cpu", weights_only=False)
        # Try to get epoch time, fallback to total time if necessary
        return cp.get("elapsed_epoch_time", cp.get("elapsed_total_time"))
    except (RuntimeError, KeyError, FileNotFoundError) as e:
        print(f"Skipping {target_file}: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    logging.info(f"Looking for runs in: {base_checkpoint_dir}")

    # Extract Data
    plot_data = {}
    for name, config in CONFIG.items():
        x_vals = []
        y_vals = []

        for p in PERCENTAGES:
            folder_name = f"{config['prefix']}{p}"
            run_path = os.path.join(base_checkpoint_dir, folder_name)

            epoch_time = load_epoch_time(run_path)
            if epoch_time is not None:
                x_vals.append(p)
                y_vals.append(epoch_time)

        plot_data[name] = {
            "x": x_vals,
            "y": y_vals,
            "color": config["color"],
            "style": config["style"]
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, data in plot_data.items():
        if not data["x"]:
            logging.warning(f"No data to plot for {name}")
            continue

        ax.plot(
            data["x"],
            data["y"],
            label=name,
            color=data["color"],
            linestyle=data["style"],
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.85,
        )

    # Axis Formatting
    ax.set_xlabel("Percentage of Corpus (%)", labelpad=10.0)
    ax.set_ylabel("Time per Epoch (seconds)", labelpad=10.0)

    # Ensure X-axis shows the percentages nicely
    ax.xaxis.set_major_locator(mticker.FixedLocator(PERCENTAGES))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

    ax.tick_params(axis="both", labelcolor="black", labelsize=14)

    # Custom Legend
    legend_handles = [
        Line2D([0], [0],
               color=data["color"],
               linestyle=data["style"],
               linewidth=2.5,
               marker="o",
               markersize=6,
               label=name)
        for name, data in plot_data.items() if data["x"]
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.2),
        ncol=2,
        frameon=True,
        edgecolor="black",
        framealpha=1,
        fontsize=16,
    )

    plt.tight_layout()
    # Adjust bottom to make room for the legend
    plt.subplots_adjust(bottom=0.20)

    # Save
    save_path = os.path.join(base_checkpoint_dir, "epoch_time_scaling_comparison.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    logging.info(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()