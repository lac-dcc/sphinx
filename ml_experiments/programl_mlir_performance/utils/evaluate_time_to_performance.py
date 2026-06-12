import torch
import os
import glob
import logging
from config.params import params

# --- CONFIGURATION ---
RUNS_CONFIG = {
    "StableHLO": "TrainingTime_40Epochs_StableHLO",
    "Linalg": "50%",
    "FTE6": "FT6",
    "FTE7": "FT7",
    "FTE8": "FT8"
}

# RUNS_CONFIG = {
#     "StableHLO": "TestAccuracy_60Epochs_StableHLO",
#     "Linalg": "30%",
#     "FTE6": "FT6",
#     "FTE7": "FT7",
#     "FTE8": "FT8"
# }

THRESHOLDS = [0.90, 0.95, 0.99]


# ---------------------

def format_time(seconds):
    """Converts seconds to a readable 'Xm Ys' format."""
    if seconds is None:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


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
                'val_tau': cp['val_tau'],
                'elapsed_total_time': cp['elapsed_total_time']
            })
        except (RuntimeError, KeyError, FileNotFoundError) as e:
            print(f"Skipping {f}: {e}")

    # Sort sequentially by epoch
    history.sort(key=lambda x: x['epoch'])
    return history


def get_threshold_metrics(history, target_tau):
    """
    Finds the first epoch that meets/exceeds the target tau.
    Returns a dictionary with both the time and the actual tau achieved.
    """
    for h in history:
        if h['val_tau'] >= target_tau:
            return {
                'time': h['elapsed_total_time'],
                'achieved_tau': h['val_tau']
            }
    return None


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Resolve base checkpoint directory from params
    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    if not os.path.exists(base_checkpoint_dir):
        logging.error(f"Base checkpoint directory not found: {base_checkpoint_dir}")
        return

    logging.info(f"Analyzing checkpoints in: {base_checkpoint_dir}")

    # Extract all data
    data = {}
    for run_name, folder_name in RUNS_CONFIG.items():
        run_path = os.path.join(base_checkpoint_dir, folder_name)
        history = load_run_history(run_path)
        if history:
            peak_tau = max(h['val_tau'] for h in history)
            final_time = history[-1]['elapsed_total_time']

            data[run_name] = {
                "history": history,
                "peak_tau": peak_tau,
                "final_time": final_time
            }
        else:
            data[run_name] = None

    # Print the Markdown Table Header
    print("\n" + "=" * 120)
    print("### Time-to-Performance Table (With Achieved $\\tau$)\n")
    print(
        "| Method | 90% Threshold (Pre + FT = Total) | 95% Threshold (Pre + FT = Total) | 99% Threshold (Pre + FT = Total) | Absolute Peak $\\tau$ | End-to-End Time |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    # Process Linalg (Baseline) - No Pre-training
    if data.get("Linalg"):
        res = data["Linalg"]
        row_str = f"| **Linalg (Scratch)** | "

        for pct in THRESHOLDS:
            target_tau = res["peak_tau"] * pct
            metrics = get_threshold_metrics(res["history"], target_tau)

            if metrics is not None:
                t_str = format_time(metrics['time'])
                tau_str = f"$\\tau$={metrics['achieved_tau']:.3f}"
                row_str += f"0m + {t_str} = **{t_str}** ({tau_str}) | "
            else:
                row_str += "N/A | "

        row_str += f"**{res['peak_tau']:.3f}** | {format_time(res['final_time'])} |"
        print(row_str)

    # Process FTE Configurations
    stable_res = data.get("StableHLO")
    if not stable_res:
        logging.error("StableHLO data missing! Cannot calculate transfer times.")
        return

    for fte_name in ["FTE6", "FTE7", "FTE8"]:
        if data.get(fte_name):
            fte_res = data[fte_name]
            row_str = f"| **{fte_name} (Transfer)** | "

            for pct in THRESHOLDS:
                target_tau_stable = stable_res["peak_tau"] * pct
                target_tau_fte = fte_res["peak_tau"] * pct

                metrics_pre = get_threshold_metrics(stable_res["history"], target_tau_stable)
                metrics_ft = get_threshold_metrics(fte_res["history"], target_tau_fte)

                if metrics_pre is not None and metrics_ft is not None:
                    total_time = metrics_pre['time'] + metrics_ft['time']
                    t_pre_str = format_time(metrics_pre['time'])
                    t_ft_str = format_time(metrics_ft['time'])
                    total_str = format_time(total_time)
                    tau_str = f"$\\tau$={metrics_ft['achieved_tau']:.3f}"

                    row_str += f"{t_pre_str} + {t_ft_str} = **{total_str}** ({tau_str}) | "
                else:
                    row_str += "N/A | "

            row_str += f"**{fte_res['peak_tau']:.3f}** | {format_time(fte_res['final_time'])} |"
            print(row_str)

    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()