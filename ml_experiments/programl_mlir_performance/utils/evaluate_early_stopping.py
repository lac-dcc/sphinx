import torch
import os
import glob
import logging
from config.params import params

# --- CONFIGURATION ---
# RUNS_CONFIG = {
#     "StableHLO": "TrainingTime_40Epochs_StableHLO",
#     "Linalg": "50%",
#     "FTE6": "FT6",
#     "FTE7": "FT7",
#     "FTE8": "FT8"
# }

RUNS_CONFIG = {
    "StableHLO": "TestAccuracy_60Epochs_StableHLO",
    "Linalg": "30%",
    "FTE6": "FT6",
    "FTE7": "FT7",
    "FTE8": "FT8"
}

# Early Stopping Parameters
PATIENCE = 5
MIN_DELTA = 0.005


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
            # Load lightly to save memory
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


def simulate_early_stopping(history, patience=5, min_delta=0.005):
    """
    Simulates early stopping. Returns the time spent waiting to trigger the stop,
    and the best tau achieved before the plateau.
    """
    best_tau = -float('inf')
    wait = 0
    best_epoch = 0

    for h in history:
        current_tau = h['val_tau']

        if current_tau >= best_tau + min_delta:
            best_tau = current_tau
            wait = 0
            best_epoch = h['epoch']
        else:
            wait += 1

        if wait >= patience:
            return {
                'stopped_epoch': h['epoch'],
                'best_epoch': best_epoch,
                'final_tau': best_tau,
                'time_spent': h['elapsed_total_time']
            }

    # If it never stops early (runs out of epochs), return the final state
    return {
        'stopped_epoch': history[-1]['epoch'],
        'best_epoch': best_epoch,
        'final_tau': max(h['val_tau'] for h in history),
        'time_spent': history[-1]['elapsed_total_time']
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Resolve base checkpoint directory from params
    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    if not os.path.exists(base_checkpoint_dir):
        logging.error(f"Base checkpoint directory not found: {base_checkpoint_dir}")
        return

    logging.info(f"Applying Early Stopping (Patience: {PATIENCE}, Min Delta: {MIN_DELTA})")

    # Extract data and simulate early stopping for all runs
    data = {}
    for run_name, folder_name in RUNS_CONFIG.items():
        run_path = os.path.join(base_checkpoint_dir, folder_name)
        history = load_run_history(run_path)

        if history:
            data[run_name] = simulate_early_stopping(history, patience=PATIENCE, min_delta=MIN_DELTA)
        else:
            data[run_name] = None

    # Print the Markdown Table Header
    print("\n" + "=" * 90)
    print("### Early Stopping Analysis (Real-World Deployment)")
    print(f"*Parameters: Patience = {PATIENCE} epochs, Min Delta = {MIN_DELTA} $\\tau$*\n")
    print("| Method | Stopped @ Epoch | Final $\\tau$ | Active Compute Time (Pre + FT = Total) |")
    print("| :--- | :--- | :--- | :--- |")

    # 1. Process Linalg (Baseline) - No Pre-training
    if data.get("Linalg"):
        res = data["Linalg"]
        time_str = f"0m + {format_time(res['time_spent'])} = **{format_time(res['time_spent'])}**"
        print(f"| **Linalg (Scratch)** | {res['stopped_epoch']} | {res['final_tau']:.3f} | {time_str} |")

    # 2. Process FTE Configurations
    stable_res = data.get("StableHLO")
    if not stable_res:
        logging.error("StableHLO data missing! Cannot calculate transfer times.")
        return

    for fte_name in ["FTE6", "FTE7", "FTE8"]:
        if data.get(fte_name):
            fte_res = data[fte_name]

            # Combine the early stopping times
            t_pre = stable_res['time_spent']
            t_ft = fte_res['time_spent']
            total = t_pre + t_ft

            time_str = f"{format_time(t_pre)} + {format_time(t_ft)} = **{format_time(total)}**"
            print(
                f"| **{fte_name} (Transfer)** | {fte_res['stopped_epoch']} | {fte_res['final_tau']:.3f} | {time_str} |")

    print("=" * 90 + "\n")

    # Bonus: Print the StableHLO one-time cost context
    if stable_res:
        print(
            f"*(Note: StableHLO Pre-training triggered early stopping at Epoch {stable_res['stopped_epoch']}. This {format_time(stable_res['time_spent'])} one-time cost is amortized across the FT totals above.)*")


if __name__ == "__main__":
    main()