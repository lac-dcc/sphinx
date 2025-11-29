import torch
import os
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm


def create_training_plot_seaborn(checkpoints_dir: str, output_image_path: str = "training_plot.png"):
    """
    Scans a directory of PyTorch checkpoints, extracts training metrics,
    and generates a dual-axis plot using Seaborn/Matplotlib.

    Args:
        checkpoints_dir (str): The path to the directory containing .pt files.
        output_image_path (str): The path to save the output PNG image.
    """
    if not os.path.isdir(checkpoints_dir):
        print(f"Error: Directory not found at '{checkpoints_dir}'")
        return

    metrics_data = []
    best_model_metrics = None

    print(f"Scanning directory: {checkpoints_dir}")
    for filename in tqdm(os.listdir(checkpoints_dir), desc="Reading checkpoints"):
        if not filename.endswith('.pt'):
            continue

        file_path = os.path.join(checkpoints_dir, filename)

        try:
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint and 'f1_score' in checkpoint:
                metric_point = {
                    'epoch': checkpoint.get('epoch', 0),
                    'f1': checkpoint.get('f1_score', 0.0),
                    'loss': checkpoint.get('loss', 0.0)
                }
                if filename == 'best_model.pt':
                    best_model_metrics = metric_point
                elif filename.startswith('liveness_'):
                    metrics_data.append(metric_point)
        except Exception as e:
            print(f"Warning: Could not read or parse file '{filename}'. Error: {e}")

    if not metrics_data:
        print("No valid periodic checkpoints found to plot.")
        return

    # Convert to a Pandas DataFrame for easy sorting and manipulation
    df = pd.DataFrame(metrics_data).sort_values(by='epoch').reset_index(drop=True)

    # --- Calculate Summary Metrics ---
    df_1m = df[df['epoch'] <= 1000]
    best_f1_1m = df_1m['f1'].max() if not df_1m.empty else "N/A"

    formatted_f1 = f"{best_f1_1m:.4f}" if isinstance(best_f1_1m, float) else "N/A"
    print(f"Best F1 Score: {formatted_f1}")

    # --- Create the Plot using Seaborn and Matplotlib ---
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # --- Plot F1 Score on the primary Y-axis (ax1) ---
    sns.lineplot(data=df, x='epoch', y='f1', ax=ax1, color='royalblue', label='Validation F1 Score', linewidth=3)
    ax1.set_xlabel("Number of Epochs", fontsize=20, labelpad=15)
    ax1.set_ylabel("Validation F1 Score", color='royalblue', fontsize=20, labelpad=15)
    ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.set_yticks([x * 0.05 for x in range(0, 21)])
    ax1.set_ylim(0, 1.0)
    # ax1.set_xscale('log')
    # ax1.set_xticks([10000, 100000, 1000000, 2000000])
    ax1.xaxis.set_major_formatter(mticker.EngFormatter())  # Format x-axis ticks (e.g., 500k)

    # --- Create a secondary Y-axis for Loss ---
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='epoch', y='loss', ax=ax2, color='darkorange', label='Validation Loss', linewidth=3)
    ax2.set_ylabel("Validation Loss", color='darkorange', fontsize=20, labelpad=15)
    ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=18)
    ax2.set_ylim(0, 0.2)
    ax2.set_yticks(numpy.arange(0, 0.21, 0.01))
    ax2.grid(False)  # Turn off the grid for the second axis to keep it clean

    # --- Plot the best model checkpoint ---
    if best_model_metrics:
        ax1.plot(best_model_metrics['epoch'], best_model_metrics['f1'],
                 marker='o', markersize=10, color='gold',
                 markeredgecolor='black', label=f"Best F1 Checkpoint ({best_model_metrics['f1']:.4f})")

    # --- Final Touches ---
    fig.suptitle("MLIR To ProGraML Liveness Training Progress", fontsize=28, fontweight='bold')

    # Create a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.get_legend().remove()  # Remove the individual legend from the second axis
    ax1.legend(lines + lines2, labels + labels2, loc='center right', fontsize=14,
               bbox_to_anchor=(0.98, 0.35), ncol=1, fancybox=True, shadow=True)

    # Add summary text to the plot
    best_f1_1m_str = f"{best_f1_1m:.4f}" if isinstance(best_f1_1m, float) else "N/A"
    summary_text = f"Best F1: {best_f1_1m:.4f}"

    plt.figtext(0.97, 0.085, summary_text, ha="right", va="top", fontsize=16,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "aliceblue", "edgecolor": "black", "linewidth": 1})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make space for title and legend

    # --- Save and Show the Plot ---
    output_image_path = os.path.join(checkpoints_dir, output_image_path)
    print(f"\nSaving plot to '{output_image_path}'...")
    plt.savefig(output_image_path, dpi=300)

    print("Displaying plot...")
    plt.show()


if __name__ == '__main__':
    CHECKPOINTS_DIR = "/home/douglasvc/Desktop/LivenessTest/checkpoints"

    create_training_plot_seaborn(CHECKPOINTS_DIR)