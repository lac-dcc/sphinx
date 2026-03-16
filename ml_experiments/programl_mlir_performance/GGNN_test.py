import os
import sys
import math
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from functools import partial
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
import torch.nn.functional as functional
from torch_geometric.loader import DataLoader as PygDataLoader
import glob

import GGNN_train
from config.params import params

tqdm_stdout = partial(tqdm, file=sys.stdout)

target_metric_name = params.model.target_performance_metric.replace("_", " ").title()

def plot_comprehensive_analysis(preds, targets, run_dir, filename="analysis_panel.png"):
    """
    Generates a 3-panel figure with Scatter, Residuals, and Top-K analysis.
    """
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        "Test Set Analysis",
        fontsize=22,
        fontweight="bold"
    )

    # 1. SCATTER PLOT
    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.2, s=3, color='blue', label='Models')
    low = min(np.min(targets), np.min(preds))
    high = max(np.max(targets), np.max(preds))
    ax.plot([low, high], [low, high], 'r--', linewidth=2, label='Perfect Prediction')

    spearman_corr, _ = spearmanr(targets, preds)
    ax.set_title(f"Predicted vs Actual {target_metric_name}", fontsize=14)
    ax.set_xlabel(f"Ground Truth {target_metric_name}")
    ax.set_ylabel(f"Predicted {target_metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. RESIDUAL PLOT
    ax = axes[1]
    residuals = preds - targets
    ax.scatter(targets, residuals, alpha=0.2, s=3, color='green')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Residual Analysis\n(Prediction - Truth)', fontsize=14)
    ax.set_xlabel(f'Ground Truth {target_metric_name}')
    ax.set_ylabel('Error')
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, alpha=0.3)

    # 3. TOP-K EFFICIENCY
    ax = axes[2]
    sorted_indices = np.argsort(preds)
    sorted_targets = targets[sorted_indices]

    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
    k_values = [k for k in k_values if k <= len(targets)]
    best_found = [np.min(sorted_targets[:k]) for k in k_values]

    random_curve = []
    for k in k_values:
        trials = [np.min(np.random.choice(targets, k, replace=False)) for _ in range(50)]
        random_curve.append(np.mean(trials))

    ax.plot(k_values, best_found, 'b-o', linewidth=2, label='Our Predictor')
    ax.plot(k_values, random_curve, 'k--', linewidth=2, label='Random Search')
    ax.axhline(np.min(targets), color='green', linestyle=':', label='Global Optimum')
    ax.set_xscale('log')
    ax.set_title('Search Efficiency (Top-K)', fontsize=14)
    ax.set_xlabel('Number of Candidates Selected (K)')
    ax.set_ylabel(f'Best True {target_metric_name} Found')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(run_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    logging.info(f"Comprehensive analysis saved to {save_path}")


def evaluate_with_predictions(model, loader, device, t_mean, t_std):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm_stdout(loader, desc="Testing", leave=False):
            batch = batch.to(device)
            preds_norm = model(batch)
            preds_real = (preds_norm * t_std) + t_mean
            targets_real = batch.y
            all_preds.append(preds_real.cpu())
            all_targets.append(targets_real.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = functional.mse_loss(preds, targets).item()
    rmse = math.sqrt(mse)
    mae = functional.l1_loss(preds, targets).item()

    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item() if ss_tot.item() != 0 else 0.0

    target_range = torch.max(targets) - torch.min(targets)
    nrmse = (rmse / target_range.item()) if target_range.item() > 1e-6 else 0.0

    preds_np = preds.numpy()
    targets_np = targets.numpy()
    ktau, _ = kendalltau(targets_np, preds_np)
    spearman, _ = spearmanr(targets_np, preds_np)

    return rmse, mae, r2, nrmse, ktau, spearman, preds_np, targets_np


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    run_folder_name = sys.argv[1] if len(sys.argv) > 1 else "TrainingTime_20Epochs_Linalg"

    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    run_dir = os.path.join(base_checkpoint_dir, run_folder_name)
    model_path = os.path.join(run_dir, "best_model.pt")

    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}")
        return

    device = torch.device(params.environment.device)
    logging.info(f"Testing Model: {model_path}")
    logging.info(f"Device: {device}")

    logging.info("Computing normalization stats from Training Set...")
    t_mean, t_std = GGNN_train.compute_baselines_and_stats(
        params.paths.labels,
        params.paths.train_graphs_txt
    )

    t_mean_dev = torch.tensor(t_mean, device=device, dtype=torch.float)
    t_std_dev = torch.tensor(t_std, device=device, dtype=torch.float)

    test_split_path = params.paths.test_graphs_txt
    if not os.path.exists(test_split_path):
        logging.error(f"Test split file not found at: {test_split_path}")
        return

    GGNN_train.log_blank_line()
    logging.info(f"Loading Test Data from: {test_split_path}")
    test_dataset = GGNN_train.ProGraMLPygDataset(
        split_file_path=test_split_path,
        processed_dir=params.paths.processed
    )

    test_loader = PygDataLoader(
        test_dataset,
        batch_size=params.training.graph_level_batch_size,
        shuffle=False,
        num_workers=params.environment.num_workers
    )

    vocab_size = params.model.expected_vocab_size
    model = GGNN_train.ProGraMLNetPyG(vocab_size=vocab_size, device=device)

    logging.info("Loading weights...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logging.info(f"Model Loaded. (Training Best Tau: {checkpoint.get('val_tau', 'N/A'):.4f})")

    logging.info("Running Evaluation on Test Set...")
    test_rmse, test_mae, test_r2, test_nrmse, test_tau, test_spearman, preds, targets = evaluate_with_predictions(
        model, test_loader, device, t_mean_dev, t_std_dev
    )

    GGNN_train.log_blank_line()
    logging.info(f"TRAINING BEST RESULTS (Validation)")
    if 'val_rmse' in checkpoint:
        logging.info(f"RMSE:           {checkpoint['val_rmse']:.5f}")
        logging.info(f"MAE:            {checkpoint['val_mae']:.5f}")
        logging.info(f"R²:             {checkpoint.get('val_r2', 0):.5f}")
        logging.info(f"NRMSE:          {checkpoint['val_nrmse']:.5f}")
        logging.info(f"Kendall's Tau:  {checkpoint['val_tau']:.5f}")
        logging.info(f"Spearman's Rho: {checkpoint['val_spearman']:.5f}")
    else:
        logging.info("Metrics not found in checkpoint dict.")

    GGNN_train.log_blank_line()
    logging.info(f"FINAL TEST SET RESULTS")
    logging.info(f"RMSE:           {test_rmse:.5f}")
    logging.info(f"MAE:            {test_mae:.5f}")
    logging.info(f"R²:             {test_r2:.5f}")
    logging.info(f"NRMSE:          {test_nrmse:.5f}")
    logging.info(f"Kendall's Tau:  {test_tau:.5f}")
    logging.info(f"Spearman's Rho: {test_spearman:.5f}")

    # 3. Plot Test Set Analysis
    plot_comprehensive_analysis(preds, targets, run_dir)


if __name__ == "__main__":
    main()