import os
import math
import sys
import time
import logging
import json
import numpy as np
from tqdm import tqdm
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.amp import GradScaler, autocast

from torch_geometric.data import Dataset as PygDataset
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool

from scipy.stats import kendalltau, spearmanr

from config.params import params

import preprocess


tqdm_stdout = partial(tqdm, file=sys.stdout)


def setup_logging(run_dir):
    log_file = os.path.join(run_dir, 'train.log')

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    logging.info(f"Logging initialized. Writing to {log_file}")


def log_blank_line():
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'stream'):
            handler.stream.write('\n')
            handler.flush()


def format_elapsed_time(seconds: float) -> str:
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


def get_next_run_dir(base_checkpoint_dir):
    base_path = Path(base_checkpoint_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        run_name = f"run_{i}"
        run_path = base_path / run_name
        if not run_path.exists():
            run_path.mkdir()
            return str(run_path)
        i += 1


class ProGraMLPygDataset(PygDataset):
    def __init__(self, split_file_path, processed_dir=None):
        self._custom_processed_dir = processed_dir if processed_dir else params.paths.processed
        self.split_file_path = split_file_path

        self.file_list = self._load_file_list()

        logging.info(f"Initialized Dataset from {split_file_path}")
        logging.info(f"Found {len(self.file_list)} samples.")

        super().__init__(
            root=None,
            transform=None,
            pre_transform=None
        )

    def _load_file_list(self):
        if not os.path.exists(self.split_file_path):
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")

        files = []
        with open(self.split_file_path, 'r') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue

                if not name.endswith('.pt'):
                    name = f"{name}.pt"

                files.append(name)
        files.sort()
        return files

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.processed_dir, filename)

        try:
            data = torch.load(file_path, weights_only=False)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed graph not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

    def process(self):
        pass

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.file_list

    @property
    def processed_dir(self):
        return self._custom_processed_dir


def compute_baselines_and_stats(metrics_path, split_file_path):
    with open(split_file_path, 'r') as f:
        train_models = set(line.strip().replace('.pt', '') for line in f if line.strip())

    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    train_metrics = []
    for name, data in all_metrics.items():
        if name in train_models:
            train_metrics.append(data[params.model.target_performance_metric])

    y = np.array(train_metrics)

    min_val = np.min(y)
    max_val = np.max(y)
    mean = np.mean(y)
    std = np.std(y)
    median = np.median(y)

    log_blank_line()
    logging.info(f"DATASET STATISTICS (Train Split N={len(y)})")
    logging.info(f"Metric:  {params.model.target_performance_metric}")
    logging.info(f"Min:     {min_val:.4f}")
    logging.info(f"Max:     {max_val:.4f}")
    logging.info(f"Mean:    {mean:.4f}")
    logging.info(f"Std Dev: {std:.4f}")
    logging.info(f"Median:  {median:.4f}")

    mean_pred_mae = np.mean(np.abs(y - mean))
    median_pred_mae = np.mean(np.abs(y - median))
    mean_pred_rmse = np.sqrt(np.mean((y - mean) ** 2))

    logging.info(f"BASELINE: Mean Predictor MAE:   {mean_pred_mae:.5f}")
    logging.info(f"BASELINE: Mean Predictor RMSE:  {mean_pred_rmse:.5f} (Target to beat)")
    logging.info(f"BASELINE: Median Predictor MAE: {median_pred_mae:.5f}")

    return mean, std


def get_sinusoidal_positional_embeddings(max_pos, embedding_dim):
    position = torch.arange(max_pos).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pos_emb = torch.zeros(max_pos, embedding_dim)
    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)
    return pos_emb


class ProGraMLGGNNLayer(MessagePassing):
    def __init__(self, hidden_dim, num_edge_types, positional_embedding_dim):
        super(ProGraMLGGNNLayer, self).__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.edge_type_mlps = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)
        ])

        self.position_gating_mlp = nn.Sequential(
            nn.Linear(positional_embedding_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_type, edge_positions_embedded):
        aggregated_messages = self.propagate(edge_index, x=x, edge_type=edge_type,
                                             edge_positions_embedded=edge_positions_embedded)
        updated_x = self.gru_cell(aggregated_messages, x)
        return updated_x

    # noinspection PyMethodOverriding
    def message(self, x_j, edge_type, edge_positions_embedded):
        position_gate = 2 * self.position_gating_mlp(edge_positions_embedded)
        gated_x_j = x_j * position_gate

        messages = torch.zeros_like(gated_x_j)
        for i in range(self.num_edge_types):
            type_mask = (edge_type == i)
            masked_x = gated_x_j[type_mask]
            messages[type_mask] = self.edge_type_mlps[i](masked_x).to(messages.dtype)

        return messages


class ProGraMLNetPyG(nn.Module):
    def __init__(self, vocab_size, device):
        super(ProGraMLNetPyG, self).__init__()
        self.ggnn_iterations = params.model.ggnn_iterations
        self.device = device

        self.node_emb_dim = params.model.node_embedding_dim
        self.hidden_dim = self.node_emb_dim

        self.node_text_embedding = nn.Embedding(vocab_size, self.node_emb_dim, padding_idx=0)

        self.edge_positional_encodings = get_sinusoidal_positional_embeddings(
            params.model.max_edge_position + 1, params.model.positional_embedding_dim
        ).to(device)

        self.ggnn_layer = ProGraMLGGNNLayer(
            hidden_dim=self.hidden_dim,
            num_edge_types=params.model.num_edge_types,
            positional_embedding_dim=params.model.positional_embedding_dim
        )

        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, data):
        x = self.node_text_embedding(data.x)
        edge_pos_embeds = self.edge_positional_encodings[data.edge_positions]

        curr_x = x
        for _ in range(self.ggnn_iterations):
            curr_x = self.ggnn_layer(curr_x, data.edge_index, data.edge_type, edge_pos_embeds)

        graph_embedding = global_add_pool(curr_x, data.batch)

        out = self.readout_mlp(graph_embedding)
        return out.view(-1)


def train_by_epoch(model, train_loader, val_loader, optimizer, scheduler,
                   criterion, device, t_mean, t_std, run_dir, global_start_time):
    epochs = params.training.epochs
    checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    os.makedirs(checkpoint_dir, exist_ok=True)

    scaler = GradScaler('cuda')
    # Track Best Ranking (Kendall Tau) instead of RMSE
    best_val_tau = -1.0

    best_model_path = os.path.join(run_dir, "best_model.pt")

    logging.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # --- TRAIN ---
        model.train()
        total_loss = 0.0
        total_sq_error = 0.0
        total_graphs = 0

        pbar = tqdm_stdout(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Z-Score Normalization: (y - mean) / std
            targets_norm = (batch.y - t_mean) / t_std

            with autocast('cuda'):
                preds = model(batch)
                loss = criterion(preds, targets_norm)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            preds_real = (preds.detach() * t_std) + t_mean
            total_sq_error += (preds_real - batch.y).pow(2).sum().item()

            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            total_graphs += batch_size

            pbar.set_description(f"Epoch {epoch} [Train] Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / total_graphs
        avg_train_rmse = math.sqrt(total_sq_error / total_graphs)

        # --- VALIDATE ---
        val_rmse, val_mae, val_r2, val_nrmse, val_tau, val_spearman = evaluate(model, val_loader, device, t_mean, t_std)
        scheduler.step(val_tau)
        current_lr = optimizer.param_groups[0]['lr']

        # --- LOGGING ---
        log_blank_line()
        logging.info(f"Epoch {epoch}/{epochs} Results:")
        logging.info(f"Train MSE:  {avg_train_loss:.5f}  (Z-Score Loss)")
        logging.info(f"Train RMSE: {avg_train_rmse:.5f}")
        logging.info(f"Current LR: {current_lr:.2e}")
        logging.info(f"Val RMSE:   {val_rmse:.5f}")
        logging.info(f"Val MAE:    {val_mae:.5f}")
        logging.info(f"Val R²:     {val_r2:.4f}   (Explained Var)")
        logging.info(f"Val NRMSE:  {val_nrmse:.4f}   (Error %)")
        logging.info(f"Val Tau:    {val_tau:.4f}   (Ranking Metric)")
        logging.info(f"Val Spear:  {val_spearman:.4f}")

        epoch_end_time = time.time()
        elapsed_epoch_time = epoch_end_time - epoch_start_time
        elapsed_total_time = epoch_end_time - global_start_time
        formatted_epoch_time = format_elapsed_time(elapsed_epoch_time)
        formatted_total_time = format_elapsed_time(elapsed_total_time)
        logging.info(f"Epoch time: {formatted_epoch_time} ({elapsed_epoch_time:.2f} seconds)")
        logging.info(f"Total time: {formatted_total_time} ({elapsed_total_time:.2f} seconds)")

        logging.info(f"GPU Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

        # --- CHECKPOINT ---
        if val_tau > best_val_tau:
            best_val_tau = val_tau
            is_best = True
        else:
            is_best = False

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'grad_scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_mse': avg_train_loss,
            'train_rmse': avg_train_rmse,
            'current_lr': current_lr,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'val_nrmse': val_nrmse,
            'val_tau': val_tau,
            'val_spearman': val_spearman,
            'best_val_tau': best_val_tau,
            'elapsed_epoch_time': elapsed_epoch_time,
            'elapsed_total_time': elapsed_total_time,
            'batch_size': params.training.graph_level_batch_size,
            'config': {
                'node_dim': model.node_emb_dim,
                'hidden_dim': model.hidden_dim,
                'learning_rate': params.training.learning_rate
            }
        }

        epoch_path = os.path.join(run_dir, f"epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_path)

        if is_best:
            torch.save(checkpoint, best_model_path)
            logging.info(f"New Best Model Saved! (Tau: {val_tau:.4f})")


def evaluate(model, loader, device, t_mean, t_std):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm_stdout(loader, desc="Validating", leave=False):
            batch = batch.to(device)

            # Predict Z-Score
            preds_norm = model(batch)

            # De-normalize: pred * std + mean
            preds_real = (preds_norm * t_std) + t_mean
            targets_real = batch.y

            all_preds.append(preds_real.cpu())
            all_targets.append(targets_real.cpu())

    # Concatenate once
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Standard Regression metrics
    mse = functional.mse_loss(preds, targets).item()
    rmse = math.sqrt(mse)
    mae = functional.l1_loss(preds, targets).item()

    # R² Score (Coefficient of Determination)
    # R2 = 1 - (SS_res / SS_tot)
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)

    if ss_tot.item() == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot).item()

    # NRMSE (Normalized by Range)
    target_range = torch.max(targets) - torch.min(targets)
    if target_range.item() > 1e-6:
        nrmse = rmse / target_range.item()
    else:
        nrmse = 0.0

    # Ranking metrics (scipy expects numpy)
    preds_np = preds.numpy()
    targets_np = targets.numpy()

    if np.std(preds_np) == 0 or np.std(targets_np) == 0:
        return rmse, mae, r2, nrmse, 0.0, 0.0

    ktau, _ = kendalltau(targets_np, preds_np)
    spearman, _ = spearmanr(targets_np, preds_np)

    return rmse, mae, r2, nrmse, ktau, spearman


def main():
    start_time = time.time()

    torch.cuda.empty_cache()

    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    run_dir = get_next_run_dir(base_checkpoint_dir)
    setup_logging(run_dir)

    device = torch.device(params.environment.device)
    logging.info(f"Run Directory: {run_dir}")
    logging.info(f"Using device: {params.environment.device}")

    log_blank_line()
    preprocess.run_preprocessing()
    log_blank_line()

    train_dataset = ProGraMLPygDataset(
        split_file_path=params.paths.train_graphs_txt,
        processed_dir=params.paths.processed
    )

    val_dataset = ProGraMLPygDataset(
        split_file_path=params.paths.validation_graphs_txt,
        processed_dir=params.paths.processed
    )

    train_loader = PygDataLoader(
        train_dataset,
        batch_size=params.training.graph_level_batch_size,
        shuffle=True,
        num_workers=params.environment.num_workers,
        pin_memory=True
    )

    val_loader = PygDataLoader(
        val_dataset,
        batch_size=params.training.graph_level_batch_size,
        shuffle=False,
        num_workers=params.environment.num_workers,
        pin_memory=True
    )

    t_mean, t_std = compute_baselines_and_stats(params.paths.labels, params.paths.train_graphs_txt)
    t_mean_dev = torch.tensor(t_mean, device=device, dtype=torch.float)
    t_std_dev = torch.tensor(t_std, device=device, dtype=torch.float)

    model = ProGraMLNetPyG(vocab_size=params.model.expected_vocab_size, device=device).to(device)

    log_blank_line()
    logging.info(f"Model Initialized. Vocab: {params.model.expected_vocab_size}, Params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=params.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    criterion = nn.MSELoss()

    train_by_epoch(model, train_loader, val_loader, optimizer, scheduler,
                   criterion, device, t_mean_dev, t_std_dev, run_dir, start_time)

    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = format_elapsed_time(elapsed_time)
    log_blank_line()
    logging.info(f"Total Execution time: {formatted_time} ({elapsed_time:.2f} seconds)")


if __name__ == '__main__':
    main()