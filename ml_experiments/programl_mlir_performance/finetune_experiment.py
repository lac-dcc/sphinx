import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
import time

from GGNN_train import (
    ProGraMLNetPyG,
    ProGraMLPygDataset,
    train_by_epoch,
    setup_logging,
    get_next_run_dir,
    compute_baselines_and_stats,
    log_blank_line,
    PygDataLoader,
    format_elapsed_time
)
from config.params import params

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

# Path to the BEST checkpoint from best (StableHLO) training run
# PRETRAINED_CHECKPOINT_PATH = os.path.join(os.path.dirname(params.paths.checkpoint), "run_1/best_model.pt")
PRETRAINED_CHECKPOINT_PATH = os.path.join(os.path.dirname(params.paths.checkpoint), "TrainingTime_20Epochs_StableHLO/best_model.pt")

# SELECT EXPERIMENT:
EXPERIMENT_NAMES = {
    1: "Naive Transfer (Load Model + Random Embeddings + Train All)",
    2: "Frozen GNN (Load Model + Random Embeddings + Freeze All GNN + Train MLP/Embeddings)",
    3: "Partial Freeze GNN 1 (Load Model + Random Embeddings + Freeze Edge and Position MLPs + Train GRU/MLP/Embeddings)",
    4: "Partial Freeze GNN 2 (Load Model + Random Embeddings + Freeze Edge MLPs Only + Train GRU/Position MLP/MLP/Embeddings)",
    5: "Smart Init (Load Model + Mean-Centered Embeddings + Train All)",
    6: "Frozen Smart Init (Load Model + Mean-Centered Embeddings + Freeze All GNN + Train MLP/Embeddings)",
    7: "Partial Freeze Smart Init 1 (Load Model + Mean-Centered Embeddings + Freeze Edge and Position MLPs + Train GRU/MLP/Embeddings)",
    8: "Partial Freeze Smart Init 2 (Load Model + Mean-Centered Embeddings + Freeze Edge MLPs Only + Train GRU/Position MLP/MLP/Embeddings)",
}

EXPERIMENT_ID = 1

# ==============================================================================

def load_pretrained_surgery(model, checkpoint_path):
    """
    Loads weights from the old model but handles the embedding size mismatch
    and performs initialization surgery based on the experiment type.
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    log_blank_line()
    logging.info(f"Loading pretrained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # 1. Isolate and remove the old embedding weights
    # We cannot load them directly because the shape doesn't match the new vocab
    emb_key = 'node_text_embedding.weight'
    old_emb_weights = state_dict[emb_key]
    del state_dict[emb_key]

    # 2. Load the GNN backbone and Readout MLP
    model.load_state_dict(state_dict, strict=False)
    logging.info("Backbone (GNN + MLP) weights loaded successfully.")

    # 3. Handle Embedding Initialization based on Experiment
    new_emb_layer = model.node_text_embedding.weight.data

    if EXPERIMENT_ID in [1, 2, 3, 4]:
        # Naive / Frozen: Leave as Random Initialization (Default PyTorch)
        logging.info("Embeddings initialized RANDOMLY (Standard Transfer).")

    elif EXPERIMENT_ID in [5, 6, 7, 8]:
        # Smart Init: Initialize new embeddings to the MEAN of the old ones
        mean_vector = torch.mean(old_emb_weights, dim=0)

        # Assign this mean vector to every token in the new vocabulary
        new_emb_layer[:] = mean_vector
        logging.info("Embeddings initialized via SURGERY (Mean-Centered).")

    return model


def freeze_gnn_layers(model):
    """
    Freezes the Message Passing layers (The 'Brain') so only the
    Vocabulary (Embeddings) and Translation (Readout) are learned.
    """
    logging.info("FREEZING GNN Backbone...")

    for param in model.ggnn_layer.edge_type_mlps.parameters():
        param.requires_grad = False

    if EXPERIMENT_ID in [2, 3, 6, 7]:
        for param in model.ggnn_layer.position_gating_mlp.parameters():
            param.requires_grad = False

    if EXPERIMENT_ID in [2, 6]:
        for param in model.ggnn_layer.gru_cell.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"GNN Frozen. Trainable Params: {trainable:,} / {total:,} ({(trainable / total):.1%})")


def main():
    start_time = time.time()

    # 1. Setup
    torch.cuda.empty_cache()
    base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    run_dir = get_next_run_dir(base_checkpoint_dir)
    setup_logging(run_dir)

    device = torch.device(params.environment.device)
    logging.info(f"Starting Fine-Tuning | EXPERIMENT ID: {EXPERIMENT_ID} - {EXPERIMENT_NAMES[EXPERIMENT_ID]}")
    logging.info(f"Run Directory: {run_dir}")

    # 2. Data Preparation (Load NEW Linalg Dataset)
    # Ensure params.paths points to the NEW dataset files
    log_blank_line()
    logging.info("Loading New Dataset...")

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
        num_workers=params.environment.num_workers
    )
    val_loader = PygDataLoader(
        val_dataset,
        batch_size=params.training.graph_level_batch_size,
        shuffle=False,
        num_workers=params.environment.num_workers
    )

    # Compute stats for the NEW dataset
    t_mean, t_std = compute_baselines_and_stats(params.paths.labels, params.paths.train_graphs_txt)
    t_mean_dev = torch.tensor(t_mean, device=device, dtype=torch.float)
    t_std_dev = torch.tensor(t_std, device=device, dtype=torch.float)

    # 3. Model Initialization
    # params.model.expected_vocab_size must match the NEW vocabulary size
    model = ProGraMLNetPyG(vocab_size=params.model.expected_vocab_size, device=device).to(device)

    # 4. PERFORM SURGERY (Load Pretrained Weights)
    model = load_pretrained_surgery(model, PRETRAINED_CHECKPOINT_PATH)

    # 5. Apply Experiment-Specific Logic
    learning_rate = params.training.learning_rate

    if EXPERIMENT_ID not in [1, 5]:
        freeze_gnn_layers(model)

    # 6. Optimizer Setup
    # Filter ensures we don't try to optimize frozen parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    criterion = nn.MSELoss()

    # 7. Start Training
    log_blank_line()
    logging.info("Starting Fine-Tuning Loop...")
    train_by_epoch(model, train_loader, val_loader, optimizer, criterion, device, t_mean_dev, t_std_dev, run_dir, start_time)

    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = format_elapsed_time(elapsed_time)
    log_blank_line()
    logging.info(f"Total Execution time: {formatted_time} ({elapsed_time:.2f} seconds)")
    log_blank_line()
    log_blank_line()

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # main()

    for exp_id in range(1, 9):
        EXPERIMENT_ID = exp_id
        main()