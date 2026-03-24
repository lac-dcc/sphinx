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

import preprocess

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

SPLITS_DIR = params.paths.splits_txt + "/incremental_linalg_experiment"

# Percentages and runs to loop over
PERCENTAGES = [1, 5, 10, 20, 30, 40, 50, 60]
NUM_RUNS = 3


# ==============================================================================

def run_baseline_training(train_split_path, val_split_path, percentage, run_num):
    """
    Trains a model FROM SCRATCH using the specified data splits.
    Monkey-patches the config to load the correct .txt files.
    """
    start_time = time.time()
    torch.cuda.empty_cache()

    # --- MONKEY PATCH PARAMS ---
    original_train_txt = params.paths.train_graphs_txt
    original_val_txt = params.paths.validation_graphs_txt

    params.paths.train_graphs_txt = train_split_path
    params.paths.validation_graphs_txt = val_split_path

    try:
        # Setup Logging & Directory
        base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
        run_dir = get_next_run_dir(base_checkpoint_dir)
        setup_logging(run_dir)

        device = torch.device(params.environment.device)
        log_blank_line()
        logging.info(f"BASELINE EXPERIMENT | TRAIN FROM SCRATCH")
        logging.info(f"DATA: {percentage}% | RUN: {run_num}/{NUM_RUNS}")
        logging.info(f"Train Split: {train_split_path}")
        logging.info(f"Run Directory: {run_dir}")
        log_blank_line()
        preprocess.run_preprocessing()
        log_blank_line()

        # Load Data
        logging.info("Initializing Datasets...")
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

        # Compute stats (Mean/Std needed for Z-score loss)
        t_mean, t_std = compute_baselines_and_stats(params.paths.labels, params.paths.train_graphs_txt)
        t_mean_dev = torch.tensor(t_mean, device=device, dtype=torch.float)
        t_std_dev = torch.tensor(t_std, device=device, dtype=torch.float)

        # ------------------------------------------------------------------
        # INITIALIZE PURE, BLANK-SLATE MODEL
        # No pretrained weights are loaded here. Everything is random init.
        # ------------------------------------------------------------------
        logging.info("Initializing fresh ProGraMLNetPyG model from scratch...")
        model = ProGraMLNetPyG(vocab_size=params.model.expected_vocab_size, device=device).to(device)

        # Setup Optimizer (All parameters are trainable)
        learning_rate = params.training.learning_rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        # Train
        logging.info(f"Starting Training Loop for {percentage}% data...")
        train_by_epoch(model, train_loader, val_loader, optimizer, scheduler, criterion, device, t_mean_dev, t_std_dev,
                       run_dir, start_time)

        elapsed_time = time.time() - start_time
        logging.info(f"Run {run_num} for {percentage}% finished in {format_elapsed_time(elapsed_time)}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Training failed for {percentage}% Run {run_num}: {e}")
        raise e
    finally:
        # --- RESTORE PARAMS ---
        # Ensures the next loop iteration starts with a clean slate
        params.paths.train_graphs_txt = original_train_txt
        params.paths.validation_graphs_txt = original_val_txt


if __name__ == '__main__':
    fixed_val_path = os.path.join(SPLITS_DIR, "fixed_validation_linalg.txt")

    if not os.path.exists(fixed_val_path):
        print(f"ERROR: Fixed validation file not found at {fixed_val_path}")
        sys.exit(1)

    print(f"Starting Baseline Incremental Pipeline. Total runs: {len(PERCENTAGES) * NUM_RUNS}")

    for p in PERCENTAGES:
        for run in range(1, NUM_RUNS + 1):
            train_filename = f"train_{p}percent_run{run}.txt"
            train_split_path = os.path.join(SPLITS_DIR, train_filename)

            if not os.path.exists(train_split_path):
                print(f"WARNING: Split file missing: {train_split_path}. Skipping.")
                continue

            run_baseline_training(train_split_path, fixed_val_path, p, run)

    print("All baseline incremental training runs completed successfully!")