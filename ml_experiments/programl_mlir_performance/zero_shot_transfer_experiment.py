import os
import sys
import torch
import logging
from torch_geometric.loader import DataLoader as PygDataLoader

# Import existing modules
import GGNN_train
import GGNN_test
import preprocess
from config.params import params

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# New folders for this experiment
TRANSFER_GRAPHS_DIR = os.path.join(os.path.dirname(params.paths.graphs), "graphs_linalg")
TRANSFER_PROCESSED_DIR = os.path.join(os.path.dirname(params.paths.processed), "processed_training_time_linalg")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def run_preprocessing_for_transfer():
    """
    Monkey-patches the global params to point to the new folders,
    then runs the existing preprocessing logic.
    """
    logging.info("Step 2: Preprocessing Linalg Graphs into PyG Data...")

    # --- MONKEY PATCH PARAMS ---
    # We temporarily trick the system into thinking the 'graphs' folder
    # is our new 'graphs_linalg' folder.
    original_graphs_path = params.paths.graphs
    original_processed_path = params.paths.processed

    params.paths.graphs = TRANSFER_GRAPHS_DIR
    params.paths.processed = TRANSFER_PROCESSED_DIR

    try:
        # Run the standard preprocessing
        # This will read from TRANSFER_GRAPHS_DIR and write to TRANSFER_PROCESSED_DIR
        preprocess.run_preprocessing()
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise e
    finally:
        # Restore params just in case (though we exit after this usually)
        params.paths.graphs = original_graphs_path
        params.paths.processed = original_processed_path


def main():
    setup_logging()

    test_split_path = params.paths.test_graphs_txt

    run_preprocessing_for_transfer()

    logging.info("Step 3: Running Inference on Transfer Data...")

    device = torch.device(params.environment.device)
    run_dir = os.path.join(os.path.dirname(params.paths.checkpoint), "run_1")
    model_path = os.path.join(run_dir, "best_model.pt")

    t_mean, t_std = GGNN_train.compute_baselines_and_stats(
        params.paths.labels,
        params.paths.train_graphs_txt
    )
    t_mean_dev = torch.tensor(t_mean, device=device, dtype=torch.float)
    t_std_dev = torch.tensor(t_std, device=device, dtype=torch.float)

    # Create Dataset pointing to the NEW processed folder
    test_dataset = GGNN_train.ProGraMLPygDataset(
        split_file_path=test_split_path,
        processed_dir=TRANSFER_PROCESSED_DIR
    )

    test_loader = PygDataLoader(
        test_dataset,
        batch_size=params.training.graph_level_batch_size,
        shuffle=False,
        num_workers=params.environment.num_workers
    )

    # Initialize Model
    vocab_size = params.model.expected_vocab_size
    model = GGNN_train.ProGraMLNetPyG(vocab_size=vocab_size, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Run Eval
    # We reuse the function from your GGNN_test.py
    rmse, mae, r2, nrmse, tau, spearman, preds, targets = GGNN_test.evaluate_with_predictions(
        model, test_loader, device, t_mean_dev, t_std_dev
    )

    GGNN_train.log_blank_line()
    logging.info(f"TRANSFER LEARNING RESULTS (StableHLO -> Linalg)")
    logging.info(f"RMSE:          {rmse:.5f}")
    logging.info(f"MAE:           {mae:.5f}")
    logging.info(f"R²:            {r2:.5f}")
    logging.info(f"NRMSE:         {nrmse:.5f}")
    logging.info(f"Kendall's Tau: {tau:.5f}")
    logging.info(f"Spearman:      {spearman:.5f}")
    GGNN_train.log_blank_line()

    # Plot
    GGNN_test.plot_comprehensive_analysis(preds, targets, run_dir, "transfer_analysis_linalg.png")


if __name__ == "__main__":
    main()