import os
import time
import logging

import torch
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from config.params import params

from GGNN_train import (ProGraMLPygDataset, ProGraMLNetPyG, evaluate, PygDataLoader,
                        log_blank_line, load_vocabulary_from_csv, load_filenames_from_txt)


def search_filenames():
    base_graph_dir = Path("/home/douglasvc/Desktop/LivenessTest/graphs/dataflow/graphs")
    labels_dir = Path("/home/douglasvc/Desktop/LivenessTest/labels/liveness")

    logging.info(f"Scanning base graph directory: {base_graph_dir}")
    logging.info(f"Checking for labels in: {labels_dir}")
    log_blank_line()
    logging.info(f"Processing real test data...")
    log_blank_line()

    if not os.path.isdir(base_graph_dir):
        logging.critical(f"Warning: Directory not found: {base_graph_dir}")
        return []

    if not os.path.isdir(labels_dir):
        logging.critical(f"Warning: Directory not found: {labels_dir}")
        return []

    potential_graph_files = sorted(os.listdir(base_graph_dir))

    valid_graph_filenames = []
    count_missing_labels = 0
    for filename in tqdm(potential_graph_files, desc=f"Verifying files"):
        # Ensure we're only looking at graph files with the correct suffix
        if filename.endswith("ProgramGraph.pb"):
            # Derive basename and construct the expected label filename
            basename = filename[:-len("ProgramGraph.pb")]
            label_filename = f"{basename}ProgramGraphFeaturesList.pb"
            label_filepath = os.path.join(labels_dir, label_filename)

            # Check if the corresponding label file exists
            if os.path.exists(label_filepath) and os.path.getsize(label_filepath) > 0:
                valid_graph_filenames.append(basename)
            else:
                count_missing_labels += 1

    logging.info(f"Missing labels for {count_missing_labels} graph files.")

    if not valid_graph_filenames:
        logging.critical(f"Warning: No valid graph/label pairs found.")
        return []

    return valid_graph_filenames


def print_eval_info(model, test_real_loader, device):
    logging.info("--- Starting Manual Inspection Loop (All Samples) ---")
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for i, batch in enumerate(test_real_loader):

            logging.info(f"\n\n--- Inspecting Sample Index: {i} ---")

            batch = batch.to(device)
            if batch.num_nodes == 0:
                continue

            out_logits = model(batch)

            # Get probabilities (0.0 to 1.0) and predictions (0 or 1)
            out_probs = torch.sigmoid(out_logits)
            out_preds = (out_probs > 0.5).int()

            # Ground-truth labels
            labels = batch.y.int()

            root_node_index = torch.argmax(batch.node_selectors[:, 1]).item()

            logging.info(f"Graph has {batch.num_nodes} nodes.")
            logging.info(f"Root Node for this sample: {root_node_index}")
            logging.info(f"True 'Live-Out' Nodes: {labels.sum().item()}")
            logging.info(f"Predicted 'Live-Out' Nodes: {out_preds.sum().item()}")
            log_blank_line()

            for node_idx in range(batch.num_nodes):
                node_label = labels[node_idx].item()
                node_pred = out_preds[node_idx].item()

                if node_label == 1 or node_pred == 1:
                    logging.info(
                        f"  Node {node_idx:3d}:   Label = {node_label}   |   Prediction = {node_pred}   |   Prob = {out_probs[node_idx].item():.4f}")

    logging.info("--- Full inspection complete. ---")

def main():
    device = torch.device(params.environment.device)
    logging.info(f"Using device: {params.environment.device}")

    # VOCAB_SIZE = 2230 + 1 for unknown token (ProGraML paper, Page 7, Table 1)
    token_to_idx, idx_to_token, vocab_size = load_vocabulary_from_csv(params.paths.vocab)
    assert vocab_size == 2231  # Ensure the vocabulary size matches the expected size

    log_blank_line()
    test_graphs_filenames = search_filenames()
    test_real_dataset = ProGraMLPygDataset(
        file_basenames=test_graphs_filenames,
        graph_src_dir="/home/douglasvc/Desktop/LivenessTest/graphs/dataflow/graphs",
        split='liveness',
        token_to_idx=token_to_idx,
        max_samples=None,
        max_steps=30,
        max_sample_size_mb=None,
        reprocess=True
    )

    test_real_loader = PygDataLoader(
        test_real_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=params.training.graph_level_batch_size,
        num_workers=params.environment.num_workers
    )

    log_blank_line()
    model = ProGraMLNetPyG(
        vocab_size=vocab_size,
        device=device,
    ).to(device)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    best_model_path = "/home/douglasvc/Desktop/Results/Liveness_batch=32_bestf1score=09737/best_model.pt"

    logging.info(f"Loading best model weights from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path).get('model_state_dict', {}))

    # print_eval_info(model, test_real_loader, device)

    criterion = nn.BCEWithLogitsLoss()

    log_blank_line()
    logging.info("--- Evaluating on Real Test Set (T=30) ---")
    model.ggnn_iterations = 30
    _, _, test_prec_30, test_recall_30, test_f1_30 = evaluate(model, test_real_loader, criterion, device)
    logging.info(f"Final F1 Score for Real Set (T=30): {test_f1_30:.4f} - Precision: {test_prec_30:.4f} - Recall: {test_recall_30:.4f}")


if __name__ == '__main__':
    start_time = time.time()

    torch.cuda.empty_cache()
    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    log_blank_line()
    logging.info(f"Execution time: {elapsed_time:.5f} seconds")