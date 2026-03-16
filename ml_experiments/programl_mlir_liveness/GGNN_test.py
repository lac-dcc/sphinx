import time
import logging

import torch
import torch.nn as nn

from config.params import params

from GGNN_train import (ProGraMLPygDataset, ProGraMLNetPyG, evaluate, PygDataLoader,
                        log_blank_line, load_vocabulary_from_csv, load_filenames_from_txt)


def main():
    device = torch.device(params.environment.device)
    logging.info(f"Using device: {params.environment.device}")

    token_to_idx, idx_to_token, vocab_size = load_vocabulary_from_csv(params.paths.vocab)
    assert vocab_size == 608  # Ensure the vocabulary size matches the expected size

    log_blank_line()
    test_graphs_filenames = load_filenames_from_txt(params.paths.test_graphs_txt)
    test_dataset = ProGraMLPygDataset(
        file_basenames=test_graphs_filenames,
        graph_src_dir=params.paths.test_graphs,
        split='test',
        token_to_idx=token_to_idx,
        max_samples=None,
        max_steps=30,
        max_sample_size_mb=None,
        reprocess=False
    )

    test_loader = PygDataLoader(
        test_dataset,
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

    best_model_path = "/home/douglasvc/Desktop/LivenessTest/checkpoints/best_model.pt"
    logging.info(f"Loading best model weights from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path).get('model_state_dict', {}))

    criterion = nn.BCEWithLogitsLoss()

    log_blank_line()
    logging.info("--- Evaluating on Test Set (T=30) ---")
    model.ggnn_iterations = 30
    _, _, test_prec_30, test_recall_30, test_f1_30 = evaluate(model, test_loader, criterion, device)
    logging.info(f"Final F1 Score for Test Set (T=30): {test_f1_30:.4f} - Precision: {test_prec_30:.4f} - Recall: {test_recall_30:.4f}")


if __name__ == '__main__':
    start_time = time.time()

    torch.cuda.empty_cache()
    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    log_blank_line()
    logging.info(f"Execution time: {elapsed_time:.5f} seconds")