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

    # VOCAB_SIZE = 2230 + 1 for unknown token (ProGraML paper, Page 7, Table 1)
    token_to_idx, idx_to_token, vocab_size = load_vocabulary_from_csv(params.paths.vocab)
    assert vocab_size == 2231  # Ensure the vocabulary size matches the expected size

    log_blank_line()
    test_graphs_filenames = load_filenames_from_txt(params.paths.test_graphs_txt)
    test_ddf30_dataset = ProGraMLPygDataset(
        file_basenames=test_graphs_filenames,
        graph_src_dir=params.paths.test_graphs,
        split='test/DDF30',
        token_to_idx=token_to_idx,
        max_samples=None,
        max_steps=30,
        max_sample_size_mb=None,
        reprocess=False
    )

    log_blank_line()
    test_ddf60_dataset = ProGraMLPygDataset(
        file_basenames=test_graphs_filenames,
        graph_src_dir=params.paths.test_graphs,
        split='test/DDF60',
        token_to_idx=token_to_idx,
        max_samples=None,
        max_steps=60,
        max_sample_size_mb=None,
        reprocess=False
    )

    log_blank_line()
    test_ddf_dataset = ProGraMLPygDataset(
        file_basenames=test_graphs_filenames,
        graph_src_dir=params.paths.test_graphs,
        split='test/DDF',
        token_to_idx=token_to_idx,
        max_samples=None,
        max_steps=None,
        max_sample_size_mb=None,
        reprocess=False
    )

    test_ddf30_loader = PygDataLoader(
        test_ddf30_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=params.training.graph_level_batch_size,
        num_workers=params.environment.num_workers
    )

    test_ddf60_loader = PygDataLoader(
        test_ddf60_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=params.training.graph_level_batch_size,
        num_workers=params.environment.num_workers
    )

    test_ddf_loader = PygDataLoader(
        test_ddf_dataset,
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

    criterion = nn.BCEWithLogitsLoss()

    log_blank_line()
    logging.info("--- Evaluating on DDF-30 Test Set (T=30) ---")
    model.ggnn_iterations = 30
    _, _, test_prec_30, test_recall_30, test_f1_30 = evaluate(model, test_ddf30_loader, criterion, device)
    logging.info(f"Final F1 Score for DDF-30 (T=30): {test_f1_30:.4f} - Precision: {test_prec_30:.4f} - Recall: {test_recall_30:.4f}")

    log_blank_line()
    logging.info("--- Evaluating on DDF-60 Test Set (T=60) ---")
    model.ggnn_iterations = 30
    _, _, test_prec_60, test_recall_60, test_f1_60 = evaluate(model, test_ddf60_loader, criterion, device)
    logging.info(f"Final F1 Score for DDF-60 (T=60): {test_f1_60:.4f} - Precision: {test_prec_60:.4f} - Recall: {test_recall_60:.4f}")

    log_blank_line()
    logging.info("--- Evaluating on Full DDF Test Set (T=200) ---")
    model.ggnn_iterations = 200
    _, _, test_prec_200, test_recall_200, test_f1_200 = evaluate(model, test_ddf_loader, criterion, device)
    logging.info(f"Final F1 Score for DDF-200 (T=200): {test_f1_200:.4f} - Precision: {test_prec_200:.4f} - Recall: {test_recall_200:.4f}")


if __name__ == '__main__':
    start_time = time.time()

    torch.cuda.empty_cache()
    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    log_blank_line()
    logging.info(f"Execution time: {elapsed_time:.5f} seconds")