"""
Evaluate the shipped checkpoints on the test sets and write the test-set
metrics of Table 2 to results/test_metrics.json.

Needs the dataset (SPHINX_DATA). A GPU is recommended but not required (set
SPHINX_DEVICE=cpu).

    python -m reproduce.eval_checkpoints            # FTE1-8, Linalg-Target, baselines
    python -m reproduce.eval_checkpoints --source   # also StableHLO-Source and Zero-Shot

The default needs the archives labels and nb101_graphs_linalg; --source also
needs nb101_graphs_stablehlo.

Columns:
  Mean Pred.        RMSE of predicting the training mean, on the stablehlo training
                    split (the baseline line GGNN_train.py logs for the source model)
  Lin. Reg.         linear regression on the number of trainable parameters, fitted
                    on the stablehlo training split, scored on the stablehlo test split
  StableHLO-Source  the source model on the stablehlo test set
  Zero-Shot         the source model on the linalg test set, with the stablehlo
                    vocabulary (every linalg opcode maps to the unknown entry);
                    predictions are de-normalised with the linalg training statistics
  FTE1-8, Linalg-Target  on the linalg test set
"""

import json
import logging
import os
import sys

import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression

from config.params import params
from reproduce import common as c
from reproduce import datasets

RUNS = {
    "training_time": ("1.FineTuning_Batch128_TrainingTime",
                      "TrainingTime_40Epochs_StableHLO", "TrainingTime_100Epochs_Linalg"),
    "test_accuracy": ("3.FineTuning_Batch128_TestAccuracy",
                      "TestAccuracy_60Epochs_StableHLO", "TestAccuracy_100Epochs_Linalg"),
}
CHECKPOINT = "best_tau_model.pt"


def evaluate(run_dirs, stats_split=None):
    """Evaluate checkpoints on params.paths.test_graphs_txt.

    stats_split is the training split whose label mean and standard deviation
    de-normalise the predictions (default: the configured training split).
    """
    import GGNN_test
    import GGNN_train

    datasets.preprocess_split(params.paths.test_graphs_txt)
    t_mean, t_std = GGNN_train.compute_baselines_and_stats(
        params.paths.labels, stats_split or params.paths.train_graphs_txt)
    device = torch.device(params.environment.device)
    t_mean = torch.tensor(t_mean, device=device, dtype=torch.float)
    t_std = torch.tensor(t_std, device=device, dtype=torch.float)

    dataset = GGNN_train.ProGraMLPygDataset(
        split_file_path=params.paths.test_graphs_txt, processed_dir=params.paths.processed)
    loader = GGNN_train.PygDataLoader(
        dataset, batch_size=params.training.graph_level_batch_size, shuffle=False,
        num_workers=min(8, params.environment.num_workers))

    out = {}
    for key, run_dir in run_dirs:
        model = GGNN_train.ProGraMLNetPyG(
            vocab_size=params.model.expected_vocab_size, device=device).to(device)
        ckpt = torch.load(os.path.join(run_dir, CHECKPOINT), map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        rmse, mae, r2, nrmse, tau, spearman, _, _ = GGNN_test.evaluate_with_predictions(
            model, loader, device, t_mean, t_std)
        out[key] = {"rmse": float(rmse), "r2": float(r2), "tau": float(tau),
                    "n_test": len(dataset), "checkpoint": CHECKPOINT}
        report(key, out[key])
        del model
    return out


def report(key, m):
    print(f"{key}: RMSE {m['rmse']:.4f}  R2 {m['r2']:.4f}  tau {m['tau']:.4f}  "
          f"(n={m['n_test']})", flush=True)


def baselines(target):
    """Mean predictor and linear regression on the parameter count (no graphs needed)."""
    with open(params.paths.labels) as f:
        labels = json.load(f)

    def load(split):
        with open(split) as f:
            names = [line.strip().replace(".pt", "") for line in f if line.strip()]
        x = np.array([labels[n]["trainable_parameters"] for n in names], dtype=float)
        y = np.array([labels[n][target] for n in names], dtype=float)
        return x.reshape(-1, 1), y

    x_train, y_train = load(params.paths.train_graphs_txt)
    x_test, y_test = load(params.paths.test_graphs_txt)

    pred = LinearRegression().fit(x_train, y_train).predict(x_test)
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    out = {
        # A constant predictor has no ranking signal: R2 and tau are 0 by definition.
        f"baseline/mean/{target}": {
            "rmse": float(np.sqrt(np.mean((y_train - y_train.mean()) ** 2))),
            "r2": 0.0, "tau": 0.0, "n_test": len(y_train)},
        f"baseline/linear/{target}": {
            "rmse": float(np.sqrt(np.mean((y_test - pred) ** 2))),
            "r2": 1.0 - ss_res / ss_tot, "tau": float(kendalltau(y_test, pred)[0]),
            "n_test": len(y_test)},
    }
    for key, m in out.items():
        report(key, m)
    return out


def main():
    logging.basicConfig(level=logging.WARNING)
    with_source = "--source" in sys.argv[1:]
    path = c.results_path("test_metrics.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    for target, (folder, source, scratch) in RUNS.items():
        datasets.configure("nb101", "stablehlo", target, "stablehlo")
        results.update(baselines(target))

        datasets.configure("nb101", "linalg", target, "linalg")
        subs = [f"FT{n}" for n in range(1, 9)] + [scratch]
        results.update(evaluate([(f"{folder}/{s}", c.results_path(folder, s)) for s in subs]))

        if with_source:
            source_run = [(f"{folder}/{source}", c.results_path(folder, source))]
            datasets.configure("nb101", "stablehlo", target, "stablehlo")
            results.update(evaluate(source_run))

            datasets.configure("nb101", "linalg", target, "linalg", vocab_dialect="stablehlo")
            zero = evaluate([(f"zero_shot/{target}", c.results_path(folder, source))],
                            stats_split=f"{datasets.layout.SPLITS['nb101']}/train_files_linalg.txt")
            results.update(zero)

        with open(path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    print(f"written {path}")


if __name__ == "__main__":
    main()
