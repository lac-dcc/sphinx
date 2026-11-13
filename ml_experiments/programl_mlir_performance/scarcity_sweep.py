"""
Data-scarcity sweep: scratch vs FTE6 across training-set fractions.

Runs the same protocol as the paper's Section 5.2 on NAS-Bench-201. For each
fraction of the target training split, trains a model from scratch and a
fine-tuned model, and records the best validation Kendall tau.

FTE6 is mean-centered init with the Edge and Position MLPs frozen. The IDs are
the paper's FTE numbers, as in finetune_experiment.py.

Run from the same directory as GGNN_train.py.
"""

import json
import logging
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from GGNN_train import (
    ProGraMLNetPyG,
    ProGraMLPygDataset,
    train_by_epoch,
    setup_logging,
    get_next_run_dir,
    compute_baselines_and_stats,
    log_blank_line,
    PygDataLoader,
    format_elapsed_time,
)
from config.params import params

import preprocess
import finetune_experiment as fte


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Fractions of the FULL target training split to use.
FRACTIONS = (0.80, 1.00)

# Repeats per (fraction, arm). The paper trains each configuration three times.
SEEDS = (0, 1, 2)

# Which finetune_experiment configurations to run (the paper's FTE numbers):
#   6 = FTE6: mean-centered init, Edge + Position MLPs frozen
#   8 = FTE8: mean-centered init, everything trainable
# Running both gives the selective-freezing vs full-fine-tuning contrast.
FT_EXPERIMENT_IDS = (6, 8)

# Arms to run.
RUN_SCRATCH = True
RUN_FINETUNE = True

# Prediction targets to sweep, in order. Each needs its own preprocessed cache,
# because the label is baked into the .pt files, and its own pre-trained
# backbone. Paths are relative to the checkpoint directory.
CHECKPOINT_DIR = os.path.dirname(params.paths.checkpoint)

if os.environ.get("SPHINX_DATA"):
    # Published layout: run with SPHINX_BENCHMARK=nb201 SPHINX_DIALECT=linalg.
    # The backbones are the shipped NAS-Bench-101 stablehlo source models.
    _cache = os.environ.get("SPHINX_CACHE", os.path.join(os.environ["SPHINX_DATA"], "processed"))
    _results = os.environ.get("SPHINX_RESULTS", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))
    _backbones = os.path.join(_results, "12.NasBench201_From101", "checkpoints")
    _processed = {m: os.path.join(_cache, f"nb201_linalg_{m}")
                  for m in ("training_time", "test_accuracy")}
    # The sweep draws from all NAS-Bench-201 architectures (60/20/20), not from
    # the 10% target pool of the stablehlo -> linalg pipeline.
    _splits = os.path.join(params.paths.splits_txt, "finetune_experiment")
    params.paths.train_graphs_txt = f"{_splits}/NasBench201_train_files.txt"
    params.paths.validation_graphs_txt = f"{_splits}/NasBench201_validation_files.txt"
    params.paths.test_graphs_txt = f"{_splits}/NasBench201_test_files.txt"
else:
    _backbones = CHECKPOINT_DIR
    _processed = {m: f"{params.paths.dataset}/processed_{m}_linalg"
                  for m in ("training_time", "test_accuracy")}

TARGETS = (
    {
        "metric": "training_time",
        "processed": _processed["training_time"],
        "backbone": os.path.join(
            _backbones, "TrainingTime_40Epochs_StableHLO/best_r2_model.pt"
        ),
    },
    {
        "metric": "test_accuracy",
        "processed": _processed["test_accuracy"],
        "backbone": os.path.join(
            _backbones, "TestAccuracy_60Epochs_StableHLO/best_r2_model.pt"
        ),
    },
)

# Where the generated subset split files go.
SUBSET_DIR = os.path.join(params.paths.splits_txt, "scarcity_sweep")

# Where the summary lands. Holds every target; records carry a "target" field.
RESULTS_PATH = os.path.join(CHECKPOINT_DIR, "scarcity_sweep_results.json")

# Delete per-epoch checkpoints after each run. train_by_epoch writes one file
# per epoch; 36 runs x 100 epochs will otherwise fill the disk.
PRUNE_EPOCH_CHECKPOINTS = False

# ==============================================================================


def build_subset(full_split_path, fraction, seed):
    """
    Write a subset split file and return its path and size.

    Subsets are nested: for a given seed the full split is shuffled once and
    every fraction takes a prefix of that order, so the 5% set contains the 1%
    set. Independent draws per fraction would let sampling noise produce
    non-monotonic points in the curve, which is hard to distinguish from a real
    effect.
    """
    with open(full_split_path) as f:
        names = [line.strip() for line in f if line.strip()]

    # Sort first so the shuffled order depends only on the seed, not on the
    # order the split file happens to be written in.
    names.sort()
    rng = random.Random(seed)
    rng.shuffle(names)

    n = max(1, int(round(len(names) * fraction)))
    picked = sorted(names[:n])

    os.makedirs(SUBSET_DIR, exist_ok=True)
    pct = f"{fraction * 100:g}".replace(".", "p")
    path = os.path.join(SUBSET_DIR, f"train_{pct}pct_seed{seed}.txt")
    with open(path, "w") as f:
        f.write("\n".join(picked) + "\n")

    return path, n


def read_best(run_dir):
    """Pull the best validation metrics out of the saved best-tau checkpoint."""
    out = {}
    for key, fname in (
        ("tau", "best_tau_model.pt"),
        ("rmse", "best_rmse_model.pt"),
        ("r2", "best_r2_model.pt"),
    ):
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        out[f"best_{key}"] = ckpt.get(f"best_val_{key}")
        if key == "tau":
            out["epoch_at_best_tau"] = ckpt.get("epoch")
    return out


def prune_epoch_checkpoints(run_dir):
    for name in os.listdir(run_dir):
        if name.startswith("epoch_") and name.endswith(".pt"):
            try:
                os.remove(os.path.join(run_dir, name))
            except OSError:
                pass


def run_one(arm, train_split_path, n_train, fraction, seed, ft_id=None):
    """Train one model. arm is 'scratch' or 'ft<id>'."""
    start_time = time.time()
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    random.seed(seed)

    # Point the shared config at this subset for the duration of the run.
    original_train_split = params.paths.train_graphs_txt
    params.paths.train_graphs_txt = train_split_path

    try:
        base_checkpoint_dir = os.path.dirname(params.paths.checkpoint)
        run_dir = get_next_run_dir(base_checkpoint_dir)
        setup_logging(run_dir)

        device = torch.device(params.environment.device)

        logging.info(
            f"SWEEP | target={params.model.target_performance_metric} | arm={arm} | "
            f"fraction={fraction:.2%} | n_train={n_train} | seed={seed}"
        )
        if ft_id is not None:
            logging.info(
                f"Fine-tuning config {ft_id}: {fte.EXPERIMENT_NAMES[ft_id]}"
            )
        logging.info(f"Train split: {train_split_path}")
        logging.info(f"Run Directory: {run_dir}")

        train_dataset = ProGraMLPygDataset(
            split_file_path=params.paths.train_graphs_txt,
            processed_dir=params.paths.processed,
        )
        val_dataset = ProGraMLPygDataset(
            split_file_path=params.paths.validation_graphs_txt,
            processed_dir=params.paths.processed,
        )

        train_loader = PygDataLoader(
            train_dataset,
            batch_size=params.training.graph_level_batch_size,
            shuffle=True,
            num_workers=params.environment.num_workers,
            pin_memory=True,
        )
        val_loader = PygDataLoader(
            val_dataset,
            batch_size=params.training.graph_level_batch_size,
            shuffle=False,
            num_workers=params.environment.num_workers,
            pin_memory=True,
        )

        t_mean, t_std = compute_baselines_and_stats(
            params.paths.labels, params.paths.train_graphs_txt
        )
        t_mean_dev = torch.tensor(t_mean, device=device, dtype=torch.float)
        t_std_dev = torch.tensor(t_std, device=device, dtype=torch.float)

        model = ProGraMLNetPyG(
            vocab_size=params.model.expected_vocab_size, device=device
        ).to(device)

        if ft_id is not None:
            # load_pretrained_surgery and freeze_gnn_layers read the module-level
            # EXPERIMENT_ID at call time, so set it before calling them.
            fte.EXPERIMENT_ID = ft_id
            model = fte.load_pretrained_surgery(
                model, fte.PRETRAINED_CHECKPOINT_PATH
            )
            if ft_id not in fte.TRAIN_ALL_IDS:
                fte.freeze_gnn_layers(model)
        else:
            logging.info(
                f"Model Initialized. Vocab: {params.model.expected_vocab_size}, "
                f"Params: {sum(p.numel() for p in model.parameters())}"
            )

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params.training.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        log_blank_line()
        train_by_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            t_mean_dev,
            t_std_dev,
            run_dir,
            start_time,
        )

        result = read_best(run_dir)
        result.update(
            {
                "arm": arm,
                "fraction": fraction,
                "n_train": n_train,
                "seed": seed,
                "run_dir": run_dir,
                "elapsed_seconds": time.time() - start_time,
                "target": params.model.target_performance_metric,
            }
        )
        if ft_id is not None:
            result["ft_experiment_id"] = ft_id

        logging.info(
            f"SWEEP RESULT | target={params.model.target_performance_metric} | "
            f"arm={arm} | fraction={fraction:.2%} | seed={seed} | "
            f"best tau {result.get('best_tau')}"
        )

        del model
        torch.cuda.empty_cache()

        if PRUNE_EPOCH_CHECKPOINTS:
            prune_epoch_checkpoints(run_dir)

        return result

    finally:
        params.paths.train_graphs_txt = original_train_split


def main():
    full_split = params.paths.train_graphs_txt

    print(f"Full training split: {full_split}")
    print(f"Graphs:              {params.paths.graphs}")
    print(f"Vocab:               {params.paths.vocab}")
    print(f"Labels:              {params.paths.labels}")
    print(f"Targets:             {[t['metric'] for t in TARGETS]}")
    print()

    # Fail early on anything missing rather than hours in.
    with open(params.paths.labels) as f:
        sample = next(iter(json.load(f).values()))
    for target in TARGETS:
        if target["metric"] not in sample:
            print(
                f"ERROR: '{target['metric']}' is not a field in {params.paths.labels}. "
                f"Available: {sorted(sample)}"
            )
            sys.exit(1)
        if RUN_FINETUNE and not os.path.exists(target["backbone"]):
            print(f"ERROR: backbone not found: {target['backbone']}")
            sys.exit(1)

    # (arm label, finetune experiment id or None)
    arms = []
    if RUN_SCRATCH:
        arms.append(("scratch", None))
    if RUN_FINETUNE:
        arms += [(f"ft{i}", i) for i in FT_EXPERIMENT_IDS]

    total = len(TARGETS) * len(FRACTIONS) * len(SEEDS) * len(arms)
    print(f"{total} runs planned.\n")

    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} runs already recorded.\n")

    def arm_key(r):
        # Records written before multiple FT configs used arm == "finetune".
        if r["arm"] == "finetune":
            return f"ft{r.get('ft_experiment_id')}"
        return r["arm"]

    done = {
        (r.get("target"), arm_key(r), r["fraction"], r["seed"]) for r in results
    }
    i = 0

    for target in TARGETS:
        metric = target["metric"]

        # Each target needs its own label in the preprocessed graphs and its own
        # source backbone, so switch both before touching the data.
        params.model.target_performance_metric = metric
        params.paths.processed = target["processed"]
        fte.PRETRAINED_CHECKPOINT_PATH = target["backbone"]

        print(f"=== target: {metric}")
        print(f"    processed: {params.paths.processed}")
        if RUN_FINETUNE:
            print(f"    backbone:  {target['backbone']}")

        log_blank_line()
        preprocess.run_preprocessing()
        log_blank_line()

        for fraction in FRACTIONS:
            for seed in SEEDS:
                train_split_path, n_train = build_subset(full_split, fraction, seed)
                for arm, ft_id in arms:
                    i += 1
                    if (metric, arm, fraction, seed) in done:
                        print(f"[{i}/{total}] skip {metric} {arm} {fraction:.0%} seed{seed}")
                        continue

                    print(
                        f"[{i}/{total}] {metric} | {arm} | {fraction:.0%} "
                        f"({n_train} graphs) | seed {seed}",
                        flush=True,
                    )
                    try:
                        r = run_one(
                            arm, train_split_path, n_train, fraction, seed, ft_id
                        )
                    except Exception as e:
                        logging.error(f"run failed: {e}")
                        print(f"    FAILED: {e}", flush=True)
                        continue

                    results.append(r)
                    with open(RESULTS_PATH, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"    best tau {r.get('best_tau'):.4f}", flush=True)

    print(f"\nResults written to {RESULTS_PATH}")
    summarize(results)


def summarize(results):
    import statistics

    def arm_key(r):
        if r["arm"] == "finetune":
            return f"ft{r.get('ft_experiment_id')}"
        return r["arm"]

    by = {}
    for r in results:
        by.setdefault((r.get("target"), r["fraction"], arm_key(r)), []).append(r)

    arm_order = ["scratch"] + [f"ft{i}" for i in FT_EXPERIMENT_IDS]

    for target in dict.fromkeys(r.get("target") for r in results):
        print(f"\n=== {target}")
        print(f"{'fraction':>10} {'n':>7} {'arm':>10} {'tau mean':>10} "
              f"{'tau sd':>8} {'tau max':>9} {'runs':>5}")
        fractions = sorted({k[1] for k in by if k[0] == target})
        for fraction in fractions:
            for arm in arm_order:
                rows = by.get((target, fraction, arm))
                if not rows:
                    continue
                taus = [r["best_tau"] for r in rows if r.get("best_tau") is not None]
                if not taus:
                    continue
                sd = statistics.stdev(taus) if len(taus) > 1 else 0.0
                print(
                    f"{fraction:>9.0%} {rows[0]['n_train']:>7} {arm:>10} "
                    f"{statistics.mean(taus):>10.4f} {sd:>8.4f} "
                    f"{max(taus):>9.4f} {len(taus):>5}"
                )


if __name__ == "__main__":
    main()
