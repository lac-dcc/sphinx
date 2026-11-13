"""
Full 90/10 NAS-Bench-201 pipeline: pre-train on stablehlo, fine-tune on linalg.

For each prediction target:
  1. train a model from scratch on the stablehlo source pool  (8,253 graphs)
  2. train a linalg scratch baseline on the target pool         (916 graphs)
  3. fine-tune the source model onto linalg under all 8 FTE configurations

The two dialects differ in two config values that must be set before the model
is constructed:

    max_edge_position   stablehlo 8, linalg 6
    expected_vocab_size stablehlo 49, linalg 58

Neither requires weight surgery. edge_positional_encodings is a sinusoidal
table built at construction and not part of state_dict, and the vocabulary only
sizes node_text_embedding, which load_pretrained_surgery drops and rebuilds.

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

SPLITS = "splits/finetune_experiment_201"

DIALECTS = {
    "stablehlo": {
        "vocab": "vocab/FTE_NasBench201_StableHLO.csv",
        "graphs": f"{params.paths.dataset}/graphs/stablehlo",
        "max_edge_position": 8,
        "vocab_size": 49,
        "train": f"{SPLITS}/train_files_stablehlo.txt",
        "validation": f"{SPLITS}/validation_files_stablehlo.txt",
        "test": f"{SPLITS}/test_files_stablehlo.txt",
    },
    "linalg": {
        "vocab": "vocab/FTE_NasBench201_Linalg.csv",
        "graphs": f"{params.paths.dataset}/graphs/linalg",
        "max_edge_position": 6,
        "vocab_size": 58,
        "train": f"{SPLITS}/train_files_linalg.txt",
        "validation": f"{SPLITS}/validation_files_linalg.txt",
        "test": f"{SPLITS}/test_files_linalg.txt",
    },
    "arith": {
        "vocab": "vocab/FTE_NasBench201_Arith.csv",
        "graphs": f"{params.paths.dataset}/graphs/arith",
        "max_edge_position": 5,
        "vocab_size": 54,
        # The 90/10 partition is over architectures, so the source pool uses the
        # stablehlo name lists and the target pool the linalg ones, whichever
        # dialect they are read in.
        "train": f"{SPLITS}/train_files_stablehlo.txt",
        "validation": f"{SPLITS}/validation_files_stablehlo.txt",
        "test": f"{SPLITS}/test_files_stablehlo.txt",
    },
    "llvm": {
        "vocab": "vocab/FTE_NasBench201_LLVM.csv",
        "graphs": f"{params.paths.dataset}/graphs/llvm",
        "max_edge_position": 8,
        "vocab_size": 52,
        "train": f"{SPLITS}/train_files_linalg.txt",
        "validation": f"{SPLITS}/validation_files_linalg.txt",
        "test": f"{SPLITS}/test_files_linalg.txt",
    },
}

# Which pair to run. The source dialect uses the 90% architecture pool, the
# target dialect the 10% pool, so a dialect's split files depend on the side it
# is used on. Change these two lines to switch pairs.
#   stablehlo -> linalg           (done)
#   arith     -> llvm             (this run)
SOURCE_DIALECT = "arith"
TARGET_DIALECT = "llvm"

TARGETS = ("training_time", "test_accuracy")

# This pair was run with a batch size of 64: at 128 the llvm graphs do not fit
# in 32 GB of GPU memory. SPHINX_BATCH_SIZE overrides it.
BATCH_SIZE = 64
if not os.environ.get("SPHINX_BATCH_SIZE"):
    params.training.graph_level_batch_size = BATCH_SIZE

# All eight configurations, in finetune_experiment.py's numbering.
FT_EXPERIMENT_IDS = (1, 2, 3, 4, 5, 6, 7, 8)

# Also train a scratch baseline on the target dialect.
RUN_TARGET_SCRATCH = True

# Which saved checkpoint from the source run becomes the backbone.
BACKBONE_CHECKPOINT = "best_r2_model.pt"

CHECKPOINT_DIR = os.path.dirname(params.paths.checkpoint)
RESULTS_PATH = os.path.join(
    CHECKPOINT_DIR,
    f"pipeline_9010_{SOURCE_DIALECT}_to_{TARGET_DIALECT}_results.json",
)

# train_by_epoch writes a checkpoint per epoch; 20 runs x 100 epochs adds up.
PRUNE_EPOCH_CHECKPOINTS = False

SEED = 0

# ==============================================================================


def processed_dir(dialect, target):
    return f"{params.paths.dataset}/processed_{target}_{dialect}_201"


def apply_dialect(dialect, target):
    """Point params at one dialect and one target. Must precede model construction."""
    cfg = DIALECTS[dialect]
    params.model.target_performance_metric = target
    params.model.max_edge_position = cfg["max_edge_position"]
    params.model.expected_vocab_size = cfg["vocab_size"]
    params.paths.vocab = cfg["vocab"]
    params.paths.graphs = cfg["graphs"]
    params.paths.processed = processed_dir(dialect, target)
    params.paths.train_graphs_txt = cfg["train"]
    params.paths.validation_graphs_txt = cfg["validation"]
    params.paths.test_graphs_txt = cfg["test"]


def read_best(run_dir):
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


def prune(run_dir):
    for name in os.listdir(run_dir):
        if name.startswith("epoch_") and name.endswith(".pt"):
            try:
                os.remove(os.path.join(run_dir, name))
            except OSError:
                pass


def train_one(dialect, target, arm, ft_id=None, backbone=None):
    """One training run. arm is 'source', 'scratch' or 'ft<id>'."""
    start_time = time.time()
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    random.seed(SEED)

    apply_dialect(dialect, target)

    run_dir = get_next_run_dir(CHECKPOINT_DIR)
    setup_logging(run_dir)

    device = torch.device(params.environment.device)

    logging.info(
        f"PIPELINE | target={target} | dialect={dialect} | arm={arm} | "
        f"vocab={params.model.expected_vocab_size} | "
        f"max_edge_position={params.model.max_edge_position}"
    )
    if ft_id is not None:
        logging.info(f"Config {ft_id}: {fte.EXPERIMENT_NAMES[ft_id]}")
        logging.info(f"Backbone: {backbone}")
    logging.info(f"Graphs: {params.paths.graphs}")
    logging.info(f"Train split: {params.paths.train_graphs_txt}")
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
        fte.EXPERIMENT_ID = ft_id
        model = fte.load_pretrained_surgery(model, backbone)
        if ft_id not in fte.TRAIN_ALL_IDS:
            fte.freeze_gnn_layers(model)
    else:
        logging.info(
            f"Model Initialized. Params: {sum(p.numel() for p in model.parameters())}"
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
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        device, t_mean_dev, t_std_dev, run_dir, start_time,
    )

    result = read_best(run_dir)
    result.update({
        "target": target,
        "dialect": dialect,
        "arm": arm,
        "n_train": len(train_dataset),
        "run_dir": run_dir,
        "elapsed_seconds": time.time() - start_time,
    })
    if ft_id is not None:
        result["ft_experiment_id"] = ft_id
        result["backbone"] = backbone

    logging.info(
        f"PIPELINE RESULT | {target} | {dialect} | {arm} | "
        f"best tau {result.get('best_tau')}"
    )
    logging.info(f"Total: {format_elapsed_time(time.time() - start_time)}")

    del model
    torch.cuda.empty_cache()

    if PRUNE_EPOCH_CHECKPOINTS:
        prune(run_dir)

    return result


def preflight():
    problems = []

    with open(params.paths.labels) as f:
        sample = next(iter(json.load(f).values()))
    for target in TARGETS:
        if target not in sample:
            problems.append(
                f"'{target}' not a field in {params.paths.labels}; "
                f"available: {sorted(sample)}"
            )

    for dialect in (SOURCE_DIALECT, TARGET_DIALECT):
        cfg = DIALECTS[dialect]
        if not os.path.isdir(cfg["graphs"]):
            problems.append(f"graphs dir missing: {cfg['graphs']}")
        if not os.path.exists(cfg["vocab"]):
            problems.append(f"vocab missing: {cfg['vocab']}")
        for split in ("train", "validation", "test"):
            if not os.path.exists(cfg[split]):
                problems.append(f"split missing: {cfg[split]}")

    if problems:
        print("Preflight failed:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)

    print(f"Preflight OK. Pair: {SOURCE_DIALECT} -> {TARGET_DIALECT}")
    for role, dialect in (("source", SOURCE_DIALECT), ("target", TARGET_DIALECT)):
        cfg = DIALECTS[dialect]
        n = len(open(cfg["train"]).read().split())
        print(
            f"  {role:<7} {dialect:<10} vocab {cfg['vocab_size']:>3}  "
            f"max_edge_position {cfg['max_edge_position']}  "
            f"graphs {cfg['graphs']}  train {n}"
        )
    print()


def main():
    preflight()

    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} runs already recorded.\n")

    def key(r):
        return (r["target"], r["dialect"], r["arm"])

    done = {key(r): r for r in results}

    def save():
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    planned = len(TARGETS) * (
        1 + (1 if RUN_TARGET_SCRATCH else 0) + len(FT_EXPERIMENT_IDS)
    )
    print(f"{planned} runs planned.\n")
    i = 0

    for target in TARGETS:
        print(f"=== target: {target}")

        # ---- 1. source pre-training on stablehlo
        i += 1
        src_key = (target, SOURCE_DIALECT, "source")
        if src_key in done:
            source_run_dir = done[src_key]["run_dir"]
            print(f"[{i}/{planned}] skip source pre-training ({source_run_dir})")
        else:
            print(f"[{i}/{planned}] pre-training on {SOURCE_DIALECT}", flush=True)
            apply_dialect(SOURCE_DIALECT, target)
            log_blank_line()
            preprocess.run_preprocessing()
            log_blank_line()
            r = train_one(SOURCE_DIALECT, target, "source")
            results.append(r)
            done[src_key] = r
            save()
            source_run_dir = r["run_dir"]
            print(f"    best tau {r.get('best_tau'):.4f}", flush=True)

        backbone = os.path.join(source_run_dir, BACKBONE_CHECKPOINT)
        if not os.path.exists(backbone):
            print(f"    ERROR: backbone not found at {backbone}")
            sys.exit(1)

        # ---- target dialect: preprocess once
        apply_dialect(TARGET_DIALECT, target)
        log_blank_line()
        preprocess.run_preprocessing()
        log_blank_line()

        # ---- 2. target scratch baseline
        if RUN_TARGET_SCRATCH:
            i += 1
            k = (target, TARGET_DIALECT, "scratch")
            if k in done:
                print(f"[{i}/{planned}] skip {TARGET_DIALECT} scratch")
            else:
                print(f"[{i}/{planned}] {TARGET_DIALECT} scratch", flush=True)
                try:
                    r = train_one(TARGET_DIALECT, target, "scratch")
                    results.append(r)
                    done[k] = r
                    save()
                    print(f"    best tau {r.get('best_tau'):.4f}", flush=True)
                except Exception as e:
                    logging.error(f"run failed: {e}")
                    print(f"    FAILED: {e}", flush=True)

        # ---- 3. all eight fine-tuning configurations
        for ft_id in FT_EXPERIMENT_IDS:
            i += 1
            arm = f"ft{ft_id}"
            k = (target, TARGET_DIALECT, arm)
            if k in done:
                print(f"[{i}/{planned}] skip {arm}")
                continue
            print(f"[{i}/{planned}] {arm}: {fte.EXPERIMENT_NAMES[ft_id]}", flush=True)
            try:
                r = train_one(TARGET_DIALECT, target, arm, ft_id, backbone)
                results.append(r)
                done[k] = r
                save()
                print(f"    best tau {r.get('best_tau'):.4f}", flush=True)
            except Exception as e:
                logging.error(f"run failed: {e}")
                print(f"    FAILED: {e}", flush=True)

    print(f"\nResults written to {RESULTS_PATH}")
    summarize(results)


def summarize(results):
    for target in dict.fromkeys(r["target"] for r in results):
        print(f"\n=== {target}")
        print(f"{'arm':>10} {'dialect':>11} {'n_train':>8} {'tau':>8} "
              f"{'rmse':>10} {'r2':>8}")
        rows = [r for r in results if r["target"] == target]
        order = {"source": 0, "scratch": 1}
        rows.sort(key=lambda r: (order.get(r["arm"], 2), r["arm"]))
        for r in rows:
            tau = r.get("best_tau")
            rmse = r.get("best_rmse")
            r2 = r.get("best_r2")
            print(
                f"{r['arm']:>10} {r['dialect']:>11} {r['n_train']:>8} "
                f"{tau if tau is None else f'{tau:.4f}':>8} "
                f"{rmse if rmse is None else f'{rmse:.4f}':>10} "
                f"{r2 if r2 is None else f'{r2:.4f}':>8}"
            )


if __name__ == "__main__":
    main()
