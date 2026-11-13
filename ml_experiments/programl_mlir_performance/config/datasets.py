"""
Layout of the published dataset and the model settings of each dialect.

    $SPHINX_DATA/nb101/all_metrics.json
    $SPHINX_DATA/nb101/graphs/{stablehlo,linalg,arith,llvm,linalg_full}/
    $SPHINX_DATA/nb201/all_metrics_cifar10-valid.json
    $SPHINX_DATA/nb201/graphs/{stablehlo,linalg,arith,llvm}/

max_edge_position is a property of the graphs of a dialect; the values below
were obtained with utils/find_max_edge_position.py. Vocabulary sizes are read
from the vocabulary files.
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

MAX_EDGE_POSITION = {"stablehlo": 8, "linalg": 6, "linalg_full": 6, "arith": 5, "llvm": 8}

VOCAB = {
    ("nb101", "stablehlo"): "vocab/FTE_programl_mlir_vocab_stablehlo.csv",
    ("nb101", "linalg"): "vocab/FTE_programl_mlir_vocab_linalg.csv",
    ("nb101", "linalg_full"): "vocab/ILE_programl_mlir_vocab_linalg.csv",
    ("nb101", "arith"): "vocab/FTE_programl_mlir_vocab_arith.csv",
    ("nb101", "llvm"): "vocab/FTE_programl_mlir_vocab_llvm.csv",
    ("nb201", "stablehlo"): "vocab/FTE_NasBench201_StableHLO.csv",
    ("nb201", "linalg"): "vocab/FTE_NasBench201_Linalg.csv",
    ("nb201", "arith"): "vocab/FTE_NasBench201_Arith.csv",
    ("nb201", "llvm"): "vocab/FTE_NasBench201_LLVM.csv",
}
LABELS = {"nb101": "all_metrics.json", "nb201": "all_metrics_cifar10-valid.json"}
SPLITS = {"nb101": "splits/finetune_experiment", "nb201": "splits/finetune_experiment_201"}


def vocab_size(path):
    """Number of entries preprocess.load_vocabulary_from_csv builds from a file."""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT, path)
    tokens = {"<unknown>"}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) > 3:
                tokens.add(row[3].strip())
    return len(tokens)


def profile(root, benchmark, dialect, target, pool=None, cache=None, vocab_dialect=None):
    """Settings for one benchmark/dialect/target as plain values.

    pool selects the split files: "stablehlo" is the 90% source pool and
    "linalg" the 10% target pool, whichever dialect the graphs are in.
    vocab_dialect overrides the vocabulary (used for zero-shot evaluation).
    """
    if pool is None:
        pool = "stablehlo" if dialect == "stablehlo" else "linalg"
    vdialect = vocab_dialect or dialect
    vocab = VOCAB[(benchmark, vdialect)]
    cache = cache or os.path.join(root, "processed")
    tag = f"{benchmark}_{dialect}_{target}" + (f"_vocab_{vocab_dialect}" if vocab_dialect else "")
    split = SPLITS[benchmark]
    return {
        "target_performance_metric": target,
        "max_edge_position": MAX_EDGE_POSITION[vdialect],
        "expected_vocab_size": vocab_size(vocab),
        "vocab": vocab,
        "dataset": os.path.join(root, benchmark),
        "graphs": os.path.join(root, benchmark, "graphs", dialect),
        "labels": os.path.join(root, benchmark, LABELS[benchmark]),
        "processed": os.path.join(cache, tag),
        "train_graphs_txt": f"{split}/train_files_{pool}.txt",
        "validation_graphs_txt": f"{split}/validation_files_{pool}.txt",
        "test_graphs_txt": f"{split}/test_files_{pool}.txt",
    }


def apply(params, settings):
    for key, value in settings.items():
        holder = params.model if hasattr(params.model, key) else params.paths
        setattr(holder, key, value)
