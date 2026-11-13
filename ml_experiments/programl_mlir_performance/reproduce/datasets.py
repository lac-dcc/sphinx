"""
Point the global params at one benchmark/dialect/target of the published
dataset (layout and settings: config/datasets.py), and preprocess only the
graphs a script needs.
"""

import json
import multiprocessing
import os

from config import datasets as layout
from config.params import params


def data_root():
    root = os.environ.get("SPHINX_DATA")
    if not root:
        raise SystemExit("Set SPHINX_DATA to the directory holding the extracted dataset.")
    return root


def configure(benchmark, dialect, target, pool, vocab_dialect=None):
    settings = layout.profile(data_root(), benchmark, dialect, target, pool=pool,
                              cache=os.environ.get("SPHINX_CACHE"),
                              vocab_dialect=vocab_dialect)
    layout.apply(params, settings)


def preprocess_split(split_file):
    """Preprocess the graphs named in one split file (preprocess.run_preprocessing
    converts a whole directory, which is more than an evaluation needs)."""
    import preprocess

    os.makedirs(params.paths.processed, exist_ok=True)
    with open(params.paths.labels) as f:
        labels = json.load(f)
    preprocess.load_vocabulary_from_csv(params.paths.vocab)
    target = params.model.target_performance_metric
    tasks = []
    with open(split_file) as f:
        for line in f:
            name = line.strip().replace(".pt", "")
            if not name:
                continue
            out = os.path.join(params.paths.processed, f"{name}.pt")
            if not os.path.exists(out):
                tasks.append((name, os.path.join(params.paths.graphs, f"{name}.ProgramGraph.pb"),
                              out, labels[name][target], params.model.num_edge_types,
                              params.model.max_edge_position))
    if tasks:
        with multiprocessing.Pool(params.environment.num_workers) as pool:
            results = pool.map(preprocess.process_single_graph, tasks, chunksize=256)
        failed = sum(r not in ("processed", "skipped") for r in results)
        if failed:
            raise SystemExit(f"{failed} graphs of {split_file} could not be preprocessed")
