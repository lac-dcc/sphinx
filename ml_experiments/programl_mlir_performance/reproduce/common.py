"""
Shared helpers for the reproduce scripts.

Locations are taken from two environment variables:

    SPHINX_RESULTS  directory holding the shipped logs, JSONs and checkpoints
                    (default: <repository root>/results)
    SPHINX_DATA     directory holding the extracted dataset archives
                    (only needed by the scripts that evaluate checkpoints)

Naming note. The experiment ID in the scripts, logs, folder names and JSON
files is the paper's configuration number: ID 6 is FTE6.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.environ.get("SPHINX_RESULTS", os.path.join(REPO_ROOT, "results"))
OUTPUT = os.environ.get("SPHINX_OUTPUT", os.path.join(REPO_ROOT, "reproduced"))

_EPOCH = re.compile(r"Epoch (\d+)/(\d+) Results:")
_FIELDS = {
    "val_rmse": re.compile(r"Val RMSE:\s+([-\d.eE+]+)"),
    "val_r2": re.compile(r"Val R²:\s+([-\d.eE+]+)"),
    "val_tau": re.compile(r"Val Tau:\s+([-\d.eE+]+)"),
    "epoch_seconds": re.compile(r"Epoch time: .*\(([\d.]+) seconds\)"),
    "total_seconds": re.compile(r"Total time: .*\(([\d.]+) seconds\)"),
}
_EXPERIMENT = re.compile(r"EXPERIMENT ID: (\d+)")
_SAMPLES = re.compile(r"Found (\d+) samples")
_METRIC = re.compile(r"Metric:\s+(\w+)")
_TRAIN_SPLIT = re.compile(r"Initialized Dataset from (\S+)")


def results_path(*parts):
    return os.path.join(RESULTS, *parts)


def output_path(name):
    os.makedirs(OUTPUT, exist_ok=True)
    return os.path.join(OUTPUT, name)


def parse_train_log(path):
    """Parse a train.log written by GGNN_train.train_by_epoch.

    Returns a dict with the run header (experiment id, training-set
    size, target metric, training split) and one entry per epoch holding the
    validation metrics and the timings.
    """
    run = {"path": path, "experiment_id": None, "n_train": None,
           "metric": None, "train_split": None, "epochs": []}
    current = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _EPOCH.search(line)
            if m:
                current = {"epoch": int(m.group(1))}
                run["epochs"].append(current)
                continue
            if current is None:
                if run["experiment_id"] is None:
                    m = _EXPERIMENT.search(line)
                    if m:
                        run["experiment_id"] = int(m.group(1))
                if run["n_train"] is None:
                    m = _SAMPLES.search(line)
                    if m:
                        run["n_train"] = int(m.group(1))
                if run["train_split"] is None:
                    m = _TRAIN_SPLIT.search(line)
                    if m:
                        run["train_split"] = m.group(1)
                if run["metric"] is None:
                    m = _METRIC.search(line)
                    if m:
                        run["metric"] = m.group(1)
                continue
            for key, pattern in _FIELDS.items():
                if key not in current:
                    m = pattern.search(line)
                    if m:
                        current[key] = float(m.group(1))
    run["epochs"] = [e for e in run["epochs"] if "val_tau" in e and "total_seconds" in e]
    return run


def load_run(*parts):
    path = results_path(*parts, "train.log")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Set SPHINX_RESULTS to the extracted results archive."
        )
    return parse_train_log(path)


def peak(run, key="val_tau"):
    best = max if key != "val_rmse" else min
    return best(e[key] for e in run["epochs"])


def mean_epoch_seconds(run):
    values = [e["epoch_seconds"] for e in run["epochs"]]
    return sum(values) / len(values)


def total_seconds(run):
    return run["epochs"][-1]["total_seconds"]


def fmt_short(seconds):
    """1h42m / 52m04s / 31s, as in Table 2."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_long(seconds):
    """1h 38m 52s / 9m 39s, as in the time-to-performance table."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def print_table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
