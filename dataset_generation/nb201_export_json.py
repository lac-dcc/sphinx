"""
Export NAS-Bench-201 into a JSON file consumable by the StableHLO conversion
script, normalised to the NAS-Bench-101 field names and units.

Setup:
    pip install nas-bench-201
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    # download NAS-Bench-201-v1_1-096897.pth (4.7 GB)

Edit the configuration block below and hit Run. Loading the .pth takes several
minutes and a few GB of RAM. It only has to happen once.

Normalisation applied to the raw API output:
  - accuracies are divided by 100, so they are fractions in [0, 1] as in NB101
  - `params` and `flops` are multiplied by 1e6, so they are raw counts rather
    than millions
  - `test_accuracy` always refers to the FULL original test set. On cifar100 and
    ImageNet16-120 the API's `test-*` keys cover only half of it (the other half
    is the validation split) and the full set is under `valtest-*`; that is
    resolved here, with the half split kept as `half_test_accuracy`.
  - times and latency are seconds and pass through unchanged
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm


# -----------------------
# Configuration
# -----------------------
# Paths come from the environment (see README.md in this folder):
#   NB201_PTH   the NAS-Bench-201 file, default ./NAS-Bench-201-v1_1-096897.pth
#   GEN_ROOT    working folder, default ./dataset_build
#   GEN_LIMIT   export only the first N architectures (for a quick test)
import os
BENCHMARK_PATH = os.environ.get("NB201_PTH", "NAS-Bench-201-v1_1-096897.pth")
OUT_PATH = str(Path(os.environ.get("GEN_ROOT", "dataset_build")).resolve() / "nb201" / "nasbench201_full.json")

# Training budget. NAS-Bench-201 records only two: "12" (the short budget) and
# "200" (the full one behind every headline number in the paper). There is no
# NB201 equivalent of the 4/36 checkpoints in NAS-Bench-101; "200" is the
# analogue of the 108-epoch NB101 setting.
HP = "200"

# False averages over the available trials, True picks one at random.
RANDOM_TRIAL = False

# Debug: set to an int to export only the first N architectures.
LIMIT = int(os.environ["GEN_LIMIT"]) if os.environ.get("GEN_LIMIT") else None

DATASETS = ("cifar10-valid", "cifar10", "cifar100", "ImageNet16-120")

# torch >= 2.6 defaults torch.load to weights_only=True, which cannot unpickle
# the NAS-Bench-201 file. Only leave this on because the benchmark file comes
# from a source you trust.
ALLOW_UNSAFE_TORCH_LOAD = True


# -----------------------
# torch.load compatibility shim (must run before nas_201_api is imported)
# -----------------------
if ALLOW_UNSAFE_TORCH_LOAD:
    import torch

    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

from nas_201_api import NASBench201API as API


# -----------------------
# Normalisation helpers
# -----------------------
def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_fraction(value):
    """NB201 reports accuracy in percent, NB101 in [0, 1]."""
    value = as_float(value)
    return None if value is None else value / 100.0


def as_count(value):
    """NB201 reports params and flops in millions."""
    value = as_float(value)
    return None if value is None else int(round(value * 1e6))


def get_compute_costs(meta_info, dataset):
    """The API spells this method differently across versions."""
    for name in ("get_compute_costs", "get_comput_costs"):
        method = getattr(meta_info, name, None)
        if method is not None:
            return method(dataset)
    raise AttributeError("no compute-cost accessor on ArchResults")


def normalise(info, costs):
    """Map one dataset's raw API output onto the NB101 field names."""
    has_valtest = "valtest-accuracy" in info

    if has_valtest:
        # cifar100 / ImageNet16-120: `test-*` is half the original test set,
        # `valtest-*` is all of it.
        full_test_acc = info.get("valtest-accuracy")
        full_test_loss = info.get("valtest-loss")
        half_test_acc = info.get("test-accuracy")
        half_test_loss = info.get("test-loss")
    else:
        # cifar10 / cifar10-valid: `test-*` is already the full test set.
        full_test_acc = info.get("test-accuracy")
        full_test_loss = info.get("test-loss")
        half_test_acc = None
        half_test_loss = None

    return {
        "trainable_parameters": as_count(costs.get("params")),
        "flops": as_count(costs.get("flops")),
        "latency": as_float(costs.get("latency")),
        "training_time": as_float(info.get("train-all-time")),
        "training_time_per_epoch": as_float(info.get("train-per-time")),
        "training_accuracy": as_fraction(info.get("train-accuracy")),
        "training_loss": as_float(info.get("train-loss")),
        "validation_accuracy": as_fraction(info.get("valid-accuracy")),
        "validation_loss": as_float(info.get("valid-loss")),
        "test_accuracy": as_fraction(full_test_acc),
        "test_loss": as_float(full_test_loss),
        "half_test_accuracy": as_fraction(half_test_acc),
        "half_test_loss": as_float(half_test_loss),
    }


def collect_record(api, index):
    arch_str = api[index]

    record = {
        "arch_index": index,
        "arch_str": arch_str,
        # NB101 keyed records by module hash; arch_str is the NB201 equivalent
        # unique identifier, mirrored here so downstream code keyed on "hash"
        # keeps working.
        "hash": arch_str,
    }

    try:
        meta_info = api.query_meta_info_by_index(index, hp=HP)
    except TypeError:
        # Older API without the hp keyword.
        meta_info = api.query_meta_info_by_index(index)

    for dataset in DATASETS:
        try:
            info = api.get_more_info(
                index, dataset, iepoch=None, hp=HP, is_random=RANDOM_TRIAL
            )
        except Exception as e:
            record[dataset] = {"error": f"get_more_info: {e}"}
            continue

        try:
            costs = get_compute_costs(meta_info, dataset)
        except Exception:
            costs = {}

        record[dataset] = normalise(info, costs)

    return record


# -----------------------
# Main
# -----------------------
def main():
    print("Loading benchmark file (this takes a while)...", flush=True)
    api = API(BENCHMARK_PATH, verbose=False)
    total = len(api)
    print(f"Loaded {total} architectures. Querying with hp={HP}.")

    indices = range(total if LIMIT is None else min(LIMIT, total))

    records = []
    failures = 0
    for index in tqdm(indices, unit="arch"):
        try:
            records.append(collect_record(api, index))
        except Exception as e:
            failures += 1
            print(f"\n[{index}] failed: {e}", file=sys.stderr)

    if records:
        sample = records[0]
        print("\nSample record:")
        print(f"  arch_str: {sample['arch_str']}")
        for dataset in DATASETS:
            block = sample.get(dataset, {})
            print(
                f"  {dataset}: params={block.get('trainable_parameters')} "
                f"train={block.get('training_accuracy')} "
                f"valid={block.get('validation_accuracy')} "
                f"test={block.get('test_accuracy')}"
            )

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f)

    print(f"\nWrote {len(records)} records to {OUT_PATH} ({failures} failures).")


if __name__ == "__main__":
    main()
