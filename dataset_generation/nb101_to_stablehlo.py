import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import multiprocessing
from contextlib import contextmanager
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.compiler.mlir.tensorflow_to_stablehlo.python import (
    pywrap_tensorflow_to_stablehlo as tensorflow_to_stablehlo,
)


# -----------------------
# Configuration
# -----------------------
# Paths come from the environment (see README.md in this folder):
#   GEN_ROOT       working folder, default ./dataset_build
#   GEN_LIMIT      convert only the first N architectures (for a quick test)
#   STABLEHLO_OPT  path to stablehlo-opt, default: found on the PATH
GEN_ROOT = Path(os.environ.get("GEN_ROOT", "dataset_build")).resolve()
JSON_PATH = str(GEN_ROOT / "nb101" / "nasbench_full.json")
METRICS_PATH = str(GEN_ROOT / "nb101")
OUTPUT_ROOT = str(GEN_ROOT / "nb101" / "mlir" / "stablehlo")
STABLEHLO_OPT_PATH = os.environ.get("STABLEHLO_OPT", "stablehlo-opt")

# Limit for test run (Set to None for full run)
TEST_LIMIT = int(os.environ["GEN_LIMIT"]) if os.environ.get("GEN_LIMIT") else None

NUM_FILTERS = 16
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
TEMP_DIR_BASE = "/dev/shm" if os.path.exists("/dev/shm") else None


# -----------------------
# Helpers
# -----------------------
@contextmanager
def suppress_output():
    """Redirects C++ and Python logs to /dev/null."""
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(sys.stdout.fileno())
        old_stderr = os.dup(sys.stderr.fileno())
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout, sys.stdout.fileno())
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stdout)
            os.close(old_stderr)


def extract_metrics(rec):
    """
    Extracts pre-calculated metrics directly from the flattened JSON record.
    """
    return {
        "hash": rec.get("hash", "unknown"),
        "trainable_parameters": rec.get("trainable_parameters", 0),
        "training_time": rec.get("training_time", 0.0),
        "training_accuracy": rec.get("training_accuracy", 0.0),
        "validation_accuracy": rec.get("validation_accuracy", 0.0),
        "test_accuracy": rec.get("test_accuracy", 0.0)
    }


def build_nasbench101_model(adj, ops, num_filters=NUM_FILTERS, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    num_nodes = len(ops)
    inp = layers.Input(shape=input_shape, name="input_image")

    stem = tf.keras.Sequential([
        layers.Conv2D(num_filters, 3, padding="same", use_bias=False, name="stem_conv"),
        layers.BatchNormalization(name="stem_bn"),
        layers.ReLU(name="stem_relu"),
    ], name="stem")

    node_outputs = [None] * num_nodes
    node_outputs[0] = stem(inp)

    node_ops = [None] * num_nodes
    for i, op in enumerate(ops):
        if i == 0 or i == num_nodes - 1: continue

        if op == "conv3x3-bn-relu":
            node_ops[i] = tf.keras.Sequential([
                layers.Conv2D(num_filters, 3, padding="same", use_bias=False, name=f"conv3x3_{i}"),
                layers.BatchNormalization(name=f"bn_{i}"),
                layers.ReLU(name=f"relu_{i}"),
            ], name=f"conv3x3_{i}")
        elif op == "conv1x1-bn-relu":
            node_ops[i] = tf.keras.Sequential([
                layers.Conv2D(num_filters, 1, padding="same", use_bias=False, name=f"conv1x1_{i}"),
                layers.BatchNormalization(name=f"bn_{i}"),
                layers.ReLU(name=f"relu_{i}"),
            ], name=f"conv1x1_{i}")
        elif op == "maxpool3x3":
            node_ops[i] = layers.MaxPool2D(pool_size=3, strides=1, padding="same", name=f"maxpool_{i}")

    import numpy as np
    adj_arr = np.array(adj, dtype=int)
    for i in range(1, num_nodes):
        parents = list(np.where(adj_arr[:, i] == 1)[0])

        if len(parents) == 0:
            incoming = tf.zeros_like(node_outputs[0])
        elif len(parents) == 1:
            incoming = node_outputs[parents[0]]
        else:
            parent_tensors = [node_outputs[p] for p in parents]
            if i == num_nodes - 1:
                incoming = layers.Concatenate(name="concat_out")(parent_tensors)
            else:
                incoming = layers.Add(name=f"add_{i}")(parent_tensors)

        if i == num_nodes - 1:
            node_outputs[i] = incoming
        else:
            node_outputs[i] = node_ops[i](incoming)

    x = node_outputs[-1]
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    logits = layers.Dense(num_classes, activation=None, name="classifier")(x)

    return Model(inputs=inp, outputs=logits, name="nasbench101_cell")


# -----------------------
# Worker Function
# -----------------------
def process_architecture(args):
    idx, rec = args
    model_name = f"model_{idx + 1}"
    final_path = Path(OUTPUT_ROOT) / f"{model_name}.mlir"

    clean_metrics = extract_metrics(rec)

    # Resume capability: if file exists and is valid, skip conversion
    if final_path.exists() and final_path.stat().st_size > 0:
        return True, model_name, clean_metrics

    try:
        adj = rec["matrix"]
        ops = rec["ops"]

        model = build_nasbench101_model(adj, ops)

        with tempfile.TemporaryDirectory(dir=TEMP_DIR_BASE) as tmp_dir:
            saved_model_path = os.path.join(tmp_dir, "saved_model")

            with suppress_output():
                model.export(saved_model_path)

            if tensorflow_to_stablehlo is None:
                return False, model_name, "No TF Binding"

            with suppress_output():
                bc_bytes = tensorflow_to_stablehlo.savedmodel_to_stablehlo(input_path=saved_model_path)

            result = subprocess.run(
                [STABLEHLO_OPT_PATH, "-"],
                input=bc_bytes,
                capture_output=True,
                check=True
            )
            mlir_text = result.stdout

            with open(final_path, "wb") as f:
                f.write(mlir_text)

        return True, model_name, clean_metrics

    except Exception as e:
        return False, model_name, str(e)


# -----------------------
# Main
# -----------------------
def main():
    if not Path(JSON_PATH).exists():
        print(f"ERROR: JSON file not found at {JSON_PATH}", file=sys.stderr)
        sys.exit(1)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    print("Loading dataset JSON...", end=" ", flush=True)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    print(f"Done. Found {len(data)} architectures.")

    tasks = list(enumerate(data))

    if TEST_LIMIT:
        print(f"DEBUG: Limiting run to first {TEST_LIMIT} models.")
        tasks = tasks[:TEST_LIMIT]

    max_workers = os.cpu_count() or 1
    num_workers = max(1, max_workers - 2)

    print(f"Starting parallel conversion with {num_workers} workers.")
    print(f"Saving to: {OUTPUT_ROOT}")

    all_metrics = {}
    errors = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for success, name, payload in tqdm(
                pool.imap_unordered(process_architecture, tasks, chunksize=50),
                total=len(tasks),
                unit="model"
        ):
            if success:
                all_metrics[name] = payload
            else:
                errors.append(f"{name}: {payload}")

    metrics_path = Path(METRICS_PATH) / "all_metrics.json"
    print(f"\nSaving consolidated metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=None)

    if errors:
        print(f"\nFinished with {len(errors)} errors.")
    else:
        print("\nFinished successfully with 0 errors.")


if __name__ == "__main__":
    main()