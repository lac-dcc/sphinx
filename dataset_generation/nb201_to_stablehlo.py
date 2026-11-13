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

# NB201 runs a separate module per edge, so two identical avg pools on the same
# source node cost time twice. Grappler would fold them into one.
tf.config.optimizer.set_experimental_options({
    "common_subgraph_elimination": False,
})


# -----------------------
# Configuration
# -----------------------
# Paths come from the environment (see README.md in this folder):
#   GEN_ROOT       working folder, default ./dataset_build
#   GEN_LIMIT      convert only the first N architectures (for a quick test)
#   STABLEHLO_OPT  path to stablehlo-opt, default: found on the PATH
GEN_ROOT = Path(os.environ.get("GEN_ROOT", "dataset_build")).resolve()
JSON_PATH = str(GEN_ROOT / "nb201" / "nasbench201_full.json")
METRICS_PATH = str(GEN_ROOT / "nb201")
OUTPUT_ROOT = str(GEN_ROOT / "nb201" / "mlir" / "stablehlo")
STABLEHLO_OPT_PATH = os.environ.get("STABLEHLO_OPT", "stablehlo-opt")

# Limit for test run (Set to None for full run)
TEST_LIMIT = int(os.environ["GEN_LIMIT"]) if os.environ.get("GEN_LIMIT") else None

# Which NAS-Bench-201 setting the emitted graphs correspond to. This fixes the
# input shape and the classifier width, and selects which metric block is
# written out, so the MLIR and its labels always describe the same network.
#
#   cifar10-valid  trained on 25k, has train / valid / test  <- NB101 analogue
#   cifar10        trained on 50k, no validation split
#   cifar100       100 classes
#   ImageNet16-120 16x16 input, 120 classes
DATASET = "cifar10-valid"

DATASET_SPECS = {
    "cifar10-valid": {"input_shape": (32, 32, 3), "num_classes": 10},
    "cifar10": {"input_shape": (32, 32, 3), "num_classes": 10},
    "cifar100": {"input_shape": (32, 32, 3), "num_classes": 100},
    "ImageNet16-120": {"input_shape": (16, 16, 3), "num_classes": 120},
}

INPUT_SHAPE = DATASET_SPECS[DATASET]["input_shape"]
NUM_CLASSES = DATASET_SPECS[DATASET]["num_classes"]

# None leaves the batch dimension dynamic, which is what the NAS-Bench-101 set
# was generated with: tensor<?x32x32x3xf32>, and a graph carrying shape.shape_of
# / shape.broadcast / dynamic_broadcast_in_dim plumbing. Keep it at None so both
# benchmarks share one convention. An int pins the batch instead, which removes
# all of that plumbing and produces a structurally different graph.
BATCH_SIZE = None

# Elide constants above this many elements. The NB101 set was generated with the
# weights inline, so None keeps the two sets consistent; set an int only if the
# downstream preprocessing drops constant payloads for both.
ELIDE_ELEMENTS_LARGER_THAN = 50

# 341 of the 15,625 cells have no live path from node 0 to node 3 once the
# 'none' edges are removed. Their output is zeros, so the whole network
# constant-folds to a single `return dense<0.0>` and carries no structure.
SKIP_DEAD_ARCHS = True

# Outputs are separated per dataset, since the graphs differ between them.
OUTPUT_DIR = Path(OUTPUT_ROOT)
METRICS_FILE = Path(METRICS_PATH) / f"all_metrics_{DATASET}.json"
ERRORS_FILE = Path(METRICS_PATH) / f"errors_{DATASET}.log"

# "full" reproduces the NAS-Bench-201 macro skeleton (stem + 3 stages of N cells
# separated by residual reduction blocks + head). "cell" emits stem + a single
# cell + head, which is the same size and shape as the NASBench-101 script.
MACRO_MODE = "cell"

INIT_CHANNELS = 16
CELLS_PER_STAGE = 5

TEMP_DIR_BASE = "/dev/shm" if os.path.exists("/dev/shm") else None

OP_NAMES = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")

# Field set carried into all_metrics.json, matching the NAS-Bench-101 run.
METRIC_FIELDS = (
    "trainable_parameters",
    "flops",
    "latency",
    "training_time",
    "training_time_per_epoch",
    "training_accuracy",
    "training_loss",
    "validation_accuracy",
    "validation_loss",
    "test_accuracy",
    "test_loss",
    "half_test_accuracy",
    "half_test_loss",
)


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
    Flattens the selected dataset's block onto the NB101 field names. The
    exporter has already normalised units, so accuracies are fractions and
    parameter and flop counts are raw integers.
    """
    block = rec.get(DATASET, {})

    metrics = {
        "hash": rec.get("hash", rec.get("arch_str", "unknown")),
        "arch_index": rec.get("arch_index"),
        "arch_str": rec.get("arch_str"),
        "dataset": DATASET,
    }

    for field in METRIC_FIELDS:
        metrics[field] = block.get(field)

    return metrics


def parse_arch_str(arch_str):
    """
    '|nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|+|...|'
    -> [[(op, src), ...], ...] with one entry per intermediate node 1..3.
    """
    nodes = []
    for node_str in arch_str.split("+"):
        edges = []
        for token in node_str.split("|"):
            token = token.strip()
            if not token:
                continue
            op, src = token.rsplit("~", 1)
            if op not in OP_NAMES:
                raise ValueError(f"unknown operation {op!r} in {arch_str!r}")
            edges.append((op, int(src)))
        if edges:
            nodes.append(edges)

    for j, edges in enumerate(nodes, start=1):
        if {src for _, src in edges} != set(range(j)):
            raise ValueError(f"node {j} has malformed edge set in {arch_str!r}")

    return nodes


def has_live_path(nodes):
    """True if the output node is reachable from the input over non-none edges."""
    live = {0}
    for j, edges in enumerate(nodes, start=1):
        if any(op != "none" and src in live for op, src in edges):
            live.add(j)
    return len(nodes) in live


# -----------------------
# Model construction
# -----------------------
def relu_conv_bn(channels, kernel, stride, name):
    """NAS-Bench-201 ReLUConvBN: activation first, no conv bias."""
    return tf.keras.Sequential(
        [
            layers.ReLU(name=f"{name}_relu"),
            layers.Conv2D(
                channels,
                kernel,
                strides=stride,
                padding="same",
                use_bias=False,
                name=f"{name}_conv",
            ),
            layers.BatchNormalization(name=f"{name}_bn"),
        ],
        name=name,
    )


def avg_pool_3x3(x, name):
    """
    Explicit pad + VALID pool rather than SAME pool. With SAME padding TF builds
    the divisor by reduce_window over a tensor of ones shaped like the input,
    which becomes an illegal dynamically-shaped mhlo.constant under a dynamic
    batch. VALID padding uses a scalar divisor instead.

    Side effect: border pixels divide by 9 including the padded zeros
    (count_include_pad=True), where the reference excludes them. Values only,
    not structure.
    """
    x = layers.ZeroPadding2D(padding=1, name=f"{name}_pad")(x)
    return layers.AveragePooling2D(3, strides=1, padding="valid", name=name)(x)


def build_cell(x, nodes, channels, prefix):
    """
    Fixed 4-node complete DAG. Node j is the sum of op(node_i) over all i < j,
    the cell output is node 3. Stride is always 1 inside a cell.

    Returns (output, intermediates). NAS-Bench-201 computes every node in the
    forward pass, so every non-'none' edge costs time even when its result never
    reaches node 3. Exporting only the final output lets TF and MLIR eliminate
    those ops, which understates the cost of 33.6% of the search space. The
    intermediates are returned so the caller can keep them alive as extra model
    outputs; that adds no computation, only return values.
    """
    node_outputs = [x]

    for j, edges in enumerate(nodes, start=1):
        incoming = []
        for op, src in edges:
            # 'none' contributes a zero tensor to the sum, so it is dropped from
            # the graph instead of materialising a multiply-by-zero.
            if op == "none":
                continue

            h = node_outputs[src]
            name = f"{prefix}_n{j}_e{src}_{op}"

            if op == "skip_connect":
                incoming.append(h)
            elif op == "avg_pool_3x3":
                incoming.append(avg_pool_3x3(h, name))
            else:
                kernel = 3 if op == "nor_conv_3x3" else 1
                incoming.append(relu_conv_bn(channels, kernel, 1, name)(h))

        if not incoming:
            node_outputs.append(
                layers.Lambda(tf.zeros_like, name=f"{prefix}_n{j}_zero")(node_outputs[0])
            )
        elif len(incoming) == 1:
            node_outputs.append(incoming[0])
        else:
            node_outputs.append(layers.Add(name=f"{prefix}_n{j}_add")(incoming))

    return node_outputs[-1], node_outputs[1:-1]


def resnet_basicblock(x, channels, stride, prefix):
    """Reduction block sitting between the cell stages."""
    main = relu_conv_bn(channels, 3, stride, f"{prefix}_conv_a")(x)
    main = relu_conv_bn(channels, 3, 1, f"{prefix}_conv_b")(main)

    shortcut = layers.AveragePooling2D(
        2, strides=stride, padding="valid", name=f"{prefix}_down_pool"
    )(x)
    shortcut = layers.Conv2D(
        channels, 1, strides=1, padding="valid", use_bias=False, name=f"{prefix}_down_conv"
    )(shortcut)

    return layers.Add(name=f"{prefix}_add")([main, shortcut])


def build_nasbench201_model(
    arch_str,
    macro_mode=MACRO_MODE,
    init_channels=INIT_CHANNELS,
    cells_per_stage=CELLS_PER_STAGE,
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
):
    nodes = parse_arch_str(arch_str)

    inp = layers.Input(shape=input_shape, batch_size=BATCH_SIZE, name="input_image")

    channels = init_channels
    x = layers.Conv2D(channels, 3, padding="same", use_bias=False, name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)

    # Kept alive as extra outputs so nothing downstream eliminates them.
    aux = []

    if macro_mode == "cell":
        x, extra = build_cell(x, nodes, channels, "cell")
        aux += extra
    elif macro_mode == "full":
        for stage in range(3):
            if stage > 0:
                channels *= 2
                x = resnet_basicblock(x, channels, 2, f"reduction_{stage}")
            for pos in range(cells_per_stage):
                x, extra = build_cell(x, nodes, channels, f"stage{stage}_cell{pos}")
                aux += extra
    else:
        raise ValueError(f"unknown MACRO_MODE {macro_mode!r}")

    x = layers.BatchNormalization(name="last_bn")(x)
    x = layers.ReLU(name="last_relu")(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    logits = layers.Dense(num_classes, activation=None, name="classifier")(x)

    return Model(inputs=inp, outputs=[logits] + aux, name="nasbench201_net")


def export_saved_model(model, saved_model_path):
    """Plain export for a dynamic batch, pinned input signature otherwise."""
    if BATCH_SIZE is None:
        model.export(saved_model_path)
        return

    spec = tf.TensorSpec(
        shape=(BATCH_SIZE,) + tuple(INPUT_SHAPE),
        dtype=tf.float32,
        name="input_image",
    )
    archive = tf.keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(name="serve", fn=model.call, input_signature=[spec])
    archive.write_out(saved_model_path)


# -----------------------
# Worker Function
# -----------------------
def process_architecture(args):
    idx, rec = args

    arch_str = rec.get("arch_str")
    arch_index = rec.get("arch_index", idx)

    if not arch_str:
        return False, f"model_{idx}", "record has no arch_str"

    model_name = f"model_{arch_index}"
    final_path = OUTPUT_DIR / f"{model_name}.mlir"

    clean_metrics = extract_metrics(rec)

    # Resume capability: if file exists and is valid, skip conversion
    if final_path.exists() and final_path.stat().st_size > 0:
        return True, model_name, clean_metrics

    try:
        # Long-lived workers build thousands of graphs; without this the Keras
        # global state grows until the process is killed.
        tf.keras.backend.clear_session()

        model = build_nasbench201_model(arch_str)

        with tempfile.TemporaryDirectory(dir=TEMP_DIR_BASE) as tmp_dir:
            saved_model_path = os.path.join(tmp_dir, "saved_model")

            with suppress_output():
                export_saved_model(model, saved_model_path)

            if tensorflow_to_stablehlo is None:
                return False, model_name, "No TF Binding"

            with suppress_output():
                bc_bytes = tensorflow_to_stablehlo.savedmodel_to_stablehlo(
                    input_path=saved_model_path
                )

            cmd = [STABLEHLO_OPT_PATH]
            if ELIDE_ELEMENTS_LARGER_THAN is not None:
                cmd.append(
                    f"--mlir-elide-elementsattrs-if-larger={ELIDE_ELEMENTS_LARGER_THAN}"
                )
            cmd.append("-")

            result = subprocess.run(
                cmd,
                input=bc_bytes,
                capture_output=True,
                check=True,
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

    if DATASET not in DATASET_SPECS:
        print(f"ERROR: unknown DATASET {DATASET!r}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset JSON...", end=" ", flush=True)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Accept either a list of records or a dict keyed by arch index / arch string.
    if isinstance(data, dict):
        data = list(data.values())
    print(f"Done. Found {len(data)} architectures.")

    if data and DATASET not in data[0]:
        print(
            f"WARNING: first record has no '{DATASET}' block; metrics will be null.",
            file=sys.stderr,
        )

    tasks = list(enumerate(data))

    if SKIP_DEAD_ARCHS:
        kept = []
        dropped = 0
        for idx, rec in tasks:
            try:
                if has_live_path(parse_arch_str(rec["arch_str"])):
                    kept.append((idx, rec))
                else:
                    dropped += 1
            except Exception:
                kept.append((idx, rec))
        tasks = kept
        print(f"Skipped {dropped} architectures with no live path to the output.")

    if TEST_LIMIT:
        print(f"DEBUG: Limiting run to first {TEST_LIMIT} models.")
        tasks = tasks[:TEST_LIMIT]

    max_workers = os.cpu_count() or 1
    num_workers = max(1, max_workers - 2)

    batch_desc = "dynamic batch" if BATCH_SIZE is None else f"batch {BATCH_SIZE}"
    print(
        f"Starting parallel conversion with {num_workers} workers "
        f"({MACRO_MODE} mode, {DATASET}, {batch_desc}, input {INPUT_SHAPE}, "
        f"{NUM_CLASSES} classes)."
    )
    print(f"Saving to: {OUTPUT_DIR}")

    all_metrics = {}
    errors = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for success, name, payload in tqdm(
            pool.imap_unordered(process_architecture, tasks, chunksize=4),
            total=len(tasks),
            unit="model",
        ):
            if success:
                all_metrics[name] = payload
            else:
                errors.append(f"{name}: {payload}")

    print(f"\nSaving consolidated metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=None)

    if errors:
        print(f"\nFinished with {len(errors)} errors.")
        with open(ERRORS_FILE, "w") as f:
            f.write("\n".join(errors))
        print(f"Error log written to {ERRORS_FILE}")
    else:
        print("\nFinished successfully with 0 errors.")


if __name__ == "__main__":
    main()
