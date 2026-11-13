import json
import os
import sys
from pathlib import Path
import multiprocessing
from contextlib import contextmanager
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, Model

import tf2onnx
import onnx

# -----------------------
# Configuration
# -----------------------
# Paths come from the environment (see README.md in this folder):
#   GEN_ROOT       working folder, default ./dataset_build
#   GEN_LIMIT      convert only the first N architectures (for a quick test)
GEN_ROOT = Path(os.environ.get("GEN_ROOT", "dataset_build")).resolve()

JSON_PATH = GEN_ROOT / "nb201" / "nasbench201_full.json"
METRICS_PATH = GEN_ROOT / "nb201" / "onnx"
OUTPUT_ROOT = GEN_ROOT / "nb201" / "onnx"

# Limit for test run (Set to None for full run)
TEST_LIMIT = int(os.environ["GEN_LIMIT"]) if os.environ.get("GEN_LIMIT") else None

DATASET = "cifar10-valid"

DATASET_SPECS = {
    "cifar10-valid": {"input_shape": (32, 32, 3), "num_classes": 10},
    "cifar10": {"input_shape": (32, 32, 3), "num_classes": 10},
    "cifar100": {"input_shape": (32, 32, 3), "num_classes": 100},
    "ImageNet16-120": {"input_shape": (16, 16, 3), "num_classes": 120},
}

INPUT_SHAPE = DATASET_SPECS[DATASET]["input_shape"]
NUM_CLASSES = DATASET_SPECS[DATASET]["num_classes"]

# Matches the NB101 ONNX script, which also pinned the Keras input at 1.
# tf2onnx still emits a symbolic batch dimension downstream.
BATCH_SIZE = 1

MACRO_MODE = "cell"
INIT_CHANNELS = 16
CELLS_PER_STAGE = 5

SKIP_DEAD_ARCHS = True

OP_NAMES = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")

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
    """Explicit pad + VALID pool, matching the StableHLO route's builder."""
    x = layers.ZeroPadding2D(padding=1, name=f"{name}_pad")(x)
    return layers.AveragePooling2D(3, strides=1, padding="valid", name=name)(x)


def build_cell(x, nodes, channels, prefix):
    """
    Returns (output, intermediates). NAS-Bench-201 computes every node in the
    forward pass, so every non-'none' edge costs time even when its result never
    reaches node 3. Exporting only the final output lets those ops be eliminated,
    which understates the cost of 33.6% of the search space. The intermediates
    are returned so the caller can keep them alive as extra model outputs; that
    adds no computation, only return values.
    """
    node_outputs = [x]

    for j, edges in enumerate(nodes, start=1):
        incoming = []
        for op, src in edges:
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

    inp = layers.Input(batch_size=BATCH_SIZE, shape=input_shape, name="input_image")

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
    final_path = OUTPUT_ROOT / f"{model_name}.onnx"

    clean_metrics = extract_metrics(rec)

    # Resume capability: if file exists and is valid, skip conversion
    if final_path.exists() and final_path.stat().st_size > 0:
        return True, model_name, clean_metrics

    try:
        with suppress_output():
            model = build_nasbench201_model(arch_str)

            final_path.parent.mkdir(parents=True, exist_ok=True)
            tf2onnx.convert.from_keras(model, output_path=str(final_path))
            tf.keras.backend.clear_session()

        return True, model_name, clean_metrics

    except Exception as e:
        return False, model_name, str(e)


# -----------------------
# Main
# -----------------------
def main():
    multiprocessing.set_start_method("spawn", force=True)

    if not JSON_PATH.exists():
        print(f"ERROR: JSON file not found at {JSON_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading dataset JSON...", end=" ", flush=True)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    print(f"Done. Found {len(data)} architectures.")

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

    print(f"Starting parallel conversion with {num_workers} workers ({MACRO_MODE} mode, {DATASET}).")
    print(f"Saving to: {OUTPUT_ROOT}")

    all_metrics = {}
    errors = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for success, name, payload in tqdm(
                pool.imap_unordered(process_architecture, tasks, chunksize=4),
                total=len(tasks),
                unit="model"
        ):
            if success:
                all_metrics[name] = payload
            else:
                errors.append(f"{name}: {payload}")

    metrics_path = METRICS_PATH / f"all_metrics_onnx_{DATASET}.json"
    print(f"\nSaving consolidated metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=None)

    if errors:
        print(f"\nFinished with {len(errors)} errors.")
        errors_path = METRICS_PATH / f"errors_onnx_{DATASET}.log"
        with open(errors_path, "w") as f:
            f.write("\n".join(errors))
        print(f"Error log written to {errors_path}")
    else:
        print("\nFinished successfully with 0 errors.")


if __name__ == "__main__":
    main()
