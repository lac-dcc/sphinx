"""
Export NAS-Bench-101 into the JSON file read by nb101_to_stablehlo.py and
nb101_to_onnx.py.

Input: nasbench_only108.tfrecord (0.5 GB), the 108-epoch subset published with
NAS-Bench-101: https://github.com/google-research/nasbench

Output: one record per architecture, sorted by hash, with the adjacency matrix,
the operations, and the metrics after 108 epochs averaged over the three
training runs recorded in the benchmark:

    {"hash", "matrix", "ops", "trainable_parameters", "training_time",
     "training_accuracy", "validation_accuracy", "test_accuracy"}

The architecture at position i of the file becomes model_<i+1> in the dataset.

The file is read directly, without the nasbench package, which needs
TensorFlow 1. Each row of the tfrecord is a JSON list
[hash, epochs, adjacency, operations, metrics], where metrics is a
base64-encoded ModelMetrics protocol buffer (nasbench/lib/model_metrics.proto).

Paths come from the environment (see README.md in this folder):
  NB101_TFRECORD  the benchmark file, default ./nasbench_only108.tfrecord
  GEN_ROOT        working folder, default ./dataset_build
  GEN_LIMIT       export only the first N architectures (for a quick test)
"""

import base64
import json
import os
import struct
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tqdm import tqdm

TFRECORD_PATH = os.environ.get("NB101_TFRECORD", "nasbench_only108.tfrecord")
OUT_PATH = Path(os.environ.get("GEN_ROOT", "dataset_build")).resolve() / "nb101" / "nasbench_full.json"
LIMIT = int(os.environ["GEN_LIMIT"]) if os.environ.get("GEN_LIMIT") else None

EPOCHS = 108


# -----------------------
# Minimal protocol-buffer reader for ModelMetrics
# -----------------------
#   message ModelMetrics {
#     repeated EvaluationData evaluation_data = 1;
#     optional int32 trainable_parameters = 2;
#     optional double total_time = 3;
#   }
#   message EvaluationData {
#     optional double current_epoch = 1;
#     optional double training_time = 2;
#     optional double train_accuracy = 3;
#     optional double validation_accuracy = 4;
#     optional double test_accuracy = 5;
#     optional string checkpoint_path = 6;
#   }
def read_varint(buf, pos):
    value, shift = 0, 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def read_fields(buf):
    """Yields (field number, value) for every field of a serialized message."""
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, pos = read_varint(buf, pos)
        elif wire_type == 1:
            value = struct.unpack("<d", buf[pos:pos + 8])[0]
            pos += 8
        elif wire_type == 2:
            length, pos = read_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire_type == 5:
            value = struct.unpack("<f", buf[pos:pos + 4])[0]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        yield number, value


def parse_metrics(raw):
    evaluations, trainable_parameters = [], 0
    for number, value in read_fields(base64.b64decode(raw)):
        if number == 1:
            fields = dict(read_fields(value))
            evaluations.append({
                "training_time": fields.get(2, 0.0),
                "training_accuracy": fields.get(3, 0.0),
                "validation_accuracy": fields.get(4, 0.0),
                "test_accuracy": fields.get(5, 0.0),
            })
        elif number == 2:
            trainable_parameters = value
    # The last evaluation is the one at the end of training.
    return trainable_parameters, evaluations[-1]


# -----------------------
# Main
# -----------------------
def main():
    if not Path(TFRECORD_PATH).exists():
        print(f"ERROR: benchmark file not found at {TFRECORD_PATH}", file=sys.stderr)
        sys.exit(1)

    architectures = {}
    print("Reading the benchmark file...")
    for row in tqdm(tf.data.TFRecordDataset(TFRECORD_PATH), unit="row"):
        module_hash, epochs, adjacency, operations, metrics = json.loads(row.numpy().decode("utf-8"))
        if epochs != EPOCHS:
            continue

        if module_hash not in architectures:
            dim = int(round(len(adjacency) ** 0.5))
            architectures[module_hash] = {
                "matrix": [[int(adjacency[r * dim + c]) for c in range(dim)] for r in range(dim)],
                "ops": operations.split(","),
                "runs": [],
            }

        trainable_parameters, final = parse_metrics(metrics)
        architectures[module_hash]["trainable_parameters"] = trainable_parameters
        architectures[module_hash]["runs"].append(final)

    records = []
    for module_hash in sorted(architectures):
        arch = architectures[module_hash]
        runs = arch["runs"]
        record = {
            "hash": module_hash,
            "matrix": arch["matrix"],
            "ops": arch["ops"],
            "trainable_parameters": arch["trainable_parameters"],
        }
        for field in ("training_time", "training_accuracy", "validation_accuracy", "test_accuracy"):
            record[field] = sum(run[field] for run in runs) / len(runs)
        records.append(record)

    if LIMIT:
        records = records[:LIMIT]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f)

    print(f"Wrote {len(records)} architectures to {OUT_PATH}.")


if __name__ == "__main__":
    main()
