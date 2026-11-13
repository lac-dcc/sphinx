# How the dataset was generated

These are the scripts that turned NAS-Bench-101 and NAS-Bench-201 into the MLIR programs of the dataset. Nothing in the paper's results needs them: the programs and their graphs are in the dataset archives. They are here so that the dataset can be inspected, extended or rebuilt.

## The steps

```
NAS-Bench file ──> JSON of architectures ──> Keras model ──┬──> stablehlo ──> linalg
                                                           │
                                                           └──> ONNX ──┬──> scf/memref/arith
                                                                       └──> llvm
```

| Step | Script | Environment |
|---|---|---|
| NAS-Bench-101 file to JSON | `nb101_export_json.py` | stablehlo |
| NAS-Bench-201 file to JSON | `nb201_export_json.py` | stablehlo |
| JSON to stablehlo, and the labels | `nb101_to_stablehlo.py`, `nb201_to_stablehlo.py` | stablehlo |
| stablehlo to linalg | `stablehlo_to_linalg.sh` | `stablehlo-opt` only |
| JSON to ONNX | `nb101_to_onnx.py`, `nb201_to_onnx.py` | onnx |
| ONNX to scf/memref/arith | `onnx_to_arith.py` | onnx |
| ONNX to llvm | `onnx_to_llvm.py` | onnx |

The `stablehlo` and `linalg` programs come from TensorFlow's StableHLO exporter, and `linalg` is a direct lowering of `stablehlo`. The `scf/memref/arith` and `llvm` programs take a different road: the same Keras model is exported to ONNX, imported with torch-mlir, and lowered with `mlir-opt`. They are therefore not a lowering of the `stablehlo` files, but an independent compilation of the same networks.

Each architecture keeps one name through every step (`model_<n>`), which is what ties a program in one dialect to the same network in another and to its labels.

## Environments

Two Python environments are needed, because the two exporters need different versions:

```bash
python3.12 -m venv env-stablehlo && env-stablehlo/bin/pip install -r requirements-stablehlo.txt
python3.11 -m venv env-onnx      && env-onnx/bin/pip install -r requirements-onnx.txt
```

And two tools:

- `stablehlo-opt`, from StableHLO at commit `1ef9e390`. The `sphinx-tool` image (see `ARTIFACT.md`) has it on the PATH.
- `mlir-opt` from LLVM 20.1.6, for the ONNX route.

## Settings

All scripts read and write under one working folder and take their settings from environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `GEN_ROOT` | working folder | `./dataset_build` |
| `GEN_LIMIT` | convert only the first N architectures | all |
| `GEN_BENCHMARK` | `nb101` or `nb201`, for `onnx_to_arith.py` and `onnx_to_llvm.py` | `nb101` |
| `STABLEHLO_OPT`, `MLIR_OPT` | paths to the two tools | found on the PATH |
| `NB101_TFRECORD` | `nasbench_only108.tfrecord` from [NAS-Bench-101](https://github.com/google-research/nasbench) (0.5 GB) | `./nasbench_only108.tfrecord` |
| `NB201_PTH` | `NAS-Bench-201-v1_1-096897.pth` from [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) (4.7 GB) | `./NAS-Bench-201-v1_1-096897.pth` |

The working folder ends up like this, for each benchmark:

```
dataset_build/nb101/
  nasbench_full.json      architectures and metrics
  all_metrics.json        labels, one entry per model
  mlir/stablehlo/         model_<n>.mlir
  mlir/linalg/
  onnx/                   model_<n>.onnx
  mlir/arith/
  mlir/llvm/
```

## Running it

A quick run on the first 100 architectures of NAS-Bench-101 takes a few minutes:

```bash
export GEN_LIMIT=100
env-stablehlo/bin/python nb101_export_json.py
env-stablehlo/bin/python nb101_to_stablehlo.py
./stablehlo_to_linalg.sh nb101

env-onnx/bin/python nb101_to_onnx.py
source env-onnx/bin/activate        # puts the torch-mlir tools on the PATH
python onnx_to_arith.py
python onnx_to_llvm.py
```

For NAS-Bench-201, use the `nb201_` scripts, `./stablehlo_to_linalg.sh nb201` and `GEN_BENCHMARK=nb201`.

To turn the programs into graphs, run the graph generator on a folder with an `mlir/` subfolder (see the main `README.md`).

Without `GEN_LIMIT` the scripts process a whole benchmark, which takes many hours for NAS-Bench-101 (423,624 architectures). In the dataset, `stablehlo` covers all architectures, while `linalg`, `scf/memref/arith` and `llvm` cover the 10% of them listed in the split files; `linalg_full` is `linalg` for all of them. NAS-Bench-201 has 15,284 architectures because cells with no path from input to output are skipped.

## What to expect when regenerating

We reran each step on a sample (200 architectures of NAS-Bench-101 and 150 of NAS-Bench-201, unless noted) and compared with the files the dataset was built from:

| Step | NAS-Bench-101 | NAS-Bench-201 |
|---|---|---|
| JSON of architectures | identical (all 423,624) | identical (30 architectures) |
| labels | identical | identical |
| stablehlo | identical apart from the weights | identical |
| linalg | identical | identical |
| scf/memref/arith, llvm | identical | 142 of 150 identical |

- `nb101_export_json.py` was rewritten for this artifact, because the original script was not kept. Its output is identical to the JSON the dataset was built from.
- The networks are built with freshly initialised weights, so weight values differ on every run. They are elided in the dataset and are not read by the model.
- The ONNX export does not always list the operations of a NAS-Bench-201 cell in the same order. In the 8 files that differed, the program has the same operations in a different order, in some cases with one buffer allocation more or less.
