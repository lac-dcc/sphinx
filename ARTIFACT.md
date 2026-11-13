# Artifact: Transfer Learning Across MLIR Dialects (CGO 2027)

This artifact reproduces the tables and figures of the paper from the logs, checkpoints and datasets of the original experiments.

It supports three levels of reproduction:

| Level | What it does | Needs | Time |
|---|---|---|---|
| 1. Results | Rebuilds every table and figure from the shipped training logs and JSON files | Python, no GPU, no download | under 1 minute |
| 2. Checkpoints | Re-evaluates the shipped models on the test sets (Table 2 test metrics) | `results.tar.zst` and dataset archives, GPU recommended | about 5 minutes |
| 3. Training | Re-runs pre-training and fine-tuning | dataset archives, GPU | hours, see below |

## Suggested path for reviewers

This table is only an overview of what can be run in a given amount of time. The commands for each row are given step by step further down: first [Setup](#setup), then one section per level ([Level 1](#level-1-tables-and-figures-from-the-shipped-results), [Level 2](#level-2-re-evaluating-the-shipped-checkpoints), [Level 3](#level-3-training)).

| Time | What to run | What it checks |
|---|---|---|
| 1 minute, no GPU | Level 1 | Every table and figure is rebuilt from the original logs |
| 5 minutes on a GPU (about 2 hours on a CPU), 25 GB of disk | Level 2 with `--source` | The metrics of Table 2 are recomputed from the trained models and match to the printed precision |
| 1 GPU-hour, 7 GB of disk | Level 3, fine-tuning of FTE6 on linalg | One column of Table 2 is retrained from the shipped source model |
| 2.5 GPU-hours, 6 GB of disk | Level 3, `pipeline_9010.py` on NAS-Bench-201 | A whole experiment is retrained end to end: source model, scratch baseline and the eight configurations, for both tasks |
| about 20 GPU-hours, 32 GB GPU | Level 3, all of Table 2 | Table 2 is retrained in full: the two source models, the two linalg baselines and the eight configurations, for both tasks |

Retrained models do not reproduce the reported numbers exactly; the claims to check are the gap to the scratch baseline and the broad ordering of the configurations. As a reference, our own reruns on the paper's hardware gave:

- FTE6 on NAS-Bench-101 linalg (training time): best validation τ 0.9650, against 0.9626 in the shipped run.
- `pipeline_9010.py`: the 20 runs differ from the shipped ones by 0.017 in validation τ on average and by at most 0.045. The pipeline is seeded, but GPU training is not bit-reproducible, and the fine-tuned runs inherit the differences of the retrained source model. In the rerun, as in the shipped results, all eight configurations beat the scratch baseline on test accuracy (0.56 to 0.71 against 0.55), and FTE2, FTE3, FTE4 and FTE8 beat it on training time. Which single configuration comes first changes between runs.

## Contents

- `mlir_to_programl/`: the C++ tool that converts MLIR to ProGraML graphs.
- `ml_experiments/programl_mlir_performance/`: the GGNN model, training and experiment scripts.
  - `reproduce/`: one entry point per table and figure (this artifact's workflow).
  - `splits/`, `vocab/`: the exact split files and vocabularies of the paper.
- `dataset_generation/`: the scripts that turned NAS-Bench-101 and NAS-Bench-201 into MLIR programs, with their own README. Not needed for any result.
- `results/`: the training logs, result JSON files and figure PDFs of every experiment in the paper (28 MB). They are in the repository, so level 1 works right after a clone. The model checkpoints are not; they come from `results.tar.zst`.
- `artifact/`: the tool that built the dataset archives, the Dockerfiles and pinned requirements.
- Archives (Zenodo, https://doi.org/10.5281/zenodo.24175862):
  - `results.tar.zst`: the `results/` folder of the repository plus the best checkpoints of every run (1,110 files, 0.7 GB).
  - 18 dataset archives and `labels.tar.zst`: ProGraML graphs and MLIR sources of NAS-Bench-101 and NAS-Bench-201 in every dialect, with checksums in `MANIFEST.json`.

The archives are small to download but large once extracted, because the graphs compress very well:

| Archive | Download | Extracted | Files |
|---|---|---|---|
| `results.tar.zst` | 0.3 GB | 0.75 GB | 1,500 |
| `labels.tar.zst` | 16 MB | 0.1 GB | 2 |
| `nb101_graphs_stablehlo` | 67 MB | 11.2 GB | 381,261 |
| `nb101_graphs_linalg` | 17 MB | 3.0 GB | 42,363 |
| `nb101_graphs_arith` | 22 MB | 2.3 GB | 42,363 |
| `nb101_graphs_llvm` | 265 MB | 9.4 GB | 42,363 |
| `nb101_graphs_linalg_full` | 168 MB | 30.2 GB | 423,624 |
| `nb101_mlir_stablehlo` | 31 MB | 4.9 GB | 423,624 |
| `nb101_mlir_linalg` | 5 MB | 0.9 GB | 42,363 |
| `nb101_mlir_arith` | 8 MB | 0.8 GB | 42,363 |
| `nb101_mlir_llvm` | 52 MB | 3.3 GB | 42,363 |
| `nb101_mlir_linalg_full` | 45 MB | 9.5 GB | 423,624 |
| `nb201_graphs_stablehlo` | 3 MB | 0.4 GB | 15,284 |
| `nb201_graphs_linalg` | 8 MB | 1.1 GB | 15,284 |
| `nb201_graphs_arith` | 8 MB | 0.8 GB | 15,284 |
| `nb201_graphs_llvm` | 90 MB | 3.1 GB | 15,284 |
| `nb201_mlir_stablehlo` | 1 MB | 0.2 GB | 15,284 |
| `nb201_mlir_linalg` | 2 MB | 0.3 GB | 15,284 |
| `nb201_mlir_arith` | 3 MB | 0.3 GB | 15,284 |
| `nb201_mlir_llvm` | 18 MB | 1.1 GB | 15,284 |
| All of them | 1.1 GB | 84 GB | 2.0 million |

Nothing needs all of them at once. Level 1 needs no archive. Level 2 needs `results.tar.zst` for the checkpoints, `labels` and `nb101_graphs_linalg`, plus `nb101_graphs_stablehlo` for the `--source` columns. Training needs `labels` and the graph archives of the dialects involved. The `mlir` archives are the program sources and are not read by any script; they are there for inspection and to regenerate graphs.

## Setup

There are two ways to set up the environment. Pick one; the commands of the three levels are the same in both.

### Option A: with Docker

Build the image from the repository root and start a container, mounting the folder where you downloaded the archives:

```bash
docker build -t sphinx-artifact -f artifact/Dockerfile .
docker run --rm -it --gpus all --shm-size=8g -v /path/to/archives:/archives sphinx-artifact
```

- `--gpus all` needs the NVIDIA Container Toolkit. Drop it to run on the CPU.
- `--shm-size=8g` is needed by the data-loader workers.
- The image is 8 GB. Inside the container the repository is at `/sphinx`, the archives are at `/archives`, and `SPHINX_DATA` is already set to `/sphinx/data`.

Level 1 can be run right away. For levels 2 and 3, extract the checkpoints and the dataset archives you need inside the container:

```bash
cd /sphinx && zstd -dc /archives/results.tar.zst | tar -x
mkdir -p /sphinx/data && for f in labels nb101_graphs_linalg; do zstd -dc /archives/$f.tar.zst | tar -x -C /sphinx/data; done
cd /sphinx/ml_experiments/programl_mlir_performance
```

### Option B: without Docker

You need Python 3.12 and the `zstd` tool. From the repository root:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r artifact/requirements.txt
```

Level 1 can be run right away. For levels 2 and 3, extract the checkpoints into the repository root (they go into the existing `results/` folder), then the dataset archives you need into one folder, and point `SPHINX_DATA` at it:

```bash
zstd -dc /path/to/archives/results.tar.zst | tar -x
mkdir data && for f in labels nb101_graphs_linalg; do zstd -dc /path/to/archives/$f.tar.zst | tar -x -C data; done
export SPHINX_DATA=$PWD/data
cd ml_experiments/programl_mlir_performance
```

### In both options

- All commands of the three levels run from `ml_experiments/programl_mlir_performance/`.
- The two archives above (`labels`, `nb101_graphs_linalg`; 3 GB extracted) are enough for the default level 2. Add other archives to the `for` loop as the later steps ask for them; the table in [Contents](#contents) gives their extracted sizes.

### Optional: the graph generator

The graph generator has an image of its own. No result of the paper needs it, since the generated graphs are in the dataset archives. The image builds LLVM/MLIR 22 and StableHLO from source at the versions used for the paper (12 minutes on 64 cores, about 10 GB of disk during the build, 0.8 GB image) and checks the tool on the sample programs:

```bash
docker build -t sphinx-tool -f artifact/Dockerfile.tool .
docker run --rm sphinx-tool artifact/check_tool.sh
```

Expected: the 16 linalg and 16 stablehlo graphs it writes are identical to the shipped ones in `sample_programs/`. To convert other programs, mount a folder that has an `mlir/` subfolder; the graphs are written to `graphs/` next to it:

```bash
docker run --rm -v /path/to/folder:/work sphinx-tool mlir_to_programl /work
```

## Level 1: tables and figures from the shipped results

```bash
python -m reproduce.table2       # Table 2
python -m reproduce.table_time   # time-to-performance table
python -m reproduce.table3       # NAS-Bench-201 table
python -m reproduce.figures      # all figures, written to reproduced/
```

| Paper item | Command | Source in `results/` |
|---|---|---|
| Table 2 (NAS-Bench-101, FTE1 to FTE8) | `reproduce.table2` | `1.*`, `3.*`, `test_metrics.json` |
| Figure, RQ2 (1%, 5%, 10%) | `reproduce.figures rq2` | `2.*`, `4.*` |
| Figure, RQ3 (all configurations) | `reproduce.figures rq3` | `1.*`, `3.*` |
| Figure, RQ5 (time per epoch) | `reproduce.figures rq5` | `5.TimeComparison/TimeComparison` |
| Time-to-performance table | `reproduce.table_time` | `5.TimeComparison` |
| Figure, RQ6 (scf/memref/arith, llvm) | `reproduce.figures rq6` | `6.*` to `9.*` |
| NAS-Bench-201 table, RQ7 | `reproduce.table3` | `13.*`, `14.*` |
| Figure, RQ8 (cross-dataset) | `reproduce.figures rq8` | `12.*`, `15.*` |

The figures carry the same data as the paper's with plain matplotlib styling. The original PDFs are in `results/` for comparison.

## Level 2: re-evaluating the shipped checkpoints

```bash
python -m reproduce.eval_checkpoints            # FTE1 to FTE8, Linalg-Target, Mean Pred., Lin. Reg.
python -m reproduce.eval_checkpoints --source   # also StableHLO-Source and Zero-Shot
python -m reproduce.table2                      # prints Table 2 with the new values
```

The default needs the archives `labels` and `nb101_graphs_linalg` (3 GB extracted); `--source` also needs `nb101_graphs_stablehlo` (11 GB extracted, and about 10 GB more for its preprocessed test graphs). Only the test graphs are preprocessed (seconds). Evaluation of one model on the 8,474-graph linalg test set takes about 12 seconds on an RTX 4070 SUPER, and the whole command with `--source` about 5 minutes.

The scripts run on the GPU by default. To run on the CPU instead, set the environment variable `SPHINX_DEVICE` to `cpu` in front of the command. On the CPU the command with `--source` takes about 2 hours:

```bash
SPHINX_DEVICE=cpu python -m reproduce.eval_checkpoints --source
```

Expected: every metric cell of Table 2 (RMSE, R², Kendall's τ, 13 columns, both tasks) matches the paper to the printed precision. One cell differs in the last digit by rounding (Zero-Shot R² for test accuracy, -18.919 against -18.918).

How the columns are computed (see the docstring of `reproduce/eval_checkpoints.py`). Mean Pred. and Lin. Reg. are computed on the stablehlo pool: they are the reference for the StableHLO-Source model, showing that it learns more than a trivial or single-feature predictor.

- Mean Pred.: RMSE of predicting the training mean on the stablehlo training split.
- Lin. Reg.: fitted on the stablehlo training split, scored on the stablehlo test split.
- StableHLO-Source: the source model on the stablehlo test split.
- Zero-Shot: the source model on the linalg test split, with the stablehlo vocabulary; predictions are de-normalised with the linalg training statistics.
- FTE1 to FTE8 and Linalg-Target: on the linalg test split.

## Level 3: training

Training uses the original scripts. With `SPHINX_DATA` set, the dataset paths, vocabulary and per-dialect model settings come from environment variables instead of `config/config.yaml`:

| Variable | Meaning | Default |
|---|---|---|
| `SPHINX_BENCHMARK` | `nb101` or `nb201` | `nb101` |
| `SPHINX_DIALECT` | `stablehlo`, `linalg`, `arith`, `llvm`, `linalg_full` | `stablehlo` |
| `SPHINX_TARGET` | `training_time` or `test_accuracy` | `training_time` |
| `SPHINX_POOL` | which split files to use: `stablehlo` (the 90% pool) or `linalg` (the 10% pool) | `stablehlo` for stablehlo, `linalg` for every other dialect |
| `SPHINX_RUNS` | where run folders and checkpoints are written | `$SPHINX_DATA/runs` |
| `SPHINX_CACHE` | where preprocessed graphs are cached | `$SPHINX_DATA/processed` |
| `SPHINX_EPOCHS`, `SPHINX_BATCH_SIZE`, `SPHINX_DEVICE`, `SPHINX_WORKERS` | training overrides | from `config.yaml` |
| `SPHINX_ITERATIONS` | GGNN message-passing iterations | 30 |
| `SPHINX_POOLING` | graph pooling, `add` or `mean` | `add` |

Pre-train the source model (writes `$SPHINX_RUNS/run_<n>`):

```bash
SPHINX_DIALECT=stablehlo SPHINX_TARGET=training_time python GGNN_train.py
```

Fine-tune on linalg from a source checkpoint, here the shipped one, for FTE6:

```bash
SPHINX_DIALECT=linalg SPHINX_TARGET=training_time SPHINX_FT_IDS=6 SPHINX_BACKBONE=../../results/1.FineTuning_Batch128_TrainingTime/TrainingTime_40Epochs_StableHLO/best_tau_model.pt python finetune_experiment.py
```

Each part of this command can be changed independently:

- `SPHINX_FT_IDS` takes FTE numbers, one or several (`SPHINX_FT_IDS=2,6,8`). Without it, all eight configurations run one after the other.
- `SPHINX_DIALECT` is the target dialect: `linalg`, `arith` (scf/memref/arith) or `llvm`. The vocabulary and the model settings of that dialect are picked automatically.
- `SPHINX_TARGET` is what the model predicts: `training_time` or `test_accuracy`.
- `SPHINX_BENCHMARK` switches between NAS-Bench-101 and NAS-Bench-201.
- `SPHINX_BACKBONE` is the source model. It can be a shipped checkpoint or one you trained yourself with `GGNN_train.py`, on any dialect.

This also means you can run experiments that are not in the paper. The paper always starts from stablehlo on NAS-Bench-101; to try, for example, transferring from linalg to scf/memref/arith, train a source model on linalg and fine-tune it on arith (needs `nb101_graphs_linalg` and `nb101_graphs_arith`):

```bash
SPHINX_DIALECT=linalg python GGNN_train.py     # writes $SPHINX_DATA/runs/run_<n>/best_tau_model.pt
SPHINX_DIALECT=arith SPHINX_BACKBONE=$SPHINX_DATA/runs/run_<n>/best_tau_model.pt python finetune_experiment.py
```

The full NAS-Bench-201 pipeline (source model, scratch baseline and the eight configurations, both tasks):

```bash
SPHINX_BENCHMARK=nb201 python pipeline_9010.py     # stablehlo -> linalg
SPHINX_BENCHMARK=nb201 python pipeline_9010_2.py   # scf/memref/arith -> llvm
```

Add `SPHINX_EPOCHS=1` to any of these for a quick functional check. The other experiment scripts (`incremental_linalg_experiment.py` for the growing-subset runs, `scarcity_sweep.py` for RQ8) take the same variables, plus the configuration block at the top of each script.

Each training run saves three checkpoints: `best_tau_model.pt`, `best_r2_model.pt` and `best_rmse_model.pt`. The results reported in the paper are always those of `best_tau_model.pt`.

GPU memory: training ran on a 32 GB GPU with a batch size of 128, except `pipeline_9010_2.py` (scf/memref/arith to llvm on NAS-Bench-201), which ran with 64 because the llvm graphs do not fit in 32 GB at 128. Peak use at these settings is 11 GB for the NAS-Bench-101 linalg runs and 19 GB for the NAS-Bench-201 llvm runs.

If a run stops with a CUDA out-of-memory error, your GPU has less memory than the run needs at the paper's batch size. On a 12 GB GPU, for example, only the fine-tuning configurations that freeze layers fit at 128. You can make any run fit by putting a smaller batch size in front of the command, for example `SPHINX_BATCH_SIZE=32`. The run then completes, but it is no longer the same experiment as in the paper: the batch size changes how the model trains, so its numbers should be compared with the reported ones only for the general trend, not value by value.

Disk: the first run on a dataset writes a preprocessed copy of its graphs to `SPHINX_CACHE`, once per prediction target, of about the same size as the extracted graphs (4 GB for `nb101_graphs_linalg`, 13 GB for `nb101_graphs_stablehlo`, 37 GB for `nb101_graphs_linalg_full`). Extract only the graph archives an experiment needs; the MLIR archives are not used by training.

Reference times on the paper's hardware (RTX 5090, Threadripper 7970X): pre-training 1h42m (training time, 40 epochs) and 2h33m (test accuracy, 60 epochs); each fine-tuning run 52 to 60 minutes; one NAS-Bench-201 run 5 to 13 minutes. The whole `pipeline_9010.py` takes 2h10m, and one epoch of every run of `incremental_linalg_experiment.py` takes 33 minutes.

The NAS-Bench-101 training scripts do not fix a random seed. The NAS-Bench-201 pipeline and the sweep are seeded, which fixes the initial weights and the order of the data but does not make GPU training bit-reproducible. See the suggested path above for how far a rerun lands from the shipped numbers.

## Adding a dialect or a benchmark

The model does not depend on any dialect. A training run reads four things, and a new dialect or a new set of programs only has to provide them:

| Input | Location | Format |
|---|---|---|
| Graphs | `$SPHINX_DATA/<benchmark>/graphs/<dialect>/` | One `<name>.ProgramGraph.pb` per program, a ProGraML `ProgramGraph` protocol buffer as written by `mlir_to_programl`. The model uses the text of each node and the source, target, flow and position of each edge. |
| Labels | `$SPHINX_DATA/<benchmark>/<labels>.json` | One JSON object: `{"<name>": {"training_time": 925.8, "test_accuracy": 0.90, ...}, ...}`. Any numeric field can be the prediction target. |
| Vocabulary | `vocab/<file>.csv` | Tab-separated with a header line; the fourth column is the node text. A node text gets the number of its line as its index, and index 0 is kept for texts not in the file. The first three columns (cumulative frequencies and count) are informative. |
| Splits | `splits/<folder>/{train,validation,test}_files_<pool>.txt` | One program name per line, without extension (`model_12`). |

The steps for a new dialect are:

1. Write the programs in the new dialect as `.mlir` files, one per program, in `<folder>/mlir/`. The `dataset_generation/` folder has the lowering commands we used for linalg, scf/memref/arith and llvm.
2. Convert them to graphs with `mlir_to_programl <folder>` (see [Optional: the graph generator](#optional-the-graph-generator)), and place the resulting `graphs/` folder at `$SPHINX_DATA/<benchmark>/graphs/<dialect>/`. The generator has no dialect-specific code: it reads operation names, operands, results and regions. Every dialect of upstream MLIR is already registered, and StableHLO is added in `mlir_to_programl/main.cpp`; a dialect from another project needs one more `registry.insert` line there.
3. Build the vocabulary with `python -m utils.create_vocab <graph_dir> vocab/<file>.csv`.
4. Get the largest edge position with `python -m utils.find_max_edge_position <graph_dir>`.
5. Register the dialect in `config/datasets.py`: its vocabulary in `VOCAB` and its edge position in `MAX_EDGE_POSITION`.
6. Train with `SPHINX_DIALECT=<dialect>`, from scratch (`GGNN_train.py`) or from a model of another dialect (`finetune_experiment.py` with `SPHINX_BACKBONE`).

A new dialect of NAS-Bench-101 or NAS-Bench-201 reuses the labels and the split files of the benchmark. A new benchmark also needs a labels file, split files (`python -m utils.create_split_txt_files <graph_dir> splits/<folder> <pool>` writes a 60/20/20 split) and its entries in `LABELS` and `SPLITS`.

About the split pools: each benchmark of the paper is divided into two disjoint sets of programs, called `stablehlo` (90%, where the source models are trained) and `linalg` (10%, used for every target dialect). The names come from the first experiment and do not restrict the dialect: the arith and llvm models are trained on the `linalg` pool. A dialect that has split files of its own (`train_files_<dialect>.txt`) uses them by default, and `SPHINX_POOL` selects a pool explicitly.

### Worked example

This example adds a dialect and a benchmark using only files that are in the repository, so it needs no download and no GPU, and runs in under a minute. The benchmark, `toy`, is the 16 networks of `sample_programs/`. The dialect, `newdialect`, is their linalg version, treated as if the artifact had never seen it. With 16 programs the models learn nothing useful; the point is to show every step.

Place the graphs and the labels (from `ml_experiments/programl_mlir_performance`):

```bash
export SPHINX_DATA=/tmp/sphinx_toy
mkdir -p $SPHINX_DATA/toy/graphs
cp -r ../../sample_programs/linalg/programl $SPHINX_DATA/toy/graphs/newdialect
cp -r ../../sample_programs/stablehlo/programl $SPHINX_DATA/toy/graphs/stablehlo
cp ../../sample_programs/labels.json $SPHINX_DATA/toy/labels.json
```

Build the vocabulary, the edge position and the split files:

```bash
python -m utils.create_vocab $SPHINX_DATA/toy/graphs/newdialect vocab/toy_newdialect.csv
python -m utils.find_max_edge_position $SPHINX_DATA/toy/graphs/newdialect
python -m utils.create_split_txt_files $SPHINX_DATA/toy/graphs/newdialect splits/toy newdialect
```

Expected output:

```
16 graphs, 60 node texts, written to vocab/toy_newdialect.csv
/tmp/sphinx_toy/toy/graphs/newdialect: 16 graphs, max_edge_position = 6
splits/toy/train_files_newdialect.txt: 9 programs
splits/toy/validation_files_newdialect.txt: 3 programs
splits/toy/test_files_newdialect.txt: 4 programs
```

Register them by adding these lines at the end of `config/datasets.py`:

```python
MAX_EDGE_POSITION["newdialect"] = 6
VOCAB[("toy", "newdialect")] = "vocab/toy_newdialect.csv"
VOCAB[("toy", "stablehlo")] = "vocab/FTE_programl_mlir_vocab_stablehlo.csv"
LABELS["toy"] = "labels.json"
SPLITS["toy"] = "splits/toy"
```

The third line makes the stablehlo version of the same programs available as a source dialect, with the vocabulary the paper uses for it.

Train a model on the new dialect:

```bash
export SPHINX_BENCHMARK=toy SPHINX_EPOCHS=2 SPHINX_DEVICE=cpu SPHINX_WORKERS=2
SPHINX_DIALECT=newdialect python GGNN_train.py
```

Transfer to it from another dialect: train a source model on stablehlo, then fine-tune it on the new dialect with configuration FTE6. In this toy benchmark both dialects share one set of split files, so the source run names the pool:

```bash
SPHINX_DIALECT=stablehlo SPHINX_POOL=newdialect python GGNN_train.py
SPHINX_DIALECT=newdialect SPHINX_FT_IDS=6 SPHINX_BACKBONE=$SPHINX_DATA/runs/run_2/best_tau_model.pt python finetune_experiment.py
```

Predict another quantity by naming any numeric field of the labels file:

```bash
SPHINX_DIALECT=newdialect SPHINX_TARGET=trainable_parameters python GGNN_train.py
```

Each run writes a folder `$SPHINX_DATA/runs/run_<n>` with:

- `train.log`: the settings, the statistics of the training labels, and for every epoch the training loss and the validation RMSE, MAE, R², Kendall's τ and Spearman correlation, followed by the epoch time. These are the files in `results/` that the `reproduce` scripts parse.
- `best_tau_model.pt`, `best_r2_model.pt`, `best_rmse_model.pt`: the model at the best epoch for each validation metric, and `epoch_<n>.pt` for every epoch.

The first run on a dialect also writes the preprocessed graphs (one PyTorch Geometric `Data` file per program) to `$SPHINX_DATA/processed/<benchmark>_<dialect>_<target>`.

### Other changes

- Model and optimiser settings (embedding sizes, learning rate, batch size, epochs) are in `config/config.yaml`.
- The eight fine-tuning configurations are defined in `finetune_experiment.py`: `EXPERIMENT_NAMES` lists them, `load_pretrained_surgery` decides how the embeddings of the new vocabulary are initialised, and `freeze_gnn_layers` decides which layers are frozen. A new configuration is a new ID handled in these two functions.
- The amount of training data is set by the training split file: a shorter list, with the same validation and test files, trains on a subset of the programs.

## Settings that matter

- The reported models are `best_tau_model.pt` (best validation τ).
- The default setting is sum pooling, 30 GGNN iterations and a batch size of 128. The larger graphs of the lower dialects did not fit at this setting for every experiment, so some runs used a lighter one. To retrain or evaluate a shipped checkpoint of those runs, set the variables in the last column.

| Results | Experiment | Pooling | Iterations | Batch | Variables |
|---|---|---|---|---|---|
| `1.*` to `5.*`, `10.*` | NAS-Bench-101, stablehlo and linalg | sum | 30 | 128 | defaults |
| `6.*` | NAS-Bench-101, scf/memref/arith, training time | sum | 30 | 128 | defaults |
| `7.*` | NAS-Bench-101, scf/memref/arith, test accuracy | sum | 10 | 128 | `SPHINX_ITERATIONS=10` |
| `8.*`, `9.*` | NAS-Bench-101, llvm | mean | 10 | 128 | `SPHINX_POOLING=mean SPHINX_ITERATIONS=10` |
| `12.*`, `13.*`, `15.*` | NAS-Bench-201, stablehlo and linalg | sum | 30 | 128 | defaults |
| `14.*` | NAS-Bench-201, scf/memref/arith to llvm | sum | 30 | 64 | set in `pipeline_9010_2.py` |
- `max_edge_position` depends on the dialect: stablehlo 8, linalg 6, scf/memref/arith 5, llvm 8 (`utils/find_max_edge_position.py` recomputes it).
- Vocabulary sizes are read from the files in `vocab/`.

## Naming of the fine-tuning configurations

The experiment ID in the scripts, logs, JSON files and `FT<n>` / `FTE<n>` folders is the paper's configuration number: ID 6 is FTE6.

## Dataset notes

- Split sizes (NAS-Bench-101): stablehlo 228,756 / 76,252 / 76,253 and linalg 25,417 / 8,472 / 8,474 (train / validation / test).
- Large tensor constants (network weights) are elided in the shipped graphs and MLIR files, in the form MLIR prints with `--mlir-elide-elementsattrs-if-larger`. The model reads only node labels and edges, which are unchanged; `artifact/package_dataset.py --verify` checks this.
- `nb101_graphs_linalg_full` holds all 423,624 architectures lowered to linalg, used by the experiments on growing training subsets (RQ2).
- `dataset_generation/README.md` describes how the MLIR programs were produced from the two benchmarks and how to rerun it. The stablehlo and linalg programs come from TensorFlow's StableHLO exporter; the scf/memref/arith and llvm programs are compiled from the same networks through ONNX and torch-mlir.
- To regenerate graphs from MLIR, use the `sphinx-tool` image (see Setup) or build `mlir_to_programl/` yourself (see `README.md`), and run it on a folder with an `mlir/` subfolder.
