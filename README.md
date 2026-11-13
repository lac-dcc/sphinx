<p align="center">
  <img alt="Project Banner" src="assets/img/ProjSphinx.png" width="100%" height="auto"/></br>
</p>

**SPHINX** studies how to reuse machine-learning performance predictors across [MLIR](https://mlir.llvm.org/) dialects. A program in MLIR changes shape as it is lowered: the same neural network looks different in `stablehlo`, in `linalg` and in `llvm`. A predictor trained on one of these levels normally has to be retrained from scratch for the next one. We represent MLIR programs as [ProGraML](https://proceedings.mlr.press/v139/cummins21a.html) graphs, train a graph neural network on one dialect, and fine-tune it on another, so that most of what the model learned carries over.

This repository holds the code behind the paper *Transfer Learning Across MLIR Dialects* (CGO 2027): the tool that turns MLIR into graphs, the models, and the experiment scripts.

## How it works

1. `mlir_to_programl` reads an MLIR module in any dialect and writes a ProGraML graph, with nodes for operations and values and edges for control, data and call relations.
2. A Gated Graph Neural Network (GGNN) reads the graph and predicts a property of the program. In the paper the programs are the networks of NAS-Bench-101 and NAS-Bench-201, and the properties are their training time and test accuracy.
3. The model is first trained on a source dialect, where data is plentiful. It is then fine-tuned on a target dialect with a new vocabulary, keeping or freezing different parts of the network. The paper compares eight such configurations (FTE1 to FTE8) against training on the target from scratch.

## Repository structure

- `mlir_to_programl/` (C++): the graph generator.
- `ml_experiments/programl_mlir_performance/` (Python): the GGNN, the training and fine-tuning scripts, and the experiments of the paper.
- `ml_experiments/programl_llvm_experiment/` (Python): a reproduction of the liveness experiment of the ProGraML paper, on LLVM IR, as they ran it.
- `ml_experiments/programl_mlir_liveness/` (Python): the same experiment on MLIR, with graphs produced by our generator, to check that we get the same results over MLIR as over LLVM IR.
- `sample_programs/`: a few NAS-Bench programs in `stablehlo` and `linalg`, with their graphs.
- `dataset_generation/`: the scripts that turn NAS-Bench-101 and NAS-Bench-201 into MLIR programs in each dialect. See its [README](dataset_generation/README.md).
- `artifact/`: Dockerfiles, pinned requirements and packaging scripts for the paper's artifact.

The two liveness experiments were sanity checks made before the work in the paper, and are not part of it.

## Reproducing the paper

[ARTIFACT.md](ARTIFACT.md) has the full instructions. In short, the datasets, training logs and model checkpoints are archived on Zenodo (https://doi.org/10.5281/zenodo.24175862), and the paper can be checked at three levels: rebuilding the tables and figures from the logs (under a minute, no GPU), re-evaluating the trained models on the test sets, and training again.

## The graph generator

### With Docker

The image builds LLVM/MLIR and StableHLO from source at the versions we used, so nothing else has to be installed:

```bash
docker build -t sphinx-tool -f artifact/Dockerfile.tool .
docker run --rm sphinx-tool artifact/check_tool.sh
```

The second command converts the sample programs and compares the result with the graphs in this repository.

### From source

You need:

- CMake 3.20 or newer and a C++20 compiler
- LLVM/MLIR (tested with LLVM 22)
- Protobuf and Abseil
- Optionally [StableHLO](https://github.com/openxla/stablehlo), to read programs in the `stablehlo` dialect

The build looks for StableHLO in `$HOME/stablehlo`, and uses the LLVM build inside it (`$HOME/stablehlo/llvm-build`) when there is one. Otherwise it looks for LLVM in `$HOME/llvm_install`. If your LLVM is somewhere else, point CMake at it:

```bash
export CMAKE_PREFIX_PATH=/path/to/your/llvm/lib/cmake:$CMAKE_PREFIX_PATH
```

Then build:

```bash
cd mlir_to_programl
mkdir build && cd build
cmake ..
make -j8
```

### Usage

Convert one file. Without an output path, the graph is written next to the input with the extension `.ProgramGraph.pb`:

```bash
./mlir_to_programl <input.mlir> [output.ProgramGraph.pb]
```

Convert a whole dataset. The folder must contain an `mlir/` subfolder; the graphs are written to `graphs/` beside it, and errors go to `conversion.log`:

```bash
./mlir_to_programl <dataset_folder>
```

For example, from `mlir_to_programl/build`:

```bash
./mlir_to_programl ../../sample_programs/linalg
```

## The experiments

The experiments need Python 3.12. A GPU is not required to look at results, but training without one is not practical.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r artifact/requirements.txt
```

`artifact/requirements.txt` pins the versions used for the paper. `ml_experiments/requirements.txt` lists the same packages without versions, if you prefer newer ones.

The scripts live in `ml_experiments/programl_mlir_performance/`:

- `GGNN_train.py` trains a model on one dialect.
- `finetune_experiment.py` fine-tunes a trained model on another dialect, in any of the eight configurations.
- `pipeline_9010.py` and `pipeline_9010_2.py` run the whole sequence (source model, scratch baseline, fine-tuning) on NAS-Bench-201.
- `incremental_linalg_experiment.py` and `scarcity_sweep.py` train on growing fractions of the data.
- `reproduce/` rebuilds the paper's tables and figures.

The benchmark, dialect and prediction target are chosen with environment variables. For example, to train on the `stablehlo` programs of NAS-Bench-101 and then fine-tune on `linalg`:

```bash
cd ml_experiments/programl_mlir_performance
export SPHINX_DATA=/path/to/extracted/dataset
SPHINX_DIALECT=stablehlo python GGNN_train.py
SPHINX_DIALECT=linalg SPHINX_BACKBONE=/path/to/best_tau_model.pt python finetune_experiment.py
```

[ARTIFACT.md](ARTIFACT.md) lists every variable and the settings of each experiment, and shows how to add a new dialect or benchmark, with a worked example that runs on the sample programs.

## Sample programs

`sample_programs/` has 16 NAS-Bench-101 networks in `stablehlo` and the same 16 lowered to `linalg`, each with the graph the generator produces for it:

```
sample_programs/
  stablehlo/
    mlir/       model_<id>.mlir
    programl/   model_<id>.ProgramGraph.pb
  linalg/
    mlir/       model_<id>.mlir
    programl/   model_<id>.ProgramGraph.pb
  labels.json   training time and accuracy of the 16 networks
```

They are a quick way to try the generator and to see what the graphs look like.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{coutinho2027transfer,
  title     = {Transfer Learning Across {MLIR} Dialects},
  author    = {Coutinho, Douglas Viana and Ros{\'a}rio, Vanderson and Pereira, Fernando M. Quint{\~a}o},
  booktitle = {Proceedings of the IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
  year      = {2027}
}
```

## License

The code is released under the GPL-3.0 license (see [LICENSE](LICENSE)). The datasets are derived from NAS-Bench-101 (Apache-2.0) and NAS-Bench-201 (MIT) and are released under the same terms.
