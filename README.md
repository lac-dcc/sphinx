# SPHINX: Transferring Optimization Predictors Across MLIR Dialects Using Program Graphs

**SPHINX** is a research project focused on transferring machine learning-based optimization predictors across [**MLIR**](https://mlir.llvm.org/) dialects by representing programs as [**ProGraML**](https://proceedings.mlr.press/v139/cummins21a.html) graphs. This repository contains both the core C++ graph generation infrastructure and the Python-based machine learning experiments.

## Repository Structure

This monorepo is divided into three main components:

- **`mlir_to_programl/`** (C++): The core tool that translates MLIR code into ProGraML graph representations.
- **`ml_experiments/`** (Python): Machine learning models, training pipelines, and experiment scripts used in our evaluation.
- **`sample_programs/`**: A small collection of MLIR programs derived from the NASBench dataset, provided for testing and demonstration purposes.

---

## Getting Started

### 1. The Graph Generator (C++)
Located in `mlir_to_programl/`. This tool reads MLIR files and outputs the corresponding graphs.

**Prerequisites:**

- **CMake >= 3.20**
- **C++20 compiler** (GCC 10+ or Clang 10+)
- **LLVM/MLIR >= 20.0** (Installed and configured)
- **Google Protobuf**
- **Abseil (abseil-cpp)**

**Build Instructions:**
   1. **Configure LLVM Path:** The build system expects to find your LLVM installation. By default, it looks in `$HOME/llvm_install`. If your LLVM is installed elsewhere, export the path before building:
     
      ```bash
      export CMAKE_PREFIX_PATH=/path/to/your/llvm/lib/cmake:$CMAKE_PREFIX_PATH
      ```
     
   2. **Build:**
      ```bash
      cd mlir_to_programl
      mkdir build && cd build
      cmake ..
      make -j8
      ```

**Usage:**
   1. **Single File Mode:** Converts a single MLIR file. If the output path is omitted, it defaults to replacing the extension with `.ProgramGraph.pb`.
      
      ```bash
      ./mlir_to_programl <input.mlir> [output.ProgramGraph.pb]
      ```
     
   2. **Dataset Mode:** Processes an entire directory. It detects if the input is a folder and automatically converts all contained MLIR files.
      
      ```bash
      ./mlir_to_programl <dataset_folder>
      ```
   
### 2. The Experiments (Python)

Located in `ml_experiments/`. Contains GNN models along with training and evaluation scripts.

**Prerequisites:**
- **Python 3.8+**
- **CUDA (optional, but recommended for training)**

**Setup:**

We provide a unified environment for all experiments:

```bash
cd ml_experiments

# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (PyTorch, PyG, etc.)
pip install -r requirements.txt
```

### 3. Sample Programs

Located in `sample_programs/`. These files are provided for testing and demonstration purposes. The programs originate from architectures in the NASBench dataset and were converted into MLIR as part of our experimental pipeline. They include programs represented in different MLIR dialects used throughout the project.

Example structure:

```
sample_programs/
  stablehlo/
    mlir/
      model_1.mlir
      model_2.mlir
    programl/
   	  model_1.ProgramGraph.pb
   	  model_2.ProgramGraph.pb
  linalg/
    mlir/
      model_1.mlir
      model_2.mlir
    programl/
   	  model_1.ProgramGraph.pb
   	  model_2.ProgramGraph.pb
```

You can use these programs to quickly test the graph generator:

```bash
cd mlir_to_programl/build
./mlir_to_programl ../../sample_programs/linalg
```

This will generate the corresponding ProGraML graphs for all MLIR files in the directory.
