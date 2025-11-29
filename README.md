# SPHINX: Transferring Optimization Predictors Across MLIR Dialects Using Program Graphs

**SPHINX** is a framework for representing MLIR programs as **ProGraML** graphs to enable machine learning-based optimization across different dialects. This repository contains both the core C++ graph generation infrastructure and the Python-based machine learning experiments.

## Repository Structure

This monorepo is divided into two main components:

- **`mlir-to-programl/`** (C++): The core tool that translates MLIR code into ProGraML graph representations.
- **`ml-experiments/`** (Python): Machine learning models, training scripts, and reproduction experiments.

---

## Getting Started

### 1. The Graph Generator (C++)
Located in `mlir-to-programl/`. This tool reads MLIR files and outputs the corresponding graphs.

**Prerequisites:**
- **CMake >= 3.20**
- **C++20 Compiler** (GCC 10+ or Clang 10+)
- **LLVM/MLIR** (Installed and configured)
- **Google Protobuf**
- **Google Abseil (Abseil-cpp)**

**Build Instructions:**
   1. **Configure LLVM Path:** The build system expects to find your LLVM installation. By default, it looks in `$HOME/llvm_install`. If your LLVM is installed elsewhere, export the path before building:
     
      ```bash
      export CMAKE_PREFIX_PATH=/path/to/your/llvm/lib/cmake:$CMAKE_PREFIX_PATH
      ```
     
   2. **Build:**
      ```bash
      cd mlir-to-programl
      mkdir build && cd build
      cmake ..
      make -j8
      ```

**Usage:**
   1. **Single File Mode:** Converts a single MLIR file. If the output path is omitted, it defaults to replacing the extension with `.ProgramGraph.pb`.
      
      ```bash
      ./mlir-to-programl <input.mlir> [output.ProgramGraph.pb]
      ```
     
   2. **Dataset Mode:** Processes an entire directory. It detects if the input is a folder and automatically converts all contained MLIR files.
      
      ```bash
      ./mlir-to-programl <dataset_folder>
      ```
   
---

### 2. The Experiments (Python)

Located in `ml-experiments/`. Contains GNN models and training/evaluation scripts.

**Prerequisites:**
- **Python 3.8+**
- **CUDA (optional but recommended for training)**

**Setup:**

We provide a unified environment for all experiments:

   ```bash
   cd ml-experiments
  
   # 1. Create a virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # 2. Install dependencies (PyTorch, PyG, etc.)
   pip install -r requirements.txt
   ```

---
