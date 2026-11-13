# Optional image for the graph generator (mlir_to_programl). The results of the
# paper do not need it: the generated graphs are in the dataset archives.
# It builds LLVM/MLIR and StableHLO from source at the versions used for the
# paper, which takes 12 minutes on 64 cores and about 10 GB of disk during the build.
#
# Build from the repository root:
#   docker build -t sphinx-tool -f artifact/Dockerfile.tool .
# Check the tool on the sample programs:
#   docker run --rm sphinx-tool artifact/check_tool.sh
# Convert your own programs (a folder with an mlir/ subfolder; graphs/ is written next to it):
#   docker run --rm -v /path/to/folder:/work sphinx-tool mlir_to_programl /work
FROM ubuntu:24.04 AS build

ARG STABLEHLO_COMMIT=1ef9e390b5295e676d2b864fe1924bc2f3f4cf0f
# The LLVM commit that this StableHLO version pins in build_tools/llvm_version.txt.
ARG LLVM_COMMIT=0c2701fe7fa002e1befc5f86c268a7964f96d286

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build lld git ca-certificates python3 \
       protobuf-compiler libprotobuf-dev libabsl-dev \
    && rm -rf /var/lib/apt/lists/*

# The tool's CMakeLists.txt looks for StableHLO and its LLVM build in $HOME/stablehlo.
WORKDIR /root/stablehlo
RUN git init -q . && git remote add origin https://github.com/openxla/stablehlo \
    && git fetch -q --depth 1 origin ${STABLEHLO_COMMIT} && git checkout -q FETCH_HEAD \
    && test "$(cat build_tools/llvm_version.txt)" = "${LLVM_COMMIT}"
RUN mkdir llvm-project && cd llvm-project && git init -q . \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch -q --depth 1 origin ${LLVM_COMMIT} && git checkout -q FETCH_HEAD

RUN cmake -GNinja -S llvm-project/llvm -B llvm-build \
      -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_LLD=ON \
      -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host -DLLVM_INSTALL_UTILS=ON \
      -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
    && cmake --build llvm-build

RUN cmake -GNinja -S . -B build \
      -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_LLD=ON \
      -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
      -DMLIR_DIR=/root/stablehlo/llvm-build/lib/cmake/mlir \
    && cmake --build build

COPY mlir_to_programl /sphinx/mlir_to_programl
RUN cmake -GNinja -S /sphinx/mlir_to_programl -B /sphinx/mlir_to_programl/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /sphinx/mlir_to_programl/build

FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends libprotobuf-dev libabsl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /sphinx/mlir_to_programl/build/mlir_to_programl /root/stablehlo/llvm-build/bin/mlir-opt /root/stablehlo/build/bin/stablehlo-opt /usr/local/bin/
WORKDIR /sphinx
COPY mlir_to_programl mlir_to_programl
COPY sample_programs sample_programs
COPY artifact/check_tool.sh artifact/check_tool.sh
CMD ["bash"]
