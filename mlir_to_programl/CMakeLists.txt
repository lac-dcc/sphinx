cmake_minimum_required(VERSION 3.20)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

project(mlir_to_programl CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(STABLEHLO_ROOT "$ENV{HOME}/stablehlo" CACHE PATH "Path to StableHLO root directory")
if(EXISTS "${STABLEHLO_ROOT}")
    set(STABLEHLO_FOUND TRUE)
    message(STATUS "StableHLO found at: ${STABLEHLO_ROOT}")
else()
    set(STABLEHLO_FOUND FALSE)
    message(STATUS "StableHLO NOT found. Building without StableHLO support.")
endif()

if(STABLEHLO_FOUND AND EXISTS "${STABLEHLO_ROOT}/llvm-build/lib/cmake/llvm/LLVMConfig.cmake")
    message(STATUS "Using StableHLO LLVM build")
    set(LLVM_DIR "${STABLEHLO_ROOT}/llvm-build/lib/cmake/llvm")
    set(MLIR_DIR "${STABLEHLO_ROOT}/llvm-build/lib/cmake/mlir")
else()
    # Set this to your LLVM/MLIR build directory if not using StableHLO or you have a separate LLVM/MLIR installation.
    message(STATUS "Falling back to local LLVM install")
    set(LLVM_DIR "$ENV{HOME}/llvm_install/lib/cmake/llvm")
    set(MLIR_DIR "$ENV{HOME}/llvm_install/lib/cmake/mlir")
endif()

find_program(LLD_FOUND lld)
if(LLD_FOUND)
    add_link_options(-fuse-ld=lld)
    message(STATUS "Using lld linker to accelerate linking.")
endif()

find_package(Protobuf REQUIRED)
find_package(absl REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto_cpp")
file(GLOB PROTO_FILES "${PROTO_SRC_DIR}/*.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    ${PROTO_FILES}
    PROTOC_OUT_DIR ${PROTO_CPP_DIR}
)

set(LABM8_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/labm8/cpp")
file(GLOB LABM8_SRCS "${LABM8_SRC_DIR}/*.cc")

if(STABLEHLO_FOUND)
    file(GLOB STABLEHLO_ALL_LIBS "${STABLEHLO_ROOT}/build/lib/libStablehlo*.a")
endif()

add_executable(mlir_to_programl
    main.cpp
    MLIRToProGraMLBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/program_graph_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/create_features.cc
    ${PROTO_SRCS}
    ${LABM8_SRCS}
)

target_include_directories(mlir_to_programl PRIVATE
    ${PROTOBUF_INCLUDE_DIRS}
    ${PROTO_CPP_DIR}

    ${MLIR_INCLUDE_DIRS}
    ${MLIR_GENERATED_INCLUDE_DIRS}

    ${LLVM_INCLUDE_DIRS}
    ${LLVM_GENERATED_INCLUDE_DIRS}

    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

if(STABLEHLO_FOUND)
    target_include_directories(mlir_to_programl PRIVATE
        "${STABLEHLO_ROOT}"
        "${STABLEHLO_ROOT}/build"
    )
    target_compile_definitions(mlir_to_programl PRIVATE USE_STABLEHLO)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(core_mlir_libs
    MLIRRegisterAllDialects
    MLIRIR
    MLIRSupport
)
set(MLIR_LIBRARIES
    ${dialect_libs}
    ${core_mlir_libs}
)

add_subdirectory(third_party/fmt)

target_link_libraries(mlir_to_programl PRIVATE
    ${PROTOBUF_LIBRARIES}
    ${MLIR_LIBRARIES}
    absl::flat_hash_map
    absl::flat_hash_set
    fmt::fmt
)

if(STABLEHLO_FOUND)
    target_link_libraries(mlir_to_programl PRIVATE
        "-Wl,--start-group"
        ${STABLEHLO_ALL_LIBS}
        "-Wl,--end-group"
    )
endif()