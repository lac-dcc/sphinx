cmake_minimum_required(VERSION 3.20)
project(mlir-to-programl CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/llvm_install/lib/cmake" ${CMAKE_PREFIX_PATH})

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    message(STATUS "Using ccache to accelerate builds: ${CCACHE_FOUND}")
endif()

find_program(LLD_FOUND lld)
if(LLD_FOUND)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
    message(STATUS "Using lld linker to accelerate linking.")
endif()

find_package(Protobuf REQUIRED)
find_package(absl REQUIRED CONFIG)
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

add_executable(mlir-to-programl
        main.cpp
        MLIRToProGraMLBuilder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/program_graph_builder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/create_features.cc
        ${PROTO_SRCS}
        ${LABM8_SRCS}
)

target_include_directories(mlir-to-programl PRIVATE
        ${PROTOBUF_INCLUDE_DIRS}
        ${PROTO_CPP_DIR}
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(core_mlir_libs
        MLIRIR
        MLIRSupport
)
set(MLIR_LIBRARIES
        ${dialect_libs}
        ${core_mlir_libs}
)

add_subdirectory(third_party/fmt)

target_link_libraries(mlir-to-programl PRIVATE
        ${PROTOBUF_LIBRARIES}
        ${MLIR_LIBRARIES}
        absl::flat_hash_map
        absl::flat_hash_set
        fmt::fmt
)