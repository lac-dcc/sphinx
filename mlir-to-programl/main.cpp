#include <fstream>
#include <iostream>
#include <filesystem>
#include <thread>

#include "llvm/Support/SourceMgr.h"

#include "mlir/Parser/Parser.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/Verifier.h"

#include "MLIRToProGraMLBuilder.h"


mlir::OwningOpRef<mlir::ModuleOp> parseMlirFile(
    mlir::MLIRContext &context,
    const std::filesystem::path &inputPath
) {
    llvm::SourceMgr sourceMgr;

    #ifdef NDEBUG
        mlir::ScopedDiagnosticHandler silenceHandler {&context, [](mlir::Diagnostic &){}};
    #else
        mlir::SourceMgrDiagnosticHandler sourceMgrHandler {sourceMgr, &context};
    #endif

    auto buffer {llvm::MemoryBuffer::getFile(inputPath.string())};
    if (!buffer) {
        std::cerr << "Failed to read input file: " << inputPath << "\n";
        return nullptr;
    }
    sourceMgr.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());

    return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}


bool serializeGraphToFile(
    const programl::ProgramGraph &graph,
    const std::filesystem::path &outputPath
) {
    if (outputPath.has_parent_path())
        std::filesystem::create_directories(outputPath.parent_path());

    std::ofstream ofs {outputPath, std::ios::binary};
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << outputPath << "\n";
        return false;
    }

    if (!graph.SerializeToOstream(&ofs)) {
        std::cerr << "Failed to serialize ProgramGraph to: " << outputPath << "\n";
        return false;
    }

    return true;
}


bool convertMlirToGraph(const std::filesystem::path &inputPath, const std::filesystem::path &outputPath) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::MLIRContext context {registry};

    const mlir::OwningOpRef module {parseMlirFile(context, inputPath)};
    if (!module) {
        std::cerr << "Failed to parse MLIR file: " << inputPath << "\n";
        return false;
    }

    if (failed(mlir::verify(*module)))
        return false;

    if (llvm::failed(mlir::verify(*module))) {
        std::cerr << "MLIR failed to verify the file: " << inputPath << "\n";
        return false;
    }

    MLIRToProGraMLBuilder builder;
    const programl::ProgramGraph &graph {builder.Build(*module)};

    if (!serializeGraphToFile(graph, outputPath)) {
        std::cerr << "Error writing output file, exiting.\n";
        return false;
    }

    return true;
}


void processDataset(const std::filesystem::path& datasetPath) {
    const auto mlirSourcePath {datasetPath / "mlir"};
    const auto graphsDestPath {datasetPath / "graphs"};
    const auto allGraphsPath {graphsDestPath / "all_graphs"};

    if (!std::filesystem::exists(mlirSourcePath) || !std::filesystem::is_directory(mlirSourcePath)) {
        std::cerr << "Error: 'mlir' subdirectory not found in " << datasetPath << "\n";
        exit(4);
    }

    std::filesystem::create_directories(allGraphsPath);

    std::cout << "Collecting files from " << mlirSourcePath << "...\n";

    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> filesToProcess;
    for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(mlirSourcePath)) {
        if (dirEntry.is_regular_file() && dirEntry.path().extension() == ".mlir") {
            const auto& inputPath {dirEntry.path()};
            const auto relativePath {std::filesystem::relative(inputPath, mlirSourcePath)};
            auto outputPath {graphsDestPath / relativePath};
            outputPath.replace_extension(".ProgramGraph.pb");
            filesToProcess.emplace_back(inputPath, outputPath);
        }
    }

    if (filesToProcess.empty()) {
        std::cout << "No files found to process.\n";
        return;
    }

    const size_t totalFiles {filesToProcess.size()};
    std::cout << "Found " << totalFiles << " files. Starting parallel conversion...\n";

    std::atomic<size_t> fileIndex {0};
    std::atomic<int> successCount {0};
    std::atomic<int> failureCount {0};
    std::atomic<int> copyFailureCount {0};
    std::mutex coutMutex;

    auto worker_fn = [&]() {
        while (true) {
            const size_t currentIndex {fileIndex.fetch_add(1)};
            if (currentIndex >= totalFiles)
                break;

            const auto& [input, output] {filesToProcess[currentIndex]};

            {
                std::lock_guard<std::mutex> lock {coutMutex};
                std::cout << "Processing [" << currentIndex + 1 << "/" << totalFiles << "]: " << input.filename().string() << "\n";
            }

            if (convertMlirToGraph(input, output)) {
                ++successCount;

                try {
                    const auto allGraphsDest {allGraphsPath / output.filename()};
                    std::filesystem::copy_file(output, allGraphsDest, std::filesystem::copy_options::overwrite_existing);
                } catch (const std::filesystem::filesystem_error& e) {
                    std::lock_guard<std::mutex> lock {coutMutex};
                    std::cerr << "-> [POST-PROCESS] Failed to copy " << output.filename().string()
                              << " to all_graphs: " << e.what() << "\n";
                    ++copyFailureCount;
                }
            } else {
                ++failureCount;
                std::lock_guard<std::mutex> lock {coutMutex};
                std::cerr << "-> Failed to convert " << input << "\n";
            }
        }
    };

    const unsigned int numThreads {std::max(1u, std::thread::hardware_concurrency())};
    std::cout << "Using " << numThreads << " threads.\n";
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < numThreads; ++i)
        threads.emplace_back(worker_fn);

    for (auto& thread : threads)
        thread.join();

    std::cout << "----------------------------------------\n";
    std::cout << "Processing complete.\n";
    std::cout << "Successfully converted: " << successCount << "\n";
    std::cout << "Failed to convert: " << failureCount << "\n";
    std::cout << "Failed to copy: " << copyFailureCount << "\n";
}


int main(const int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  Single file mode: mlir-to-programl <input.mlir> [output.ProgramGraph.pb]\n"
                  << "  Dataset mode:     mlir-to-programl <dataset_folder>\n";
        return 1;
    }

    const std::filesystem::path pathArg {argv[1]};

    if (!std::filesystem::exists(pathArg)) {
        std::cerr << "Error: Provided path does not exist: " << pathArg << "\n";
        return 2;
    }

    if (std::filesystem::is_directory(pathArg)) {
        // --- Dataset Mode ---
        std::cout << "Dataset mode activated.\n";
        processDataset(pathArg);
    } else if (std::filesystem::is_regular_file(pathArg)) {
        // --- Single File Mode ---
        std::cout << "Single file mode activated.\n";
        std::filesystem::path outputPath {};
        if (argc >= 3) {
            outputPath = argv[2];
        } else {
            outputPath = pathArg;
            outputPath.replace_extension(".ProgramGraph.pb");
        }

        std::cout << "Processing: " << pathArg << " -> " << outputPath << "\n";

        if (!convertMlirToGraph(pathArg, outputPath)) {
            std::cerr << "Error converting file, exiting.\n";
            return 3;
        }

        std::cout << "Successfully wrote MLIR ProgramGraph to " << outputPath << "\n";
    } else {
        std::cerr << "Error: Input path is not a regular file or a directory: " << pathArg << "\n";
        return 2;
    }

    return 0;
}