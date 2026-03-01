#include <fstream>
#include <iostream>
#include <filesystem>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <format>
#include <chrono>

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Parser/Parser.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Diagnostics.h"

#include "MLIRToProGraMLBuilder.h"

#ifdef USE_STABLEHLO
    #include "stablehlo/dialect/StablehloOps.h"
#endif

mlir::OwningOpRef<mlir::ModuleOp> parseMlirFile(
    mlir::MLIRContext &context,
    const std::filesystem::path &inputPath,
    std::string &localLog
) {
    llvm::SourceMgr sourceMgr;

    auto buffer {llvm::MemoryBuffer::getFile(inputPath.string())};
    if (!buffer) {
        localLog += "Failed to read input file: " + inputPath.string() + "\n";
        return nullptr;
    }
    sourceMgr.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());

    return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}


bool serializeGraphToFile(
    const programl::ProgramGraph &graph,
    const std::filesystem::path &outputPath,
    std::string &localLog
) {
    if (outputPath.has_parent_path())
        std::filesystem::create_directories(outputPath.parent_path());

    std::ofstream ofs {outputPath, std::ios::binary};
    if (!ofs.is_open()) {
        localLog += "Failed to open output file: " + outputPath.string() + "\n";
        return false;
    }

    if (!graph.SerializeToOstream(&ofs)) {
        localLog += "Failed to serialize ProgramGraph to: " + outputPath.string() + "\n";
        return false;
    }

    return true;
}


bool convertMlirToGraph(
    const std::filesystem::path &inputPath,
    const std::filesystem::path &outputPath,
    std::ofstream &globalLogFile,
    std::mutex &logMutex,
    const std::string &logHeaderName
) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    #ifdef USE_STABLEHLO
        registry.insert<mlir::stablehlo::StablehloDialect>();
    #endif
    mlir::MLIRContext context {registry};

    std::string localLog;
    llvm::raw_string_ostream logStream(localLog);

    mlir::ScopedDiagnosticHandler diagHandler(&context, [&](const mlir::Diagnostic &diag) {
        std::string severity;
        switch (diag.getSeverity()) {
            case mlir::DiagnosticSeverity::Note: severity = "NOTE"; break;
            case mlir::DiagnosticSeverity::Warning: severity = "WARNING"; break;
            case mlir::DiagnosticSeverity::Error: severity = "ERROR"; break;
            case mlir::DiagnosticSeverity::Remark: severity = "REMARK"; break;
        }
        logStream << "  [" << severity << "] " << diag << "\n";
        return mlir::success();
    });

    bool success {true};
    const mlir::OwningOpRef module {parseMlirFile(context, inputPath, localLog)};

    if (!module) {
        localLog += "Failed to parse MLIR file\n";
        success = false;
    } else if (mlir::failed(mlir::verify(*module))) {
        localLog += "MLIR failed to verify the file\n";
        success = false;
    } else {
        MLIRToProGraMLBuilder builder;
        const programl::ProgramGraph &graph {builder.Build(*module)};

        if (!serializeGraphToFile(graph, outputPath, localLog)) {
            success = false;
        }
    }

    if (!localLog.empty()) {
        std::lock_guard<std::mutex> lock(logMutex);
        globalLogFile << "=== " << logHeaderName << " ===\n"
                      << localLog << "\n";
    }

    return success;
}


void processDataset(const std::filesystem::path& datasetPath, bool createAllGraphs) {
    const auto startTime {std::chrono::steady_clock::now()};

    const auto mlirSourcePath {datasetPath / "mlir"};
    const auto graphsDestPath {datasetPath / "graphs"};
    const auto allGraphsPath {graphsDestPath / "all_graphs"};
    const auto logFilePath {datasetPath / "conversion.log"};

    if (!std::filesystem::exists(mlirSourcePath) || !std::filesystem::is_directory(mlirSourcePath)) {
        std::cerr << "Error: 'mlir' subdirectory not found in " << datasetPath << "\n";
        exit(4);
    }

    if (createAllGraphs)
        std::filesystem::create_directories(allGraphsPath);

    std::ofstream logFile(logFilePath, std::ios::out | std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file at " << logFilePath << "\n";
        exit(5);
    }

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
        std::cout << "No files found to process\n";
        return;
    }

    const size_t totalFiles {filesToProcess.size()};
    std::cout << "Found " << totalFiles << " files. Detailed logs will be written to " << logFilePath << "\n";

    std::atomic<size_t> fileIndex {0};
    std::atomic<size_t> completedCount {0};
    std::atomic<int> successCount {0};
    std::atomic<int> failureCount {0};
    std::atomic<int> copyFailureCount {0};

    std::mutex terminalMutex;
    std::mutex logMutex;

    auto worker_fn = [&]() {
        while (true) {
            const size_t currentIndex {fileIndex.fetch_add(1)};
            if (currentIndex >= totalFiles)
                break;

            const auto& [input, output] {filesToProcess[currentIndex]};
            std::string relativePathStr {std::filesystem::relative(input, mlirSourcePath).string()};

            if (convertMlirToGraph(input, output, logFile, logMutex, relativePathStr)) {
                ++successCount;
                if (createAllGraphs) {
                    try {
                        const auto allGraphsDest {allGraphsPath / output.filename()};
                        std::filesystem::copy_file(output, allGraphsDest, std::filesystem::copy_options::overwrite_existing);
                    } catch (const std::filesystem::filesystem_error& e) {
                        std::lock_guard<std::mutex> lock {logMutex};
                        logFile << "=== " << relativePathStr << " ===\n"
                                << "  [POST-PROCESS ERROR] Failed to copy to all_graphs: " << e.what() << "\n\n";
                        ++copyFailureCount;
                    }
                }
            } else {
                ++failureCount;
            }

            {
                const size_t completed {completedCount.fetch_add(1) + 1};
                const double percentage {static_cast<double>(completed) * 100.0 / static_cast<double>(totalFiles)};

                std::lock_guard<std::mutex> lock {terminalMutex};
                std::cout << std::format(
                    "\rProgress: [{}/{}] ({:.2f}%) | Success: {} | Failed: {}",
                    completed,
                    totalFiles,
                    percentage,
                    successCount.load(),
                    failureCount.load()
                ) << std::flush;
            }
        }
    };

    const unsigned int numThreads {std::max(1u, std::thread::hardware_concurrency() - 1)};
    std::cout << "Using " << numThreads << " threads\n";
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < numThreads; ++i)
        threads.emplace_back(worker_fn);

    for (auto& thread : threads)
        thread.join();

    const auto endTime {std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsedSeconds {endTime - startTime};
    const double totalSecondsPrecise {elapsedSeconds.count()};
    const auto totalSecondsInt {static_cast<long long>(totalSecondsPrecise)};
    const auto hours {totalSecondsInt / 3600};
    const auto minutes {(totalSecondsInt % 3600) / 60};
    const auto seconds {totalSecondsInt % 60};

    std::cout << std::format("\nProcessing complete in {}h {}m {}s ({:.3f} s)\n",
                             hours, minutes, seconds, totalSecondsPrecise);
    if (createAllGraphs && copyFailureCount.load() > 0)
        std::cout << "Failed to copy to all_graphs folder: " << copyFailureCount.load() << "\n";
}


int main(const int argc, char **argv) {
    bool createAllGraphs {false};
    std::vector<std::string> positionalArgs;

    for (int i = 1; i < argc; ++i) {
        std::string arg {argv[i]};
        if (arg == "--all-graphs-folder") {
            createAllGraphs = true;
        } else {
            positionalArgs.push_back(arg);
        }
    }

    if (positionalArgs.empty()) {
        std::cerr << "Usage:\n"
                  << "  Single file mode: mlir-to-programl <input.mlir> [output.ProgramGraph.pb]\n"
                  << "  Dataset mode:     mlir-to-programl [--all-graphs-folder] <dataset_folder>\n";
        return 1;
    }

    const std::filesystem::path pathArg {positionalArgs[0]};

    if (!std::filesystem::exists(pathArg)) {
        std::cerr << "Error: Provided path does not exist: " << pathArg << "\n";
        return 2;
    }

    if (std::filesystem::is_directory(pathArg)) {
        // --- Dataset Mode ---
        std::cout << "Dataset mode activated\n";
        processDataset(pathArg, createAllGraphs);
    } else if (std::filesystem::is_regular_file(pathArg)) {
        // --- Single File Mode ---
        std::cout << "Single file mode activated\n";
        std::filesystem::path outputPath {};
        if (argc >= 3) {
            outputPath = positionalArgs[1];
        } else {
            outputPath = pathArg;
            outputPath.replace_extension(".ProgramGraph.pb");
        }

        std::filesystem::path logPath {pathArg.parent_path() / "conversion.log"};
        std::ofstream logFile(logPath, std::ios::out | std::ios::trunc);
        std::mutex logMutex;

        std::cout << "Processing: " << pathArg << " -> " << outputPath << "\n";
        std::cout << "Logs will be written to: " << logPath << "\n";

        if (!convertMlirToGraph(pathArg, outputPath, logFile, logMutex, pathArg.filename().string())) {
            std::cerr << "Error converting file. Check the log file for details\n";
            return 3;
        }

        std::cout << "Successfully wrote MLIR ProgramGraph to " << outputPath << "\n";
    } else {
        std::cerr << "Error: Input path is not a regular file or a directory: " << pathArg << "\n";
        return 2;
    }

    return 0;
}