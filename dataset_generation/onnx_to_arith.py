"""
ONNX -> scf/memref/arith MLIR, for NAS-Bench-101 and NAS-Bench-201.

torch-mlir-import-onnx, then torch-mlir-opt to linalg-on-tensors, then mlir-opt stopping after --lower-affine,
so the result is in the scf, memref and arith dialects.
"""

import os
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


# -----------------------
# Configuration
# -----------------------
# Paths come from the environment (see README.md in this folder):
#   GEN_ROOT       working folder, default ./dataset_build
#   GEN_LIMIT      convert only the first N architectures (for a quick test)
#   GEN_BENCHMARK  nb101 (default) or nb201
#   MLIR_OPT       path to mlir-opt, default: found on the PATH
GEN_ROOT = Path(os.environ.get("GEN_ROOT", "dataset_build")).resolve()
BENCHMARK = os.environ.get("GEN_BENCHMARK", "nb101")

ONNX_DIR = GEN_ROOT / BENCHMARK / "onnx"
OUTPUT_DIR = GEN_ROOT / BENCHMARK / "mlir" / "arith"
ERRORS_FILE = GEN_ROOT / BENCHMARK / "arith_failures.log"

MLIR_OPT = os.environ.get("MLIR_OPT", "mlir-opt")
TORCH_MLIR_IMPORT_ONNX = "torch-mlir-import-onnx"
TORCH_MLIR_OPT = "torch-mlir-opt"

# Set to an int for a debug run.
TEST_LIMIT = int(os.environ["GEN_LIMIT"]) if os.environ.get("GEN_LIMIT") else None

NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)


TORCH_ARGS = [
    "--torch-onnx-to-torch-backend-pipeline",
    "--torch-backend-to-linalg-on-tensors-backend-pipeline",
    "--refback-generalize-tensor-concat",
]

MLIR_ARGS = [
    "--mlir-elide-elementsattrs-if-larger=50",
    "--convert-tensor-to-linalg",
    # Bufferization
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--buffer-deallocation-pipeline",
    "--convert-bufferization-to-memref",
    # Lower linalg
    "--convert-linalg-to-loops",
    # Required memref lowering
    "--expand-strided-metadata",
    # Cleanup
    "--lower-affine",
    "--canonicalize",
    "--cse",
]


def convert_model(onnx_path: Path):
    model_name = onnx_path.stem
    out_path = OUTPUT_DIR / f"{model_name}.mlir"

    if out_path.exists() and out_path.stat().st_size > 0:
        return True, model_name, "skipped"

    try:
        import_proc = subprocess.Popen(
            [TORCH_MLIR_IMPORT_ONNX, str(onnx_path), "-o", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        torch_proc = subprocess.Popen(
            [TORCH_MLIR_OPT] + TORCH_ARGS,
            stdin=import_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Close the parent's copy so SIGPIPE propagates correctly.
        import_proc.stdout.close()

        with open(out_path, "w") as output_file:
            mlir_proc = subprocess.Popen(
                [MLIR_OPT] + MLIR_ARGS,
                stdin=torch_proc.stdout,
                stdout=output_file,
                stderr=subprocess.PIPE,
                text=True,
            )

            torch_proc.stdout.close()

            _, import_err = import_proc.communicate()
            _, torch_err = torch_proc.communicate()
            _, mlir_err = mlir_proc.communicate()

        if import_proc.returncode != 0:
            raise RuntimeError(f"torch-mlir-import-onnx failed:\n{import_err}")
        if torch_proc.returncode != 0:
            raise RuntimeError(f"torch-mlir-opt failed:\n{torch_err}")
        if mlir_proc.returncode != 0:
            raise RuntimeError(f"mlir-opt failed:\n{mlir_err}")

        return True, model_name, "converted"

    except Exception as e:
        out_path.unlink(missing_ok=True)
        return False, model_name, str(e)


def main():
    if not ONNX_DIR.exists():
        print(f"ERROR: ONNX directory not found: {ONNX_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(ONNX_DIR.glob("*.onnx"))
    if not onnx_files:
        print(f"ERROR: no ONNX files in {ONNX_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(onnx_files)} ONNX models.")

    if TEST_LIMIT:
        onnx_files = onnx_files[:TEST_LIMIT]
        print(f"DEBUG: limiting run to first {TEST_LIMIT} models.")

    print(f"Starting conversion with {NUM_WORKERS} workers.")
    print(f"Saving to: {OUTPUT_DIR}")

    errors = []

    with Pool(processes=NUM_WORKERS) as pool:
        for success, name, msg in tqdm(
            pool.imap_unordered(convert_model, onnx_files, chunksize=4),
            total=len(onnx_files),
            unit="model",
        ):
            if not success:
                errors.append(f"{name}: {msg}")

    written = len(list(OUTPUT_DIR.glob("*.mlir")))
    print(f"\nWrote {written} files.")

    if errors:
        print(f"Finished with {len(errors)} errors.")
        with open(ERRORS_FILE, "w") as f:
            f.write("\n".join(errors))
        print(f"Error log written to {ERRORS_FILE}")
    else:
        print("Finished successfully with 0 errors.")


if __name__ == "__main__":
    main()
