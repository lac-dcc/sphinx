import os
import sys
import subprocess
import logging
from tqdm import tqdm
from pathlib import Path

from config.params import params

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# The binary to perform the dialect conversion
OPT_BINARY = "/home/douglasvc/stablehlo/build/bin/stablehlo-opt"

# Flags provided by you
OPT_FLAGS = [
    # "--shape-legalize-to-stablehlo",
    "--stablehlo-aggressive-folder",
    "--stablehlo-aggressive-simplification",
    "--stablehlo-legalize-to-linalg=enable-primitive-ops=true",
    "--mlir-elide-elementsattrs-if-larger=50"
]

# New folders for this experiment
TRANSFER_GRAPHS_DIR = os.path.join(os.path.dirname(params.paths.graphs), "mlir_linalg")
source = Path("/home/douglasvc/Desktop/NASBench_Dataset/mlir")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def legalize_graphs(file_list):
    logging.info(f"Step 1: Legalizing {len(file_list)} graphs to Linalg...")
    logging.info(f"Source: {params.paths.graphs}")
    logging.info(f"Destination: {TRANSFER_GRAPHS_DIR}")

    os.makedirs(TRANSFER_GRAPHS_DIR, exist_ok=True)

    success_count = 0
    error_count = 0

    for filename in tqdm(file_list, desc="Legalizing"):
        if not filename.endswith('.mlir'):
            src_filename = f"{filename}.mlir"
        else:
            src_filename = filename

        src_path = os.path.join(source, src_filename)
        dst_path = os.path.join(TRANSFER_GRAPHS_DIR, src_filename)

        # skip if already exists (resume capability)
        if os.path.exists(dst_path):
            success_count += 1
            continue

        if not os.path.exists(src_path):
            # Try finding it without extension or just skip
            continue

        try:
            with open(dst_path, 'w') as outfile:
                # Construct command
                cmd = [OPT_BINARY, src_path] + OPT_FLAGS
                subprocess.run(cmd, stdout=outfile, stderr=None, check=True, text=True)
            success_count += 1
        except subprocess.CalledProcessError:
            error_count += 1
            # If failed, maybe delete the empty file
            if os.path.exists(dst_path):
                os.remove(dst_path)

    logging.info(f"Legalization Complete. Success: {success_count}, Failures: {error_count}")


def main():
    setup_logging()

    # 1. Load Test File List
    test_split_path = params.paths.validation_graphs_txt

    logging.info(f"Reading test split from: {test_split_path}")
    with open(test_split_path, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    # 2. Legalize (Convert Dialects)
    legalize_graphs(file_list)


if __name__ == "__main__":
    main()