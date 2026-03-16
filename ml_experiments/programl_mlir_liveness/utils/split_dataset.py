import shutil
import random
import math
from tqdm import tqdm
from ..config.params import params
from pathlib import Path

ROOT_DIR = Path(params.paths.graphs)  # <-- Change this!

# Source directory for graph files
GRAPH_SRC_DIR = ROOT_DIR / "graphs"

# Destination directories
TRAIN_DIR = ROOT_DIR / "train"
VALID_DIR = ROOT_DIR / "validation"
TEST_DIR = ROOT_DIR / "test"

# Split Ratios
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
# TEST_RATIO is the remainder (0.2)

def main():
    print(f"--- Graph Splitter Script ---")
    print(f"Root Directory:    {ROOT_DIR}")
    print(f"Source Graphs:     {GRAPH_SRC_DIR}")
    print("\nIMPORTANT: This script will ONLY COPY graph files.")
    print("Label files will NOT be moved.")
    print("-" * 30)

    # --- 1. Check Source Directory ---
    if not GRAPH_SRC_DIR.is_dir():
        print(f"Error: Source directory not found:")
        print(f"{GRAPH_SRC_DIR}")
        print("Please check the 'ROOT_DIR' variable in this script.")
        return

    # --- 2. Get All Graph Files ---
    try:
        # Find all .pb files in the source directory
        graph_files = sorted(list(GRAPH_SRC_DIR.glob("*.ProgramGraph.pb")))

        if not graph_files:
            print(f"Error: No '*.ProgramGraph.pb' files found in {GRAPH_SRC_DIR}")
            return

        print(f"Found {len(graph_files)} graph files to split.")

    except Exception as e:
        print(f"Error scanning for files: {e}")
        return

    # --- 3. Shuffle and Split the File List ---
    random.seed(42)  # Use a fixed seed for a reproducible split
    random.shuffle(graph_files)

    total_count = len(graph_files)
    train_count = math.ceil(total_count * TRAIN_RATIO)
    valid_count = math.ceil(total_count * VALID_RATIO)

    split_1 = train_count
    split_2 = train_count + valid_count

    # These are lists of Path objects
    train_files = graph_files[:split_1]
    valid_files = graph_files[split_1:split_2]
    test_files = graph_files[split_2:]

    print("\n--- Split Summary ---")
    print(f"Training set:   {len(train_files)} files")
    print(f"Validation set: {len(valid_files)} files")
    print(f"Test set:       {len(test_files)} files")
    print(f"Total:          {len(train_files) + len(valid_files) + len(test_files)} files")

    # --- 4. Define the Copy Function ---
    def copy_files_to_dest(file_list, destination_folder):
        """Copies graph files to the destination, creating a 'graphs' subfolder."""

        # Create the 'graphs' sub-directory that your training script expects
        dest_graph_dir = destination_folder
        dest_graph_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        for src_path in tqdm(file_list, desc=f"Copying to {destination_folder.name}"):
            dest_path = dest_graph_dir / src_path.name

            try:
                shutil.copy(str(src_path), str(dest_path))
                copied_count += 1
            except Exception as e:
                print(f"\nError copying file {src_path.name}: {e}")

        print(f"Successfully copied {copied_count} files to {dest_graph_dir}")

    # --- 5. Run the Copy ---
    try:
        print("\nStarting file copy process...")
        copy_files_to_dest(train_files, TRAIN_DIR)
        copy_files_to_dest(valid_files, VALID_DIR)
        copy_files_to_dest(test_files, TEST_DIR)
        print("\nAll graph files have been split successfully!")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")


if __name__ == "__main__":
    main()