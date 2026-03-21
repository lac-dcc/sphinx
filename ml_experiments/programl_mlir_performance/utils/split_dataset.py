import os
import random
from tqdm import tqdm

# --- CONFIGURATION ---
MLIR_STABLEHLO_DIR = "/home/douglasvc/Desktop/NASBench_Dataset/mlir"
MLIR_LINALG_DIR = "/home/douglasvc/Desktop/NASBench_Dataset/mlir_linalg"

SPLIT_RATIO = 0.90
SEED = 42

DRY_RUN = False


# ---------------------

def main():
    print(f"Scanning directory: {MLIR_STABLEHLO_DIR}...")

    # 1. Get all filenames (assuming both folders have the exact same files)
    try:
        all_files = [f for f in os.listdir(MLIR_STABLEHLO_DIR) if os.path.isfile(os.path.join(MLIR_STABLEHLO_DIR, f))]
    except FileNotFoundError:
        print(f"Error: Could not find directory {MLIR_STABLEHLO_DIR}")
        return

    num_files = len(all_files)
    if num_files == 0:
        print("No files found! Check your paths.")
        return

    print(f"Found {num_files:,} files.")

    # 2. Sort to guarantee determinism, then shuffle with a fixed seed
    all_files.sort()
    random.seed(SEED)
    random.shuffle(all_files)

    # 3. Calculate the split index
    split_idx = int(num_files * SPLIT_RATIO)

    # 90% StableHLO Set -> These stay in MLIR, so we DELETE them from LINALG
    files_to_delete_from_linalg = all_files[:split_idx]

    # 10% Linalg Set -> These stay in LINALG, so we DELETE them from MLIR
    files_to_delete_from_mlir = all_files[split_idx:]

    print("\n" + "=" * 40)
    print("SPLIT PLAN:")
    print(
        f"StableHLO (90%): {len(files_to_delete_from_linalg):,} files will be KEPT in mlir, DELETED from mlir_linalg.")
    print(f"Linalg (10%):    {len(files_to_delete_from_mlir):,} files will be KEPT in mlir_linalg, DELETED from mlir.")
    print("=" * 40 + "\n")

    if DRY_RUN:
        print("DRY_RUN is True. No files will be deleted.")
        print("Change DRY_RUN = False in the script to execute the deletion.")
        return

    # 4. Execute Deletions
    print(f"Deleting 10% ({len(files_to_delete_from_mlir):,}) from the StableHLO (mlir) folder...")
    mlir_deleted = 0
    for f in tqdm(files_to_delete_from_mlir):
        file_path = os.path.join(MLIR_STABLEHLO_DIR, f)
        if os.path.exists(file_path):
            os.remove(file_path)
            mlir_deleted += 1

    print(f"Deleting 90% ({len(files_to_delete_from_linalg):,}) from the Linalg (mlir_linalg) folder...")
    linalg_deleted = 0
    for f in tqdm(files_to_delete_from_linalg):
        file_path = os.path.join(MLIR_LINALG_DIR, f)
        if os.path.exists(file_path):
            os.remove(file_path)
            linalg_deleted += 1

    print("\nOperation Complete!")
    print(f"Actually deleted {mlir_deleted:,} files from {MLIR_STABLEHLO_DIR}")
    print(f"Actually deleted {linalg_deleted:,} files from {MLIR_LINALG_DIR}")


if __name__ == "__main__":
    main()