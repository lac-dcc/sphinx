import os
import random
from pathlib import Path
from config.params import params


def create_dataset_splits(seed=42):
    print("=" * 60)
    print("CREATING DATASET SPLITS (STABLEHLO & LINALG)")
    print("=" * 60)

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    # Test is remainder (0.2)

    random.seed(seed)

    mlir_dir = Path("/home/douglasvc/Desktop/NASBench_Dataset/mlir")
    linalg_dir = Path("/home/douglasvc/Desktop/NASBench_Dataset/mlir_linalg")

    output_dir = Path(params.paths.splits_txt)
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_folder(folder_path, suffix):
        print(f"\nScanning directory: {folder_path}...")

        if not folder_path.exists():
            print(f"Error: Directory {folder_path} not found.")
            return

        # Read files and drop the extension to keep just the 'model_#' name
        valid_models = []
        for f in os.listdir(folder_path):
            file_path = folder_path / f
            if os.path.isfile(file_path):
                name_without_ext = os.path.splitext(f)[0]
                valid_models.append(name_without_ext)

        print(f"Found {len(valid_models)} models for {suffix}.")

        # Shuffle deterministically
        random.shuffle(valid_models)

        n_total = len(valid_models)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_set = valid_models[:n_train]
        val_set = valid_models[n_train:n_train + n_val]
        test_set = valid_models[n_train + n_val:]

        print(f"Split sizes for {suffix}:")
        print(f"  Train: {len(train_set):,}")
        print(f"  Val:   {len(val_set):,}")
        print(f"  Test:  {len(test_set):,}")

        save_list(f"train_files_{suffix}.txt", train_set)
        save_list(f"validation_files_{suffix}.txt", val_set)
        save_list(f"test_files_{suffix}.txt", test_set)

    def save_list(filename, data_list):
        path = output_dir / filename
        with open(path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")
        print(f"  -> Saved {path}")

    # Process the 90% set
    process_folder(mlir_dir, "stablehlo")

    # Process the 10% set
    process_folder(linalg_dir, "linalg")

    print("\n" + "=" * 60)
    print("ALL SPLITS CREATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    create_dataset_splits()