import json
import random
import os
from pathlib import Path
from programl_mlir_performance.config.params import params


def create_dataset_splits(seed=42):
    print("=" * 60)
    print("CREATING DATASET SPLITS")
    print("=" * 60)

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    # Test is remainder (0.2)

    random.seed(seed)

    metrics_path = Path(params.paths.labels)
    output_dir = Path(params.paths.splits_txt)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading models from {metrics_path}...")
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    valid_models = [k for k in all_metrics.keys()]

    print(f"Found {len(valid_models)} models.")

    random.shuffle(valid_models)

    n_total = len(valid_models)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_set = valid_models[:n_train]
    val_set = valid_models[n_train:n_train + n_val]
    test_set = valid_models[n_train + n_val:]

    print(f"Split sizes:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    def save_list(filename, data_list):
        path = output_dir / filename
        with open(path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")
        print(f"Saved {path}")

    save_list("train_files.txt", train_set)
    save_list("validation_files.txt", val_set)
    save_list("test_files.txt", test_set)


if __name__ == "__main__":
    create_dataset_splits()