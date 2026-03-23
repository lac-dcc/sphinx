import os
import random


def generate_incremental_splits(graph_dir, output_dir, file_extension=".ProgramGraph.pb", seed=42):
    random.seed(seed)

    print(f"Scanning directory: {graph_dir} ...")
    all_files = []
    for filename in os.listdir(graph_dir):
        if filename.endswith(file_extension):
            model_name = filename[:-len(file_extension)]
            all_files.append(model_name)

    all_files.sort()

    total_n = len(all_files)
    if total_n == 0:
        print(f"Error: No files ending in '{file_extension}' found in {graph_dir}")
        return

    print(f"Total graphs found: {total_n}")

    random.shuffle(all_files)

    val_size = int(total_n * 0.20)
    test_size = int(total_n * 0.20)

    val_files = all_files[:val_size]
    test_files = all_files[val_size: val_size + test_size]

    available_train_pool = all_files[val_size + test_size:]

    print(f"Fixed Validation Set: {len(val_files)} graphs (20%)")
    print(f"Fixed Test Set:       {len(test_files)} graphs (20%)")
    print(f"Available Train Pool: {len(available_train_pool)} graphs (60%)")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "fixed_validation_linalg.txt"), 'w') as f:
        f.write('\n'.join(val_files))
    with open(os.path.join(output_dir, "fixed_test_linalg.txt"), 'w') as f:
        f.write('\n'.join(test_files))

    train_proportions = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    num_runs = 3

    for p in train_proportions:
        train_size = int(total_n * p)
        p_label = int(p * 100)
        print(f"\nGenerating splits for {p_label}% (Size = {train_size} graphs)...")

        for run in range(1, num_runs + 1):
            current_split = random.sample(available_train_pool, train_size)

            filename = f"train_{p_label}percent_run{run}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                f.write('\n'.join(current_split))

            print(f"  -> Saved {filename}")

    print("\nAll splits generated successfully!")


if __name__ == "__main__":
    GRAPH_DIRECTORY = "/home/douglasvc/Downloads/graphs_linalg"
    OUTPUT_DIRECTORY = "splits/incremental_linalg_experiment"

    generate_incremental_splits(GRAPH_DIRECTORY, OUTPUT_DIRECTORY, file_extension=".ProgramGraph.pb")