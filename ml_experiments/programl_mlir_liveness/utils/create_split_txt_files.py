from ..config.params import params

import os
from tqdm import tqdm


def create_split_files(output_dir="."):
    """
    Scans subdirectories (train, validation, test), and for each graph file,
    checks if a corresponding label file exists. If it does, the graph
    filename is written to the appropriate split .txt file.

    File naming convention expected:
    - Graph file: <basename>ProgramGraph.pb
    - Label file: <basename>ProgramGraphFeaturesList.pb

    Args:
        output_dir (str): Directory where the .txt files will be saved.
    """
    base_graph_dir = params.paths.graphs
    labels_dir = params.paths.labels

    print(f"Scanning base graph directory: {base_graph_dir}")
    print(f"Checking for labels in: {labels_dir}")

    splits = ["train", "validation", "test"]

    for split in splits:
        print(f"\nProcessing '{split}' split...")
        split_graph_path = os.path.join(base_graph_dir, split)
        output_filepath = os.path.join(output_dir, f"{split}_files.txt")

        if not os.path.isdir(split_graph_path):
            print(f"Warning: Directory not found for split '{split}' at: {split_graph_path}")
            continue

        potential_graph_files = sorted(os.listdir(split_graph_path))

        valid_graph_filenames = []
        count_missing_labels = 0
        for filename in tqdm(potential_graph_files, desc=f"Verifying '{split}' files"):
            # Ensure we're only looking at graph files with the correct suffix
            if filename.endswith("ProgramGraph.pb"):
                # Derive basename and construct the expected label filename
                basename = filename[:-len("ProgramGraph.pb")]
                label_filename = f"{basename}ProgramGraphFeaturesList.pb"
                label_filepath = os.path.join(labels_dir, label_filename)

                # Check if the corresponding label file exists
                if os.path.exists(label_filepath):
                    valid_graph_filenames.append(basename)
                else:
                    count_missing_labels += 1

        print(f"Missing labels for {count_missing_labels} graph files in split '{split}'.")

        if not valid_graph_filenames:
            print(f"Warning: No valid graph/label pairs found for split '{split}'.")
            continue

        # Write the list of valid graph filenames to the output .txt file
        with open(output_filepath, 'w') as f:
            for filename in valid_graph_filenames:
                f.write(f"{filename}\n")

        print(f"Successfully wrote {len(valid_graph_filenames)} verified filenames to {output_filepath}")


if __name__ == '__main__':

    create_split_files(output_dir = "../datasets/")

    print("\nData split files have been created successfully!")