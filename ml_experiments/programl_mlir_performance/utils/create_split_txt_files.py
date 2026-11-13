"""
Write the train, validation and test split files of a set of programs.

A split file lists one program per line, by the name of its graph without the
extension (model_12). The programs of <graph_dir> are shuffled with a fixed
seed and divided 60% / 20% / 20% into

    <output_dir>/train_files_<pool>.txt
    <output_dir>/validation_files_<pool>.txt
    <output_dir>/test_files_<pool>.txt

<pool> names the set of programs. The paper divides each benchmark into two
pools, named after the dialects they were first used with: stablehlo (90% of
the programs, used to train the source models) and linalg (10%, used for every
target dialect). The names are sorted before shuffling, so a rerun on the same
directory gives the same files on any machine.

Run from the same directory as GGNN_train.py:

    python -m utils.create_split_txt_files <graph_dir> <output_dir> <pool>
"""

import argparse
import os
import random

SUFFIX = ".ProgramGraph.pb"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph_dir")
    parser.add_argument("output_dir")
    parser.add_argument("pool")
    parser.add_argument("--train", type=float, default=0.6, help="fraction for training (default 0.6)")
    parser.add_argument("--validation", type=float, default=0.2,
                        help="fraction for validation (default 0.2); the rest is the test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = sorted(name[:-len(SUFFIX)] for name in os.listdir(args.graph_dir) if name.endswith(SUFFIX))
    if not names:
        raise SystemExit(f"No {SUFFIX} files in {args.graph_dir}")
    random.Random(args.seed).shuffle(names)

    n_train = int(len(names) * args.train)
    n_validation = int(len(names) * args.validation)
    parts = {
        "train": names[:n_train],
        "validation": names[n_train:n_train + n_validation],
        "test": names[n_train + n_validation:],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for part, members in parts.items():
        path = os.path.join(args.output_dir, f"{part}_files_{args.pool}.txt")
        with open(path, "w") as f:
            f.writelines(f"{name}\n" for name in members)
        print(f"{path}: {len(members)} programs")


if __name__ == "__main__":
    main()
