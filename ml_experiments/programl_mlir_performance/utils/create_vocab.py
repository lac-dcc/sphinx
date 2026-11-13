"""
Build the vocabulary file of a dialect from a directory of ProGraML graphs.

The vocabulary is the frequency table of the node texts of all the graphs, in
descending order of frequency (ties in ascending order of text). The file is
tab-separated with a header line and four columns:

    cumulative_frequency  cumulative_node_frequency  count  text

The model only reads the last column: a node text gets the index of its line,
and index 0 is kept for texts that are not in the file.

The files in vocab/ were made with the create_vocab tool of ProGraML
(tasks/dataflow/dataset/create_vocab.cc); this script writes the same file.

Run from the same directory as GGNN_train.py:

    python -m utils.create_vocab <graph_dir> <output.csv>
"""

import multiprocessing
import os
import sys
from collections import Counter

import utils.proto_python.program_graph_pb2 as program_graph_pb2


def count_texts(path):
    graph = program_graph_pb2.ProgramGraph()
    with open(path, "rb") as f:
        graph.ParseFromString(f.read())
    return Counter(node.text for node in graph.node)


def build(graph_dir, workers):
    files = [
        os.path.join(graph_dir, name)
        for name in os.listdir(graph_dir)
        if name.endswith(".ProgramGraph.pb")
    ]
    total = Counter()
    with multiprocessing.Pool(workers) as pool:
        for counts in pool.imap_unordered(count_texts, files, chunksize=512):
            total.update(counts)
    return len(files), total


def write(counts, path):
    total = sum(counts.values())
    cumulative = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("cumulative_frequency\tcumulative_node_frequency\tcount\ttext\n")
        for text, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            cumulative += count
            share = f"{cumulative / total:.4g}"
            f.write(f"{share}\t{share}\t{count}\t{text}\n")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    graph_dir, output = sys.argv[1:]
    workers = int(os.environ.get("SPHINX_WORKERS", os.cpu_count() or 1))
    graphs, counts = build(graph_dir, workers)
    if not graphs:
        sys.exit(f"No .ProgramGraph.pb files in {graph_dir}")
    write(counts, output)
    print(f"{graphs} graphs, {len(counts)} node texts, written to {output}")


if __name__ == "__main__":
    main()
