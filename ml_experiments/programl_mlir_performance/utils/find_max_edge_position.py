"""
Report the largest edge position in a directory of ProGraML graphs.

The model's max_edge_position must be set per dialect to the value found here
(positions above it are clamped during preprocessing).

Run from the same directory as GGNN_train.py:

    python -m utils.find_max_edge_position <graph_dir> [<graph_dir> ...]
"""

import multiprocessing
import os
import sys

import utils.proto_python.program_graph_pb2 as program_graph_pb2


def max_position(path):
    graph = program_graph_pb2.ProgramGraph()
    with open(path, "rb") as f:
        graph.ParseFromString(f.read())
    return max((edge.position for edge in graph.edge), default=0)


def scan(graph_dir, workers):
    files = [
        os.path.join(graph_dir, name)
        for name in os.listdir(graph_dir)
        if name.endswith(".ProgramGraph.pb")
    ]
    with multiprocessing.Pool(workers) as pool:
        best = max(pool.imap_unordered(max_position, files, chunksize=512), default=0)
    return len(files), best


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    workers = int(os.environ.get("SPHINX_WORKERS", os.cpu_count() or 1))
    for graph_dir in sys.argv[1:]:
        count, best = scan(graph_dir, workers)
        print(f"{graph_dir}: {count} graphs, max_edge_position = {best}", flush=True)


if __name__ == "__main__":
    main()
