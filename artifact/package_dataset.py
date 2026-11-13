"""
Build the dataset archives of the artifact.

Each archive holds one benchmark/kind/dialect and extracts to the layout that
reproduce/datasets.py expects:

    nb101/graphs/<dialect>/model_<id>.ProgramGraph.pb
    nb101/mlir/<dialect>/model_<id>.mlir

Large tensor constants (network weights printed as hexadecimal blobs) are
replaced by `dense_resource<__elided__>`, the form MLIR itself prints with
--mlir-elide-elementsattrs-if-larger. In the graphs this only touches the
`full_text` feature of constant nodes; node labels and edges, which are all the
model reads, are unchanged (checked with --verify).

    python artifact/package_dataset.py --verify            # check a sample, write nothing
    python artifact/package_dataset.py <out_dir> [name...] # build archives

Run from the repository root.
"""

import hashlib
import io
import json
import multiprocessing
import os
import random
import re
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "ml_experiments", "programl_mlir_performance"))
import utils.proto_python.program_graph_pb2 as program_graph_pb2  # noqa: E402

D = os.environ.get("SPHINX_SOURCE", os.path.expanduser("~/Desktop"))

# archive name -> (source directory, path inside the archive)
SOURCES = {
    "nb101_graphs_stablehlo": (f"{D}/Backup/DatasetFT/graphs", "nb101/graphs/stablehlo"),
    "nb101_graphs_linalg": (f"{D}/Backup/DatasetFT/graphs_linalg_101", "nb101/graphs/linalg"),
    "nb101_graphs_arith": (f"{D}/Backup/DatasetFT/graphs_arith", "nb101/graphs/arith"),
    "nb101_graphs_llvm": (f"{D}/Backup/DatasetFT/graphs_llvm", "nb101/graphs/llvm"),
    "nb101_graphs_linalg_full": (f"{D}/Backup/DatasetIncremental/graphs_linalg", "nb101/graphs/linalg_full"),
    "nb101_mlir_stablehlo": (f"{D}/Backup/mlir", "nb101/mlir/stablehlo"),
    "nb101_mlir_linalg": (f"{D}/Backup/DatasetFT/mlir_linalg", "nb101/mlir/linalg"),
    "nb101_mlir_arith": (f"{D}/Backup/DatasetFT/mlir_arith", "nb101/mlir/arith"),
    "nb101_mlir_llvm": (f"{D}/Backup/DatasetFT/mlir_llvm", "nb101/mlir/llvm"),
    "nb101_mlir_linalg_full": (f"{D}/Backup/DatasetIncremental/mlir_linalg", "nb101/mlir/linalg_full"),
    "nb201_graphs_stablehlo": (f"{D}/NASBench_Dataset/graphs", "nb201/graphs/stablehlo"),
    "nb201_graphs_linalg": (f"{D}/NASBench_Dataset/graphs_linalg", "nb201/graphs/linalg"),
    "nb201_graphs_arith": (f"{D}/NASBench_Dataset/graphs_arith", "nb201/graphs/arith"),
    "nb201_graphs_llvm": (f"{D}/NASBench_Dataset/graphs_llvm", "nb201/graphs/llvm"),
    "nb201_mlir_stablehlo": (f"{D}/NASBench201_Dataset/mlir/cifar10-valid", "nb201/mlir/stablehlo"),
    "nb201_mlir_linalg": (f"{D}/NASBench201_Dataset/linalg", "nb201/mlir/linalg"),
    "nb201_mlir_arith": (f"{D}/NASBench201_Dataset/arith", "nb201/mlir/arith"),
    "nb201_mlir_llvm": (f"{D}/NASBench201_Dataset/llvm", "nb201/mlir/llvm"),
}
LABELS = {
    "nb101/all_metrics.json": f"{D}/NASBench_Dataset/all_metrics.json",
    "nb201/all_metrics_cifar10-valid.json": f"{D}/NASBench_Dataset/all_metrics_cifar10-valid.json",
}

HEX_BLOB = re.compile(rb'dense<"0x[0-9A-Fa-f]{64,}">')
ELIDED = b"dense_resource<__elided__>"


def model_id(name):
    return int(name.split("_")[1].split(".")[0])


def elide_graph(blob):
    graph = program_graph_pb2.ProgramGraph()
    graph.ParseFromString(blob)
    changed = False
    for node in graph.node:
        feature = node.features.feature.get("full_text")
        if feature is None:
            continue
        values = feature.bytes_list.value
        for i, value in enumerate(values):
            if b'dense<"0x' in value:
                new = HEX_BLOB.sub(ELIDED, value)
                if new != value:
                    values[i] = new
                    changed = True
    return graph.SerializeToString() if changed else blob


def transform(path):
    with open(path, "rb") as f:
        blob = f.read()
    if path.endswith(".pb"):
        return elide_graph(blob)
    return HEX_BLOB.sub(ELIDED, blob)


def model_inputs(blob):
    """What preprocess.py reads from a graph."""
    graph = program_graph_pb2.ProgramGraph()
    graph.ParseFromString(blob)
    return ([(n.type, n.text, n.function, n.block) for n in graph.node],
            [(e.flow, e.position, e.source, e.target) for e in graph.edge])


def verify(samples=300):
    random.seed(0)
    for name, (src, _) in SOURCES.items():
        files = random.sample(sorted(os.listdir(src)), samples)
        before = after = changed = 0
        for fname in files:
            path = os.path.join(src, fname)
            raw = open(path, "rb").read()
            new = transform(path)
            before += len(raw)
            after += len(new)
            changed += new != raw
            assert b'dense<"0x' not in new or not HEX_BLOB.search(new), path
            if fname.endswith(".pb"):
                assert model_inputs(raw) == model_inputs(new), path
        print(f"{name}: {changed}/{samples} files changed, "
              f"{before / samples / 1e3:.1f} KB -> {after / samples / 1e3:.1f} KB per file, "
              f"model inputs identical", flush=True)


def build(name, out_dir, workers):
    src, arc = SOURCES[name]
    files = sorted(os.listdir(src), key=model_id)
    out = os.path.join(out_dir, f"{name}.tar.zst")
    start = time.time()
    with open(out + ".part", "wb") as sink:
        zstd = subprocess.Popen(["zstd", "-12", "--long=27", "-T0", "-q", "-c"],
                                stdin=subprocess.PIPE, stdout=sink)
        total = 0
        with tarfile.open(fileobj=zstd.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar, \
                multiprocessing.Pool(workers) as pool:
            paths = [os.path.join(src, f) for f in files]
            for fname, blob in zip(files, pool.imap(transform, paths, chunksize=256)):
                info = tarfile.TarInfo(f"{arc}/{fname}")
                info.size = len(blob)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(blob))
                total += len(blob)
        zstd.stdin.close()
        if zstd.wait() != 0:
            raise RuntimeError(f"zstd failed for {name}")
    os.replace(out + ".part", out)
    entry = {"archive": os.path.basename(out), "files": len(files),
             "uncompressed_bytes": total, "compressed_bytes": os.path.getsize(out),
             "extracts_to": arc}
    print(f"{name}: {len(files)} files, {total / 1e9:.2f} GB -> "
          f"{entry['compressed_bytes'] / 1e9:.3f} GB in {time.time() - start:.0f}s", flush=True)
    return entry


def build_labels(out_dir):
    out = os.path.join(out_dir, "labels.tar.zst")
    with open(out, "wb") as sink:
        zstd = subprocess.Popen(["zstd", "-19", "-T0", "-q", "-c"], stdin=subprocess.PIPE, stdout=sink)
        with tarfile.open(fileobj=zstd.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for arc, src in LABELS.items():
                info = tarfile.TarInfo(arc)
                info.size = os.path.getsize(src)
                info.mode = 0o644
                with open(src, "rb") as f:
                    tar.addfile(info, f)
        zstd.stdin.close()
        zstd.wait()
    return {"archive": "labels.tar.zst", "files": len(LABELS),
            "compressed_bytes": os.path.getsize(out), "extracts_to": "nb101/, nb201/"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--verify":
        verify()
        return
    out_dir = sys.argv[1]
    names = sys.argv[2:] or list(SOURCES)
    os.makedirs(out_dir, exist_ok=True)
    workers = int(os.environ.get("SPHINX_WORKERS", os.cpu_count() or 1))
    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest["labels"] = build_labels(out_dir)
    for name in names:
        manifest[name] = build(name, out_dir, workers)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    for entry in manifest.values():
        entry["sha256"] = sha256(os.path.join(out_dir, entry["archive"]))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
