import os
import sys
import json
import csv
import torch
import multiprocessing
import logging
from functools import partial
from pathlib import Path
from torch_geometric.data import Data
from tqdm import tqdm

import utils.proto_python.program_graph_pb2 as program_graph_pb2

from config.params import params

VOCAB_DICT = {}


def load_vocabulary_from_csv(csv_path, token_column_index=3, unknown_token="<unknown>", delimiter='\t'):
    global VOCAB_DICT
    token_to_idx_dict = {unknown_token: 0}
    next_idx = 1
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader) # Skip header
            for row in reader:
                if len(row) > token_column_index:
                    token = row[token_column_index].strip()
                    if token not in token_to_idx_dict:
                        token_to_idx_dict[token] = next_idx
                        next_idx += 1
    except FileNotFoundError:
        logging.critical(f"Error: File not found at {csv_path}")
    except IndexError:
        logging.critical(f"Error: Token column index {token_column_index} out of range for a row in {csv_path}.")
    except Exception as e:
        logging.critical(f"Could not load vocabulary from {csv_path}: {e}")

    assert len(token_to_idx_dict) == params.model.expected_vocab_size
    VOCAB_DICT = token_to_idx_dict
    return token_to_idx_dict


def process_single_graph(args):
    model_name, graph_path, out_path, performance_label, num_edge_types, max_edge_pos = args

    if os.path.exists(out_path):
        return "skipped"

    try:
        with open(graph_path, 'rb') as f:
            graph_def = program_graph_pb2.ProgramGraph()
            graph_def.ParseFromString(f.read())

        num_nodes = len(graph_def.node)
        if num_nodes == 0:
            return "empty"

        unknown_idx = VOCAB_DICT.get("<unknown>", 0)
        x_indices = [VOCAB_DICT.get(node.text, unknown_idx) for node in graph_def.node]
        x = torch.tensor(x_indices, dtype=torch.long)

        edge_src, edge_tgt = [], []
        edge_type, edge_pos = [], []

        half_types = num_edge_types // 2

        for edge in graph_def.edge:
            # Forward Edge
            edge_src.append(edge.source)
            edge_tgt.append(edge.target)
            edge_type.append(edge.flow)
            edge_pos.append(edge.position)

            # Backward Edge
            edge_src.append(edge.target)
            edge_tgt.append(edge.source)
            edge_type.append(edge.flow + half_types)  # Offset type for backward
            edge_pos.append(edge.position)

        edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        edge_attr_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr_pos = torch.tensor(edge_pos, dtype=torch.long).clamp(0, max_edge_pos)

        y = torch.tensor([float(performance_label)], dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_attr_type,
            edge_positions=edge_attr_pos,
            y=y,
            num_nodes=num_nodes
        )

        torch.save(data, out_path)
        return "processed"

    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        return "error"


def run_preprocessing():
    logging.info("PRE-PROCESSING CHECK")

    raw_dir = Path(params.paths.graphs)
    processed_dir = Path(params.paths.processed)
    metrics_path = Path(params.paths.labels)
    vocab_path = Path(params.paths.vocab)

    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading metrics from {metrics_path}...")
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    logging.info(f"Loading vocab from {vocab_path}...")
    load_vocabulary_from_csv(vocab_path)
    logging.info(f"Vocab size: {len(VOCAB_DICT)}")

    tasks = []
    skipped_count = 0

    logging.info(f"Scanning files directly from directory: {raw_dir}...")

    pb_files = [f for f in os.listdir(raw_dir) if f.endswith('.ProgramGraph.pb')]
    try:
        pb_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    except (IndexError, ValueError):
        pb_files.sort()

    for pb_filename in pb_files:
        model_name = pb_filename.split('.ProgramGraph.pb')[0]
        pt_filename = f"{model_name}.pt"

        in_path = raw_dir / pb_filename
        out_path = processed_dir / pt_filename

        if out_path.exists():
            skipped_count += 1
            continue

        if model_name not in all_metrics:
            logging.warning(f"Label missing for {model_name} in metrics. Skipping.")
            continue

        performance_metric = all_metrics[model_name].get(params.model.target_performance_metric, 0.0)

        tasks.append((
            model_name,
            str(in_path),
            str(out_path),
            performance_metric,
            params.model.num_edge_types,
            params.model.max_edge_position
        ))

    logging.info(f"Found {len(all_metrics)} total models.")
    logging.info(f"Skipping {skipped_count} already processed files.")
    logging.info(f"Queuing {len(tasks)} files for processing...")

    if not tasks:
        logging.info("All files processed. Ready to train.")
        return

    num_workers = params.environment.num_workers

    def init_worker(vocab_p):
        load_vocabulary_from_csv(vocab_p)

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(str(vocab_path),)) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_graph, tasks, chunksize=100),
            total=len(tasks),
            desc="Preprocessing",
            unit="graph"
        ))

    logging.info("Preprocessing complete.")