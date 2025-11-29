import os
import math
import shutil
import sys
import csv
import time
import logging
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.amp import GradScaler, autocast

from torch_geometric.data import Data, Dataset as PygDataset
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.nn.conv import MessagePassing

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

import utils.proto_python.program_graph_pb2 as program_graph_pb2
import utils.proto_python.util_pb2 as util_pb2

from config.params import params


tqdm_stdout = partial(tqdm, file=sys.stdout)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/ggnn_run_info.log', mode='w')
    ]
)


def log_blank_line():
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'stream'):
            handler.stream.write('\n')
            handler.flush()


class ProGraMLPygDataset(PygDataset):
    """
    This Dataset class implements a one-time pre-processing step
    with manual control. It checks for a metadata file to see if processing
    is complete. If not, it runs a processing method to convert raw .pb files
    into a set of processed .pt files.
    """

    def __init__(self, file_basenames: list, graph_src_dir: str, split: str,
                 token_to_idx: dict, max_samples: int = None, max_steps: int = None,
                 max_sample_size_mb: float = None, reprocess: bool = False):

        super().__init__()

        self.processed_path = os.path.join(params.paths.processed, split)
        self.file_basenames = file_basenames
        self.graph_src_dir = graph_src_dir
        self.label_src_dir = params.paths.labels

        self.label_key = "data_flow_value"
        self.selector_key = "data_flow_root_node"
        self.step_count_key = "data_flow_step_count"
        self.active_node_count_key = "data_flow_active_node_count"

        self.token_to_idx = token_to_idx
        self.node_selector_dim = params.model.node_selector_dim
        self.num_edge_types = params.model.num_edge_types
        self.max_edge_position = params.model.max_edge_position
        self.max_samples = max_samples
        self.max_steps = max_steps
        self.unknown_token_idx = self.token_to_idx.get("<unknown>", 0)

        # Store all parameters that define this dataset version
        self.unique_id = {
            "token_to_idx_len": len(self.token_to_idx),
            "node_selector_dim": self.node_selector_dim,
            "num_edge_types": self.num_edge_types,
            "max_edge_position": self.max_edge_position,
            "max_steps": self.max_steps,
            "max_samples": self.max_samples
        }

        metadata_path = os.path.join(self.processed_path, 'pre_process_info.pt')

        self.max_tensor_bytes = int(max_sample_size_mb * 1024 * 1024) if max_sample_size_mb is not None else None

        run_processing = True
        if not reprocess and os.path.exists(metadata_path):
            logging.info(f"Found existing metadata at {metadata_path}. Checking parameters...")
            metadata = torch.load(metadata_path)
            # Only skip processing if the parameters match exactly
            if metadata.get("params") == self.unique_id:
                self.num_samples = metadata.get("num_samples")
                logging.info(f"Parameters match. Loading existing dataset with {self.num_samples} samples.")
                run_processing = False
            else:
                logging.warning("Parameters have changed. Reprocessing dataset.")

        if run_processing:
            self._process_data()

        # Load the final metadata to get the sample count
        final_metadata = torch.load(metadata_path)
        self.num_samples = final_metadata.get("num_samples", 0)
        logging.info(f"Dataset ready. Found {self.num_samples} samples.")

    def _process_data(self):
        """
        The private processing method. Runs once to convert all raw .pb data
        into individual .pt files for fast loading.
        """
        logging.info(f"Processing raw data and saving individual .pt files to '{self.processed_path}'...")
        os.makedirs(self.processed_path, exist_ok=True)

        idx = 0
        max_samples = self.max_samples

        for basename in tqdm_stdout(self.file_basenames, desc="Processing graphs"):
            if max_samples is not None and idx >= max_samples:
                logging.info(f"Reached max_samples limit of {max_samples}. Stopping processing.")
                break

            graph_filename = f"{basename}ProgramGraph.pb"
            label_filename = f"{basename}ProgramGraphFeaturesList.pb"

            graph_file_path = os.path.join(self.graph_src_dir, graph_filename)
            label_file_path = os.path.join(self.label_src_dir, label_filename)

            if not (os.path.exists(label_file_path) and os.path.exists(graph_file_path)):
                continue

            graph_proto = program_graph_pb2.ProgramGraph()
            with open(graph_file_path, 'rb') as f:
                graph_proto.ParseFromString(f.read())

            features_list_proto = util_pb2.ProgramGraphFeaturesList()
            with open(label_file_path, 'rb') as f:
                features_list_proto.ParseFromString(f.read())

            for graph_features in features_list_proto.graph:
                if max_samples is not None and idx >= max_samples:
                    break

                data = self._create_data_object(graph_proto, graph_features)

                if data:
                    torch.save(data, os.path.join(self.processed_path, f'data_{idx}.pt'))
                    idx += 1

        logging.info(f"Saving metadata for {idx} processed samples.")
        torch.save({'params': self.unique_id, 'num_samples': idx},
                   os.path.join(self.processed_path, 'pre_process_info.pt'))

    def _create_data_object(self, graph_proto, label_features):
        """Helper function to convert protobufs to a single PyG Data object."""
        # Check step count filter first
        if self.max_steps is not None:
            if self.step_count_key in label_features.features.feature:
                step_val = label_features.features.feature[self.step_count_key].int64_list.value
                if step_val and int(step_val[0]) > self.max_steps:
                    return None  # Return None to skip this sample

        num_nodes = len(graph_proto.node)
        if num_nodes == 0: return None

        # Node Text, Selectors, and Labels
        node_texts_indices = [self.token_to_idx.get(node.text, self.unknown_token_idx)
                              for node in graph_proto.node]

        node_features_map = label_features.node_features.feature_list
        labels_list = node_features_map.get(self.label_key)
        selectors_list = node_features_map.get(self.selector_key)

        node_labels = torch.zeros(num_nodes, dtype=torch.float)
        if labels_list and len(labels_list.feature) == num_nodes:
            for i, feature in enumerate(labels_list.feature):
                if feature.int64_list.value:
                    node_labels[i] = float(feature.int64_list.value[0])

        node_selectors_raw = torch.zeros(num_nodes, dtype=torch.long)
        if selectors_list and len(selectors_list.feature) == num_nodes:
            for i, feature in enumerate(selectors_list.feature):
                if feature.int64_list.value and feature.int64_list.value[0] == 1:
                    node_selectors_raw[i] = 1

        # Final Tensors
        x_text_indices = torch.LongTensor(node_texts_indices)
        node_selectors = functional.one_hot(node_selectors_raw, num_classes=self.node_selector_dim).float()
        y_labels = node_labels.unsqueeze(1)

        # Edge Features
        edge_src, edge_tgt, edge_type, edge_pos = [], [], [], []
        num_base_edge_types = self.num_edge_types // 2
        for edge_p in graph_proto.edge:
            edge_src.append(edge_p.source)
            edge_tgt.append(edge_p.target)
            edge_type.append(edge_p.flow)
            edge_pos.append(edge_p.position)
            edge_src.append(edge_p.target)
            edge_tgt.append(edge_p.source)
            edge_type.append(edge_p.flow + num_base_edge_types)
            edge_pos.append(edge_p.position)

        edge_index = torch.LongTensor([edge_src, edge_tgt])
        edge_type = torch.LongTensor(edge_type)
        edge_positions = torch.LongTensor(edge_pos).clamp(0, self.max_edge_position)

        data = Data(
            x_text_indices=x_text_indices, # node text to index (through vocab)
            node_selectors=node_selectors, # indicates which vertice is taken as reference
            edge_index=edge_index, # Indices of source and target nodes for edges
            edge_type=edge_type,
            edge_positions=edge_positions,
            y=y_labels,
            num_nodes=num_nodes # amount of vertices in the graph
        )

        if self.max_tensor_bytes is not None:
            total_bytes = 0
            for key, item in data:
                if torch.is_tensor(item):
                    total_bytes += item.nelement() * item.element_size()

            if total_bytes > self.max_tensor_bytes:
                # This sample is too large, skip it by returning None
                return None

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.processed_path, f'data_{idx}.pt'),
            weights_only=False
        )
        return data


def get_sinusoidal_positional_embeddings(max_pos, embedding_dim):
    """Generates sinusoidal positional embeddings."""
    position = torch.arange(max_pos).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pos_emb = torch.zeros(max_pos, embedding_dim)
    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)
    return pos_emb


# noinspection PyAbstractClass
class ProGraMLGGNNLayer(MessagePassing):
    """The Gated Graph Neural Network layer, adapted for ProGraML."""

    def __init__(self, hidden_dim, num_edge_types, positional_embedding_dim):
        super(ProGraMLGGNNLayer, self).__init__(aggr='mean')
        self.hidden_dim = hidden_dim

        self.edge_type_mlps = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)
        ])

        self.position_gating_mlp = nn.Sequential(
            nn.Linear(positional_embedding_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_type, edge_positions_embedded):
        aggregated_messages = self.propagate(edge_index, x=x, edge_type=edge_type,
                                             edge_positions_embedded=edge_positions_embedded)
        updated_x = self.gru_cell(aggregated_messages, x)
        return updated_x

    # noinspection PyMethodOverriding
    def message(self, x_j, edge_type, edge_positions_embedded):
        position_gate = 2 * self.position_gating_mlp(edge_positions_embedded)
        gated_x_j = x_j * position_gate

        messages = torch.zeros_like(gated_x_j)
        for i in range(len(self.edge_type_mlps)):
            type_mask = (edge_type == i)
            # noinspection PyUnresolvedReferences
            if type_mask.any():
                mlp_output = self.edge_type_mlps[i](gated_x_j[type_mask])
                messages[type_mask] = mlp_output.to(messages.dtype)
                # messages[type_mask] = self.edge_type_mlps[i](gated_x_j[type_mask])
        return messages


class ProGraMLReadout(nn.Module):
    """The readout head for producing final node classifications."""

    def __init__(self, hidden_dim, initial_embedding_dim):
        super(ProGraMLReadout, self).__init__()
        # Using a simpler but effective readout: MLP on concatenated features
        # This is a common and robust alternative to the paper's specific formula.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + initial_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_t, h_0):
        combined_features = torch.cat([h_t, h_0], dim=-1)
        return self.mlp(combined_features)


class ProGraMLNetPyG(nn.Module):
    """The complete ProGraML Graph Neural Network model."""

    def __init__(self, vocab_size, device):
        super(ProGraMLNetPyG, self).__init__()
        self.ggnn_iterations = params.model.ggnn_iterations
        self.device = device
        self.initial_embedding_dim = params.model.node_embedding_dim + params.model.node_selector_dim

        self.node_text_embedding = nn.Embedding(vocab_size, params.model.node_embedding_dim, padding_idx=0)

        self.edge_positional_encodings = get_sinusoidal_positional_embeddings(
            params.model.max_edge_position + 1, params.model.positional_embedding_dim
        ).to(device)

        self.ggnn_layer = ProGraMLGGNNLayer(
            hidden_dim=self.initial_embedding_dim,
            num_edge_types=params.model.num_edge_types,
            positional_embedding_dim=params.model.positional_embedding_dim
        )

        self.readout = ProGraMLReadout(
            hidden_dim=self.initial_embedding_dim,
            initial_embedding_dim=self.initial_embedding_dim
        )

    def forward(self, data):
        text_embeds = self.node_text_embedding(data.x_text_indices)
        h_0 = torch.cat([text_embeds, data.node_selectors], dim=-1)

        edge_pos_embeds = self.edge_positional_encodings[data.edge_positions]

        h_current = h_0
        for _ in range(self.ggnn_iterations):
            h_current = self.ggnn_layer(h_current, data.edge_index, data.edge_type, edge_pos_embeds)
        h_t = h_current

        logits = self.readout(h_t, h_0)
        return logits


def train_by_epoch(model, train_loader, val_loader, optimizer, criterion, device):
    """
    Trains the model for a fixed number of epochs, evaluating
    after each epoch and saving the best checkpoint based on validation F1 score.
    """
    epochs = params.training.epochs
    checkpoint_dir = os.path.dirname(params.paths.checkpoint)
    best_checkpoint_path = params.paths.checkpoint

    os.makedirs(checkpoint_dir, exist_ok=True)

    scaler = GradScaler('cuda')  # For mixed-precision
    best_val_f1 = -1.0

    logging.info(f"Starting training for {epochs} epochs...")
    logging.info(f"Batch size: {train_loader.batch_size}. Validation after every epoch.")
    log_blank_line()

    for epoch in range(1, epochs + 1):

        # --- 1. Training Phase ---
        model.train()
        total_train_loss = 0.0
        total_train_nodes = 0

        train_pbar = tqdm_stdout(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", position=0, leave=False,
                                 ncols=100)

        for batch in train_pbar:
            batch = batch.to(device)
            if batch.num_nodes == 0:
                continue

            optimizer.zero_grad()

            with autocast('cuda'):
                out = model(batch)
                loss = criterion(out, batch.y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item() * batch.num_nodes
            total_train_nodes += batch.num_nodes

            avg_loss_so_far = total_train_loss / total_train_nodes if total_train_nodes > 0 else 0
            train_pbar.set_description(f"Epoch {epoch}/{epochs} [Train] Loss: {avg_loss_so_far:.4f}")

        avg_train_loss = total_train_loss / total_train_nodes if total_train_nodes > 0 else 0

        # --- 2. Validation Phase ---
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)

        # Log results for this epoch
        log_blank_line()
        logging.info(f"Epoch {epoch}/{epochs} Complete")
        logging.info(f"  Train Loss: {avg_train_loss:.4f}")
        logging.info(
            f"  Val Loss:   {val_loss:.4f} | Val F1: {val_f1:.4f} | "
            f"Val Prec: {val_prec:.4f} | Val Rec: {val_recall:.4f}"
        )

        # --- 3. Checkpointing Phase ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1_score': val_f1,
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_recall,
            'loss': val_loss
        }

        periodic_path = os.path.join(checkpoint_dir, f"liveness_epoch_{epoch}_f1={val_f1:.3f}.pt")
        torch.save(checkpoint, periodic_path)

        try:
            shutil.copy(params.paths.logs, checkpoint_dir)
        except Exception as e:
            logging.warning(f"Could not copy log file: {e}")

        if val_f1 > best_val_f1:
            logging.info(
                f"New best validation F1 score: {val_f1:.4f} (previously {best_val_f1:.4f}). Saving best model...")
            best_val_f1 = val_f1
            torch.save(checkpoint, best_checkpoint_path)

        log_blank_line()

    # --- End of Training ---
    logging.info("Training complete.")
    logging.info(f"Best model was saved to '{best_checkpoint_path}' with F1 score: {best_val_f1:.4f}")


def evaluate(model, loader, criterion, device):
    """
    Evaluates the performance of a model on a given dataset loader.
    """
    model.eval()
    total_loss = 0
    total_nodes = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm_stdout(loader, desc="Validating", position=0, leave=False, ncols=100):
            batch = batch.to(device)
            if batch.num_nodes == 0:
                continue

            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

            preds = (torch.sigmoid(out) > 0.5)
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    if not all_preds:
        return 0, 0, 0, 0, 0

    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_labels = torch.cat(all_labels, dim=0).numpy().flatten()

    logging.info("\n" + classification_report(all_labels, all_preds, zero_division=0, digits=4))

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    avg_loss = total_loss / total_nodes if total_nodes > 0 else 0
    return avg_loss, accuracy, precision, recall, f1


def load_vocabulary_from_csv(csv_path, token_column_index=3, unknown_token="<unknown>", delimiter='\t'):
    token_to_idx_dict = {unknown_token: 0}
    idx_to_token_dict = {0: unknown_token}
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
                        idx_to_token_dict[next_idx] = token
                        next_idx += 1
        logging.info(f"Loaded vocabulary of size {len(token_to_idx_dict)} from {csv_path}")
    except FileNotFoundError:
        logging.critical(f"Error: File not found at {csv_path}")
    except IndexError:
        logging.critical(f"Error: Token column index {token_column_index} out of range for a row in {csv_path}.")
    except Exception as e:
        logging.critical(f"Could not load vocabulary from {csv_path}: {e}")

    return token_to_idx_dict, idx_to_token_dict, len(token_to_idx_dict)


def load_filenames_from_txt(filepath):
    """Reads a .txt file and returns a list of filenames."""
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.critical(f"Warning: File not found at {filepath}")
        return []


def main():
    device = torch.device(params.environment.device)
    logging.info(f"Using device: {params.environment.device}")

    token_to_idx, idx_to_token, vocab_size = load_vocabulary_from_csv(params.paths.vocab)
    assert vocab_size == 608  # Ensure the vocabulary size matches the expected size

    log_blank_line()
    train_graphs_filenames = load_filenames_from_txt(params.paths.train_graphs_txt)
    train_dataset = ProGraMLPygDataset(
        file_basenames=train_graphs_filenames,
        graph_src_dir=params.paths.train_graphs,
        split='train',
        token_to_idx=token_to_idx,
        max_samples=None,
        max_steps=30,
        max_sample_size_mb=params.training.max_sample_size_mb,
        reprocess=True
    )

    log_blank_line()
    validation_graphs_filenames = load_filenames_from_txt(params.paths.validation_graphs_txt)
    validation_dataset = ProGraMLPygDataset(
        file_basenames=validation_graphs_filenames,
        graph_src_dir=params.paths.validation_graphs,
        split='validation',
        token_to_idx=token_to_idx,
        max_samples=None,
        max_steps=30,
        max_sample_size_mb=params.training.max_sample_size_mb,
        reprocess=True
    )

    train_loader = PygDataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=params.training.graph_level_batch_size,
        num_workers=params.environment.num_workers
    )

    validation_loader = PygDataLoader(
        validation_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=params.training.graph_level_batch_size,
        num_workers=params.environment.num_workers
    )

    log_blank_line()
    model = ProGraMLNetPyG(
        vocab_size=vocab_size,
        device=device,
    ).to(device)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=params.training.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    log_blank_line()
    train_by_epoch(model, train_loader, validation_loader, optimizer, criterion, device)


if __name__ == '__main__':
    start_time = time.time()

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    log_blank_line()
    logging.info(f"Execution time: {elapsed_time:.5f} seconds")