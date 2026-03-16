import yaml
import re
import os
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class ModelParams:
    node_embedding_dim: int
    node_selector_dim: int
    positional_embedding_dim: int
    num_edge_types: int
    max_edge_position: int
    ggnn_iterations: int


@dataclass
class TrainingParams:
    learning_rate: float
    batch_size_vertices: int
    graph_level_batch_size: int
    num_train_samples: int
    validation_interval_samples: int
    max_sample_size_mb: int


@dataclass
class EnvironmentParams:
    device: str
    num_workers: int


@dataclass
class PathParams:
    vocab: str
    logs: str
    base: str
    dataset: str
    graphs: str
    train_graphs: str
    train_graphs_txt: str
    validation_graphs: str
    validation_graphs_txt: str
    test_graphs: str
    test_graphs_txt: str
    labels: str
    processed: str
    checkpoint: str


@dataclass
class Params:
    model: ModelParams
    training: TrainingParams
    environment: EnvironmentParams
    paths: PathParams


def _get_nested_value(config: Dict, path: str) -> Any:
    """Access a nested value in a dict using a dot-separated path."""
    keys = path.split('.')
    value = config
    for key in keys:
        value = value[key]
    return value


def _resolve_config_variables(config: Dict) -> Dict:
    """Recursively and iteratively resolve `${...}` variables in the config."""
    placeholder_pattern = re.compile(r'\$\{([^}]+)\}')

    def resolver(current_value: Any) -> Any:
        if isinstance(current_value, str):
            resolved_str = current_value
            # Loop up to 10 times to resolve nested variables like ${paths.dataset} -> ${paths.base}/...
            for _ in range(10):
                # Find all placeholders and replace them with their values from the top-level config
                new_str = placeholder_pattern.sub(
                    lambda m: str(_get_nested_value(config, m.group(1))),
                    resolved_str
                )

                if new_str == resolved_str:
                    return new_str

                resolved_str = new_str

            raise ValueError(
                f"Could not resolve variables in '{current_value}' after 10 iterations. Check for circular dependencies.")

        elif isinstance(current_value, dict):
            return {k: resolver(v) for k, v in current_value.items()}

        elif isinstance(current_value, list):
            return [resolver(item) for item in current_value]

        else:
            return current_value

    return resolver(config)


def load_params() -> Params:
    """Loads parameters from a YAML file into a nested dataclass object."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "config.yaml")
    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)

    resolved_config = _resolve_config_variables(config_data)

    return Params(
        model=ModelParams(**resolved_config['model']),
        training=TrainingParams(**resolved_config['training']),
        environment=EnvironmentParams(**resolved_config['environment']),
        paths=PathParams(**resolved_config['paths'])
    )


params = load_params()