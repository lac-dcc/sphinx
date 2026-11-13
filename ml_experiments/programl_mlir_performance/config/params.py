import yaml
import re
import os
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class ModelParams:
    node_embedding_dim: int
    positional_embedding_dim: int
    num_edge_types: int
    max_edge_position: int
    ggnn_iterations: int
    expected_vocab_size: int
    target_performance_metric: str


@dataclass
class TrainingParams:
    learning_rate: float
    graph_level_batch_size: int
    epochs: int


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
    splits_txt: str
    train_graphs_txt: str
    validation_graphs_txt: str
    test_graphs_txt: str
    processed: str
    labels: str
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

    loaded = Params(
        model=ModelParams(**resolved_config['model']),
        training=TrainingParams(**resolved_config['training']),
        environment=EnvironmentParams(**resolved_config['environment']),
        paths=PathParams(**resolved_config['paths'])
    )
    _apply_environment(loaded)
    return loaded


def _apply_environment(loaded: Params) -> None:
    """Override config.yaml from environment variables.

    With SPHINX_DATA set, the dataset paths, vocabulary and per-dialect model
    settings are derived from the published dataset layout (config/datasets.py)
    instead of config.yaml:

        SPHINX_DATA       directory holding the extracted dataset archives
        SPHINX_BENCHMARK  nb101 (default) or nb201
        SPHINX_DIALECT    stablehlo (default), linalg, arith, llvm, linalg_full
        SPHINX_TARGET     training_time (default) or test_accuracy
        SPHINX_POOL       split pool, stablehlo (90%) or linalg (10%);
                          default follows the dialect
        SPHINX_RUNS       where run folders and checkpoints are written
                          (default: $SPHINX_DATA/runs)
        SPHINX_CACHE      where preprocessed graphs are cached
                          (default: $SPHINX_DATA/processed)

    Independent of SPHINX_DATA: SPHINX_DEVICE, SPHINX_WORKERS, SPHINX_EPOCHS,
    SPHINX_BATCH_SIZE, SPHINX_ITERATIONS (GGNN message-passing iterations).
    SPHINX_POOLING (add or mean) is read by the model in GGNN_train.py.
    """
    env = os.environ
    if env.get("SPHINX_DEVICE"):
        loaded.environment.device = env["SPHINX_DEVICE"]
    if env.get("SPHINX_WORKERS"):
        loaded.environment.num_workers = int(env["SPHINX_WORKERS"])
    if env.get("SPHINX_EPOCHS"):
        loaded.training.epochs = int(env["SPHINX_EPOCHS"])
    if env.get("SPHINX_ITERATIONS"):
        loaded.model.ggnn_iterations = int(env["SPHINX_ITERATIONS"])
    if env.get("SPHINX_BATCH_SIZE"):
        loaded.training.graph_level_batch_size = int(env["SPHINX_BATCH_SIZE"])

    root = env.get("SPHINX_DATA")
    if not root:
        return
    from config import datasets
    settings = datasets.profile(
        root,
        env.get("SPHINX_BENCHMARK", "nb101"),
        env.get("SPHINX_DIALECT", "stablehlo"),
        env.get("SPHINX_TARGET", "training_time"),
        pool=env.get("SPHINX_POOL"),
        cache=env.get("SPHINX_CACHE"),
    )
    datasets.apply(loaded, settings)
    runs = env.get("SPHINX_RUNS", os.path.join(root, "runs"))
    loaded.paths.base = root
    loaded.paths.checkpoint = os.path.join(runs, "best_model.pt")
    loaded.paths.logs = os.path.join(runs, "ggnn_run_info.log")


params = load_params()