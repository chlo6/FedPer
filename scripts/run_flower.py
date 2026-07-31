from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import wandb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.artifact_contract import load_validated_artifact
from redo_by_sara.config import ExperimentConfig, load_config
from redo_by_sara.federated import (
    ClientPartition,
    build_client_partitions,
    build_loader,
    client_class_ids,
    client_local_label_map,
    create_federated_model,
    create_partition_summary,
    evaluate_model,
    get_base_parameters,
    get_head_state,
    regression_r2_score,
    save_partition_summary,
    save_round_history,
    set_base_parameters,
    set_head_state,
    train_local_model,
)


def _build_wandb_config(
    config: ExperimentConfig,
    artifact: dict[str, object],
    client_subjects: dict[str, list[str]],
) -> dict[str, Any]:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    return {
        "seed": config.seed,
        "task": config.training.task,
        "artifact_name": config.artifact_name,
        "dataset_root": str(config.data.dataset_root),
        "selected_sensors": config.data.selected_sensors,
        "selected_channels": artifact["summary"]["selected_channels"],
        "num_train": artifact["summary"]["num_train"],
        "num_val": artifact["summary"]["num_val"],
        "num_test": artifact["summary"]["num_test"],
        "sample_shape": artifact["summary"]["sample_shape"],
        "num_clients": federated.num_clients,
        "num_rounds": federated.num_rounds,
        "local_epochs": federated.local_epochs,
        "client_subjects": client_subjects,
        "algorithm": "fedper",
        "shared_layers": "features",
        "personal_layers": "head",
    }


def _build_run_name(config: ExperimentConfig) -> str:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return (
        f"{config.training.task}-flower-"
        f"{federated.num_clients}c-{federated.num_rounds}r-{timestamp}"
    )


def _init_wandb(
    config: ExperimentConfig,
    artifact: dict[str, object],
    client_subjects: dict[str, list[str]],
) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.enabled:
        return None

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        tags=(config.wandb.tags or []) + ["flower", "federated-learning"],
        config=_build_wandb_config(config, artifact, client_subjects),
        name=_build_run_name(config),
        job_type="federated-train",
    )

    wandb.log(
        {
            "dataset/num_train": artifact["summary"]["num_train"],
            "dataset/num_val": artifact["summary"]["num_val"],
            "dataset/num_test": artifact["summary"]["num_test"],
        },
        step=0,
    )
    return run


def _aggregate_fit_metrics(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for key in ("train_loss", "train_score"):
        weighted_sum = 0.0
        contributed = False
        for num_examples, client_metrics in metrics:
            if key not in client_metrics:
                continue
            weighted_sum += float(client_metrics[key]) * num_examples
            contributed = True
        if contributed:
            aggregated[key] = weighted_sum / total_examples
    return aggregated


def _merge_round_rows(
    fit_rows: list[dict[str, float]],
    eval_rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for row in fit_rows + eval_rows:
        round_id = int(row["round"])
        existing = merged.setdefault(round_id, {"round": float(round_id)})
        existing.update(row)
    return [merged[round_id] for round_id in sorted(merged)]


def _confusion_matrix(
    targets: torch.Tensor,
    outputs: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    """Count true/predicted pairs for one client's local output classes."""
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    predictions = torch.argmax(outputs, dim=1)
    for true_id, predicted_id in zip(
        targets.reshape(-1).tolist(),
        predictions.reshape(-1).tolist(),
        strict=True,
    ):
        matrix[int(true_id)][int(predicted_id)] += 1
    return matrix


def _plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Save a count and row-percentage confusion matrix as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted subject")
    ax.set_ylabel("True subject")
    ax.set_title(title)

    max_value = max((max(row) for row in matrix), default=0)
    for row_index, row in enumerate(matrix):
        row_total = sum(row)
        for column_index, value in enumerate(row):
            percentage = value / row_total if row_total else 0.0
            label = f"{value}\n{percentage:.0%}"
            color = "white" if max_value and value > max_value / 2 else "black"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


class FlowerDependencyError(RuntimeError):
    pass


def _coerce_ndarrays(parameters: Any, parameters_to_ndarrays: Any) -> list[Any]:
    if isinstance(parameters, list):
        return parameters
    return parameters_to_ndarrays(parameters)


def _result_stem(config: ExperimentConfig) -> str:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    suffix = f"_{federated.result_name}" if federated.result_name else ""
    return f"{config.training.task}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Flower federated-learning sandbox.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.federated is None:
        raise ValueError("This config does not contain a federated section.")

    try:
        from flwr.client import NumPyClient
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from flwr.server import ServerConfig
        from flwr.server.strategy import FedAvg
        from flwr.simulation import start_simulation
    except ImportError as exc:
        raise FlowerDependencyError(
            "Flower is not installed in this environment. Install `flwr[simulation]` first."
        ) from exc

    artifact = load_validated_artifact(config.artifact_path, config)
    client_partitions, client_subjects = build_client_partitions(
        artifact=artifact,
        num_clients=config.federated.num_clients,
        client_subjects=config.federated.client_subjects,
    )
    partitions_by_id = {partition.client_id: partition for partition in client_partitions}
    partition_summary = create_partition_summary(
        artifact=artifact,
        client_partitions=client_partitions,
        client_subjects=client_subjects,
    )
    client_label_maps = {
        partition.client_id: client_local_label_map(
            artifact,
            partition.subject_ids,
        )
        for partition in client_partitions
    }
    partition_summary["client_local_label_maps"] = client_label_maps

    result_stem = _result_stem(config)
    partition_summary_path = config.output_dir / f"{result_stem}_federated_partitions.json"
    history_path = config.output_dir / f"{result_stem}_federated_history.csv"
    client_history_path = config.output_dir / f"{result_stem}_federated_client_history.csv"
    summary_path = config.output_dir / f"{result_stem}_federated_summary.json"
    model_path = config.output_dir / f"{result_stem}_flower_model.pt"

    experiment_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")

    client_head_dir = (
        config.output_dir
        / "fedper_heads"
        / result_stem
        / experiment_id
    )
    confusion_dir = (
        config.output_dir
        / "confusion_matrices"
        / result_stem
        / experiment_id
    )
    
    client_head_dir.mkdir(parents=True, exist_ok=False)

    save_partition_summary(partition_summary_path, partition_summary)

    run = _init_wandb(config, artifact, client_subjects)

    class FlowerSubjectClient(NumPyClient):
        def __init__(
            self,
            artifact_path: Path,
            partition: ClientPartition,
            experiment: ExperimentConfig,
        ) -> None:
            self.partition = partition
            self.experiment = experiment
            self.task = experiment.training.task
            self.local_epochs = experiment.federated.local_epochs if experiment.federated else 1
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                and experiment.federated is not None
                and experiment.federated.client_num_gpus > 0
                else "cpu"
            )
            self.artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            self.head_path = client_head_dir / f"client_{partition.client_id}_head.pt"
            self.class_ids = client_class_ids(
                self.artifact,
                partition.subject_ids,
            )
            self.output_dim = len(self.class_ids)
            self.train_loader = build_loader(
                artifact=self.artifact,
                indices=self.partition.train_indices,
                task=self.task,
                batch_size=experiment.training.batch_size,
                num_workers=experiment.training.num_workers,
                shuffle=True,
                seed=experiment.seed + int(partition.client_id),
                class_ids=(
                    self.class_ids
                    if self.task == "classification"
                    else None
                ),
            )

        def get_parameters(self, config: dict[str, Any]) -> list[Any]:
            model = create_federated_model(
                self.artifact,
                self.task,
                output_dim=self.output_dim if self.task == "classification" else None,
            )
            return get_base_parameters(model)

        def fit(
            self,
            parameters: list[Any],
            config: dict[str, Any],
        ) -> tuple[list[Any], int, dict[str, float]]:
            # Ray may create clients in a different order between runs. Seed
            # each client's initial personal head independently so the same
            # experiment seed produces the same starting head every time.
            client_seed = self.experiment.seed + int(self.partition.client_id)
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(client_seed)
            model = create_federated_model(
                self.artifact,
                self.task,
                output_dim=self.output_dim if self.task == "classification" else None,
            ).to(self.device)
            if self.head_path.exists():
                head_state = torch.load(
                    self.head_path, map_location=self.device, weights_only=True
                )
                set_head_state(model, head_state)
            set_base_parameters(model, parameters)
            train_result = train_local_model(
                model=model,
                loader=self.train_loader,
                task=self.task,
                device=self.device,
                epochs=self.local_epochs,
                learning_rate=self.experiment.training.learning_rate,
                weight_decay=self.experiment.training.weight_decay,
            )
            # Flower may recreate client objects between rounds, so persist the
            # personal head explicitly. Only the base is returned to the server.
            torch.save(get_head_state(model), self.head_path)
            metrics = {
                "client_id": self.partition.client_id,
                "train_loss": float(train_result.loss),
                "train_score": float(train_result.score),
            }
            if self.task == "regression":
                metrics["train_r2"] = float(
                    regression_r2_score(train_result.outputs, train_result.targets)
                )
            return get_base_parameters(model), len(self.partition.train_indices), metrics

        def evaluate(
            self,
            parameters: list[Any],
            config: dict[str, Any],
        ) -> tuple[float, int, dict[str, float]]:
            return 0.0, len(self.partition.train_indices), {}

    eval_rows: list[dict[str, float]] = []
    val_loader = build_loader(
        artifact=artifact,
        indices=artifact["val_indices"],
        task=config.training.task,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.seed,
    )
    server_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This fixed head makes centralized validation a representation-learning
    # proxy only. FedPer's real deployable models are global base + client head.
    #torch.manual_seed(config.seed)
    #server_proxy_model = create_federated_model(
    #    artifact, config.training.task
    #).to(server_device)

    def evaluate_fn(
        server_round: int,
        parameters: Any,
        _: dict[str, Any],
    ) -> tuple[float, dict[str, float]] | None:
        # Round 0 happens before clients have trained or saved personal heads.
        if server_round == 0:
            return None
    
        global_base_parameters = _coerce_ndarrays(
            parameters,
            parameters_to_ndarrays,
        )
    
        metadata = artifact["metadata"]
        global_val_indices = [
            int(index) for index in artifact["val_indices"]
        ]
    
        total_examples = 0
        weighted_loss = 0.0
        weighted_score = 0.0
        combined_outputs: list[torch.Tensor] = []
        combined_targets: list[torch.Tensor] = []
    
        wandb_metrics: dict[str, float] = {
            "federated/round": float(server_round),
        }
    
        for partition in client_partitions:
            client_id = partition.client_id
            class_ids = client_class_ids(artifact, partition.subject_ids)
            head_path = (
                client_head_dir
                / f"client_{client_id}_head.pt"
            )
    
            if not head_path.exists():
                print(
                    f"Warning: no saved head for client {client_id} "
                    f"during round {server_round}."
                )
                continue
    
            # Only use validation examples belonging to this client’s people.
            client_subjects = set(partition.subject_ids)
    
            client_val_indices = [
                index
                for index in global_val_indices
                if str(metadata[index]["subject_id"]) in client_subjects
            ]
    
            if not client_val_indices:
                print(
                    f"Warning: no validation examples for client {client_id}."
                )
                continue
    
            # Build the personalized model:
            # newest global base + this client’s saved local head.
            client_model = create_federated_model(
                artifact,
                config.training.task,
                output_dim=(
                    len(class_ids)
                    if config.training.task == "classification"
                    else None
                ),
            ).to(server_device)
    
            set_base_parameters(
                client_model,
                global_base_parameters,
            )
    
            client_head_state = torch.load(
                head_path,
                map_location=server_device,
                weights_only=True,
            )
    
            set_head_state(
                client_model,
                client_head_state,
            )
    
            client_val_loader = build_loader(
                artifact=artifact,
                indices=client_val_indices,
                task=config.training.task,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                shuffle=False,
                seed=config.seed + int(client_id),
                class_ids=(
                    class_ids
                    if config.training.task == "classification"
                    else None
                ),
            )
    
            client_result = evaluate_model(
                model=client_model,
                loader=client_val_loader,
                task=config.training.task,
                device=server_device,
            )
    
            num_examples = len(client_val_indices)
    
            total_examples += num_examples
            weighted_loss += client_result.loss * num_examples
            weighted_score += client_result.score * num_examples

            if config.training.task == "regression":
                client_val_r2 = float(
                    regression_r2_score(
                        client_result.outputs,
                        client_result.targets,
                    )
                )
                combined_outputs.append(client_result.outputs)
                combined_targets.append(client_result.targets)
                wandb_metrics[
                    f"personalized/client_{client_id}/val_r2"
                ] = client_val_r2
    
            # These keys automatically become W&B line graphs because
            # they are logged once after every federated round.
            wandb_metrics[
                f"personalized/client_{client_id}/val_loss"
            ] = float(client_result.loss)
    
            wandb_metrics[
                f"personalized/client_{client_id}/val_score"
            ] = float(client_result.score)
    
            wandb_metrics[
                f"personalized/client_{client_id}/val_num_examples"
            ] = float(num_examples)
    
        if total_examples == 0:
            print(
                f"Warning: no personalized validation results "
                f"for round {server_round}."
            )
            return None
    
        personalized_val_loss = weighted_loss / total_examples
        personalized_val_score = (
            math.sqrt(personalized_val_loss)
            if config.training.task == "regression"
            else weighted_score / total_examples
        )
        personalized_val_r2 = (
            float(
                regression_r2_score(
                    torch.cat(combined_outputs, dim=0),
                    torch.cat(combined_targets, dim=0),
                )
            )
            if config.training.task == "regression"
            else None
        )
    
        wandb_metrics[
            "personalized/weighted_val_loss"
        ] = float(personalized_val_loss)
    
        wandb_metrics[
            "personalized/weighted_val_score"
        ] = float(personalized_val_score)

        if personalized_val_r2 is not None:
            wandb_metrics[
                "personalized/combined_val_rmse"
            ] = float(personalized_val_score)
            wandb_metrics[
                "personalized/combined_val_r2"
            ] = personalized_val_r2
    
        if run is not None:
            wandb.log(
                wandb_metrics,
                step=server_round,
            )
    
        # Save the weighted personalized result in the CSV history.
        row = {
            "round": float(server_round),
            "val_loss": float(personalized_val_loss),
            "val_score": float(personalized_val_score),
        }
        if personalized_val_r2 is not None:
            row["val_r2"] = personalized_val_r2
        eval_rows.append(row)
    
        # Flower also records these as centralized metrics, but they are
        # now real personalized validation results instead of proxy results.
        evaluation_metrics = {
            "val_score": float(personalized_val_score),
        }
        if personalized_val_r2 is not None:
            evaluation_metrics["val_r2"] = personalized_val_r2
        return float(personalized_val_loss), evaluation_metrics

    class TrackingFedAvg(FedAvg):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.latest_parameters = kwargs.get("initial_parameters")
            self.fit_rows: list[dict[str, float]] = []
            self.client_rows: list[dict[str, float | str]] = []

        def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> tuple[Any, dict[str, float]]:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                self.latest_parameters = aggregated_parameters

            for _, fit_result in results:
                metrics = getattr(fit_result, "metrics", {}) or {}
                client_id = str(metrics.get("client_id", "unknown"))
                client_row: dict[str, float | str] = {
                    "round": float(server_round),
                    "client_id": client_id,
                    "num_examples": float(getattr(fit_result, "num_examples", 0)),
                }
                if "train_loss" in metrics:
                    client_row["train_loss"] = float(metrics["train_loss"])
                if "train_score" in metrics:
                    client_row["train_score"] = float(metrics["train_score"])
                if "train_r2" in metrics:
                    client_row["train_r2"] = float(metrics["train_r2"])
                self.client_rows.append(client_row)

                if run is not None:
                    client_log_payload = {
                        f"clients/{client_id}/train_loss": client_row.get("train_loss"),
                        f"clients/{client_id}/train_score": client_row.get("train_score"),
                        f"clients/{client_id}/num_examples": client_row["num_examples"],
                    }
                    if "train_r2" in client_row:
                        client_log_payload[f"clients/{client_id}/train_r2"] = client_row[
                            "train_r2"
                        ]
                    wandb.log(client_log_payload, step=server_round)

            row = {"round": float(server_round)}
            if "train_loss" in aggregated_metrics:
                row["train_loss"] = float(aggregated_metrics["train_loss"])
            if "train_score" in aggregated_metrics:
                row["train_score"] = float(aggregated_metrics["train_score"])
            self.fit_rows.append(row)

            if run is not None and len(row) > 1:
                wandb.log(
                    {
                        "federated/train_loss": row.get("train_loss"),
                        "federated/train_score": row.get("train_score"),
                    },
                    step=server_round,
                )
            return aggregated_parameters, aggregated_metrics

    torch.manual_seed(config.seed)
    initial_model = create_federated_model(artifact, config.training.task)
    initial_parameters = ndarrays_to_parameters(get_base_parameters(initial_model))
    strategy = TrackingFedAvg(
        fraction_fit=config.federated.fraction_fit,
        fraction_evaluate=config.federated.fraction_evaluate,
        min_fit_clients=max(1, math.ceil(config.federated.num_clients * config.federated.fraction_fit)),
        min_evaluate_clients=(
            0
            if config.federated.fraction_evaluate <= 0
            else max(1, math.ceil(config.federated.num_clients * config.federated.fraction_evaluate))
        ),
        min_available_clients=config.federated.num_clients,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=_aggregate_fit_metrics,
        initial_parameters=initial_parameters,
    )

    def client_fn(client_context: Any) -> Any:
        node_config = getattr(client_context, "node_config", None)
        client_id = str(client_context)
        if node_config is not None:
            client_id = str(
                node_config.get(
                    "partition-id",
                    node_config.get("partition_id", node_config.get("cid", client_id)),
                )
            )
        return FlowerSubjectClient(
            artifact_path=config.artifact_path,
            partition=partitions_by_id[client_id],
            experiment=config,
        ).to_client()

    try:
        start_simulation(
            client_fn=client_fn,
            num_clients=config.federated.num_clients,
            config=ServerConfig(num_rounds=config.federated.num_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": config.federated.client_num_cpus,
                "num_gpus": config.federated.client_num_gpus,
            },
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "runtime_env": {
                    "py_modules": [str(SRC / "redo_by_sara")],
                },
            },
        )

        final_parameters = parameters_to_ndarrays(strategy.latest_parameters)
        final_model = create_federated_model(artifact, config.training.task).to(server_device)
        set_base_parameters(final_model, final_parameters)
        metadata = artifact["metadata"]
        test_indices = [int(index) for index in artifact["test_indices"]]
        personalized_test_rows: list[dict[str, float | str]] = []
        personalized_confusions: list[dict[str, object]] = []
        total_test_examples = 0
        weighted_test_loss = 0.0
        weighted_test_score = 0.0
        combined_test_outputs: list[torch.Tensor] = []
        combined_test_targets: list[torch.Tensor] = []
        for partition in client_partitions:
            class_ids = client_class_ids(artifact, partition.subject_ids)
            head_path = client_head_dir / f"client_{partition.client_id}_head.pt"
            if not head_path.exists():
                continue
            subject_ids = set(partition.subject_ids)
            client_test_indices = [
                index
                for index in test_indices
                if str(metadata[index]["subject_id"]) in subject_ids
            ]
            if not client_test_indices:
                continue
            client_model = create_federated_model(
                artifact,
                config.training.task,
                output_dim=(
                    len(class_ids)
                    if config.training.task == "classification"
                    else None
                ),
            ).to(server_device)
            set_base_parameters(client_model, final_parameters)
            set_head_state(
                client_model,
                torch.load(
                    head_path, map_location=server_device, weights_only=True
                ),
            )
            client_test_loader = build_loader(
                artifact=artifact,
                indices=client_test_indices,
                task=config.training.task,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                shuffle=False,
                seed=config.seed + int(partition.client_id),
                class_ids=(
                    class_ids
                    if config.training.task == "classification"
                    else None
                ),
            )
            client_result = evaluate_model(
                model=client_model,
                loader=client_test_loader,
                task=config.training.task,
                device=server_device,
            )
            num_examples = len(client_test_indices)
            total_test_examples += num_examples
            weighted_test_loss += client_result.loss * num_examples
            weighted_test_score += client_result.score * num_examples
            client_test_row: dict[str, float | str] = {
                "client_id": partition.client_id,
                "num_examples": float(num_examples),
                "test_loss": float(client_result.loss),
                "test_score": float(client_result.score),
            }

            if config.training.task == "regression":
                client_test_row["test_r2"] = float(
                    regression_r2_score(
                        client_result.outputs,
                        client_result.targets,
                    )
                )
                combined_test_outputs.append(client_result.outputs)
                combined_test_targets.append(client_result.targets)

            if config.training.task == "classification":
                local_label_map = client_local_label_map(
                    artifact,
                    partition.subject_ids,
                )
                class_names = [
                    subject_id
                    for subject_id, _ in sorted(
                        local_label_map.items(),
                        key=lambda item: item[1],
                    )
                ]
                matrix = _confusion_matrix(
                    targets=client_result.targets,
                    outputs=client_result.outputs,
                    num_classes=len(class_names),
                )
                confusion_path = (
                    confusion_dir
                    / f"client_{partition.client_id}_test_confusion.png"
                )
                confusion_json_path = confusion_path.with_suffix(".json")
                confusion_payload: dict[str, object] = {
                    "client_id": partition.client_id,
                    "split": "test",
                    "num_examples": num_examples,
                    "accuracy": float(client_result.score),
                    "class_names": class_names,
                    "local_label_map": local_label_map,
                    "matrix": matrix,
                    "plot_path": str(confusion_path),
                }
                _plot_confusion_matrix(
                    matrix=matrix,
                    class_names=class_names,
                    title=(
                        f"FedPer Client {partition.client_id} Test Confusion "
                        f"(accuracy {client_result.score:.3f})"
                    ),
                    output_path=confusion_path,
                )
                confusion_payload["json_path"] = str(confusion_json_path)
                confusion_json_path.write_text(
                    json.dumps(confusion_payload, indent=2)
                )
                personalized_confusions.append(confusion_payload)
                client_test_row["confusion_plot_path"] = str(confusion_path)
                client_test_row["confusion_json_path"] = str(
                    confusion_json_path
                )

            personalized_test_rows.append(client_test_row)
        if total_test_examples == 0:
            raise RuntimeError("No personalized FedPer client test examples were found.")
        personalized_test_loss = weighted_test_loss / total_test_examples
        personalized_test_score = (
            math.sqrt(personalized_test_loss)
            if config.training.task == "regression"
            else weighted_test_score / total_test_examples
        )
        personalized_test_r2 = (
            float(
                regression_r2_score(
                    torch.cat(combined_test_outputs, dim=0),
                    torch.cat(combined_test_targets, dim=0),
                )
            )
            if config.training.task == "regression"
            else None
        )

        round_rows = _merge_round_rows(strategy.fit_rows, eval_rows)
        save_round_history(history_path, round_rows)
        save_round_history(client_history_path, strategy.client_rows)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "fedper",
                "base_state_dict": final_model.features.state_dict(),
                "client_head_dir": str(client_head_dir),
                "client_local_label_maps": client_label_maps,
            },
            model_path,
        )

        best_val_row = (
            min(eval_rows, key=lambda row: row["val_score"])
            if config.training.task == "regression"
            else max(eval_rows, key=lambda row: row["val_score"])
        )
        summary = {
            "task": config.training.task,
            "algorithm": "fedper",
            "partition_mode": "subject_owned_with_optional_shared_subjects",
            "result_name": config.federated.result_name,
            "model_path": str(model_path),
            "client_head_dir": str(client_head_dir),
            "validation_note": (
                "Uses the aggregated base with each client's saved local head "
                "and client-local label mapping."
            ),
            "client_local_label_maps": client_label_maps,
            "personalized_test_clients": personalized_test_rows,
            "personalized_test_confusions": personalized_confusions,
            "history_path": str(history_path),
            "client_history_path": str(client_history_path),
            "partition_summary_path": str(partition_summary_path),
            "num_clients": config.federated.num_clients,
            "num_rounds": config.federated.num_rounds,
            "local_epochs": config.federated.local_epochs,
            "best_val_round": int(best_val_row["round"]),
            "best_val_loss": float(best_val_row["val_loss"]),
            "best_val_score": float(best_val_row["val_score"]),
            "best_val_r2": (
                float(best_val_row["val_r2"])
                if "val_r2" in best_val_row
                else None
            ),
            "test_loss": float(personalized_test_loss),
            "test_score": float(personalized_test_score),
            "test_r2": personalized_test_r2,
        }
        summary_path.write_text(json.dumps(summary, indent=2))

        if run is not None:
            personalized_metrics = {
                "personalized/weighted_test_loss": personalized_test_loss,
                "personalized/weighted_test_score": personalized_test_score,
            }
            if personalized_test_r2 is not None:
                personalized_metrics[
                    "personalized/combined_test_rmse"
                ] = personalized_test_score
                personalized_metrics[
                    "personalized/combined_test_r2"
                ] = personalized_test_r2
            confusion_images: dict[str, Any] = {}
        
            for client_row in personalized_test_rows:
                client_id = client_row["client_id"]
        
                personalized_metrics[
                    f"personalized/client_{client_id}/test_loss"
                ] = client_row["test_loss"]
        
                personalized_metrics[
                    f"personalized/client_{client_id}/test_score"
                ] = client_row["test_score"]
        
                personalized_metrics[
                    f"personalized/client_{client_id}/num_examples"
                ] = client_row["num_examples"]

                if "test_r2" in client_row:
                    personalized_metrics[
                        f"personalized/client_{client_id}/test_r2"
                    ] = client_row["test_r2"]

                confusion_plot_path = client_row.get("confusion_plot_path")
                if isinstance(confusion_plot_path, str):
                    confusion_images[
                        f"personalized/client_{client_id}/test_confusion"
                    ] = wandb.Image(confusion_plot_path)
        
            wandb.log(
                {
                    **personalized_metrics,
                    **confusion_images,
                    "test_loss": summary["test_loss"],
                    "test_score": summary["test_score"],
                    "best_val_round": summary["best_val_round"],
                    "best_val_loss": summary["best_val_loss"],
                    "best_val_score": summary["best_val_score"],
                    **(
                        {
                            "best_val_r2": summary["best_val_r2"],
                            "test_r2": summary["test_r2"],
                        }
                        if config.training.task == "regression"
                        else {}
                    ),
                },
                step=config.federated.num_rounds,
            )
        
            run.summary.update(summary)
            run.summary.update(personalized_metrics)

        print(json.dumps(summary, indent=2))
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
