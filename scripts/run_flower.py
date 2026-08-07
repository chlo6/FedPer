from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
    create_federated_model,
    create_partition_summary,
    evaluate_model,
    get_parameters,
    regression_r2_score,
    save_partition_summary,
    save_round_history,
    set_parameters,
    train_local_model_with_distillation,
)
from redo_by_sara.iid_partitioning import (
    build_iid_run_partitions,
    create_iid_partition_summary,
)


def _build_wandb_config(
    config: ExperimentConfig,
    artifact: dict[str, object],
    client_subjects: dict[str, list[str]],
    iid_mode: bool = False,
) -> dict[str, Any]:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    return {
        "seed": config.seed,
        "task": config.training.task,
        "partition_mode": (
            "iid_by_subject_full_train_runs"
            if iid_mode
            else "subject_owned_with_optional_shared_subjects"
        ),
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
        "algorithm": "fedavg_kd",
        "kd_enabled": bool(
            config.knowledge_distillation
            and config.knowledge_distillation.enabled
        ),
        "kd_weight": (
            config.knowledge_distillation.weight
            if config.knowledge_distillation
            else 0.0
        ),
        "kd_temperature": (
            config.knowledge_distillation.temperature
            if config.knowledge_distillation
            else 1.0
        ),
        "kd_start_round": (
            config.knowledge_distillation.start_round
            if config.knowledge_distillation
            else 2
        ),
    }


def _build_run_name(config: ExperimentConfig, iid_mode: bool = False) -> str:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return (
        f"{config.training.task}-fedavg-kd-{'iid-' if iid_mode else ''}"
        f"{federated.num_clients}c-{federated.num_rounds}r-{timestamp}"
    )


def _init_wandb(
    config: ExperimentConfig,
    artifact: dict[str, object],
    client_subjects: dict[str, list[str]],
    iid_mode: bool = False,
) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.enabled:
        return None

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        tags=(config.wandb.tags or [])
        + ["flower", "federated-learning", "fedavg-kd"]
        + (["iid"] if iid_mode else []),
        config=_build_wandb_config(
            config, artifact, client_subjects, iid_mode=iid_mode
        ),
        name=_build_run_name(config, iid_mode=iid_mode),
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
    for key in (
        "train_loss",
        "train_score",
        "train_r2",
        "train_supervised_loss",
        "train_kd_loss",
        "train_total_loss",
        "kd_active",
    ):
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


class FlowerDependencyError(RuntimeError):
    pass


def _coerce_ndarrays(parameters: Any, parameters_to_ndarrays: Any) -> list[Any]:
    if isinstance(parameters, list):
        return parameters
    return parameters_to_ndarrays(parameters)


def _result_stem(config: ExperimentConfig, iid_mode: bool = False) -> str:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    suffix = f"_{federated.result_name}" if federated.result_name else ""
    iid_suffix = "_iid" if iid_mode else ""
    return f"{config.training.task}{iid_suffix}{suffix}"


def main(force_iid: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Run a Flower federated-learning sandbox.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--iid", action="store_true", help="Use whole-run IID clients.")
    args = parser.parse_args()
    iid_mode = force_iid or args.iid

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
    if iid_mode:
        client_partitions, client_subjects = build_iid_run_partitions(
            artifact=artifact,
            num_clients=config.federated.num_clients,
            seed=config.seed,
        )
    else:
        client_partitions, client_subjects = build_client_partitions(
            artifact=artifact,
            num_clients=config.federated.num_clients,
            client_subjects=config.federated.client_subjects,
        )
    partitions_by_id = {partition.client_id: partition for partition in client_partitions}
    partition_summary = (
        create_iid_partition_summary(
            artifact=artifact,
            client_partitions=client_partitions,
            client_subjects=client_subjects,
        )
        if iid_mode
        else create_partition_summary(
            artifact=artifact,
            client_partitions=client_partitions,
            client_subjects=client_subjects,
        )
    )

    result_stem = _result_stem(config, iid_mode=iid_mode)
    partition_summary_path = config.output_dir / f"{result_stem}_federated_partitions.json"
    history_path = config.output_dir / f"{result_stem}_federated_history.csv"
    client_history_path = config.output_dir / f"{result_stem}_federated_client_history.csv"
    summary_path = config.output_dir / f"{result_stem}_federated_summary.json"
    model_path = config.output_dir / f"{result_stem}_flower_model.pt"

    experiment_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    teacher_dir = config.output_dir / "fedavg_kd_teachers" / result_stem / experiment_id
    teacher_dir.mkdir(parents=True, exist_ok=False)

    save_partition_summary(partition_summary_path, partition_summary)

    run = _init_wandb(
        config, artifact, client_subjects, iid_mode=iid_mode
    )

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
            self.teacher_path = teacher_dir / f"client_{partition.client_id}_teacher.pt"
            self.train_loader = build_loader(
                artifact=self.artifact,
                indices=self.partition.train_indices,
                task=self.task,
                batch_size=experiment.training.batch_size,
                num_workers=experiment.training.num_workers,
                shuffle=True,
                seed=experiment.seed + int(partition.client_id),
            )

        def get_parameters(self, config: dict[str, Any]) -> list[Any]:
            model = create_federated_model(self.artifact, self.task)
            return get_parameters(model)

        def fit(
            self,
            parameters: list[Any],
            config: dict[str, Any],
        ) -> tuple[list[Any], int, dict[str, float]]:
            model = create_federated_model(self.artifact, self.task).to(self.device)
            set_parameters(model, parameters)
            server_round = int(config.get("server_round", 1))
            kd_config = self.experiment.knowledge_distillation
            use_teacher = bool(
                kd_config
                and kd_config.enabled
                and kd_config.weight > 0
                and server_round >= kd_config.start_round
                and self.teacher_path.exists()
            )
            teacher_model = None
            if use_teacher:
                teacher_model = create_federated_model(
                    self.artifact,
                    self.task,
                ).to(self.device)
                teacher_model.load_state_dict(
                    torch.load(
                        self.teacher_path,
                        map_location=self.device,
                        weights_only=True,
                    ),
                    strict=True,
                )

            train_result = train_local_model_with_distillation(
                model=model,
                loader=self.train_loader,
                task=self.task,
                device=self.device,
                epochs=self.local_epochs,
                learning_rate=self.experiment.training.learning_rate,
                weight_decay=self.experiment.training.weight_decay,
                teacher_model=teacher_model,
                distillation_weight=(kd_config.weight if kd_config else 0.0),
                temperature=(kd_config.temperature if kd_config else 1.0),
            )
            torch.save(
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                self.teacher_path,
            )
            metrics = {
                "client_id": self.partition.client_id,
                "train_loss": float(train_result.evaluation.loss),
                "train_score": float(train_result.evaluation.score),
                "train_supervised_loss": float(train_result.supervised_loss),
                "train_kd_loss": float(train_result.distillation_loss),
                "train_total_loss": float(train_result.total_loss),
                "kd_active": float(train_result.distillation_active),
            }
            if self.task == "regression":
                metrics["train_r2"] = float(
                    regression_r2_score(
                        train_result.evaluation.outputs,
                        train_result.evaluation.targets,
                    )
                )
            return get_parameters(model), len(self.partition.train_indices), metrics

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
    def evaluate_fn(server_round: int, parameters: Any, _: dict[str, Any]) -> tuple[float, dict[str, float]]:
        model = create_federated_model(artifact, config.training.task).to(server_device)
        set_parameters(model, _coerce_ndarrays(parameters, parameters_to_ndarrays))
        result = evaluate_model(
            model=model,
            loader=val_loader,
            task=config.training.task,
            device=server_device,
        )
        row = {
            "round": float(server_round),
            "val_loss": float(result.loss),
            "val_score": float(result.score),
        }
        if config.training.task == "regression":
            row["val_r2"] = float(
                regression_r2_score(result.outputs, result.targets)
            )
        eval_rows.append(row)
        if run is not None:
            log_payload = {
                "federated/round": server_round,
                "federated/val_loss": result.loss,
                "federated/val_score": result.score,
            }
            if "val_r2" in row:
                log_payload["federated/val_r2"] = row["val_r2"]
            wandb.log(log_payload, step=server_round)
        return float(result.loss), {"val_score": float(result.score)}

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
                for key in (
                    "train_supervised_loss",
                    "train_kd_loss",
                    "train_total_loss",
                    "kd_active",
                ):
                    if key in metrics:
                        client_row[key] = float(metrics[key])
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
                    for key in (
                        "train_supervised_loss",
                        "train_kd_loss",
                        "train_total_loss",
                        "kd_active",
                    ):
                        if key in client_row:
                            client_log_payload[
                                f"clients/{client_id}/{key}"
                            ] = client_row[key]
                    wandb.log(client_log_payload, step=server_round)

            row = {"round": float(server_round)}
            if "train_loss" in aggregated_metrics:
                row["train_loss"] = float(aggregated_metrics["train_loss"])
            if "train_score" in aggregated_metrics:
                row["train_score"] = float(aggregated_metrics["train_score"])
            if "train_r2" in aggregated_metrics:
                row["train_r2"] = float(aggregated_metrics["train_r2"])
            for key in (
                "train_supervised_loss",
                "train_kd_loss",
                "train_total_loss",
                "kd_active",
            ):
                if key in aggregated_metrics:
                    row[key] = float(aggregated_metrics[key])
            self.fit_rows.append(row)

            if run is not None and len(row) > 1:
                wandb.log(
                    {
                        f"federated/{key}": value
                        for key, value in row.items()
                        if key != "round"
                    },
                    step=server_round,
                )
            return aggregated_parameters, aggregated_metrics

    torch.manual_seed(config.seed)
    initial_model = create_federated_model(artifact, config.training.task)
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))
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
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
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
        set_parameters(final_model, final_parameters)
        test_loader = build_loader(
            artifact=artifact,
            indices=artifact["test_indices"],
            task=config.training.task,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
            seed=config.seed,
        )
        test_result = evaluate_model(
            model=final_model,
            loader=test_loader,
            task=config.training.task,
            device=server_device,
        )
        test_r2 = (
            float(regression_r2_score(test_result.outputs, test_result.targets))
            if config.training.task == "regression"
            else None
        )

        round_rows = _merge_round_rows(strategy.fit_rows, eval_rows)
        save_round_history(history_path, round_rows)
        save_round_history(client_history_path, strategy.client_rows)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), model_path)

        best_val_row = (
            min(eval_rows, key=lambda row: row["val_score"])
            if config.training.task == "regression"
            else max(eval_rows, key=lambda row: row["val_score"])
        )
        summary = {
            "task": config.training.task,
            "algorithm": "fedavg_kd",
            "partition_mode": (
                "iid_by_subject_full_train_runs"
                if iid_mode
                else "subject_owned_with_optional_shared_subjects"
            ),
            "result_name": config.federated.result_name,
            "model_path": str(model_path),
            "teacher_model_dir": str(teacher_dir),
            "knowledge_distillation": {
                "enabled": bool(
                    config.knowledge_distillation
                    and config.knowledge_distillation.enabled
                ),
                "weight": (
                    config.knowledge_distillation.weight
                    if config.knowledge_distillation
                    else 0.0
                ),
                "temperature": (
                    config.knowledge_distillation.temperature
                    if config.knowledge_distillation
                    else 1.0
                ),
                "start_round": (
                    config.knowledge_distillation.start_round
                    if config.knowledge_distillation
                    else 2
                ),
            },
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
            "test_loss": float(test_result.loss),
            "test_score": float(test_result.score),
            "test_r2": test_r2,
        }
        summary_path.write_text(json.dumps(summary, indent=2))

        if run is not None:
            final_log_payload = {
                "test_loss": summary["test_loss"],
                "test_score": summary["test_score"],
                "best_val_round": summary["best_val_round"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_score": summary["best_val_score"],
            }
            if summary["best_val_r2"] is not None:
                final_log_payload["best_val_r2"] = summary["best_val_r2"]
            if summary["test_r2"] is not None:
                final_log_payload["test_r2"] = summary["test_r2"]
            wandb.log(final_log_payload, step=config.federated.num_rounds)
            run.summary.update(summary)

        print(json.dumps(summary, indent=2))
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
