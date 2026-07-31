from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig


def validate_artifact(
    artifact: dict[str, Any],
    config: ExperimentConfig,
) -> None:
    """Fail early when a task config and its prebuilt window artifact disagree."""
    samples = artifact.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 3:
        raise ValueError("Artifact 'samples' must be a [examples, channels, samples] tensor.")

    expected_channels = [
        channel
        for sensor in config.data.selected_sensors
        for channel in config.data.sensor_channel_map[sensor]
    ]
    stored_channels = artifact.get("summary", {}).get(
        "selected_channels",
        artifact.get("config", {}).get("selected_channels"),
    )
    if stored_channels is not None and list(stored_channels) != expected_channels:
        raise ValueError(
            "Artifact channel order does not match data.selected_sensors: "
            f"artifact={list(stored_channels)}, config={expected_channels}."
        )
    if samples.shape[1] != len(expected_channels):
        raise ValueError(
            f"Artifact has {samples.shape[1]} channels, but the config selects "
            f"{len(expected_channels)} channels."
        )

    expected_window_samples = int(
        round(config.data.target_sample_rate * config.data.window_seconds)
    )
    if samples.shape[2] != expected_window_samples:
        raise ValueError(
            f"Artifact windows contain {samples.shape[2]} samples, but the config "
            f"expects {expected_window_samples}."
        )

    task = config.training.task
    if task == "regression":
        targets = artifact.get("regression_targets")
    elif task == "classification":
        targets = artifact.get("classification_targets")
        if not isinstance(targets, torch.Tensor):
            raise ValueError("Classification artifact has no classification_targets tensor.")
        subject_to_class = artifact.get("subject_to_class")
        if not isinstance(subject_to_class, dict) or not subject_to_class:
            raise ValueError("Classification artifact has no subject_to_class mapping.")
        expected_ids = list(range(len(subject_to_class)))
        mapped_ids = sorted(int(value) for value in subject_to_class.values())
        if mapped_ids != expected_ids:
            raise ValueError(
                "subject_to_class values must be contiguous class ids starting at zero."
            )
        observed_ids = sorted(int(value) for value in torch.unique(targets).tolist())
        if observed_ids != expected_ids:
            raise ValueError(
                f"Classification targets contain {observed_ids}, but the class mapping "
                f"defines {expected_ids}."
            )
    else:
        raise ValueError(f"Unsupported task: {task}")

    if not isinstance(targets, torch.Tensor) or len(targets) != len(samples):
        raise ValueError(f"Artifact {task} targets do not align with its samples.")

    for split in ("train", "val", "test"):
        key = f"{split}_indices"
        if key not in artifact:
            raise ValueError(f"Artifact is missing {key}.")


def load_validated_artifact(
    path: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    validate_artifact(artifact, config)
    return artifact
