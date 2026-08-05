from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_flower import _confusion_matrix
from src.redo_by_sara.federated import (
    IndexedArtifactDataset,
    client_class_ids,
    client_local_label_map,
    create_federated_model,
    get_base_parameters,
    get_head_state,
    set_base_parameters,
    set_head_state,
    train_local_model_with_distillation,
)
from src.redo_by_sara.models import SimpleCNN1D


def test_setting_fedper_base_does_not_replace_personal_head() -> None:
    source = SimpleCNN1D(in_channels=3, output_dim=4)
    client = SimpleCNN1D(in_channels=3, output_dim=4)
    head_before = get_head_state(client)

    set_base_parameters(client, get_base_parameters(source))

    for actual, expected in zip(
        get_base_parameters(client), get_base_parameters(source), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for key, value in client.head.state_dict().items():
        torch.testing.assert_close(value, head_before[key])


def test_setting_fedper_head_does_not_replace_shared_base() -> None:
    source = SimpleCNN1D(in_channels=3, output_dim=4)
    client = SimpleCNN1D(in_channels=3, output_dim=4)
    base_before = [value.copy() for value in get_base_parameters(client)]

    set_head_state(client, get_head_state(source))

    for actual, expected in zip(
        get_base_parameters(client), base_before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for key, value in client.head.state_dict().items():
        torch.testing.assert_close(value, source.head.state_dict()[key])


def _classification_artifact() -> dict[str, object]:
    return {
        "samples": torch.randn(3, 2, 256),
        "channel_mean": torch.zeros(1, 2, 1),
        "channel_std": torch.ones(1, 2, 1),
        "classification_targets": torch.tensor([0, 4, 6]),
        "subject_to_class": {
            "001": 0,
            "005": 4,
            "008": 6,
        },
    }


def test_client_dataset_remaps_global_classes_to_local_targets() -> None:
    artifact = _classification_artifact()
    class_ids = client_class_ids(artifact, ["001", "005", "008"])
    dataset = IndexedArtifactDataset(
        artifact=artifact,
        indices=[0, 1, 2],
        task="classification",
        class_ids=class_ids,
    )

    assert class_ids == [0, 4, 6]
    assert [int(dataset[index][1]) for index in range(3)] == [0, 1, 2]
    assert client_local_label_map(
        artifact,
        ["008", "001", "005"],
    ) == {
        "001": 0,
        "005": 1,
        "008": 2,
    }


def test_client_model_has_only_local_class_logits() -> None:
    artifact = _classification_artifact()
    model = create_federated_model(
        artifact,
        "classification",
        output_dim=3,
    )

    logits = model(torch.randn(2, 2, 256))

    assert logits.shape == (2, 3)


def test_personalized_confusion_matrix_uses_local_class_positions() -> None:
    targets = torch.tensor([0, 1, 2, 1])
    outputs = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 5.0, 0.0],
        ]
    )

    assert _confusion_matrix(targets, outputs, num_classes=3) == [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]


def test_historical_teacher_is_frozen_during_distillation() -> None:
    torch.manual_seed(7)
    student = SimpleCNN1D(in_channels=2, output_dim=3)
    teacher = SimpleCNN1D(in_channels=2, output_dim=3)
    teacher_before = {
        key: value.detach().clone() for key, value in teacher.state_dict().items()
    }
    student_before = {
        key: value.detach().clone() for key, value in student.state_dict().items()
    }
    loader = DataLoader(
        TensorDataset(
            torch.randn(12, 2, 64),
            torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
        ),
        batch_size=4,
        shuffle=False,
    )

    result = train_local_model_with_distillation(
        model=student,
        loader=loader,
        task="classification",
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        teacher_model=teacher,
        distillation_weight=0.5,
        temperature=2.0,
    )

    assert result.distillation_active
    assert result.distillation_loss >= 0.0
    assert result.total_loss == pytest.approx(
        result.supervised_loss + 0.5 * result.distillation_loss,
        rel=1e-5,
    )
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key])
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(
        not torch.equal(value, student_before[key])
        for key, value in student.state_dict().items()
    )


def test_first_round_without_teacher_uses_supervised_loss_only() -> None:
    model = SimpleCNN1D(in_channels=2, output_dim=1)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 2, 64), torch.randn(6, 1)),
        batch_size=3,
        shuffle=False,
    )

    result = train_local_model_with_distillation(
        model=model,
        loader=loader,
        task="regression",
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        teacher_model=None,
        distillation_weight=0.5,
        temperature=2.0,
    )

    assert not result.distillation_active
    assert result.distillation_loss == 0.0
    assert result.total_loss == pytest.approx(result.supervised_loss)
