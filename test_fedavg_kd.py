from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.redo_by_sara.federated import train_local_model_with_distillation
from src.redo_by_sara.models import SimpleCNN1D


def test_fedavg_kd_updates_student_but_not_historical_teacher() -> None:
    torch.manual_seed(11)
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


def test_round_without_teacher_is_ordinary_supervised_training() -> None:
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
