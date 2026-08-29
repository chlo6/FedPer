import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.redo_by_sara.training import run_epoch


def test_regression_metrics_use_matching_column_shapes() -> None:
    predictions = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    # Reproduce the legacy [batch, 1, 1] target shape. run_epoch must flatten
    # it to [batch, 1] rather than broadcasting predictions across the batch.
    targets = torch.tensor([[[1.0]], [[2.0]], [[2.0]], [[6.0]]])
    loader = DataLoader(TensorDataset(predictions, targets), batch_size=2)

    result = run_epoch(
        nn.Identity(),
        loader,
        nn.MSELoss(),
        optimizer=None,
        device=torch.device("cpu"),
        task="regression",
    )

    assert result.outputs.shape == (4, 1)
    assert result.targets.shape == (4, 1)
    assert math.isclose(result.loss, 1.25)
    assert math.isclose(result.score, math.sqrt(result.loss), rel_tol=1e-6)
    assert math.isclose(result.r2, 1.0 - 5.0 / 14.75, rel_tol=1e-6)
