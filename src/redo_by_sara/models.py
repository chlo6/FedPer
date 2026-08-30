from __future__ import annotations

import torch
from torch import nn


class SimpleCNN1D(nn.Module):
    """Compact CNN with a configurable FedPer shared/local boundary."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        shared_conv_blocks: int = 2,
    ) -> None:
        super().__init__()
        if shared_conv_blocks not in (1, 2):
            raise ValueError("shared_conv_blocks must be either 1 or 2.")

        conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_block_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.shared_conv_blocks = shared_conv_blocks
        if shared_conv_blocks == 1:
            self.features = conv_block_1
            local_conv_blocks: list[nn.Module] = [conv_block_2]
        else:
            self.features = nn.Sequential(conv_block_1, conv_block_2)
            local_conv_blocks = []

        self.head = nn.Sequential(
            *local_conv_blocks,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)
