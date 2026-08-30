from __future__ import annotations

import torch
from torch import nn


class SimpleCNN1D(nn.Module):
    """Compact CNN with a configurable FedPer shared/local boundary."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        num_conv_blocks: int = 2,
        shared_conv_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_conv_blocks not in (1, 2, 3):
            raise ValueError("num_conv_blocks must be 1, 2, or 3.")
        if not 1 <= shared_conv_blocks <= num_conv_blocks:
            raise ValueError(
                "shared_conv_blocks must be between 1 and num_conv_blocks."
            )

        conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_blocks: list[nn.Module] = [conv_block_1]
        if num_conv_blocks >= 2:
            conv_blocks.append(nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            ))
        if num_conv_blocks >= 3:
            conv_blocks.append(nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.ReLU(inplace=True),
            ))

        self.num_conv_blocks = num_conv_blocks
        self.shared_conv_blocks = shared_conv_blocks
        self.features = nn.Sequential(*conv_blocks[:shared_conv_blocks])
        local_conv_blocks = conv_blocks[shared_conv_blocks:]
        final_channels = (32, 64, 128)[num_conv_blocks - 1]

        self.head = nn.Sequential(
            *local_conv_blocks,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)
