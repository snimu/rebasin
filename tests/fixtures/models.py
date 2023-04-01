from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    """A Multi-Layer-Perceptron."""

    def __init__(
            self, in_features: int, out_features: int = -1, num_layers: int = 5
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        for i in range(num_layers):
            layers.append(
                nn.Linear(
                    in_features, out_features=in_features if i < num_layers - 1 else 1
                )
            )
            layers.append(
                nn.ReLU() if i < num_layers - 1
                else nn.Sequential()  # don't mess with last output
            )

        self.model = nn.Sequential(*layers)
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(inputs)
        return out


class SaveCallCount(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, x: Any) -> Any:
        self.call_count += 1
        return x
