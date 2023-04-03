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


class ModuleWithWeirdWeightAndBiasNames(nn.Module):
    """
    A model with weird weight and bias names.

    Specifically, 'weight' and 'bias' appear in different positions
    in their parameters' names.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weightabc = nn.Parameter(torch.randn(5, 5))
        self.defweightghi = nn.Parameter(torch.randn(5, 5))
        self.jklweight = nn.Parameter(torch.randn(5, 5))

        # Mix the order in bias names (except for one) to test that
        #   the actual names are used, not the position of the words 'weight' and 'bias'
        #   in those names.
        self.abcbias = nn.Parameter(torch.randn(5))  # belongs to weightabc
        self.defbiasghi = nn.Parameter(torch.randn(5))  # belongs to defweightghi
        self.biasjkl = nn.Parameter(torch.randn(5))  # belongs to jklweight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs @ self.weightabc + self.abcbias
        x = nn.ReLU()(x)
        x = x @ self.defweightghi + self.defbiasghi
        x = nn.ReLU()(x)
        x = x @ self.jklweight + self.biasjkl
        return x
