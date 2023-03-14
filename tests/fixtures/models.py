from __future__ import annotations

import torch


class MLP(torch.nn.Module):
    """A Multi-Layer-Perceptron."""

    def __init__(
            self, in_features: int, out_features: int = -1, num_layers: int = 5
    ) -> None:
        super().__init__()

        layers: list[torch.nn.Module] = []

        for i in range(num_layers):
            layers.append(
                torch.nn.Linear(in_features=in_features, out_features=in_features)
            )
            layers.append(
                torch.nn.ReLU() if i < num_layers - 1
                else torch.nn.Softmax(dim=0)
            )

        self.model = torch.nn.Sequential(*layers)
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(inputs)
        return out[:self.out_features] if self.out_features > 0 else out
