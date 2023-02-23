from __future__ import annotations

import torch


class MLP(torch.nn.Module):
    """A Multi-Layer-Perceptron."""

    def __init__(self, features: int, num_layers: int = 5) -> None:
        super().__init__()

        layers: list[torch.nn.Module] = [torch.nn.Flatten()]

        for i in range(num_layers):
            layers.append(
                torch.nn.Linear(in_features=features, out_features=features)
            )
            layers.append(torch.nn.ReLU() if i < num_layers - 1 else torch.nn.Softmax())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(inputs)
        return out
