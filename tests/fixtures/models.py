from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    """A Multi-Layer-Perceptron."""

    def __init__(
            self, in_features: int, num_layers: int = 5
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


def mlp_3b() -> MLP:
    """
    A randomly initialized MLP with 4,000 layers.
    Each layer has weights of shape (850, 850) and biases of shape (850,).

    This leads to a total of 2,893,400,000 (2.89 billion) parameters.

    The resulting model takes a :code:`torch.Tensor` of shape (850) as input
    and returns a 1-dimensional :code:`torch.Tensor`.

    |

    Usage:

    |

    :code:`import torch`

    :code:`from tests.fixtures.models import mlp_3b`

    |

    :code:`model = mlp_3b()`

    :code:`x = torch.randn(850)`

    :code:`y = model(x)`
    """
    return MLP(850, num_layers=4_000)
