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


class ModuleWithWeirdWeightAndBiasNames(nn.Module):
    """
    A model with weird weight and bias names.

    Specifically, 'weight' and 'bias' appear in different positions
    in their parameters' names.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weightabc = nn.Parameter(torch.randn(15, 15))
        self.defweightghi = nn.Parameter(torch.randn(15, 15))
        self.jklweight = nn.Parameter(torch.randn(15, 15))

        # Mix the order in bias names (except for one) to test that
        #   the actual names are used, not the position of the words 'weight' and 'bias'
        #   in those names.
        self.abcbias = nn.Parameter(torch.randn(15))  # belongs to weightabc
        self.defbiasghi = nn.Parameter(torch.randn(15))  # belongs to defweightghi
        self.biasjkl = nn.Parameter(torch.randn(15))  # belongs to jklweight

        # Create a weight and bias that don't fit to test that they are not associated.
        self.xyzweight = nn.Parameter(torch.randn(15, 15))
        self.xyzbias = nn.Parameter(torch.randn(3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs @ self.weightabc + self.abcbias
        x = nn.ReLU()(x)
        x = x @ self.defweightghi + self.defbiasghi
        x = nn.ReLU()(x)
        x = x @ self.jklweight + self.biasjkl
        x = nn.ReLU()(x)
        x = x @ self.xyzweight + self.xyzbias.sum()
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
