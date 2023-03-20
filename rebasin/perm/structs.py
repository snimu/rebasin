from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchview import ModuleNode


@dataclass
class Permutation:
    permutation: torch.Tensor  # Check if parents / children fit.
    axis: int
    bias: bool  # Only True where there is bias and when it fits the axis.
    module: nn.Module
    node: ModuleNode  # For parents and children
    parents: list[Permutation] | None
    children: list[Permutation] | None


MODULE_AXIS = {
    nn.Linear: (0, 1),
}
