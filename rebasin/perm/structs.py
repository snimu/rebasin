from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch
from torch import nn
from torchview import ModuleNode


class AppliesTo(Enum):
    """Does the permutation apply to the weight only, the bias only, or both?"""
    WEIGHT = 1
    BIAS = 2
    BOTH = 3


@dataclass
class Permutation:
    # To permute: weight[axis] = weight[axis][perm_indices]
    #   (the same works with bias, too)
    perm_indices: torch.Tensor
    axis: int

    # Does this permutation apply to the weight only, the bias only, or both?
    applies_to: AppliesTo

    # The module this permutation belongs to. Get & change weight & bias here.
    module: nn.Module

    # The ModuleNode this permutation belongs to. Find parents & children here.
    node: ModuleNode

    # The parents & children of this permutation.
    # This means that among the direct parent Modules
    #   of the Module this permutation belongs to,
    #   the corresponding permutations are chosen
    #   as parents and children,
    #   if the length of the permutation's perm_indices is
    #   equal to that of this permutation's.
    parents: list[Permutation]
    children: list[Permutation]


class AxisInfo(NamedTuple):
    """Information about which axes of a Module's weight & bias are to be permuted."""
    weight_axes: tuple[int] | tuple[int, int]
    bias_axis: int


MODULE_AXES = {
    # Linear / LazyLinear: Permute rows and columns. Bias belongs to the out_features.
    nn.Linear: AxisInfo(weight_axes=(0, 1), bias_axis=0),
    nn.LazyLinear: AxisInfo(weight_axes=(0, 1), bias_axis=0),

    # LayerNorm: One-dimensional weight.
    nn.LayerNorm: AxisInfo(weight_axes=(0,), bias_axis=0),

    # BatchNorm1d: One-dimensional weight.
    nn.BatchNorm1d: AxisInfo(weight_axes=(0,), bias_axis=0),

    # ConvNd: Permute input channels and filters (aka output channels).
    #         Bias belongs to the output channels. However, if the convolution is
    #         transposed, the dimensions are not the same;
    #         in this case, the function assigning Permutations to Modules
    #         has to create a permutation for the bias alone,
    #         in addition to the weight permutations.
    nn.Conv1d: AxisInfo(weight_axes=(0, 1), bias_axis=0),
    nn.Conv2d: AxisInfo(weight_axes=(0, 1), bias_axis=0),
    nn.Conv3d: AxisInfo(weight_axes=(0, 1), bias_axis=0),
}
