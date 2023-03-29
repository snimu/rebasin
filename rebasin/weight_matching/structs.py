from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch
from torch import nn


class AppliesTo(Enum):
    """Does the permutation apply to the weight only, the bias only, or both?"""
    WEIGHT = 1
    BIAS = 2
    BOTH = 3


class ModuleInfo(NamedTuple):
    module_b: nn.Module
    module_a: nn.Module
    axis: int
    applies_to: AppliesTo
    axis_info: AxisInfo


@dataclass
class Permutation:
    perm_indices: torch.Tensor
    modules: list[ModuleInfo]


# @dataclass
# class Permutation:
#     # To permute: weight[axis] = weight[axis][perm_indices]
#     #   (the same works with bias, too)
#     perm_indices: torch.Tensor
#     axis: int
#
#     # Does this permutation apply to the weight only, the bias only, or both?
#     applies_to: AppliesTo
#
#     # The module this permutation belongs to. Get & change weight & bias here.
#     module: nn.Module
#
#     # The ModuleNode this permutation belongs to. Find parents & children here.
#     node: ModuleNode
#
#     # The parents & children of this permutation.
#     # This means that among the direct parent Modules
#     #   of the Module this permutation belongs to,
#     #   the corresponding permutations are chosen
#     #   as parents and children,
#     #   if the length of the permutation's perm_indices is
#     #   equal to that of this permutation's.
#     parents: list[Permutation]
#     children: list[Permutation]


class AxisInfo(NamedTuple):
    """Information about which axes of a Module's weight & bias are to be permuted."""
    num_axis: int
    wax_in: int
    wax_out: int
    bax: int


MODULE_AXES = {
    # Linear / LazyLinear: Permute rows and columns. Bias belongs to the out_features.
    nn.Linear: AxisInfo(num_axis=2, wax_in=1, wax_out=0, bax=0),
    nn.LazyLinear: AxisInfo(num_axis=2, wax_in=1, wax_out=0, bax=0),

    # LayerNorm: One-dimensional weight.
    nn.LayerNorm: AxisInfo(num_axis=1, wax_in=0, wax_out=0, bax=0),

    # BatchNorm1d: One-dimensional weight.
    nn.BatchNorm1d: AxisInfo(num_axis=1, wax_in=0, wax_out=0, bax=0),

    # ConvNd: Permute input channels and filters (aka output channels).
    #         Bias belongs to the output channels. However, if the convolution is
    #         transposed, the dimensions are not the same;
    #         in this case, the function assigning Permutations to Modules
    #         has to create a permutation for the bias alone,
    #         in addition to the weight permutations.
    nn.Conv1d: AxisInfo(num_axis=2, wax_in=1, wax_out=0, bax=0),
    nn.Conv2d: AxisInfo(num_axis=2, wax_in=1, wax_out=0, bax=0),
    nn.Conv3d: AxisInfo(num_axis=2, wax_in=1, wax_out=0, bax=0),
}
