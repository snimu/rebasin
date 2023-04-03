from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch
from torch import nn


class AxisType(Enum):
    """Is the axis to be permuted an input axis, an output axis, or a bias axis?"""
    IN = 1
    OUT = 2
    BOTH = 3  # For 1d-weights
    NEITHER = 4  # For biases


class ParameterInfo(NamedTuple):
    param_a: nn.Parameter
    param_b: nn.Parameter
    name: str
    axis: int
    axis_type: AxisType
    module_id: int


@dataclass
class Permutation:
    # Permutation is done with these indices (w[axis] = w[axis][perm_indices]).
    perm_indices: torch.Tensor

    # To associate a permutation with permutations of other modules,
    #   we need to know the main module's ID.
    main_module_id: int
    parameters: list[ParameterInfo]

    def apply(self) -> None:
        """Apply the permutation to all parameters in this permutation."""
        for param_info in self.parameters:
            self.permute_parameter(
                param_info.param_b, param_info.axis, self.perm_indices
            )

    @staticmethod
    @torch.no_grad()
    def permute_parameter(
            param: nn.Parameter, axis: int, perm_indices: torch.Tensor
    ) -> None:
        """
        Permute a parameter along a given axis.

        Args:
            param:
                The parameter to permute.
            axis:
                The axis along which to permute.
            perm_indices:
                The permutation indices.
        """
        x = param.data.moveaxis(axis, 0)
        x = x[perm_indices]
        x = x.moveaxis(0, axis)
        param.data = x
