"""
The definition of a permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class Permutation:
    """A permutation."""
    perm_indices: torch.Tensor

    def __len__(self) -> int:
        return len(self.perm_indices)


@dataclass
class ModuleParameters:
    weight_a: nn.Parameter | torch.Tensor
    weight_b: nn.Parameter | torch.Tensor
    name: str
    axis_to_permutation: dict[int, Permutation]
    module_type: Any

    # A parameter is either only a weight, or a weight with associated bias.
    # This is because the weights are used for associating permutations.
    # A bias is always associated with axis 0, the output axis.
    bias_a: nn.Parameter | None = None
    bias_b: nn.Parameter | None = None

    @property
    def input_permutation(self) -> Permutation:
        if len(self.axis_to_permutation) == 1:
            return self.axis_to_permutation[0]
        else:
            return self.axis_to_permutation[1]

    @input_permutation.setter
    def input_permutation(self, perm: Permutation) -> None:
        if len(self.axis_to_permutation) == 1:
            self.axis_to_permutation[0] = perm
        else:
            self.axis_to_permutation[1] = perm

    @property
    def output_permutation(self) -> Permutation:
        return self.axis_to_permutation[0]

    @output_permutation.setter
    def output_permutation(self, perm: Permutation) -> None:
        if self.module_type is nn.LayerNorm and len(self.axis_to_permutation) > 1:
            self.axis_to_permutation[0] = Permutation(torch.arange(len(perm)))
        else:
            self.axis_to_permutation[0] = perm

    def apply_permutations(self, except_axis: int = -1) -> None:
        for axis, permutation in self.axis_to_permutation.items():
            if axis == except_axis:
                continue
            self.permute_axis(self.weight_b, axis, permutation.perm_indices)
            if self.bias_b is not None and axis == 0:  # axis 0 is output dim -> bias
                self.permute_axis(self.bias_b, axis, permutation.perm_indices)

    @staticmethod
    @torch.no_grad()
    def permute_axis(
            param: nn.Parameter | torch.Tensor, axis: int, perm_indices: torch.Tensor
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
        perm_indices = perm_indices.to(param.device)
        x = param.data.moveaxis(axis, 0)
        x = x[perm_indices]
        x = x.moveaxis(0, axis)
        param.data = x
