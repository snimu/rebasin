"""
The definition of a permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn


class ParameterInfo(NamedTuple):
    param_a: nn.Parameter
    param_b: nn.Parameter
    name: str
    axis: int


@dataclass
class Permutation:
    # Permutation is done with these indices (w[axis] = w[axis][perm_indices]).
    perm_indices: torch.Tensor
    param_infos: list[ParameterInfo]

    def apply(self) -> None:
        """Apply the permutation to all parameters in this permutation."""
        for param_info in self.param_infos:
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
        perm_indices = perm_indices.to(param.device)
        x = param.data.moveaxis(axis, 0)
        x = x[perm_indices]
        x = x.moveaxis(0, axis)
        param.data = x

    def __repr__(self) -> str:
        parameters = '\n\t\t'.join(
            f'{p.name}: axis={p.axis}, shape={p.param_b.shape}'
            for p in self.param_infos
        )
        return (
            f"Permutation("
            f"\n\tperm_indices: {self.perm_indices}"
            f"\n\tparameters: "
            f"\n\t\t{parameters}"
            f"\n)"
        )
