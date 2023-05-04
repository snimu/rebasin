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



@dataclass
class ModelInfo:
    """
    Information about a model.

    Attributes:
        name:
            The name of the model.
        contains_batch_norm:
            Whether the model contains BatchNorms.
            This is relevant because BatchNorms have to have their
            statistics recalculated after a permutation & interpolation.
        num_parameters:
            The total number of parameters in the model.
        num_weights:
            The number of weight parameters in the model.
        num_elements_per_weight:
            The number of elements in each weight parameter as a list.
        num_permutable_elements_per_weight:
            The number of elements in each weight parameter that can be permuted.
            This is the size of axes 0 and 1 for 2d-weights,
            and the size of axis 0 for 1d-weights.
        num_biases:
            The number of bias parameters in the model.
        num_elements_per_bias:
            The number of elements in each bias parameter. All are permutable.
        num_elements_per_weight_mean:
            The mean number of elements in each weight parameter.
        num_elements_per_bias_mean:
            The mean number of elements in each bias parameter.
        num_elements_per_weight_std:
            The standard deviation of the number of elements in each weight parameter.
        num_elements_per_bias_std:
            The standard deviation of the number of elements in each bias parameter.
    """
    name: str
    contains_batch_norm: bool
    num_parameters: int
    num_weights: int
    num_elements_per_weight: list[int]
    num_permutable_elements_per_weight: list[int]
    num_biases: int
    num_elements_per_bias: list[int]
    num_elements_per_weight_mean: float
    num_elements_per_bias_mean: float
    num_elements_per_weight_std: float
    num_elements_per_bias_std: float

    def __repr__(self) -> str:
        return (
            f"ModelInfo: "
            f"\n\tname: {self.name}"
            f"\n\tcontains_batch_norm: {self.contains_batch_norm}"
            f"\n\tnum_parameters: {self._pretty_num_str(self.num_parameters)}"
            f"\n\tnum_weights: {self._pretty_num_str(self.num_weights)}"
            f"\n\tnum_elements_per_weight: "
            f"{self._pretty_list_str(self.num_elements_per_weight)}"
            f"\n\tnum_permutable_elements_per_weight: "
            f"{self._pretty_list_str(self.num_permutable_elements_per_weight)}"
            f"\n\tnum_biases: {self._pretty_num_str(self.num_biases)}"
            f"\n\tnum_elements_per_bias: "
            f"{self._pretty_list_str(self.num_elements_per_bias)}"
            f"\n\tnum_elements_per_weight_mean: "
            f"{self._pretty_num_str(self.num_elements_per_weight_mean)}"
            f"\n\tnum_elements_per_bias_mean: "
            f"{self._pretty_num_str(self.num_elements_per_bias_mean)}"
            f"\n\tnum_elements_per_weight_std: "
            f"{self._pretty_num_str(self.num_elements_per_weight_std)}"
            f"\n\tnum_elements_per_bias_std: "
            f"{self._pretty_num_str(self.num_elements_per_bias_std)}"
        )

    def __str__(self) -> str:
        return (
            f"ModelInfo("
            f"name={self.name}, "
            f"contains_batch_norm={self.contains_batch_norm}, "
            f"num_parameters={self._pretty_num_str(self.num_parameters)}, "
            f"num_weights={self._pretty_num_str(self.num_weights)}, "
            f"num_elements_per_weight="
            f"{self._pretty_list_str(self.num_elements_per_weight)}, "
            f"num_permutable_elements_per_weight="
            f"{self._pretty_list_str(self.num_permutable_elements_per_weight)}, "
            f"num_biases={self._pretty_num_str(self.num_biases)}, "
            f"num_elements_per_bias="
            f"{self._pretty_list_str(self.num_elements_per_bias)}, "
            f"num_elements_per_weight_mean="
            f"{self._pretty_num_str(self.num_elements_per_weight_mean)}, "
            f"num_elements_per_bias_mean="
            f"{self._pretty_num_str(self.num_elements_per_bias_mean)}, "
            f"num_elements_per_weight_std="
            f"{self._pretty_num_str(self.num_elements_per_weight_std)}, "
            f"num_elements_per_bias_std="
            f"{self._pretty_num_str(self.num_elements_per_bias_std)}"
            f")"
        )

    @staticmethod
    def _pretty_list_str(lst: list[int]) -> str:
        plist = [ModelInfo._pretty_num_str(x) for x in lst]
        if len(plist) > 3:
            plist = plist[:3] + ["..."]
        return "[" + ", ".join([num for num in plist]) + "]"

    @staticmethod
    def _pretty_num_str(num: int | float) -> str:
        rest = 0.0
        if isinstance(num, float):
            rest = num % 1
            num = int(num)

        numstr = f"{num:_}"

        if rest:
            numstr += f".{str(rest).split('.')[1][:3]}"

        return numstr
