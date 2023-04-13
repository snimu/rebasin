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
        numstr = ""

        rest = 0.0
        if isinstance(num, float):
            rest = num % 1
            num = int(num)

        numstr_ = str(num)
        for i, char in enumerate(reversed(numstr_)):
            if i % 3 == 0 and i != 0:
                numstr += "_"
            numstr += char

        numstr = numstr[::-1]

        if rest != 0.0:
            reststr = str(rest).split(".")[1][:3]
            numstr += f".{reststr}"

        return numstr
