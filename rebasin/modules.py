from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import torch
from torch import nn


@dataclass
class Permutation:
    """A permutation."""
    perm_indices: torch.Tensor

    def __len__(self) -> int:
        return len(self.perm_indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return False
        if len(self) != len(other):
            return False
        return bool(torch.all(self.perm_indices == other.perm_indices))


@dataclass
class PermutationInfo:
    module: ModuleBase
    axis: int

    # PyTorch has some wrong typehints, like saying that Linear.weight is a Tensor.
    # To not trigger mypy,
    # parameter_a and parameter_b are nn.Parameter | torch.Tensor,
    # but we know that they are nn.Parameter.
    parameter_a: nn.Parameter | torch.Tensor
    parameter_b: nn.Parameter | torch.Tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PermutationInfo):
            return False
        if self.module is not other.module:
            return False
        if self.axis != other.axis:
            return False
        if self.parameter_a is not other.parameter_a:
            return False
        if self.parameter_b is not other.parameter_b:
            return False
        return True


class ModuleBase:
    """
    The base class for all modules.
    """

    def __init__(self, module_a: nn.Module, module_b: nn.Module) -> None:
        if not type(module_a) == type(module_b):
            raise TypeError(
                f"Module types do not match: {type(module_a)} vs {type(module_b)}"
            )
        if not isinstance(module_a, nn.Module):
            raise TypeError(f"Module is not an nn.Module: {type(module_a)}")
        self.module_a = module_a
        self.module_b = module_b
        self.module_type = type(module_a)

    @property
    def input_permutation(self) -> Permutation:
        raise NotImplementedError

    @input_permutation.setter
    def input_permutation(self, perm: Permutation) -> None:
        raise NotImplementedError

    @property
    def output_permutation(self) -> Permutation:
        raise NotImplementedError

    @output_permutation.setter
    def output_permutation(self, perm: Permutation) -> None:
        raise NotImplementedError

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        raise NotImplementedError

    def apply_permutations(self, except_axis: int = -1) -> None:
        raise NotImplementedError

    @staticmethod
    @torch.no_grad()
    def permute_parameter(
            param: nn.Parameter, axis: int, perm_indices: torch.Tensor
    ) -> None:
        raise NotImplementedError


class DefaultModule(ModuleBase):
    """
    The default module: a :class:`nn.Module`
    that has a :code`weight` and a :code:`bias` parameter.

    Used for, e.g., :class:`nn.Linear`, :class:`nn.Conv1d`, :class:`nn.Conv2d`, etc.

    Not used for, e.g., :class:`nn.MultiheadAttention`.
    """

    def __init__(self, module_a: nn.Module, module_b: nn.Module) -> None:
        super().__init__(module_a, module_b)

        if not hasattr(self.module_a, "weight"):
            raise AttributeError(f"Module has no weight: {type(self.module_a)}")
        if not hasattr(self.module_b, "weight"):
            raise AttributeError(f"Module has no weight: {type(self.module_b)}")
        if not hasattr(self.module_a, "bias"):
            raise AttributeError(f"Module has no bias: {type(self.module_a)}")
        if not hasattr(self.module_b, "bias"):
            raise AttributeError(f"Module has no bias: {type(self.module_b)}")

        if not isinstance(self.module_a.weight, nn.Parameter):
            raise TypeError(
                f"Module weight is not a parameter: {type(self.module_a.weight)}"
            )
        if not isinstance(self.module_b.weight, nn.Parameter):
            raise TypeError(
                f"Module weight is not a parameter: {type(self.module_b.weight)}"
            )
        if not isinstance(self.module_a.bias, (nn.Parameter, type(None))):
            raise TypeError(
                f"Module bias is not a parameter or None: {type(self.module_a.bias)}"
            )
        if not isinstance(self.module_b.bias, (nn.Parameter, type(None))):
            raise TypeError(
                f"Module bias is not a parameter or None: {type(self.module_b.bias)}"
            )

        if self.module_a.weight.shape != self.module_b.weight.shape:
            raise ValueError(
                f"Module weight shapes do not match: "
                f"{self.module_a.weight.shape} vs {self.module_b.weight.shape}"
            )
        if self.module_a.bias is not None:
            if module_b.bias is None:
                raise ValueError(
                    "Module A has bias, but Module B's bias is None."
                )
            if self.module_a.bias.shape != self.module_b.bias.shape:
                raise ValueError(
                    f"Module bias shapes do not match: "
                    f"{self.module_a.bias.shape} vs {self.module_b.bias.shape}"
                )

        self.axis_to_permutation = {
            0: Permutation(torch.arange(self.module_b.weight.shape[0]))
        }
        if len(self.module_a.weight.shape) > 1:
            self.axis_to_permutation[1] = Permutation(
                torch.arange(self.module_b.weight.shape[1])
            )

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

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        id_to_info: dict[int, list[PermutationInfo]] = {}
        id_to_permutation: dict[int, Permutation] = {}

        for axis, permutation in self.axis_to_permutation.items():
            id_to_permutation[id(permutation)] = permutation

            infos = [
                PermutationInfo(
                    self,
                    axis,
                    self.module_a.weight,  # type: ignore[arg-type]
                    self.module_b.weight,  # type: ignore[arg-type]
                )
            ]
            if self.module_b.bias is not None and axis == 0:
                infos.append(
                    PermutationInfo(
                        self,
                        axis,
                        self.module_a.bias,  # type: ignore[arg-type]
                        self.module_b.bias,  # type: ignore[arg-type]
                    )
                )
            if id(permutation) in id_to_info:
                id_to_info[id(permutation)].extend(infos)
            else:
                id_to_info[id(permutation)] = infos

        return [(id_to_permutation[id_], info) for id_, info in id_to_info.items()]

    def apply_permutations(self, except_axis: int = -1) -> None:
        # for mypy
        assert isinstance(self.module_a.weight, nn.Parameter)
        assert isinstance(self.module_b.bias, (nn.Parameter, type(None)))

        for axis, permutation in self.axis_to_permutation.items():
            if axis == except_axis:
                continue
            self.permute_parameter(
                self.module_b.weight,  # type: ignore[arg-type]
                axis, permutation.perm_indices
            )
            if self.module_b.bias is not None and axis == 0:  # axis 0: out-dim -> bias
                self.permute_parameter(
                    self.module_b.bias, axis, permutation.perm_indices
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


class MultiheadAttentionModule(ModuleBase):
    """
    A module for :class:`nn.MultiheadAttention`.
    """


MODULE_TYPES = Union[ModuleBase, DefaultModule, MultiheadAttentionModule]
MODULE_CONSTRUCTOR_TYPES = Union[
    type[ModuleBase], type[DefaultModule], type[MultiheadAttentionModule]
]
SPECIAL_MODULES: dict[Any, MODULE_CONSTRUCTOR_TYPES] = {
    nn.MultiheadAttention: MultiheadAttentionModule,
}


def initialize_module(module_a: nn.Module, module_b: nn.Module) -> MODULE_TYPES:
    """
    Initialize a child of :class:`ModuleBase` for a given module pair.

    :param module_a: The :class:`nn.Module` from model_a.
    :param module_b: The corresponding :class:`nn.Module` from module_b.
    :return: The correct child class of :class:`ModuleBase` for the given module pair,
    initialized with the given modules.
    """
    if type(module_a) in SPECIAL_MODULES:
        return SPECIAL_MODULES[type(module_a)](module_a, module_b)

    return DefaultModule(module_a, module_b)
