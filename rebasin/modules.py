# type: ignore
# I do type-checks in code, but mypy doesn't understand that.
# This leads to me getting an error from mypy at almost every line,
# which is annoying.
# Instead of reformatting my file such that mypy doesn't complain,
# I just ignore all the errors and make sure to test thoroughly.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import torch
from torch import nn
from torchview import ModuleNode


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

    def __init__(
            self,
            module_a: nn.Module,
            module_b: nn.Module,
            module_node_b: ModuleNode,
    ) -> None:
        if not type(module_a) == type(module_b):
            raise TypeError(
                f"Module types do not match: {type(module_a)} vs {type(module_b)}"
            )
        if not isinstance(module_a, nn.Module):
            raise TypeError(f"Module is not an nn.Module: {type(module_a)}")
        self.module_a = module_a
        self.module_b = module_b
        self.module_type = type(module_a)
        self._input_shape = module_node_b.input_shape
        self._output_shape = module_node_b.output_shape

    @property
    def input_permutation(self) -> Permutation | None:
        raise NotImplementedError

    @input_permutation.setter
    def input_permutation(self, perm: Permutation | None) -> None:
        raise NotImplementedError

    @property
    def input_permutation_shape(self) -> int:
        raise NotImplementedError

    @property
    def input_shape(self) -> list[tuple[int, ...]]:
        return self._input_shape

    @property
    def output_permutation(self) -> Permutation | None:
        raise NotImplementedError

    @output_permutation.setter
    def output_permutation(self, perm: Permutation | None) -> None:
        raise NotImplementedError

    @property
    def output_permutation_shape(self) -> int:
        raise NotImplementedError

    @property
    def output_shape(self) -> list[tuple[int, ...]]:
        return self._output_shape

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


class DefaultModule(ModuleBase):
    """
    The default module: a :class:`nn.Module`
    that has a :code`weight` and a :code:`bias` parameter.

    Used for, e.g., :class:`nn.Linear`, :class:`nn.Conv1d`, :class:`nn.Conv2d`, etc.

    Not used for, e.g., :class:`nn.MultiheadAttention`.
    """

    def __init__(
            self,
            module_a: nn.Module,
            module_b: nn.Module,
            module_node_b: ModuleNode,
    ) -> None:
        super().__init__(module_a, module_b, module_node_b)

        if (
                hasattr(self.module_a, "bias")
                and not isinstance(self.module_a.bias, (nn.Parameter, type(None)))
        ):
            raise TypeError(
                f"Module bias is not a parameter or None: {type(self.module_a.bias)}"
            )
        if (
                hasattr(self.module_b, "bias")
                and not isinstance(self.module_b.bias, (nn.Parameter, type(None)))
        ):
            raise TypeError(
                f"Module bias is not a parameter or None: {type(self.module_b.bias)}"
            )

        if self.module_a.weight.shape != self.module_b.weight.shape:
            raise ValueError(
                f"Module weight shapes do not match: "
                f"{self.module_a.weight.shape} vs {self.module_b.weight.shape}"
            )
        if hasattr(module_a, "bias") and self.module_a.bias is not None:
            if not hasattr(module_b, "bias"):
                raise AttributeError(
                    f"Module A has bias, but Module B has no bias."
                )
            if module_b.bias is None:
                raise ValueError(
                    "Module A has bias, but Module B's bias is None."
                )
            if self.module_a.bias.shape != self.module_b.bias.shape:
                raise ValueError(
                    f"Module bias shapes do not match: "
                    f"{self.module_a.bias.shape} vs {self.module_b.bias.shape}"
                )

        if hasattr(module_b, "bias") and self.module_b.bias is not None:
            if not hasattr(module_a, "bias"):
                raise AttributeError(
                    f"Module B has bias, but Module A has no bias."
                )
            if module_a.bias is None:
                raise ValueError(
                    "Module B has bias, but Module A's bias is None."
                )

        self.axis_to_permutation: dict[int, Permutation | None] = {
            0: Permutation(torch.arange(self.module_b.weight.shape[0]))
        }
        if len(self.module_a.weight.shape) > 1:
            self.axis_to_permutation[1] = Permutation(
                torch.arange(self.module_b.weight.shape[1])
            )

    @property
    def input_permutation(self) -> Permutation | None:
        return self.axis_to_permutation[1]

    @input_permutation.setter
    def input_permutation(self, perm: Permutation | None) -> None:
        self.axis_to_permutation[1] = perm

    @property
    def input_permutation_shape(self) -> int:
        return self.module_b.weight.shape[1]

    @property
    def output_permutation(self) -> Permutation | None:
        return self.axis_to_permutation[0]

    @output_permutation.setter
    def output_permutation(self, perm: Permutation | None) -> None:
        self.axis_to_permutation[0] = perm

    @property
    def output_permutation_shape(self) -> int:
        return self.module_b.weight.shape[0]

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        id_to_info: dict[int, list[PermutationInfo]] = {}
        id_to_permutation: dict[int, Permutation] = {}

        for axis, permutation in self.axis_to_permutation.items():
            if permutation is None:
                continue
            id_to_permutation[id(permutation)] = permutation

            infos = [
                PermutationInfo(
                    self,
                    axis,
                    self.module_a.weight,
                    self.module_b.weight,
                )
            ]
            if (
                    hasattr(self.module_b, "bias")
                    and self.module_b.bias is not None
                    and axis == 0
            ):
                infos.append(
                    PermutationInfo(
                        self,
                        axis,
                        self.module_a.bias,
                        self.module_b.bias,
                    )
                )
            if id(permutation) in id_to_info:
                id_to_info[id(permutation)].extend(infos)
            else:
                id_to_info[id(permutation)] = infos

        return [(id_to_permutation[id_], info) for id_, info in id_to_info.items()]

    def apply_permutations(self, except_axis: int = -1) -> None:
        for axis, permutation in self.axis_to_permutation.items():
            if permutation is None or axis == except_axis:
                continue
            self.permute_parameter(
                self.module_b.weight,
                axis, permutation.perm_indices
            )
            if (
                    hasattr(self.module_b, "bias")
                    and self.module_b.bias is not None
                    and axis == 0  # axis 0: out-dim -> bias
            ):
                self.permute_parameter(
                    self.module_b.bias,
                    axis, permutation.perm_indices
                )


class OneDimModule(ModuleBase):
    """For Modules with a 1D :code:`weight` (and :code:`bias`) attribute.
    """

    def __init__(
            self,
            module_a: nn.Module,
            module_b: nn.Module,
            module_node_b: ModuleNode,
    ) -> None:
        super().__init__(module_a, module_b, module_node_b)

        self._permutation = Permutation(torch.arange(module_b.weight.shape[0]))

    @property
    def input_permutation(self) -> Permutation | None:
        return self._permutation

    @input_permutation.setter
    def input_permutation(self, perm: Permutation | None) -> None:
        self._permutation = perm

    @property
    def input_permutation_shape(self) -> int:
        return self.module_b.weight.shape[0]

    @property
    def output_permutation(self) -> Permutation | None:
        return self._permutation

    @output_permutation.setter
    def output_permutation(self, perm: Permutation | None) -> None:
        self._permutation = perm

    @property
    def output_permutation_shape(self) -> int:
        return self.input_permutation_shape

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        if self._permutation is None:
            return []

        info = [PermutationInfo(
            self,
            0,
            self.module_a.weight,
            self.module_b.weight,
        )]
        if hasattr(self.module_b, "bias") and self.module_b.bias is not None:
            assert self.module_a.bias is not None, \
                "Module A has no bias, but Module B does."
            info.append(PermutationInfo(
                self,
                0,
                self.module_a.bias,
                self.module_b.bias,
            ))

        return [(self._permutation, info)]

    def apply_permutations(self, except_axis: int = -1) -> None:
        if except_axis == 0 or self._permutation is None:
            return
        self.permute_parameter(
            self.module_b.weight,
            0, self._permutation.perm_indices
        )
        if hasattr(self.module_b, "bias") and self.module_b.bias is not None:
            assert self.module_a.bias is not None, \
                "Module A has no bias, but Module B does."
            self.permute_parameter(
                self.module_b.bias,
                0, self._permutation.perm_indices
            )


class InputPermIsOutputPermMultiDimModule(ModuleBase):
    """A module for modules for which the input permutation and output permutation
     are both at dim 1, like :class:`nn.LayerNorm` or :class:`nn.Embedding`."""

    def __init__(
            self,
            module_a: nn.Module,
            module_b: nn.Module,
            module_node_b: ModuleNode,
    ) -> None:
        super().__init__(module_a, module_b, module_node_b)
        self._permutation = (
            Permutation(torch.arange(module_b.weight.shape[0]))
            if len(module_b.weight.shape) == 1
            else Permutation(torch.arange(module_b.weight.shape[1]))
        )

    @property
    def input_permutation(self) -> Permutation | None:
        return self._permutation

    @input_permutation.setter
    def input_permutation(self, perm: Permutation | None) -> None:
        self._permutation = perm

    @property
    def input_permutation_shape(self) -> int:
        return (
            self.module_b.weight.shape[0]
            if len(self.module_b.weight.shape) == 1
            else self.module_b.weight.shape[1]
        )

    @property
    def output_permutation(self) -> Permutation | None:
        return self._permutation

    @output_permutation.setter
    def output_permutation(self, perm: Permutation | None) -> None:
        self._permutation = perm

    @property
    def output_permutation_shape(self) -> int:
        return self.input_permutation_shape

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        if self._permutation is None:
            return []

        axis = 0 if len(self.module_b.weight.shape) == 1 else 1

        info = [PermutationInfo(
            self,
            axis,
            self.module_a.weight,
            self.module_b.weight,
        )]
        if hasattr(self.module_b, "bias") and self.module_b.bias is not None:
            assert hasattr(self.module_a, "bias") and self.module_a.bias is not None, \
                "Module A has no bias, but Module B does."
            info.append(PermutationInfo(
                self,
                0,
                self.module_a.bias,
                self.module_b.bias,
            ))

        return [(self._permutation, info)]

    def apply_permutations(self, except_axis: int = -1) -> None:
        axis = 0 if len(self.module_b.weight.shape) == 1 else 1
        if except_axis == axis or self._permutation is None:
            return

        self.permute_parameter(
            self.module_b.weight,
            axis, self._permutation.perm_indices
        )

        if hasattr(self.module_b, "bias") and self.module_b.bias is not None:
            assert hasattr(self.module_a, "bias") and self.module_a.bias is not None, \
                "Module A has no bias, but Module B does."
            self.permute_parameter(
                self.module_b.bias,
                axis, self._permutation.perm_indices
            )


class MultiheadAttentionModule(ModuleBase):
    """
    A module for :class:`nn.MultiheadAttention`.
    """

    def __init__(
            self,
            module_a: nn.Module,
            module_b: nn.Module,
            module_node_b: ModuleNode,
    ) -> None:
        super().__init__(module_a, module_b, module_node_b)

        if not isinstance(module_a, nn.MultiheadAttention):
            raise TypeError(
                f"Module A is not a nn.MultiheadAttention, but a {type(module_a)}."
            )
        if not isinstance(module_b, nn.MultiheadAttention):
            raise TypeError(
                f"Module B is not a nn.MultiheadAttention, but a {type(module_b)}."
            )

        self._input_permutation: Permutation | None = Permutation(
            torch.arange(module_b.in_proj_weight.shape[1])
        ) if module_b.in_proj_weight is not None else None

        self._output_permutation: Permutation | None = Permutation(
            torch.arange(module_b.out_proj.weight.shape[0])
        )

        self._input_shape = self.input_shape[0]

    @property
    def input_permutation(self) -> Permutation | None:
        return self._input_permutation

    @input_permutation.setter
    def input_permutation(self, perm: Permutation | None) -> None:
        if self.module_b.in_proj_weight is None:
            return

        if (
                perm is not None
                and self._input_permutation is not None
                and len(perm) != len(self._input_permutation)
        ):
            raise ValueError(
                f"Permutation length {len(perm)} does not match "
                f"input permutation length {len(self._input_permutation)}."
            )
        self._input_permutation = perm

    @property
    def input_permutation_shape(self) -> int:
        return (
            self.module_b.in_proj_weight.shape[1]
            if self.module_b.in_proj_weight is not None
            else 0
        )

    @property
    def output_permutation(self) -> Permutation | None:
        return self._output_permutation

    @output_permutation.setter
    def output_permutation(self, perm: Permutation | None) -> None:
        if (
                perm is not None
                and self._output_permutation is not None
                and len(perm) != len(self._output_permutation)
        ):
            raise ValueError(
                f"Permutation length {len(perm)} does not match "
                f"output permutation length {len(self._output_permutation)}."
            )
        self._output_permutation = perm

    @property
    def output_permutation_shape(self) -> int:
        return self.module_b.out_proj.weight.shape[0]

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        id_to_info: list[tuple[Permutation, list[PermutationInfo]]] = []
        if self.input_permutation is not None:
            id_to_info.append(
                (
                    self.input_permutation,
                    [
                    PermutationInfo(
                        self,
                        1,
                        self.module_a
                        .in_proj_weight,
                        self.module_b
                        .in_proj_weight,
                    )
                    ]
                )
            )
        if self.output_permutation is not None:
            info = [
                PermutationInfo(
                    self,
                    0,
                    self.module_a
                    .out_proj.weight,
                    self.module_b
                    .out_proj.weight,
                )
            ]

            if self.module_b.out_proj.bias is not None:
                info.append(
                    PermutationInfo(
                        self,
                        0,
                        self.module_a
                        .out_proj.bias,
                        self.module_b
                        .out_proj.bias,
                    )
                )
            id_to_info.append((self.output_permutation, info))

        return id_to_info

    def apply_permutations(self, except_axis: int = -1) -> None:
        if self.input_permutation is not None and except_axis != 1:
            self.permute_parameter(
                self.module_b.in_proj_weight,
                1, self.input_permutation.perm_indices
            )
        if self.output_permutation is not None and except_axis != 0:
            self.permute_parameter(
                self.module_b.out_proj.weight,
                0, self.output_permutation.perm_indices
            )
            if self.module_b.out_proj.bias is not None:
                self.permute_parameter(
                    self.module_b.out_proj.bias,
                    0, self.output_permutation.perm_indices
                )


MODULE_TYPES = Union[
    ModuleBase,
    DefaultModule,
    OneDimModule,
    InputPermIsOutputPermMultiDimModule,
    MultiheadAttentionModule
]

SPECIAL_MODULES: dict[Any, object] = {
    nn.LayerNorm: InputPermIsOutputPermMultiDimModule,
    nn.Embedding: InputPermIsOutputPermMultiDimModule,
    nn.MultiheadAttention: MultiheadAttentionModule,
}


def initialize_module(
        module_a: nn.Module, module_b: nn.Module, module_node_b: ModuleNode
) -> MODULE_TYPES | None:
    """
    Initialize a child of :class:`ModuleBase` for a given module pair.

    :param module_a: The :class:`nn.Module` from model_a.
    :param module_b: The corresponding :class:`nn.Module` from module_b.
    :param module_node_b: The :class:`ModuleNode` correspoinding to module_b.
    :return: The correct child class of :class:`ModuleBase` for the given module pair,
    initialized with the given modules, if the type of the Module supports permutation,
    otherwise None.
    """
    if any(isinstance(module_b, mtype) for mtype in SPECIAL_MODULES):
        constructor = SPECIAL_MODULES[type(module_a)]
        return constructor(module_a, module_b, module_node_b)
    if (
            hasattr(module_b, "weight")
            and hasattr(module_a, "weight")
            and isinstance(module_a.weight, nn.Parameter)
            and isinstance(module_b.weight, nn.Parameter)
    ):
        return (
            OneDimModule(module_a, module_b, module_node_b)
            if len(module_b.weight.shape) == 1
            else DefaultModule(module_a, module_b, module_node_b)
        )

    return None
