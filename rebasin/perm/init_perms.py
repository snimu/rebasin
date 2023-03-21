from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph

from .structs import MODULE_AXES, AppliesTo, AxisInfo, Permutation

NODE_TYPES = FunctionNode | ModuleNode | TensorNode


class PermutationInitializer:
    def __init__(self, model: nn.Module, input_data: Any) -> None:
        self.id_to_module = {id(module): module for module in model.modules()}

        self.permutations, self.id_to_permutation = \
            self._init_permutations(model, input_data)
        self._find_parents_children()

    def _init_permutations(
            self, model: nn.Module, input_data: Any
    ) -> tuple[list[Permutation], dict[int, list[Permutation]]]:
        root = list(draw_graph(model, input_data=input_data, depth=1e12).root_container)
        return self._init_permutations_recursive(
            root, permutations=[], id_to_permutation={}
        )

    def _init_permutations_recursive(
            self,
            nodes: Sequence[NODE_TYPES],
            permutations: list[Permutation],
            id_to_permutation: dict[int, list[Permutation]],
    ) -> tuple[list[Permutation], dict[int, list[Permutation]]]:
        if not nodes:
            return permutations, id_to_permutation

        for node in nodes:
            children: list[NODE_TYPES] = list(node.children)  # type: ignore[arg-type]

            if not isinstance(node, ModuleNode):
                permutations, id_to_permutation = self._init_permutations_recursive(
                    children, permutations, id_to_permutation
                )
                continue  # Only consider Modules, not Functions or Tensors

            module = self.id_to_module.get(node.compute_unit_id)
            assert module is not None, \
                "Failure in torchview: Found module with id that is not in model."

            if not hasattr(module, "weight"):
                permutations, id_to_permutation = self._init_permutations_recursive(
                    children, permutations, id_to_permutation
                )
                continue  # permute weights only

            if module.weight is None:
                permutations, id_to_permutation = self._init_permutations_recursive(
                    children, permutations, id_to_permutation
                )
                continue  # cannot permute None

            assert isinstance(module.weight.shape, torch.Size)

            if len(module.weight.shape) == 0:
                permutations, id_to_permutation = self._init_permutations_recursive(
                    children, permutations, id_to_permutation
                )
                continue  # cannot permute a scalar

            axes = MODULE_AXES.get(type(module))
            if axes is None:
                permutations, id_to_permutation = self._init_permutations_recursive(
                    children, permutations, id_to_permutation
                )
                continue  # unknown module type

            # Init permutations
            permutations, id_to_permutation = self._init_permutations_step(
                axes, module, node, permutations, id_to_permutation
            )

        return permutations, id_to_permutation

    @staticmethod
    def _init_permutations_step(
            axes: AxisInfo,
            module: nn.Module,
            node: ModuleNode,
            permutations: list[Permutation],
            id_to_permutation: dict[int, list[Permutation]]
    ) -> tuple[list[Permutation], dict[int, list[Permutation]]]:
        for w_ax in axes.weight_axes:
            if (
                    w_ax != axes.bias_axis
                    or not hasattr(module, "bias")
                    or module.bias is None
                    or len(module.bias.shape) == 0  # type: ignore[arg-type]
            ):
                assert isinstance(module.weight.shape, torch.Size)  # satisfy mypy

                # Permute weight only
                permutation = Permutation(
                    perm_indices=torch.arange(module.weight.shape[w_ax]),
                    axis=w_ax,
                    applies_to=AppliesTo.WEIGHT,
                    module=module,
                    node=node,
                    parents=[],
                    children=[],
                )
                permutations.append(permutation)
                id_to_permutation[node.compute_unit_id] = [permutation]

            # bias_axis denotes the weight_axis that the bias corresponds to,
            #   not the axis of the bias itself. It is assumed that the bias
            #   is one-dimensional.
            # Therefore, the weight shape at w_ax
            #   must be compared with the bias shape at 0.
            elif (
                    module.weight.shape[w_ax]  # type: ignore[index]
                    == module.bias.shape[0]  # type: ignore[index]
            ):
                assert isinstance(module.weight.shape, torch.Size)  # satisfy mypy

                # Permutes both weight and bias
                permutation = Permutation(
                    perm_indices=torch.arange(module.weight.shape[w_ax]),
                    axis=w_ax,
                    applies_to=AppliesTo.BOTH,
                    module=module,
                    node=node,
                    parents=[],
                    children=[],
                )
                permutations.append(permutation)
                id_to_permutation[node.compute_unit_id] = [permutation]
            else:  # w_ax == axes.bias_axis and module.bias exists and is not None
                assert isinstance(module.weight.shape, torch.Size)  # satisfy mypy
                assert isinstance(module.bias.shape, torch.Size)  # satisfy mypy

                # Independently permute weight and bias
                permutation_weight = Permutation(
                    perm_indices=torch.arange(module.weight.shape[w_ax]),
                    axis=w_ax,
                    applies_to=AppliesTo.WEIGHT,
                    module=module,
                    node=node,
                    parents=[],
                    children=[],
                )
                permutation_bias = Permutation(
                    perm_indices=torch.arange(module.bias.shape[0]),
                    axis=0,
                    applies_to=AppliesTo.BIAS,
                    module=module,
                    node=node,
                    parents=[],
                    children=[],
                )
                permutations.extend([permutation_weight, permutation_bias])
                id_to_permutation[node.compute_unit_id] = \
                    [permutation_weight, permutation_bias]

        return permutations, id_to_permutation

    def _find_parents_children(self) -> None:
        """Update the permutations with their parents & children."""
