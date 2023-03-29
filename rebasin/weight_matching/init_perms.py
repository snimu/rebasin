from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph

from rebasin.weight_matching.structs import (
    MODULE_AXES,
    AppliesTo,
    AxisInfo,
    ModuleInfo,
    Permutation,
)

NODE_TYPES = FunctionNode | ModuleNode | TensorNode


class PermutationInitializer:
    def __init__(self, model_a: nn.Module, model_b: nn.Module, input_data: Any) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.input_data = input_data

        # A permutation for each axis of the weight (and bias) of each module.
        self.permutations_init, self.id_to_permutation_init = \
            self._initialize_permutations(model_a, model_b)

        # The final permutations after merging permutations.
        self.permutations: list[Permutation] = []

        # Discover redundant permutations to remove.
        self.redundant_permutation_indices: list[int] = []
        self._merge_permutations()
        self._remove_redundant_permutations()

        # The mapping of Module-IDs to permutations.
        # Used to permute a specific module along every axis.
        self.id_to_permutations: dict[int, list[Permutation]] = {}
        self._map_module_id_to_permutations()

    def _initialize_permutations(
            self, model_a: nn.Module, model_b: nn.Module
    ) -> tuple[list[Permutation], dict[int, list[Permutation]]]:
        permutations: list[Permutation] = []
        id_to_permutation: dict[int, list[Permutation]] = {}

        for module_a, module_b in zip(
                model_a.modules(), model_b.modules(), strict=True
        ):
            assert type(module_a) == type(module_b), \
                "Failure in rebasin: Found modules of different types."

            if not hasattr(module_b, "weight"):
                continue

            if module_b.weight is None:
                continue

            assert hasattr(module_a, "weight"), \
                "Failure in rebasin: Found modules with different properties."

            if not isinstance(module_a.weight, (torch.Tensor, nn.Parameter)):
                continue
            if not isinstance(module_b.weight, (torch.Tensor, nn.Parameter)):
                continue

            assert isinstance(module_a.weight.shape, torch.Size)
            assert isinstance(module_b.weight.shape, torch.Size)

            if len(module_b.weight.shape) == 0:
                continue  # cannot permute scalars

            axes = MODULE_AXES.get(type(module_b))
            if axes is None:
                continue  # unknown module type

            # Init permutations
            perms = self._init_mod_perm(module_a, module_b, axes)
            permutations.extend(perms)
            id_to_permutation[id(module_b)] = perms

        return permutations, id_to_permutation

    def _init_mod_perm(
            self, module_a: nn.Module, module_b: nn.Module, axes: AxisInfo
    ) -> list[Permutation]:
        perms: list[Permutation] = []

        for wax in range(axes.num_axis):
            if (
                    wax != axes.bax
                    or not hasattr(module_b, "bias")
                    or module_b.bias is None
                    or len(module_b.bias.shape) == 0  # type: ignore[arg-type]
            ):
                assert isinstance(module_b.weight.shape, torch.Size)  # satisfy mypy

                # Permute weight only
                module_info = ModuleInfo(
                    module_b=module_b,
                    module_a=module_a,
                    axis=wax,
                    applies_to=AppliesTo.WEIGHT,
                    axis_info=axes
                )
                permutation = Permutation(
                    perm_indices=torch.arange(module_b.weight.shape[wax]),
                    modules=[module_info],
                )
                perms.append(permutation)

                # axes.bax denotes the weight axis that the bias corresponds to,
                #   not the axis of the bias itself. It is assumed that the bias
                #   is one-dimensional.
                # Therefore, the weight shape at wax
                #   must be compared with the bias shape at 0.
            elif (
                    module_b.weight.shape[wax]  # type: ignore[index]
                    == module_b.bias.shape[0]  # type: ignore[index]
            ):
                assert isinstance(module_b.weight.shape, torch.Size)  # satisfy mypy

                # Permutes both weight and bias
                module_info = ModuleInfo(
                    module_b=module_b,
                    module_a=module_a,
                    axis=wax,
                    applies_to=AppliesTo.BOTH,
                    axis_info=axes
                )

                permutation = Permutation(
                    perm_indices=torch.arange(module_b.weight.shape[wax]),
                    modules=[module_info],
                )
                perms.append(permutation)
            else:
                # wax == axes.bax and module.bias exists and is not None
                #   but the weight and bias shapes do not match.
                # This can happen in transposed convolutions, for example.
                assert isinstance(module_b.weight.shape, torch.Size)  # satisfy mypy
                assert isinstance(module_b.bias.shape, torch.Size)  # satisfy mypy

                # Independently permute weight and bias
                winfo = ModuleInfo(
                    module_b=module_b,
                    module_a=module_a,
                    axis=wax,
                    applies_to=AppliesTo.WEIGHT,
                    axis_info=axes
                )
                permutation_weight = Permutation(
                    perm_indices=torch.arange(module_b.weight.shape[wax]),
                    modules=[winfo]
                )

                binfo = ModuleInfo(
                    module_b=module_b,
                    module_a=module_a,
                    axis=wax,
                    applies_to=AppliesTo.BIAS,
                    axis_info=axes
                )
                permutation_bias = Permutation(
                    perm_indices=torch.arange(module_b.bias.shape[0]),
                    modules=[binfo]
                )
                perms.extend([permutation_weight, permutation_bias])

        return perms

    def _merge_permutations(self) -> None:
        """Update the permutations with their parents & children."""
        root_nodes = list(
            draw_graph(self.model_b, self.input_data, depth=1e12).root_container
        )
        self._merge_permutations_recursive(root_nodes, set())  # type: ignore[arg-type]

    def _merge_permutations_recursive(
            self, nodes: list[NODE_TYPES], visited_nodes: set[NODE_TYPES]
    ) -> None:
        for node in nodes:
            children = list(node.children)

            if not isinstance(node, ModuleNode):
                self._merge_permutations_recursive(
                    children, visited_nodes  # type: ignore[arg-type]
                )
                continue

            if node in visited_nodes:
                continue

            permutations = self.id_to_permutation_init.get(node.compute_unit_id)
            if permutations is None:
                self._merge_permutations_recursive(
                    children, visited_nodes  # type: ignore[arg-type]
                )
                continue

            visited_nodes.add(node)
            parent_modules = self._get_parent_modules(node)

            for permutation in permutations:
                self._merge(parent_modules, permutation)

            self._merge_permutations_recursive(
                children, visited_nodes  # type: ignore[arg-type]
            )

    def _get_parent_modules(self, node: NODE_TYPES) -> list[Permutation]:
        """Get the permutations of the parent modules of a node."""
        parent_modules = []
        for parent in node.parents:
            if (
                    isinstance(parent, ModuleNode)
                    and parent.compute_unit_id in self.id_to_permutation_init
            ):
                parent_modules.extend(self.id_to_permutation_init[parent.compute_unit_id])
            else:
                parent_modules.extend(
                    self._get_parent_modules(parent)  # type: ignore[arg-type]
                )
        return parent_modules

    def _merge(
            self, parent_modules: list[Permutation], permutation: Permutation
    ) -> None:
        """Merge one permutation with its parents if they fit."""
        new_permutation = Permutation(
            perm_indices=permutation.perm_indices,
            modules=permutation.modules,
        )

        perm_info = permutation.modules[0]

        for parent in parent_modules:
            if parent.perm_indices.shape != permutation.perm_indices.shape:
                continue

            # Compare last weight in parents with first in current permutation.
            parent_info = parent.modules[-1]

            if (
                    perm_info.axis == perm_info.axis_info.wax_in
                    and parent_info.axis == perm_info.axis_info.wax_out
            ):
                # Merge weight permutation. Preserve correct order.
                new_modules = copy.copy(parent.modules)
                new_modules.extend(permutation.modules)
                new_permutation.modules = new_modules

        # Don't add permutations that are subsets of other permutations.
        if self._all_modules_contained_in_other_permutation(new_permutation):
            return

        # If the new permutation is a superset of another permutation,
        #   remove the other permutation.
        redundant_permutation = \
            self._get_subset_permutation(new_permutation)
        if redundant_permutation is not None:
            self.redundant_permutation_indices.append(redundant_permutation)

        # Add the new permutation to permutations.
        self.permutations.append(new_permutation)

    def _all_modules_contained_in_other_permutation(
            self, permutation: Permutation
    ) -> bool:
        """Check if all modules in a permutation are contained in another."""
        for other_permutation in self.permutations:
            if all(
                    module in other_permutation.modules
                    for module in permutation.modules
            ):
                return True
        return False

    def _get_subset_permutation(self, permutation: Permutation) -> int | None:
        """Check if a permutation is a subset of another."""
        for i, other_permutation in enumerate(self.permutations):
            if all(
                    module in permutation.modules
                    for module in other_permutation.modules
            ):
                return i
        return None

    def _remove_redundant_permutations(self) -> None:
        """Remove redundant permutations."""
        self.redundant_permutation_indices.sort(reverse=True)
        for i in self.redundant_permutation_indices:
            self.permutations.pop(i)

    def _map_module_id_to_permutations(self) -> None:
        """Map module ids to permutations."""
        for permutation in self.permutations:
            for module_info in permutation.modules:
                id_ = id(module_info.module_b)
                if id_ in self.id_to_permutations:
                    self.id_to_permutations[id_].append(permutation)
                else:
                    self.id_to_permutations[id_] = [permutation]
