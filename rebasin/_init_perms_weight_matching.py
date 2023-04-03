from __future__ import annotations

from typing import Any, Union

import torch
from torch import nn
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph

from rebasin import util
from rebasin.structs import (
    AxisType,
    ParameterInfo,
    Permutation,
)

NODE_TYPES = Union[FunctionNode, ModuleNode, TensorNode]  # noqa


class PermutationInitializer:
    def __init__(self, model_a: nn.Module, model_b: nn.Module, input_data: Any) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.input_data = input_data

        # A permutation for each axis of the weight (and bias) of each module.
        #   These will have to be merged.
        self.permutations_init, self.id_to_permutation_init = \
            self._initialize_permutations(model_a, model_b)

        # The final permutations after merging permutations.
        self.permutations: list[Permutation] = []

        # After mergins, there will be redundant permutations.
        self.redundant_permutation_indices: list[int] = []
        self._merge_permutations()
        self._remove_redundant_permutations()

    def _initialize_permutations(
            self, model_a: nn.Module, model_b: nn.Module
) -> tuple[list[Permutation], dict[int, list[Permutation]]]:
        permutations: list[Permutation] = []
        id_to_perms: dict[int, list[Permutation]] = {}

        for module_a, module_b in zip(
                model_a.modules(), model_b.modules(), strict=True
        ):
            parameters = self._get_parameter_info(module_a, module_b)
            if parameters is None:
                continue

            # Init permutations
            perms = self._initialize_permutation_step(parameters, id(module_b))
            permutations.extend(perms)
            id_to_perms[id(module_b)] = perms

        return permutations, id_to_perms

    @staticmethod
    def _get_parameter_info(
            module_a: nn.Module, module_b: nn.Module
    ) -> list[ParameterInfo] | None:
        parameters: list[ParameterInfo] = []

        assert type(module_a) == type(module_b), \
            "Failure in rebasin: Found modules of different types."

        for (name_a, param_a), (name_b, param_b) in zip(
            module_a.named_parameters(), module_b.named_parameters(), strict=True
        ):
            assert name_a == name_b, \
                "Failure in rebasin: Found modules with different properties."
            assert type(param_a) == type(param_b), \
                "Failure in rebasin: Found modules with different properties."
            assert param_a.shape == param_b.shape, \
                "Failure in rebasin: Found modules with different properties."

            if not ("weight" in name_a or "bias" in name_a):
                continue

            if not isinstance(param_a, (torch.Tensor, nn.Parameter)):
                continue
            if not isinstance(param_b, (torch.Tensor, nn.Parameter)):
                continue

            assert isinstance(param_a.shape, torch.Size)
            assert isinstance(param_b.shape, torch.Size)

            if len(param_b.shape) == 0:
                continue  # cannot permute scalars

            # Currently, it is assumed that for weights, you always want to
            #   permute along the first two axes, unless they are 1-dimensional.
            # This is true for Convolutions,
            #   where input_channels and output_channels (filters)
            #   are the first two axes, and should be permuted.
            # For Linear Layers, both input- and output-features should be permuted.
            # Bias is assumed to be one-dimensional, so it is only permuted
            #   along the first axis.
            # Potentially, explicit exceptions to this rule will have to be added.
            if "weight" in name_a:
                axis_type = AxisType.BOTH if len(param_b.shape) == 1 else AxisType.OUT
                parameters.append(
                    ParameterInfo(
                        name=name_a,
                        param_a=param_a,
                        param_b=param_b,
                        axis=0,
                        axis_type=axis_type,
                        module_id=id(module_b),
                    )
                )
                # Only for nd-weights, where n > 1 (for example LayerNorm)
                if len(param_b.shape) > 1:
                    parameters.append(
                        ParameterInfo(
                            name=name_a,
                            param_a=param_a,
                            param_b=param_b,
                            axis=1,
                            axis_type=AxisType.IN,
                            module_id=id(module_b),
                        )
                    )
            elif "bias" in name_a:
                parameters.append(
                    ParameterInfo(
                        name=name_a,
                        param_a=param_a,
                        param_b=param_b,
                        axis=0,
                        # This is not associated with other parameters:
                        axis_type=AxisType.NEITHER,
                        module_id=id(module_b),
                    )
                )

        return parameters

    @staticmethod
    def _initialize_permutation_step(
            parameters: list[ParameterInfo], module_id: int
) -> list[Permutation]:
        """
        Turn ParameterInfo-objects into Permutation-objects.

        The difficulty here is to decide whether the same Permutation
          should contain several parameters (e.g. weight and bias),
          or just one.

        The decision is made based on the following thoughts:
          - Weight and bias should be associated into the same Permutation
              (to be more precise, axis 0 of the weight
              should be associated with the bias).
          - Some modules (e.g. MultiheadAttention) have multiple weights and biases
              with different names (e.g. "in_proj_weight", "in_proj_bias", ...).
            Different weights should have different permutations,
              and the right weight should be associated with the right bias.

        The way I'll go about this is this:
          1. Group parameters by axis.
                Weights and biases belong together. Both have axis 0.
                This is what I want to associate.
                Therefore, every parameter with axis 1 should have its own Permutation.
                The parameters with axis 0 should be grouped together.
          2. Group parameters by name.
                The parameters with axis 0 should be grouped by name.
                This is important in case that there are multiple weights (and biases)
                  in one Module.
          3. Create a Permutation-object for each group.
        """
        groups: list[list[ParameterInfo]] = [[], []]
        permutations: list[Permutation] = []

        # Group by axis
        for param_info in parameters:
            if param_info.axis == 0:
                groups[0].append(param_info)  # For further inspection
            else:
                groups[1].append(param_info)  # For immediate creation

        # Create Permutations for axis 1
        for param_info in groups[1]:
            axis_len = param_info.param_b.shape[param_info.axis]

            permutations.append(
                Permutation(
                    perm_indices=torch.arange(axis_len),
                    parameters=[param_info],
                    main_module_id=module_id,
                )
            )

        # Group by name
        for param_info in groups[0]:
            if "weight" not in param_info.name:
                continue

            bias: ParameterInfo | None = None

            # Find the corresponding bias
            for p_info in groups[0]:
                if "bias" not in p_info.name:
                    continue
                if (
                        p_info.name.replace("bias", "")
                        == param_info.name.replace("weight", "")
                ):
                    bias = p_info

            if bias is None:  # If there is no fitting bias, create perm just for weight
                permutation = Permutation(
                    perm_indices=torch.arange(param_info.param_b.shape[0]),
                    parameters=[param_info],
                    main_module_id=module_id,
                )
                permutations.append(permutation)
            elif (  # If names and shapes fit, associate weight and bias in one perm
                    # scalar bias can always be added to any weight
                    not list(bias.param_b.shape)
                    or bias.param_b.shape[0] == 1
                    # otherwise, the shapes must match
                    or param_info.param_b.shape[0] == bias.param_b.shape[0]
            ):
                permutation = Permutation(
                    perm_indices=torch.arange(param_info.param_b.shape[0]),
                    parameters=[param_info, bias],
                    main_module_id=module_id,
                )
                permutations.append(permutation)
            else:  # If the shapes don't fit, just create two permutations
                permutation1 = Permutation(
                    perm_indices=torch.arange(param_info.param_b.shape[0]),
                    parameters=[param_info],
                    main_module_id=module_id,
                )
                permutation2 = Permutation(
                    perm_indices=torch.arange(bias.param_b.shape[0]),
                    parameters=[bias],
                    main_module_id=module_id,
                )
                permutations.extend([permutation1, permutation2])

        return permutations

    def _merge_permutations(self) -> None:
        """Update the permutations with their parents & children."""
        root_nodes = list(
            draw_graph(self.model_b, self.input_data, depth=1e12).root_container
        )
        self._merge_permutations_recursive(root_nodes, set())  # type: ignore[arg-type]

    def _merge_permutations_recursive(
            self, nodes: list[NODE_TYPES], visited_nodes: set[NODE_TYPES]
    ) -> None:
        children: list[NODE_TYPES] = []

        for node in nodes:
            if node in visited_nodes:
                continue

            visited_nodes.add(node)
            children.extend(list(node.children))  # type: ignore[arg-type]

            if not isinstance(node, ModuleNode):
                continue

            permutations = self.id_to_permutation_init.get(node.compute_unit_id)
            if permutations is None:
                continue

            parent_modules = self._get_parent_modules(node)

            for permutation in permutations:
                self._merge(parent_modules, permutation)

        # Don't visit the same node twice
        children = list(set(children))
        for vnode in visited_nodes:
            if vnode in children:
                children.remove(vnode)

        # No need to recurse if there is nothing to recurse on
        if children:
            self._merge_permutations_recursive(children, visited_nodes)

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
            self, parent_perms: list[Permutation], permutation: Permutation
    ) -> None:
        """Merge one permutation with its parents if they fit."""
        new_perm = Permutation(
            perm_indices=permutation.perm_indices,
            parameters=permutation.parameters,
            main_module_id=permutation.main_module_id,
        )

        # Pick a parameter belonging to the main module to compare to the parents.
        param_info: ParameterInfo | None = None

        for p_info in permutation.parameters:
            if p_info.module_id == permutation.main_module_id:
                param_info = p_info
                break

        assert param_info is not None

        # Merge
        for parent_perm in parent_perms:
            if parent_perm.perm_indices.shape != permutation.perm_indices.shape:
                continue

            # Compare last weight in parents with first in current permutation.
            parent_info: ParameterInfo | None = None
            for p_info in parent_perm.parameters:
                if p_info.module_id == parent_perm.main_module_id:
                    parent_info = p_info
                    break

            assert parent_info is not None

            if (
                    param_info.axis_type in (AxisType.IN, AxisType.BOTH)
                    and parent_info.axis_type in (AxisType.OUT, AxisType.BOTH)
            ):
                # Merge weight permutation.
                new_perm.parameters.extend(parent_perm.parameters)

        # Don't add permutations that are subsets of other permutations.
        if self._is_subset_of_other_permutation(new_perm):
            return

        # If the new permutation is a superset of another permutation,
        #   mark the other permutation as redundant (for later removal).
        redundant_permutations = \
            self._get_subset_permutation(new_perm)
        if redundant_permutations:
            self.redundant_permutation_indices.extend(redundant_permutations)
            # Remove duplicates
            self.redundant_permutation_indices = \
                list(set(self.redundant_permutation_indices))

        # Add the new permutation to permutations.
        self.permutations.append(new_perm)

    def _is_subset_of_other_permutation(
            self, permutation: Permutation
    ) -> bool:
        """Check if all modules in a permutation are contained in another."""
        params = (p_info.param_b for p_info in permutation.parameters)

        for other_permutation in self.permutations:
            other_params = list(
                p_info.param_b for p_info in other_permutation.parameters
            )

            # Can't just compare parameters
            #   because exception is raised if their shapes differ
            all_in = True
            for param in params:
                if not util.contains_parameter(other_params, param):
                    all_in = False
                    break

            if all_in:
                return True

        return False

    def _get_subset_permutation(self, permutation: Permutation) -> list[int]:
        """Find other permutations that are subsets of this one."""
        indices: list[int] = []

        params = list(p_info.param_b for p_info in permutation.parameters)

        for i, other_permutation in enumerate(self.permutations):
            if other_permutation is permutation:
                continue

            other_params = (p_info.param_b for p_info in other_permutation.parameters)

            all_in = True
            for other_param in other_params:
                if not util.contains_parameter(params, other_param):
                    all_in = False
                    break

            if all_in:
                indices.append(i)

        return indices

    def _remove_redundant_permutations(self) -> None:
        """Remove redundant permutations."""
        self.redundant_permutation_indices.sort(reverse=True)
        for i in self.redundant_permutation_indices:
            self.permutations.pop(i)
