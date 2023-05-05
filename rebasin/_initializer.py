from __future__ import annotations

from typing import Any, NamedTuple, Union

import torch
from torch import nn
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph

from rebasin._paths import ModelPaths
from rebasin.structs import (
    ModuleParameters,
    Permutation,
)

NODE_TYPES = Union[ModuleNode, FunctionNode, TensorNode]


class PermInfo(NamedTuple):
    module_parameters: ModuleParameters
    axis: int


class PermutationInitialization:
    def __init__(
            self,
            model_a: nn.Module,
            model_b: nn.Module,
            input_data_b: Any,
            input_data_a: Any | None = None,
            enforce_identity: bool = True,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self._id_to_module_a: dict[int, nn.Module] = {
            id(m): m for m in model_a.modules()
        }
        self._id_to_module_b: dict[int, nn.Module] = {
            id(m): m for m in model_b.modules()
        }

        self.input_data_b = input_data_b
        self.input_data_a = input_data_a if input_data_a is not None else input_data_b
        self.enforce_identity = enforce_identity

        self.paths = self._trace_path()
        self.paths.merge_permutations()

        # For initializing the permutations, it makes most sense to track the modules,
        #   but for optimizing the permutations, it is more natural to go through the
        #   permutations. Therefore, I now switch to a datastructure fit for
        #   that purpose.
        self.perm_to_info: list[
            tuple[Permutation, list[tuple[int, ModuleParameters]]]
        ] = []

        # TODO: Ignore the permutations that are the identity anyways!
        self._get_permutations()

    def _trace_path(self) -> ModelPaths:
        root_a = list(
            draw_graph(self.model_a, input_data=self.input_data_a, depth=1e12)
            .root_container
        )
        root_b = list(
            draw_graph(self.model_b, input_data=self.input_data_b, depth=1e12)
            .root_container
        )

        if len(root_a) != len(root_b):
            raise ValueError(
                "Both models must have the same architecture!"
            )

        visited_nodes: set[NODE_TYPES] = set()
        working_nodes_a: list[NODE_TYPES] = root_a  # type: ignore[assignment]
        working_nodes_b: list[NODE_TYPES] = root_b  # type: ignore[assignment]
        paths: list[list[ModuleParameters]] = []

        while working_nodes_a and working_nodes_b:
            assert len(working_nodes_a) == len(working_nodes_b), \
                "Both models must have the same architecture!"

            new_working_nodes_a: list[NODE_TYPES] = []
            new_working_nodes_b: list[NODE_TYPES] = []

            for node_a, node_b in zip(working_nodes_a, working_nodes_b):  # noqa: B905
                if node_a in visited_nodes or node_b in visited_nodes:
                    continue
                path, children_a, children_b, visited = self._trace_path_from_node(
                    node_a, node_b
                )
                paths.append(path)
                new_working_nodes_a.extend(children_a)
                new_working_nodes_b.extend(children_b)
                visited_nodes.update(visited)

            working_nodes_a = new_working_nodes_a
            working_nodes_b = new_working_nodes_b

        return ModelPaths(paths, enforce_identity=self.enforce_identity)

    def _trace_path_from_node(
            self,
            node_a: NODE_TYPES,
            node_b: NODE_TYPES
    ) -> tuple[
        list[ModuleParameters],
        list[NODE_TYPES],
        list[NODE_TYPES],
        set[NODE_TYPES]
    ]:
        # Steps:
        # 1. Trace node_b until there are no children or a child has two parents
        #     or the node has multiple children
        # 2. Add the children to the children_b
        # 3. Do the same for node_a.
        # 4. In node_b, add ModuleParameters to the path when possible
        # 5. Make sure that both nodes always have the same children
        # 6. Trace visited nodes
        path: list[ModuleParameters] = []
        visited: set[NODE_TYPES] = set()

        while (
            len(node_b.children) == 1
            and len(tuple(node_b.children)[0].parents) == 1
        ):
            assert len(node_a.children) == 1, \
                "Both models must have the same architecture!"
            assert len(tuple(node_a.children)[0].parents) == 1, \
                "Both models must have the same architecture!"

            self._extend_path(path, node_a, node_b)
            visited.update([node_a, node_b])
            node_a = tuple(node_a.children)[0]  # type: ignore[assignment]
            node_b = tuple(node_b.children)[0]  # type: ignore[assignment]

        children_a: list[NODE_TYPES] = list(node_a.children)  # type: ignore[arg-type]
        children_b: list[NODE_TYPES] = list(node_b.children)  # type: ignore[arg-type]

        return path, children_a, children_b, visited

    def _extend_path(
            self,
            path: list[ModuleParameters],
            node_a: NODE_TYPES,
            node_b: NODE_TYPES
    ) -> None:
        assert type(node_a) == type(node_b), \
            "Both models must have the same architecture! " \
            f"But {type(node_a)=} and {type(node_b)=}"

        if not isinstance(node_a, ModuleNode) or not isinstance(node_b, ModuleNode):
            return

        module_a = self._id_to_module_a[node_a.compute_unit_id]
        module_b = self._id_to_module_b[node_b.compute_unit_id]

        # Only use leaf nodes
        if list(module_a.children()):
            assert module_b.children(), \
                "Both models must have the same architecture!"
            return

        w_a: nn.Parameter | None = None
        w_b: nn.Parameter | None = None
        b_a: nn.Parameter | None = None
        b_b: nn.Parameter | None = None

        for (name_a, param_a), (name_b, param_b) in zip(  # noqa: B905
                module_a.named_parameters(),
                module_b.named_parameters(),
        ):
            assert name_a == name_b, \
                "Both models must have the same architecture!"

            if "weight" in name_a:
                w_a = param_a
                w_b = param_b
            elif "bias" in name_a:
                b_a = param_a
                b_b = param_b

        if w_a is not None:
            assert w_b is not None, \
                "Both models must have the same architecture!"

            axis_to_permutation = self._get_axis_to_permutation(w_a, w_b)
            path.append(
                ModuleParameters(
                    weight_a=w_a,
                    weight_b=w_b,
                    name=module_b.__class__.__name__,
                    axis_to_permutation=axis_to_permutation,
                    module_type=type(module_b),
                    bias_a=b_a,
                    bias_b=b_b
                )
            )
        if w_b is not None:
            assert w_a is not None, \
                "Both models must have the same architecture!"

    @staticmethod
    def _get_axis_to_permutation(
            w_a: nn.Parameter, w_b: nn.Parameter
    ) -> dict[int, Permutation]:
        """Get the axis to permutation mapping."""
        assert w_a.shape == w_b.shape, \
            "Both models must have the same architecture!"

        axis_to_permutation = {
            0: Permutation(torch.arange(w_a.shape[0]))
        }
        if len(w_a.shape) > 1:
            axis_to_permutation[1] = Permutation(torch.arange(w_a.shape[1]))

        return axis_to_permutation

    def _get_permutations(self) -> None:
        """Get the permutations."""
        id_to_info: dict[
            int,
            tuple[Permutation, list[tuple[int, ModuleParameters]]]
        ] = {}

        for path in self.paths.paths:
            for module_params in path:
                for axis, perm in module_params.axis_to_permutation.items():
                    if id(perm) in id_to_info:
                        id_to_info[id(perm)][1].append((axis, module_params))
                    else:
                        id_to_info[id(perm)] = (perm, [(axis, module_params)])

        for info in id_to_info.values():
            self.perm_to_info.append(info)
