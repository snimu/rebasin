from __future__ import annotations

import copy
from typing import Any

import torch
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph


class ModelPaths:
    """
    Find all paths through the model.

    A path has the following properties:

    - It consists of ModuleNodes.
    - All its members have a "weight"-attribute.
    - The weights of the modules in the path are composable.

    Attributes:

        id_module_map:
            Map from id of ModuleNode to Module.
        root:
            Root node of the model.
        base_paths:
            All possible paths through the model consisting of ModuleNodes.
        paths:
            All paths through the model that have the properties above.
            This is the main output of this class.
    """

    def __init__(self, model: torch.nn.Module, input_data: Any) -> None:
        self.id_module_map = {
            id(module): module
            for module in model.modules()
            if hasattr(module, "weight")
            and isinstance(module.weight, (torch.Tensor, torch.nn.Parameter))
        }
        self.root = draw_graph(model, input_data=input_data, depth=10000).root_container
        self.base_paths = self._base_paths(list(self.root), [], [])
        self.paths = self._split_base_paths()

    def _base_paths(
            self,
            nodes: list[FunctionNode | ModuleNode | TensorNode],
            path: list[ModuleNode],
            paths: list[list[ModuleNode]]
    ) -> list[list[ModuleNode]]:
        """
        Follow nodes through the model and trace all possible paths.
        Only ModuleNodes with weight-attribute are saved.
        If the path splits, both possible paths are saved.
        """
        for node in nodes:
            # Not deepcopy, or recursion limit will be exceeded for large models:
            #   ModuleNodes have children, which have children, etc.
            _path = copy.copy(path)

            if not isinstance(node, (FunctionNode, ModuleNode, TensorNode)):
                raise TypeError(f"Unknown node type: {type(node)}")

            if (
                    isinstance(node, ModuleNode)
                    and hasattr(self.id_module_map.get(node.compute_unit_id), "weight")
            ):
                _path.append(node)

            if list(node.children):
                paths = self._base_paths(
                    list(node.children), _path, paths  # type: ignore[arg-type]
                )
            else:
                paths.append(_path)

        return paths

    def _split_base_paths(self) -> list[list[ModuleNode]]:
        """
        Split base paths into paths
        that contain ModuleNodes whose Modules are composable.
        """
        paths = []
        for base_path in self.base_paths:
            if len(base_path) == 1:
                paths.append(base_path)
                continue

            last_split = 0
            for i, node in enumerate(base_path):
                if i == 0:
                    continue

                if not self._is_composable(base_path[i-1], node):
                    paths.append(base_path[last_split:i])
                    last_split = i

            if last_split == 0:
                paths.append(base_path)

        return paths

    def _is_composable(self, last_node: ModuleNode, node: ModuleNode) -> bool:
        """
        Check if the Module of the last ModuleNode is composable
        with the Module of the current ModuleNode.
        """
        s0 = (
            self.id_module_map
            .get(last_node.compute_unit_id)
            .weight  # type: ignore[union-attr]
            .size()  # type: ignore[operator]
          )
        s1 = (
            self.id_module_map
            .get(node.compute_unit_id)
            .weight  # type: ignore[union-attr]
            .size()  # type: ignore[operator]
        )

        if len(s1) == 1 and (s0[-1] != s1[0]):
            return False

        if s0[-1] != s1[-2]:
            return False

        return True
