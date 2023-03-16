from __future__ import annotations

import copy
from typing import Any

import torch
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph


class ModelPaths:

    def __init__(self, model: torch.nn.Module, input_data: Any) -> None:
        self.id_module_map = {id(module): module for module in model.modules()}
        self.root = draw_graph(model, input_data=input_data, depth=1000).root_container
        self.base_paths = self._base_paths(list(self.root), [], [])

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
