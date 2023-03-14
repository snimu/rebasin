from __future__ import annotations

from typing import Any

import torch
import torchview
from torchview import draw_graph


class ComposablePaths:
    """
    Calculate paths through the model. Each path must consist of
    Modules with composable weights.
    """

    def __init__(self, model: torch.nn.Module, input_data: Any) -> None:
        # Using `draw_graph` should at some point be replaced by custom
        #  tracing. That way, graphviz would not have to be installed.
        self.model = model
        self.edge_list = self.get_edge_list(input_data)
        self.paths = self.get_paths()

    def get_edge_list(self, input_data: Any) -> list[tuple[Any, ...]]:
        edge_list = draw_graph(self.model, input_data=input_data, depth=100).edge_list
        id_module = {id(module): module for module in self.model.modules()}
        new_edge_list: list[tuple[Any, ...]] = []

        for edge in edge_list:
            left, right = edge
            new_edge: list[Any] = []

            if isinstance(left, torchview.ModuleNode):
                module = id_module.get(left.compute_unit_id)
                assert module is not None
                new_edge.append(module)
            else:
                new_edge.append(left)

            if isinstance(right, torchview.ModuleNode):
                module = id_module.get(right.compute_unit_id)
                assert module is not None
                new_edge.append(module)
            else:
                new_edge.append(right)

            new_edge_list.append(tuple(new_edge))

        return new_edge_list

    def get_paths(self) -> list[list[torch.nn.Module]]:
        state = ...
        paths: list[list[torch.nn.Module]] = []

        for edge in self.edge_list:
            left, right = edge
            ...

        return paths
