from __future__ import annotations

from typing import Any, Union

from torch import nn
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph

from rebasin.modules import (  # type: ignore[attr-defined]
    MODULE_TYPES,
    initialize_module,
)
from rebasin.paths import LinearPath, ParallelPaths, PathSequence
from rebasin.type_definitions import NODE_TYPES


class PermutationInitializer:
    def __init__(
            self,
            model_a: nn.Module,
            model_b: nn.Module,
            input_data_b: Any,
            input_data_a: Any | None = None,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.input_data_b = input_data_b
        self.input_data_a = self.input_data_b if input_data_a is None else input_data_a

        self.id_to_module_a = {id(m): m for m in self.model_a.modules()}
        self.id_to_module_b = {id(m): m for m in self.model_b.modules()}
        self._err_msg = "A and B must have the same model architecture."

        self.model_graph = self.initialize_permutations()

    def initialize_permutations(self) -> PathSequence:
        """Initialize the permutations of the model.
        """

        nextnodes_a: list[NODE_TYPES] = list(
            draw_graph(self.model_a, input_data=self.input_data_a, depth=1e12)
            .root_container
        )
        nextnodes_b: list[NODE_TYPES] = list(
            draw_graph(self.model_b, input_data=self.input_data_b, depth=1e12)
            .root_container
        )

        paths: list[LinearPath | ParallelPaths] = []

        while len(nextnodes_a) > 0 or len(nextnodes_b) > 0:
            assert len(nextnodes_a) == len(nextnodes_b), self._err_msg

            path: LinearPath | ParallelPaths
            if len(nextnodes_b) == 1:
                path, nextnodes_a, nextnodes_b = self.initialize_linear_path(
                    nextnodes_a, nextnodes_b
                )
            else:
                path, nextnodes_a, nextnodes_b = self.initialize_parallel_paths(
                    nextnodes_a, nextnodes_b
                )

            if bool(path):
                # This might be False (path empty) if we, for example,
                #   start with a nn.MultiheadAttention module.
                # Then, the three input tensors would yield a ParallelPaths,
                #   but it would be empty because those tensors combine into
                #   a single Module, the nn.MultiheadAttention.
                # In this case, we aren't interested in the empty ParallelPaths,
                #   only in the subsequent LinearPath (and what follows it).
                paths.append(path)

        return PathSequence(*paths)

    def initialize_linear_path(
            self, nextnodes_a: list[NODE_TYPES], nextnodes_b: list[NODE_TYPES]
    ) -> tuple[LinearPath, list[NODE_TYPES], list[NODE_TYPES]]:

        modules: list[MODULE_TYPES] = []
        while 0 < len(nextnodes_a) < 2:
            assert len(nextnodes_a) == len(nextnodes_b) == 1, self._err_msg

            node_a, node_b = nextnodes_a.pop(), nextnodes_b.pop()
            assert type(node_a) == type(node_b), self._err_msg

            if isinstance(node_a, ModuleNode) and isinstance(node_b, ModuleNode):
                mod_a = self.id_to_module_a[node_a.compute_unit_id]
                mod_b = self.id_to_module_b[node_b.compute_unit_id]
                module = initialize_module(mod_a, mod_b, node_b)
                if module is not None:
                    modules.append(module)

            nextnodes_a = list(node_a.children)  # type: ignore[arg-type]
            nextnodes_b = list(node_b.children)  # type: ignore[arg-type]

            # If there are multiple parents in the next nodes,
            #   then we have reached a fork in the graph.
            # This is likely because initialize_linear_path was called by
            #   initialize_parallel_paths, and we reached the end of the parallel paths.
            if nextnodes_a and len(nextnodes_a[0].parents) > 1:
                assert nextnodes_b and len(nextnodes_b[0].parents) > 1, self._err_msg
                break

        return LinearPath(*modules), nextnodes_a, nextnodes_b

    def initialize_parallel_paths(
            self, nextnodes_a: list[NODE_TYPES], nextnodes_b: list[NODE_TYPES]
    ) -> tuple[ParallelPaths, list[NODE_TYPES], list[NODE_TYPES]]:
        # In some models, there are nested parallel paths.
        # For example, in torchvision.models.efficientnet_b1,
        #   there are parallel paths which consist of a linear path
        #   next to a sequence of a linear path followed by a parallel path,
        #   which is then followed by another linear path.
        # To handle this, we first find the common final nodes of all parallel paths.
        # We can do this with only model_b, because the similarity of the
        #   model architectures is tested later when constructing the paths.
        # This is what we do second: construct the paths.

        # 1. Find common final nodes
        common_finalnodes = self._get_finalnodes(nextnodes_a, nextnodes_b)

        # 2. Construct paths
        paths: list[LinearPath | PathSequence] = []
        finalnodes_a: list[NODE_TYPES] = []
        finalnodes_b: list[NODE_TYPES] = []

        for node_a, node_b in zip(nextnodes_a, nextnodes_b):
            if [node_b] == common_finalnodes:
                # If the node is a common final node,
                #   then it is already handled by the common final nodes.
                paths.append(LinearPath())
                continue
            path, finalnodes_a, finalnodes_b = self._construct_subpath(
                node_a, node_b, common_finalnodes
            )
            assert len(nextnodes_a) == len(nextnodes_b), self._err_msg

            paths.append(path)

        return ParallelPaths(*paths), finalnodes_a, finalnodes_b

    def _get_finalnodes(
            self, nextnodes_a: list[NODE_TYPES], nextnodes_b: list[NODE_TYPES]
    ) -> list[NODE_TYPES]:
        finalnodes_per_path: list[list[list[NODE_TYPES]]] = [
            [[n]] for n in nextnodes_b
        ]

        while True:
            nextnextnodes_a: list[NODE_TYPES | None] = []
            nextnextnodes_b: list[NODE_TYPES | None] = []
            for i, (node_a, node_b) in enumerate(zip(nextnodes_a, nextnodes_b)):
                if node_b is None:
                    assert node_a is None, self._err_msg
                    nextnextnodes_a.append(None)
                    nextnextnodes_b.append(None)
                    continue

                _, finalnodes_a, finalnodes_b = self.initialize_linear_path(
                    [node_a], [node_b]
                )
                finalnodes_per_path[i].append(finalnodes_b)

                # If we have multiple next nodes, we can simply look at the first,
                #   because all nodes should end in one finalnodes list.
                # The specific structure of the subpaths
                #   will be handeled in _construct_subpath.
                nextnextnodes_a.append(finalnodes_a[0] if finalnodes_a else None)
                nextnextnodes_b.append(finalnodes_b[0] if finalnodes_b else None)

            if all(node is None for node in nextnextnodes_b):
                return []

            for finalnodes_b in finalnodes_per_path[0]:
                if all(
                        finalnodes_b in finalnodes_per_path[i]
                        for i in range(1, len(finalnodes_per_path))
                ):
                    return finalnodes_b

            nextnodes_a = nextnextnodes_a  # type: ignore[assignment]
            nextnodes_b = nextnextnodes_b  # type: ignore[assignment]

    def _construct_subpath(
            self,
            node_a: NODE_TYPES,
            node_b: NODE_TYPES,
            common_finalnodes: list[NODE_TYPES]
    ) -> tuple[LinearPath | PathSequence, list[NODE_TYPES], list[NODE_TYPES]]:
        path0, finalnodes_a, finalnodes_b = self.initialize_linear_path(
            [node_a], [node_b]
        )
        paths: list[LinearPath | ParallelPaths] = [path0]

        while finalnodes_b != common_finalnodes:
            assert len(finalnodes_a) == len(finalnodes_b), self._err_msg

            path: LinearPath | ParallelPaths

            if len(finalnodes_a) == 0:
                break
            if len(finalnodes_a) == 1:
                # If there is only one final node, we can simply construct the path
                #   from the next nodes.
                path, finalnodes_a, finalnodes_b = self.initialize_linear_path(
                    finalnodes_a, finalnodes_b
                )
            else:
                # If there are multiple final nodes,
                #   we have to construct a ParallelPaths.
                path, finalnodes_a, finalnodes_b = self.initialize_parallel_paths(
                    finalnodes_a, finalnodes_b
                )
            paths.append(path)

        if len(paths) == 0:
            raise ValueError("No paths found")

        if len(paths) == 1:
            ret_path = paths[0]
            assert isinstance(ret_path, LinearPath), \
                "Can't have PathSequence with one path only"
            return ret_path, finalnodes_a, finalnodes_b

        return PathSequence(*paths), finalnodes_a, finalnodes_b
