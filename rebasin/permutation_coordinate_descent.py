from __future__ import annotations

from typing import Any

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import]
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph

from .util import identity_tensor

NODE_TYPES = FunctionNode | ModuleNode | TensorNode


class PermutationCoordinateDescent:

    def __init__(
            self,
            model_a: torch.nn.Module,
            model_b: torch.nn.Module,
            input_data: Any,
            max_iter: int = 1000
    ) -> None:
        self.max_iter = max_iter

        self.id_to_module_a = {id(module): module for module in model_a.modules()}
        self.id_to_module_b = {id(module): module for module in model_b.modules()}

        root_a = draw_graph(model_a, input_data=input_data, depth=10000).root_container
        self.num_to_id_a, self.id_to_module_node_a, _ = (
            self._crawl_model(list(root_a), self.id_to_module_a, {}, {}, {}, 0)
        )

        root_b = draw_graph(model_b, input_data=input_data, depth=10000).root_container
        self.num_to_id_b, self.id_to_module_node_b, self.id_to_permutation = (
            self._crawl_model(list(root_b), self.id_to_module_b, {}, {}, {}, 0)
        )

        # self._find_permutations()

    def _crawl_model(
            self,
            nodes: list[NODE_TYPES],
            id_to_module: dict[int, torch.nn.Module],
            id_to_module_node: dict[int, ModuleNode],
            id_to_permutation: dict[int, torch.Tensor],
            num_to_id: dict[int, int],
            num: int
    ) -> tuple[dict[int, int], dict[int, ModuleNode], dict[int, torch.Tensor]]:
        """
        Crawl the model graph and extract information.

        Args:
            nodes:
                The nodes to crawl.
            id_to_module:
                A dictionary mapping module ids to modules.
            id_to_module_node:
                A dictionary mapping module ids to module nodes.
            id_to_permutation:
                A dictionary mapping module ids to permutation tensors.
            num_to_id:
                A dictionary mapping node numbers to module ids.
                The purpose of this dictionary is to allow for and easy
                use of random permutations.
            num:
                The current node number.

        Returns:
            Three dicts: num_to_id, id_to_module_node, and id_to_permutation.
        """
        for node in nodes:
            if (
                    isinstance(node, ModuleNode)
                    and hasattr(id_to_module.get(node.compute_unit_id), "weight")
                    and node.compute_unit_id not in id_to_module_node.keys()
            ):
                id_to_module_node[node.compute_unit_id] = node
                num_to_id[num] = node.compute_unit_id
                id_to_permutation[node.compute_unit_id] = identity_tensor(
                    id_to_module.get(node.compute_unit_id)  # type: ignore[arg-type]
                    .weight  # type: ignore[union-attr]
                )
                num += 1

            children = list(node.children)
            if children:
                num_to_id, id_to_module_node, id_to_permutation = self._crawl_model(
                    children,  # type: ignore[arg-type]
                    id_to_module,
                    id_to_module_node,
                    id_to_permutation,
                    num_to_id,
                    num
                )

        return num_to_id, id_to_module_node, id_to_permutation

    def _find_permutations(self) -> None:
        """Find permutation tensors for the weights of model_b
        until the cost is minimized. Here, the cost refers to
        the difference between the weights of model_a and model_b."""
        cost = 1e12
        for _ in range(self.max_iter):
            cost_new = self._find_permutations_step()

            if cost_new >= cost:
                break

    def _find_permutations_step(self) -> float:
        """Permute the weights of the modules in model_b once."""
        cost = 0.0
        for num in torch.randperm(len(self.num_to_id_a)):
            id_a = self.num_to_id_a[num.item()]
            id_b = self.num_to_id_b[num.item()]

            p1 = self.id_to_permutation[id_b]
            w_a1 = self.id_to_module_a[id_a].weight
            w_b1 = self.id_to_module_b[id_b].weight

            _, _, p0 = self._find_parent(id_a, id_b)
            w_a2, w_b2, p2 = self._find_child(id_a, id_b)

            # If there is no composable parent or child,
            #   then calculate the cost from the current module alone.
            # Otherwise, do the calculation using the parent and/or child.
            cost_tensor = w_a1 @ (p1 @ w_b1.T)  # TODO: only calculate this if needed
            cost_tensor = w_a1 @ (p0 @ w_b1.T) if p0 is not None else cost_tensor
            cost_tensor += (
                w_a2.T @ (p2 @ w_b2)
                if p2 is not None and w_a2 is not None and w_b2 is not None
                else torch.zeros_like(cost_tensor)
            )

            ri, ci = linear_sum_assignment(cost_tensor.detach().numpy(), maximize=True)

            # The rows should not change, only the columns.
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()

            # Permute
            self.id_to_permutation[id_b] = p1[ci]

            cost += self._calculate_cost(cost_tensor)

        cost /= len(self.num_to_id_a)
        return cost

    @staticmethod
    def _calculate_cost(cost_tensor: torch.Tensor) -> float:
        # The cost is calculated differently in the repo to the paper.
        # I don't understand how or what it means, however, so for now,
        #   I'm just using the Frobenius norm of the cost matrix,
        #   which is the same as the sum of the absolute values of the elements.
        return torch.frobenius_norm(cost_tensor).item()

    def _find_parent(  # type: ignore[empty-body]
            self,
            id_a: int,
            id_b: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
        """Find a parent for each of the two modules.

        The parent should be a module that is composable with the module.
        """
        pass  # TODO

    def _find_child(  # type: ignore[empty-body]
            self,
            id_a: int,
            id_b: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
        """Find a child for each of the two modules.

        The child should be a module that is composable with the module.
        """
        pass  # TODO


