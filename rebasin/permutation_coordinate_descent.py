from __future__ import annotations

import copy
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import]
from torchview import FunctionNode, ModuleNode, TensorNode, draw_graph

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
        self.num_to_id_a, self.id_to_module_node_a, _ = \
            self._crawl_model(list(root_a), self.id_to_module_a)

        root_b = draw_graph(model_b, input_data=input_data, depth=10000).root_container
        self.num_to_id_b, self.id_to_module_node_b, self.id_to_permutation = \
            self._crawl_model(list(root_b), self.id_to_module_b)

        # self._find_permutations()

    def _crawl_model(
            self, root: list[NODE_TYPES], id_to_module: dict[int, torch.nn.Module]
    ) -> tuple[dict[int, int], dict[int, ModuleNode], dict[int, torch.Tensor]]:
        _, num_to_id, id_to_module_node, id_to_permutation = \
            self._crawl_model_recursive(root, id_to_module, {}, {}, {}, 0)
        return num_to_id, id_to_module_node, id_to_permutation

    def _crawl_model_recursive(
            self,
            nodes: list[NODE_TYPES],
            id_to_module: dict[int, torch.nn.Module],
            id_to_module_node: dict[int, ModuleNode],
            id_to_permutation: dict[int, torch.Tensor],
            num_to_id: dict[int, int],
            num: int
    ) -> tuple[int, dict[int, int], dict[int, ModuleNode], dict[int, torch.Tensor]]:
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
            The current Module-number
            and three dicts: num_to_id, id_to_module_node, and id_to_permutation.
        """
        for node in nodes:
            if (
                    isinstance(node, ModuleNode)
                    and node.compute_unit_id not in id_to_module_node.keys()
                    and hasattr(id_to_module.get(node.compute_unit_id), "weight")
                    and len(id_to_module.get(  # type: ignore[arg-type]
                        node.compute_unit_id).weight.shape  # type: ignore[union-attr]
                    ) > 0
            ):
                id_to_module_node[node.compute_unit_id] = node
                num_to_id[num] = node.compute_unit_id

                module = id_to_module.get(node.compute_unit_id)
                assert module is not None and module.weight is not None
                assert isinstance(module.weight.shape, torch.Size)

                id_to_permutation[node.compute_unit_id] = (
                    torch.arange(module.weight.shape[-2])
                    if len(module.weight.shape) > 1
                    else torch.arange(module.weight.shape[0])
                )
                num += 1

            children = list(node.children)
            if children:
                num, num_to_id, id_to_module_node, id_to_permutation = \
                    self._crawl_model_recursive(
                        children,  # type: ignore[arg-type]
                        id_to_module,
                        id_to_module_node,
                        id_to_permutation,
                        num_to_id,
                        num
                    )

        return num, num_to_id, id_to_module_node, id_to_permutation

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

            id_child_a: int = ...  # type: ignore[assignment]
            id_child_b: int = ...  # type: ignore[assignment]
            id_parent_b: int = ...  # type: ignore[assignment]

            w_a1 = self.id_to_module_a[id_a].weight
            w_b1 = (
                self._permuted_weight(weight_id=id_b, perm_id=id_b)
                if id_parent_b is None
                else self._permuted_weight(weight_id=id_b, perm_id=id_parent_b)
            )

            w_a2 = (
                self.id_to_module_a[id_child_a].weight
                if id_child_a is not None else None
            )
            w_b2 = (
                self._permuted_weight(weight_id=id_child_b, perm_id=id_child_b)
                if id_child_b is not None else None
            )

            cost_tensor = w_a1 @ w_b1.T
            cost_tensor += \
                w_a2.T @ w_b2 if w_a2 is not None and w_b2 is not None else 0.0

            ri, ci = linear_sum_assignment(cost_tensor.detach().numpy(), maximize=True)

            # The rows should not change, only the columns.
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()

            # Update permutation
            self.id_to_permutation[id_b] = self.id_to_permutation[id_b][ci]

            cost += self._calculate_cost(cost_tensor)

        cost /= len(self.num_to_id_a)
        return cost

    @staticmethod
    def _calculate_cost(cost_tensor: torch.Tensor) -> float:
        # The cost is calculated differently in the repo to the paper.
        # I don't understand how or what it means, however, so for now,
        #   I'm just using the Frobenius norm of the cost matrix
        #   (sum of absolute values of the elements).
        # Will experiment and see if it works like this.
        # TODO: figure out what the cost is supposed to be
        return torch.frobenius_norm(cost_tensor).item()

    def _permuted_weight(
            self, weight_id: int, perm_id: int
    ) -> torch.Tensor | torch.nn.Parameter:
        """Permute the weight of a module."""
        weight = self.id_to_module_b[weight_id].weight
        assert weight is not None
        assert isinstance(weight.shape, torch.Size)

        perm_weight = copy.deepcopy(weight)
        perm_col = self.id_to_permutation[perm_id]

        perm_index = -1 if len(weight.shape) > 1 else 0
        perm_weight[perm_index] = perm_weight[perm_index][perm_col]
        return perm_weight
