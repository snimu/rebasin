from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import]

from .math import identity_tensor


class PermutationCoordinateDescent:
    """
    Find permutations by matching weights.

    Assumes an MLP with weights of same shape in each layer.
    """

    def __init__(self, model1: torch.nn.Module, model2: torch.nn.Module) -> None:
        self.model1 = model1
        self.model2 = model2
        self.wperms, self.weights, self.windices = self._init_wperms()

    def _init_wperms(
            self
    ) -> tuple[
        list[torch.Tensor],
        list[tuple[torch.nn.Parameter, torch.nn.Parameter]],
        list[int]
    ]:
        perms: list[torch.Tensor] = []
        weights: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        windices: list[int] = []

        for i, (module1, module2) in enumerate(
                zip(self.model1.modules(), self.model2.modules(), strict=True)
        ):
            ident_str = "Both models must be identical except in their weight-values."

            if hasattr(module1, "weight"):
                assert hasattr(module2, "weight"), ident_str
                w1: torch.nn.Parameter = module1.weight  # type: ignore[assignment]
                w2: torch.nn.Parameter = module2.weight  # type: ignore[assignment]
                assert w1.size() == w2.size(), ident_str

                perms.append(identity_tensor(w2))
                weights.append((w1, w2))
                windices.append(i)

            if hasattr(module2, "weight"):
                assert hasattr(module1, "weight"), ident_str

        return perms, weights, windices

    def coordinate_descent(self) -> None:
        """Calculate the permutations."""
        perm_changes = [True] * len(self.weights)

        while any(perm_changes):
            perm_changes = self._coordinate_descent_step()

    def _coordinate_descent_step(self) -> list[bool]:
        layer_nums = np.arange(len(self.weights) - 2) + 1
        shuffled_layers = np.random.default_rng().permutation(layer_nums)
        perm_changes: list[bool] = []

        for i in shuffled_layers:
            p_last = self.wperms[i-1]
            p_next = self.wperms[i+1]
            w_a, w_b = self.weights[i][0], self.weights[i][1]
            w_a_next, w_b_next = self.weights[i+1][0], self.weights[i+1][1]

            cost_matrix = (w_a @ p_last @ w_b).mT + (w_a_next @ p_next @ w_b_next)

            row_ind, col_ind = linear_sum_assignment(
                cost_matrix.cpu().detach().numpy(), maximize=True
            )

            perm = self.wperms[i][row_ind]
            perm = perm[:, col_ind]
            self.wperms[i] = perm

            changed = (
                    sorted(list(row_ind)) == list(row_ind)
                    and sorted(list(col_ind)) == col_ind
            )
            perm_changes.append(changed.any())  # type: ignore[union-attr]

        return perm_changes

    def rebasin(self) -> None:
        """Re-basin model2."""
        modules = list(self.model2.modules())

        for i, wind in enumerate(self.windices):
            perm = self.wperms[i]
            perm_last = self.wperms[i-1] if i > 0 else perm
            module = modules[wind]

            weight = torch.nn.Parameter(perm @ module.weight @ perm_last.mT)
            module.weight = weight
