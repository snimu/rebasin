from __future__ import annotations

import torch

from .math import identity_tensor


class PermutationCoordinateDescent:
    """
    Find permutations by matching weights.

    Assumes a MLP with weights of same shape in each layer.
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
