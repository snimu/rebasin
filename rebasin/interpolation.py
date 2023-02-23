from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Interpolation:
    """Interpolate between two models."""

    def __init__(
            self,
            model1: torch.nn.Module,
            model2: torch.nn.Module,
            dataloader: DataLoader[Any],
            loss_fn: torch.nn.Module,
            verbose: bool = False,
    ) -> None:
        self.model1 = model1
        self.model2 = model2
        self.model: torch.nn.Module | None = None
        self.losses: list[float] = []
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.verbose = verbose

    def interpolate(self, steps: int = 20) -> None:
        steps_ = tqdm(list(range(steps))) if self.verbose else list(range(steps))
        for step in steps_:
            weights = self._interpolate_step(percentage=step / steps)
            model = self._create_model(weights)
            loss = self._eval_model(model)

            if not self.losses:
                self.losses.append(loss)
                self.model = copy.deepcopy(model)
                continue

            if loss < min(self.losses):
                self.model = copy.deepcopy(model)

            self.losses.append(loss)

    def _interpolate_step(self, percentage: float) -> list[torch.nn.Parameter]:
        weights: list[torch.nn.Parameter] = []

        for _i, (module1, module2) in enumerate(
                zip(self.model1.modules(), self.model2.modules(), strict=True)
        ):
            if hasattr(module1, "weight"):
                assert hasattr(
                    module2, "weight"
                ), "Both models must be identical except in their weight-values."

                w1: torch.Tensor = module1.weight  # type: ignore[assignment]
                w2: torch.Tensor = module2.weight  # type: ignore[assignment]
                weight = torch.nn.Parameter(
                    torch.lerp(input=w1, end=w2, weight=percentage)
                )
                weights.append(weight)

        return weights

    def _create_model(self, weights: list[torch.nn.Parameter]) -> torch.nn.Module:
        model = copy.deepcopy(self.model1)
        weight_idx = 0

        for module in model.modules():
            if not hasattr(module, "weight"):
                continue

            module.weight = weights[weight_idx]
            weight_idx += 1

        return model

    @torch.no_grad()
    def _eval_model(self, model: torch.nn.Module) -> float:
        model.eval()
        losses: list[float] = []
        for (data, target) in self.dataloader:
            out = model(data)

            try:
                loss = float(self.loss_fn(out, target))
            except Exception as e:
                raise RuntimeError("Couldn't calculate the loss.") from e

            losses.append(loss)

        avg_loss = sum(losses) / len(losses)
        return avg_loss

