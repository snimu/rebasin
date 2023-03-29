from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .util import get_inputs_labels, recalculate_batch_norms


class Interpolation:
    def __init__(
            self,
            models: Sequence[torch.nn.Module],
            loss_fn: Any,
            train_dataloader: DataLoader[Any],
            val_dataloader: DataLoader[Any],
            input_indices: Sequence[int] | int = 0,
            output_indices: Sequence[int] | int = 1,
            savedir: Path | str | None = None,
            save_all: bool = False,
    ) -> None:
        self._sanity_checks(
            models,
            loss_fn,
            train_dataloader,
            val_dataloader,
            input_indices,
            output_indices,
            savedir,
            save_all

        )
        if savedir is not None and isinstance(savedir, str):
            savedir = Path(savedir)

        self.models = models
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.savedir = savedir
        self.save_all = save_all

        self.losses_original = [self.evaluate_model(model) for model in self.models]
        self.losses_interpolated: list[float] = []

        min_idx = int(torch.argmin(torch.tensor(self.losses_original)))
        self.best_loss = self.losses_original[min_idx]
        self.best_model = self.models[min_idx]
        self.best_model_name = f"model_{min_idx}.pt"

    def interpolate(self, steps: int = 20) -> None:
        """Interpolate between the models and save the best one or all."""

    def evaluate_model(self, model: nn.Module) -> float:
        model.eval()
        loss = 0.0
        for batch in self.val_dataloader:
            inputs, labels = get_inputs_labels(
                batch, self.input_indices, self.output_indices
            )
            y_pred = model(*inputs)
            loss += float(self.loss_fn(y_pred, *labels))
        return loss / len(self.val_dataloader)

    @staticmethod
    def _sanity_checks(
            models: Sequence[torch.nn.Module],
            loss_fn: Any,
            train_dataloader: DataLoader[Any],
            val_dataloader: DataLoader[Any],
            input_indices: Sequence[int] | int,
            output_indices: Sequence[int] | int,
            savedir: Path | str | None,
            save_all: bool = False,
    ) -> None:
        assert isinstance(models, Sequence), "Models must be a sequence"
        assert all(isinstance(model, nn.Module) for model in models), \
            "All models must be a subclass of nn.Module"

        assert callable(loss_fn), "Loss function must be callable"

        assert isinstance(train_dataloader, DataLoader)
        assert isinstance(val_dataloader, DataLoader)

        assert isinstance(input_indices, (int, Sequence))
        assert isinstance(output_indices, (int, Sequence))

        if isinstance(input_indices, Sequence):
            assert all(isinstance(i, int) for i in input_indices)
        if isinstance(output_indices, Sequence):
            assert all(isinstance(i, int) for i in output_indices)

        assert isinstance(savedir, (Path, str, type(None)))
        assert isinstance(save_all, bool)

        assert len(models) > 1, "Need at least two models to interpolate between"

        if save_all:
            assert savedir is not None



class LerpSimple(Interpolation):
    """
    Linear interpolation between two models.
    """

    def interpolate(self, steps: int = 20) -> None:
        """
        Interpolate between the models and save the best one or all.

        For two models, the interpolation looks like this
        (using `m_a` to denote model number `a`, and i_ab_c to denote
        interpolation step `c` between models number `a` and `b`):

        |

        ```
        m_1 - i_12_1 - ... - i_12_[steps] - m_2
        ```

        |

        For `n` models, the same is repeated between every other pair of models,
        meaning:

        |

        ```
        m_1 - i_12_1 - ... - i_12_[steps]
        - m_2 - i_23_1 - ... - i_23_[steps]
        - m_3 - ... - m_n
        ```

        |

        This means that the total number of interpolations is `steps * (n - 1)`.

        Args:
            steps:
                The number of steps to take between each pair of models.
        """

        for step in range(steps):
            # Interpolate between the two models
            # (step + 1) so that it never starts at zero percent.
            # (steps + 2) so that it never ends at 100 percent.
            # This is because the start and end models already exist
            percentage = (step + 1) / (steps + 2)
            self._interpolate_step(percentage=percentage, step=step)

        # Save the best model, unless self.save_all is True.
        # In that case, all models are already saved in _interpolate_step.
        if self.savedir is not None and not self.save_all:
            torch.save(
                self.best_model.state_dict(),
                self.savedir.joinpath(self.best_model_name)
            )

    def _interpolate_step(self, percentage: float, step: int) -> None:
        # Interpolate between all models
        for model_num, (model1, model2) in enumerate(
                zip(self.models[:-1], self.models[1:], strict=True)
        ):
            model_interp = copy.deepcopy(model1)
            for param1, param2, param_interp in zip(
                    model1.parameters(),
                    model2.parameters(),
                    model_interp.parameters(),
                    strict=True
            ):
                param_interp.data = torch.lerp(param1, param2, percentage)

            # Recalculate BatchNorm statistics
            recalculate_batch_norms(
                model_interp, self.train_dataloader, self.input_indices
            )

            # Evaluate
            loss = self.evaluate_model(model_interp)
            self.losses_interpolated.append(loss)

            # Save
            filename = f"interp_models_{model_num}_{model_num+1}_step_{step}.pt"

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_model = model_interp
                self.best_model_name = filename  # for later saving if not save_all

            if self.savedir is not None and self.save_all:
                torch.save(model_interp.state_dict(), self.savedir.joinpath(filename))

