from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .util import get_inputs_labels, recalculate_batch_norms


class Interpolation:
    def __init__(
            self,
            models: Sequence[torch.nn.Module],
            eval_fn: Any,
            eval_mode: str = "min",
            train_dataloader: DataLoader[Any] | None = None,
            devices: Sequence[torch.device | str] | None = None,
            device_interp: torch.device | str | None = None,
            input_indices: Sequence[int] | int = 0,
            savedir: Path | str | None = None,
            save_all: bool = False,
            verbose: bool = False,
    ) -> None:
        self._sanity_checks(
            models,
            eval_fn,
            eval_mode,
            train_dataloader,
            devices,
            device_interp,
            input_indices,
            savedir,
            save_all,
            verbose
        )
        if verbose:
            print("Setting up interpolation...")

        if savedir is not None and isinstance(savedir, str):
            savedir = Path(savedir)

        self.models = models
        self.eval_fn = eval_fn
        self.eval_mode = eval_mode
        self.idx_fn = torch.argmax if eval_mode == "max" else torch.argmin
        self.train_dataloader = train_dataloader
        self.devices: Sequence[torch.device | str | None] = (
            devices if devices is not None
            else [None] * len(models)  # type: ignore[list-item]
        )
        self.device_interp = device_interp
        self.input_indices = input_indices
        self.savedir = savedir
        self.save_all = save_all
        self.verbose = verbose

        if self.verbose:
            print("Evaluating given models...")

        if self.train_dataloader is not None:
            for model, device in zip(self.models, self.devices):  # noqa
                recalculate_batch_norms(
                    model, self.train_dataloader, self.input_indices, device, verbose
                )

        self.metrics_models = [
            self.eval_fn(m, d)
            for m, d in tqdm(
                zip(self.models, self.devices), disable=not self.verbose  # noqa: B905
            )
        ]
        if self.verbose:
            print("Done.")
        self.metrics_interpolated: list[float] = []

        best_idx = int(self.idx_fn(torch.tensor(self.metrics_models)))
        self.best_metric = self.metrics_models[best_idx]
        self.best_model = self.models[best_idx]
        self.best_model_name = f"model_{best_idx}.pt"

    def interpolate(
            self,
            steps: int = 20,
            savedir: str | Path | None = None,
            save_all: bool = False
    ) -> None:
        """Interpolate between the models and save the best one or all."""

    @staticmethod
    def _sanity_checks(
            models: Sequence[torch.nn.Module],
            eval_fn: Any,
            eval_mode: str,
            train_dataloader: DataLoader[Any] | None,
            devices: Sequence[torch.device | str] | None,
            device_interp: torch.device | str | None,
            input_indices: Sequence[int] | int,
            savedir: Path | str | None,
            save_all: bool = False,
            verbose: bool = False,
    ) -> None:
        assert isinstance(models, Sequence), "Models must be a sequence"
        assert all(isinstance(model, nn.Module) for model in models), \
            "All models must be a subclass of nn.Module"

        assert callable(eval_fn), "Eval function must be callable"
        assert isinstance(eval_mode, str)
        assert eval_mode in ["min", "max"], "Eval mode must be 'min' or 'max'"

        assert isinstance(train_dataloader, (DataLoader, type(None)))

        if devices is None:
            assert device_interp is None
        else:
            assert isinstance(devices, Sequence)
            assert all(
                isinstance(device, (str, torch.device)) for device in devices
            )
            assert len(devices) == len(models)

        if device_interp is None:  # goes both ways
            assert devices is None
        else:
            assert isinstance(device_interp, (str, torch.device))

        assert isinstance(input_indices, (int, Sequence))
        if isinstance(input_indices, Sequence):
            assert all(isinstance(i, int) for i in input_indices)

        assert isinstance(savedir, (Path, str, type(None)))
        assert isinstance(save_all, bool)

        assert len(models) > 1, "Need at least two models to interpolate between"

        if save_all:
            assert savedir is not None

        assert isinstance(verbose, bool)


class LerpSimple(Interpolation):
    """
    Linear interpolation between two models.
    """

    def interpolate(
            self,
            steps: int = 20,
            savedir: str | Path | None = None,
            save_all: bool | None = None
    ) -> None:
        r"""
        Interpolate between the models and save the best one or all
        in :code:`self.best_model`.

        For two models and `s` steps, the interpolation looks like this
        (using :math:`m_a` to denote model number :math:`a`,
        and :math:`i_{ab,c}` to denote interpolation step `c`
        between models number `a` and `b`):

        |

        :math:`m_1 - i_{12,1} - ... - i_{12,s} - m_2`

        |

        For `n` models, the same is repeated between every other pair of models,
        meaning:

        |

        :math:`m_1 - i_{12,1} - ... - i_{12,s} - m_2
        - i_{23,1} - ... - i_{23,s} - m_3 - ... - m_n`

        |

        This means that the total number of interpolations is :math:`s \cdot (n - 1)`.

        |

        Args:
            steps:
                The number of steps to take between each pair of models.

            savedir:
                The directory to save the models in. Overwrites the :code:`savedir`
                argument passed to the constructor if not None.
                If both are :code:`None`, the models are not saved.

                If :code:`not None`, the following naming schema is used for the files:

                - :code:`"model_{model_num}.pt"`
                    if the model is one of the original models.
                    :code:`model_num` is the index of the model
                    in the :code:`models` argument.
                    The original models are not saved, except the best model.

                - :code:`"interp_models_{m_i}_{m_i+1}_perc_{percentage}.pt"`
                  if the model is interpolated.
                  :math:`m_i` and :math:`m_{i+1}` are the indices of the models
                  between which the interpolation is done.
                  `percentage` is the percentage of the way through the interpolation.
                  If it is small, the model is closer to :math:`m_i`,
                  and if it is large, the model is closer to :math:`m_{i+1}`.

            save_all:
                Whether to save all interpolated models, or just the best one.
                The best model is saved either way if a `savedir` is given.
                Overwrites the `save_all` argument passed to the constructor
                if not None.
        """
        # SANITY CHECKS AND DEFAULT SETTINGS
        assert isinstance(savedir, (Path, str, type(None)))
        if savedir is None:
            savedir = self.savedir
        elif isinstance(savedir, str):
            savedir = Path(savedir)

        if save_all is None:
            save_all = self.save_all

        assert isinstance(save_all, bool)

        if self.verbose:
            print("Interpolating between models...")

        # INTERPOLATION
        for step in tqdm(range(steps), disable=not self.verbose):
            # Interpolate between the two models
            # (step + 1) so that it never starts at zero percent.
            # (steps + 2) so that it never ends at 100 percent.
            # This is because the start and end models already exist
            percentage = (step + 1) / (steps + 2)
            self._interpolate_step(
                percentage=percentage, savedir=savedir, save_all=save_all
            )

        # SAVE THE BEST MODEL
        # If all interpolated models are saved and the best model is interpolated,
        #   it is unnecessary to save it again.
        # Conversely, if not all interpolated models are saved, the best model
        #   must be saved.
        # Alternatively, save_all only leads to interpolated models being saved,
        #   so the best model must be saved again if it is not interpolated.
        save_best_model = (not save_all) or (self.best_model in self.models)

        if save_best_model and (savedir is not None):
            torch.save(
                self.best_model.state_dict(),
                savedir.joinpath(self.best_model_name)
            )

    def _interpolate_step(
            self, percentage: float, savedir: Path | None, save_all: bool
    ) -> None:
        # Interpolate between all models
        for model_num, (model1, model2) in enumerate(
                zip(self.models[:-1], self.models[1:])  # noqa: B905
        ):
            model_interp = copy.deepcopy(model1).to(self.device_interp)

            for param1, param2, param_interp in zip(  # noqa: B905
                    model1.parameters(),
                    model2.parameters(),
                    model_interp.parameters(),
            ):
                param_interp.data = torch.lerp(
                    param1.to(self.device_interp),
                    param2.to(self.device_interp),
                    percentage
                )

            # Recalculate BatchNorm statistics
            if self.train_dataloader is not None:
                recalculate_batch_norms(
                    model_interp,
                    self.train_dataloader,
                    self.input_indices,
                    device=self.device_interp,
                    verbose=self.verbose
                )

            # Evaluate
            metric = self.eval_fn(model_interp, self.device_interp)
            self.metrics_interpolated.append(metric)

            # Save
            filename = f"interp_models_{model_num}_{model_num+1}" \
                       f"_perc_{percentage:.3f}.pt"

            is_best = (
                metric < self.best_metric
                if self.eval_mode == "min"
                else metric > self.best_metric
            )
            if is_best:
                self.best_metric = metric
                self.best_model = model_interp
                self.best_model_name = filename  # for later saving if not save_all

            if savedir is not None and save_all:
                torch.save(model_interp.state_dict(), savedir.joinpath(filename))
