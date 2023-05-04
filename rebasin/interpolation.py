from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rebasin.utils import recalculate_batch_norms


class Interpolation:
    """
    Interpolate between two models.

    The base class for all interpolation classes.
    This class is not meant to be used directly.

    Args:
        models:
            The models to interpolate between.
            This can be multiple models (see explanation above).

        eval_fn:
            The function to evaluate the models with.
            Usually, this simply calculates the loss for every batch in a
            validation or test dataloader, but it can be anything,
            including accuracy, or perplexity, etc.
            By using a :code:`eval_fn`, the user has maximum control.

            Its signature must be as follows:

            .. code-block:: python
                dev eval_fn(
                    model: nn.Module, device: torch.device | str | None
                ) -> float:
                    ...

            The :code:`model` argument is necessary
            because different models will have to be
            evaluated. The :code:`device` argument is necessary because the different
            models can be evaluated on different devices
            (see the :code:`devices` argument).

        eval_mode:
            The mode to use for evaluation. If "min", the model with the lowest
            metric is chosen, if "max", the model with the highest metric is chosen.

        train_dataloader:
            The dataloader to use for training the models.
            If this is given, then the :class:`Interpolation` subclass will check if
            the model has :code:`BatchNorm` layers and if so, will
            recalculate their :code:`running_mean` and :code:`running_var`
            statistics using the training data.

            This is unfortunately strongly recommended if your model includes
            these statistics.

        devices:
            Models may need a large amount of GPU-memory.
            To avoid running out of memory,
            the argument :code:`devices` can be used to specify which device
            each of the given models in :code:`models` is on.
            As each model is evaluated using the function given in :code:`eval_fn`,
            the corresponding device is passed to :code:`eval_fn` as well.

        device_interp:
            The device to use for the interpolation.
            If this is :code:`None`, no parameter will be moved
            to a different device for interpolation, and the
            interpolated model will be created on CPU.
            Again, this argument is useful for saving on memory.

        input_indices:
            If a training dataloader is given to :code:`train_dataloader`,
            and the model has :code:`BatchNorm` layers, then the
            :code:`running_mean` and :code:`running_var` statistics
            are recalculated using the training data.
            However, :class:`Interpolation` does not know which output from
            the dataloader is the input to the model. Sometimes,
            a model takes several inputs, sometimes, and input and a target
            are given by the dataloader, etc.
            To solve this, the argument :code:`input_indices` can be used
            to provide the indices of the inputs to the model.

            Can be an integer or a sequence of integers.

            The outputs of the training dataloader corresponding to the
            indices given here will be used as inputs to the models to
            recalculate the :code:`running_mean` and :code:`running_var`

        savedir:
            The directory to save the models in.
            If :code:`None`, the models are not saved.

        save_all:
            If :code:`True`, all models are saved,
            if :code:`False`, only the best model is saved.

        verbose:
            If :code:`True`, prints the progress of the interpolation.
    """
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

        if devices is not None:
            assert isinstance(devices, Sequence)
            assert all(
                isinstance(device, (str, torch.device)) for device in devices
            )
            assert len(devices) == len(models)

        assert isinstance(device_interp, (str, torch.device, type(None)))

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
    r"""
    Linear interpolation between two models.

    For two models and `s` steps, the interpolation looks like this
    (using :math:`m_a` to denote model number :math:`a`,
    and :math:`i_{ab,c}` to denote interpolation step `c`
    between models number `a` and `b`):

    |

    .. math::

        m_1 - i_{12,1} - ... - i_{12,s} - m_2

    |

    For `n` models, the same is repeated between every other pair of models,
    meaning:

    |

    .. math::
        m_1 - i_{12,1} - ... - i_{12,s} - m_2
        - i_{23,1} - ... - i_{23,s} - m_3 - ... - m_n

    |

    This means that the total number of interpolations is :math:`s \cdot (n - 1)`.

    Args:
        models:
            The models to interpolate between.
            This can be multiple models (see explanation above).

        eval_fn:
            The function to evaluate the models with.
            Usually, this simply calculates the loss for every batch in a
            validation or test dataloader, but it can be anything,
            including accuracy, or perplexity, etc.
            By using a :code:`eval_fn`, the user has maximum control.

            Its signature must be as follows:

            .. code-block:: python
                dev eval_fn(
                    model: nn.Module, device: torch.device | str | None
                ) -> float:
                    ...

            The :code:`model` argument is necessary
            because different models will have to be
            evaluated. The :code:`device` argument is necessary because the different
            models can be evaluated on different devices
            (see the :code:`devices` argument).

        eval_mode:
            The mode to use for evaluation. If "min", the model with the lowest
            metric is chosen, if "max", the model with the highest metric is chosen.

        train_dataloader:
            The dataloader to use for training the models.
            If this is given, then :class:`LerpSimple` will check if
            the model has :code:`BatchNorm` layers and if so, will
            recalculate their :code:`running_mean` and :code:`running_var`
            statistics using the training data.

            This is unfortunately strongly recommended if your model includes
            these statistics.

        devices:
            Models may need a large amount of GPU-memory.
            To avoid running out of memory,
            the argument :code:`devices` can be used to specify which device
            each of the given models in :code:`models` is on.
            As each model is evaluated using the function given in :code:`eval_fn`,
            the corresponding device is passed to :code:`eval_fn` as well.

        device_interp:
            The device to use for the interpolation.
            If this is :code:`None`, no parameter will be moved
            to a different device for interpolation, and the
            interpolated model will be created on CPU.
            Again, this argument is useful for saving on memory.

        input_indices:
            If a training dataloader is given to :code:`train_dataloader`,
            and the model has :code:`BatchNorm` layers, then the
            :code:`running_mean` and :code:`running_var` statistics
            are recalculated using the training data.
            However, :class:`LerpSimple` does not know which output from
            the dataloader is the input to the model. Sometimes,
            a model takes several inputs, sometimes, and input and a target
            are given by the dataloader, etc.
            To solve this, the argument :code:`input_indices` can be used
            to provide the indices of the inputs to the model.

            Can be an integer or a sequence of integers.

            The outputs of the training dataloader corresponding to the
            indices given here will be used as inputs to the models to
            recalculate the :code:`running_mean` and :code:`running_var`

        savedir:
            The directory to save the models in.
            If :code:`None`, the models are not saved.

        save_all:
            If :code:`True`, all models are saved,
            if :code:`False`, only the best model is saved.

        verbose:
            If :code:`True`, prints the progress of the interpolation.
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
