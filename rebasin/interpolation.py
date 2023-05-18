from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rebasin.utils import parse_logging_level, recalculate_batch_norms


class Interpolation:
    """
    Interpolate between models.

    The base class for all interpolation classes.
    **This class is not meant to be used directly.**

    Args:
        models:
            The models to interpolate between.
            This can be multiple models (see explanation above).

            *Type:* :code:`Sequence[nn.Module]`

        eval_fn:
            The function to evaluate the interpolated models with.
            Usually, this simply calculates the loss for every batch in a
            validation or test dataloader, but it can be anything,
            including accuracy, or perplexity, etc.
            By using a :code:`eval_fn`, the user has maximum control.

            Its signature must be as follows:

                >>> def eval_fn(
                >>>     model: nn.Module, device: torch.device | str | None
                >>> ) -> float:
                >>>     ...

            The :code:`model` argument is necessary
            because different models will have to be
            evaluated. The :code:`device` argument is necessary because the different
            models can be evaluated on different devices
            (see the :code:`devices` argument).

            If the default :code:`eval_fn` is used, then the best model will always be
            :code:`models[0]`, and all metrics will be :code:`0.0`.
            In this case, it is recommended to provide a :code:`savedir`
            so that the interpolated models are saved and can be evaluated manually.

            *Type:* Callable[[nn.Module, torch.device | str | None], float]

            *Default:* :code:`lambda model, device: 0.0`

        eval_mode:
            The mode to use for evaluation. If "min", the model with the lowest
            metric is chosen, if "max", the model with the highest metric is chosen.

            *Type:* :code:`str`

            *Default:* :code:`"min"`

        train_dataloader:
            The dataloader to use for training the models.
            If this is given, then the :class:`Interpolation` subclass will check if
            the model has :code:`BatchNorm` layers and if so, will
            recalculate their :code:`running_mean` and :code:`running_var`
            statistics using the training data.

            This is unfortunately strongly recommended if your model includes
            these statistics.

            *Type:* :class:`torch.utils.data.DataLoader | None`

            *Default:* :code:`None`

        dataset_percentage:
            The percentage of the dataset to use
            for recalculating the BatchNorm statistics.
            Must be between :code:`0.0` and :code:`1.1`.

            *Type:* :code:`float`

            *Default:* :code:`1.0`

        dataset_iterations:
            An alternative to :code:`dataset_percentage`.
            If it is > 0, then this will be the maximum number of batches
            to use from the training dataloader for recalculating
            the BatchNorm statistics.

            *Type:* :code:`int`

            *Default:* :code:`-1`

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

            *Type:* :code:`Sequence[int] | int`

            *Default:* :code:`0`

        devices:
            Models may need a large amount of GPU-memory.
            To avoid running out of memory,
            the argument :code:`devices` can be used to specify which device
            each of the given models in :code:`models` is on.
            As each model is evaluated using the function given in :code:`eval_fn`,
            the corresponding device is passed to :code:`eval_fn` as well.

            *Type:* :code:`Sequence[torch.device | str | None] | None`

            *Default:* :code:`None`

        device_interp:
            The device to use for the interpolation.
            If this is :code:`None`, no parameter will be moved
            to a different device for interpolation, and the
            interpolated model will be created on CPU.
            Again, this argument is useful for saving on memory.

            *Type:* :code:`torch.device | str | None`

            *Default:* :code:`None`

        savedir:
            The directory to save the models in.
            If :code:`None`, the models are not saved.
            Otherwise, the interpolated models are saved in the given directory.

            Type: :class:`pathlib.Path` or :class:`str` or :code:`None`.

            Default: :code:`None`.

        logging_level:
            The logging level to use.

            Can be any of the following:

            - :code:`logging.DEBUG` / :code:`"DEBUG"` / :code:`"debug"` / :code:`10`
            - :code:`logging.INFO` / :code:`"INFO"` / :code:`"info"` / :code:`20`
            - :code:`logging.WARNING` / :code:`"WARNING"`
              / :code:`"warning"` / :code:`30`
            - :code:`logging.WARN`  / :code:`"WARN"` / :code:`"warn"` / :code:`30`
            - :code:`logging.ERROR` / :code:`"ERROR"` / :code:`"error"` / :code:`40`
            - :code:`logging.CRITICAL` / :code:`"CRITICAL"`
              / :code:`"critical"` / :code:`50`
            - :code:`logging.FATAL` / :code:`"FATAL"` / :code:`"fatal"` / :code:`50`

            Type: :code:`int | str`.

            Default: :code:`logging.ERROR`.
    """
    def __init__(
            self,
            models: Sequence[torch.nn.Module],
            eval_fn: Any = lambda model, device: 0.0,
            eval_mode: str = "min",
            train_dataloader: DataLoader[Any] | None = None,
            dataset_percentage: float = 1.0,
            dataset_iterations: int = -1,
            input_indices: Sequence[int] | int = 0,
            devices: Sequence[torch.device | str] | None = None,
            device_interp: torch.device | str | None = None,
            savedir: Path | str | None = None,
            logging_level: int | str = "ERROR"
    ) -> None:
        self._sanity_checks(
            models,
            eval_fn,
            eval_mode,
            train_dataloader,
            dataset_percentage,
            dataset_iterations,
            devices,
            device_interp,
            input_indices,
            savedir,
        )
        self.logging_level = parse_logging_level(logging_level)
        if self.logging_level <= logging.INFO:
            print("Setting up interpolation...")

        if savedir is not None and isinstance(savedir, str):
            savedir = Path(savedir)

        self.models = models
        self.eval_fn = eval_fn
        self.eval_mode = eval_mode
        self.idx_fn = torch.argmax if eval_mode == "max" else torch.argmin
        self.train_dataloader = train_dataloader
        self.dataset_percentage = dataset_percentage
        self.dataset_iterations = dataset_iterations
        self.devices: Sequence[torch.device | str | None] = (
            devices if devices is not None
            else [None] * len(models)  # type: ignore[list-item]
        )
        self.device_interp = device_interp
        self.input_indices = input_indices
        self.savedir = savedir

        self.metrics_interpolated: list[float] = []
        self.best_metric: float = float("inf") if eval_mode == "min" else -float("inf")

    def interpolate(self, steps: int = 20, savedir: str | Path | None = None) -> None:
        """Interpolate between the models and save the best one or all."""

    @staticmethod
    def _sanity_checks(
            models: Sequence[torch.nn.Module],
            eval_fn: Any,
            eval_mode: str,
            train_dataloader: DataLoader[Any] | None,
            dataset_percentage: float,
            dataset_iterations: int,
            devices: Sequence[torch.device | str] | None,
            device_interp: torch.device | str | None,
            input_indices: Sequence[int] | int,
            savedir: Path | str | None,
    ) -> None:
        assert isinstance(models, Sequence), "Models must be a sequence"
        assert all(isinstance(model, nn.Module) for model in models), \
            "All models must be a subclass of nn.Module"

        assert callable(eval_fn), "Eval function must be callable"
        assert isinstance(eval_mode, str)
        assert eval_mode in ["min", "max"], "Eval mode must be 'min' or 'max'"

        assert isinstance(train_dataloader, (DataLoader, type(None)))
        assert isinstance(dataset_percentage, float), "dataset_percentage must be float"
        assert 0.0 <= dataset_percentage <= 1.0, "dataset_percentage must be in [0, 1]"
        assert isinstance(dataset_iterations, int), "dataset_iterations must be int"

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

        assert len(models) > 1, "Need at least two models to interpolate between"


class LerpSimple(Interpolation):
    r"""
    Linear interpolation between models.

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

    For args, see :class:`Interpolation`.
    """

    def interpolate(self, steps: int = 20, savedir: str | Path | None = None) -> None:
        r"""
        Interpolate between the models and save the interpolated models
        if :code:`savedir` is given.

        Args:
            steps:
                The number of steps to take between each pair of models.

            savedir:
                The directory to save the models in. Overwrites the :code:`savedir`
                argument passed to the constructor if not None.
                If both are :code:`None`, the interpolated models are not saved.

                If :code:`not None`, the following naming schema is used for the files:

                :code:`"interp_models_{m_i}_{m_i+1}_perc_{percentage}.pt"`
                if the model is interpolated.
                :math:`m_i` and :math:`m_{i+1}` are the indices of the models
                between which the interpolation is done.
                `percentage` is the percentage of the way through the interpolation.
                If it is small, the model is closer to :math:`m_i`,
                and if it is large, the model is closer to :math:`m_{i+1}`.
        """
        # SANITY CHECKS AND DEFAULT SETTINGS
        assert isinstance(savedir, (Path, str, type(None)))
        if savedir is None:
            savedir = self.savedir
        elif isinstance(savedir, str):
            savedir = Path(savedir)

        if self.logging_level <= logging.INFO:
            print("Interpolating between models...")

        # INTERPOLATION
        loop = tqdm(range(steps), disable=self.logging_level > logging.INFO)
        for step in loop:
            # Interpolate between the two models
            # (step + 1) so that it never starts at zero percent.
            # (steps + 2) so that it never ends at 100 percent.
            # This is because the start and end models already exist
            percentage = (step + 1) / (steps + 2)
            self._interpolate_step(percentage=percentage, savedir=savedir, loop=loop)

    def _interpolate_step(
            self, percentage: float, savedir: Path | None, loop: tqdm[Any]
    ) -> None:
        # Interpolate between all models
        for model_num, (model1, model2) in enumerate(
                zip(self.models[:-1], self.models[1:])
        ):
            model_interp = copy.deepcopy(model1).to(self.device_interp)

            for param1, param2, param_interp in zip(
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
                    verbose=self.logging_level <= logging.INFO,
                    dataset_percentage=self.dataset_percentage,
                    loop=loop,
                    iterations=self.dataset_iterations,
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

            if savedir is not None:
                torch.save(model_interp.state_dict(), savedir.joinpath(filename))
