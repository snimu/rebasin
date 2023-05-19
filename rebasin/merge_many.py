from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from rebasin.permutation_coordinate_descent import PermutationCoordinateDescent
from rebasin.utils import parse_logging_level


class MergeMany:
    """
    Implements the MergeMany algorithm from the Git Re-basin paper.

    Args:
        models:
            The models to merge.

            **ATTENTION**: The models are modified in-place! Make sure to pass
            copies of the models if you want to keep the originals.

            *Type*: :code:`Sequence[nn.Module | str | Path]`

        working_model:
            The model to use as the working model.
            This is necessary, because the model-architecture must be known.
            This is not the case for models loaded from a file.
            **This will also be modified in-place.**

            *Type*: :code:`nn.Module`

        input_data:
            The input data to use for the models.
            Can be anything; will be passed to :class:`PermutationCoordinateDescent`'s
            :code:`input_data_b` argument. Assumed to be on the same device as
            :code:`working_model`.

            *Type*: :code:`Any`

        device:
            The device to use for the models.

    Properties:
        merged_model:
            The merged model.
            This is only available after calling :py:meth:`~MergeMany.run`.
            Before, it raises an :code:`AttributeError`.

            *Type*: :code:`nn.Module`

    """

    def __init__(
            self,
            models: Sequence[nn.Module | str | Path],
            working_model: nn.Module,
            input_data: Any,
            device: torch.device | str | None = None,
            logging_level: str | int = logging.ERROR,
    ) -> None:
        self.models = models
        self.working_model = working_model
        self.input_data = input_data
        self.device = device
        self.logging_level = parse_logging_level(logging_level)

        self._merged_model: nn.Module = nn.Sequential()
        self._merged_model_was_caclulated = False

    @property
    def merged_model(self) -> nn.Module:
        """
        The merged model.
        """
        if not self._merged_model_was_caclulated:
            raise AttributeError(
                "The merged model has not been calculated yet. "
                "Please call MergeMany.run() first."
            )
        return self._merged_model

    def run(
            self, max_iterations: int = 100, max_iterations_pcd: int = 100
    ) -> nn.Module:
        """
        Runs the MergeMany algorithm.

        Args:
            max_iterations:
                The maximum number of iterations to run the MergeMany algorithm for.

                *Type*: :code:`int`

                *Default*: :code:`100`

            max_iterations_pcd:
                The :code:`max_iterations` argument to pass to
                the :class:`PermutationCoordinateDescent`'s
                :py:meth:`~PermutationCoordinateDescent.rebasin`-method.

        Returns:
            The merged model.

        Note:
            If the model contains any :code:`BatchNorm`-layers, it is recommended
            to run :py:func:`~rebasin.utils.recalculate_batch_norms` on the final
            model to recalculate the running stats.
        """
        if self.logging_level <= logging.INFO:
            print("Running MergeMany algorithm...")

        loop = tqdm(range(max_iterations), disable=self.logging_level > logging.INFO)
        for i in loop:
            progress = self._run_iteration(max_iterations_pcd, loop)
            if not progress:
                if self.logging_level <= logging.INFO:
                    print(f"Converged. Stopping early after {i} steps.")
                break

        self._merged_model = self._get_mean_model()
        self._merged_model_was_caclulated = True
        return self.merged_model


    def _run_iteration(self, max_iterations: int, loop: tqdm[Any]) -> bool:
        progress = False
        for i in torch.randperm(len(self.models)):
            mean_model = self._get_mean_model(except_index=i)
            self._set_working_model_state_dict(index=i)

            if self.logging_level <= logging.INFO:
                loop.set_description(f"Running PCD for model {i}...")

            pcd = PermutationCoordinateDescent(
                model_a=mean_model,
                model_b=self.working_model,
                input_data_b=self.input_data,
                device_a=self.device,
                device_b=self.device,
            )
            pcd.rebasin(max_iterations=max_iterations)

            self._store_working_model(index=i)
            progress = progress or self._calculate_progress(pcd)

        return progress

    @torch.no_grad()
    def _get_mean_model(self, except_index: int = -1) -> nn.Module:
        mean_model = copy.deepcopy(self.working_model)
        first_index = 0 if except_index != 0 else 1

        for i, model in enumerate(self.models):
            if i == except_index:
                continue

            if i == first_index:
                mean_model.load_state_dict(
                    model.state_dict()
                    if isinstance(model, nn.Module)
                    else torch.load(model)
                )
                continue

            # It is fine to use self.working_model here, because it is
            #   set again to the correct values after this method is called.
            self.working_model.load_state_dict(
                model.state_dict()
                if isinstance(model, nn.Module)
                else torch.load(model)
            )

            # Merge all the parameters.
            for param, mean_param in zip(
                    self.working_model.parameters(), mean_model.parameters()
            ):
                mean_param.data.add_(param.data)

        # Divide by the number of models.
        for param in mean_model.parameters():
            param.data.div_(
                len(self.models) - 1
                if 0 <= except_index < len(self.models)
                else len(self.models)
            )

        return mean_model

    def _set_working_model_state_dict(self, index: int) -> None:
        load_model = self.models[index]
        if isinstance(load_model, (str, Path)):
            self.working_model.load_state_dict(torch.load(load_model))
        else:
            self.working_model.load_state_dict(copy.deepcopy(load_model.state_dict()))

    def _store_working_model(self, index: int) -> None:
        store_model = self.models[index]
        if isinstance(store_model, (str, Path)):
            torch.save(self.working_model.state_dict(), store_model)
        else:
            store_model.load_state_dict(copy.deepcopy(self.working_model.state_dict()))

    @staticmethod
    def _calculate_progress(pcd: PermutationCoordinateDescent) -> bool:
        # If PermutationCoordinateDescent did not find any permutation that is not
        #   the identity permutation, then we have made no progress.
        # Otherwise, we have.
        return not all(
            torch.allclose(perm.perm_indices, torch.arange(len(perm.perm_indices)))
            for perm, _ in pcd.pinit.model_graph.permutation_to_info
        )
