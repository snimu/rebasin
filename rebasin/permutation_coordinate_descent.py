from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import]
from torch import nn
from tqdm import tqdm

from rebasin import utils
from rebasin.modules import Permutation, PermutationInfo  # type: ignore[attr-defined]
from rebasin.permutation_initializer import PermutationInitializer


def calculate_progress(
        cost_mat: torch.Tensor,
        perm_old: torch.Tensor,
        perm_new: torch.Tensor,
        device: torch.device | str | None = None,
) -> bool:
    """
    Compute the progress of the permutation.

    Args:
        cost_mat:
            The cost matrix. Shape: (n, n).
        perm_old:
            The old permutation. Shape: (n,).
        perm_new:
            The new permutation. Shape: (n,).
        device:
            The device on which to perform the computation.

    Returns:
        :code:`True` if the new permutation moves the cost matrix
        closer to the identity matrix than the old permutation does,
        else :code:`False`.
    """
    cost_mat, perm_old, perm_new = (
        cost_mat.to(device), perm_old.to(device), perm_new.to(device)
    )

    # linear_sum_assignment finds the minimum cost by finding a permutation
    #   of the cost matrix that moves the highest values to the diagonal.
    # In other words, it tries to maximize the cost matrix's closeness
    #   to the identity matrix.
    # Therefore, we need to check whether the new permutation moves the
    #   cost matrix closer to the identity matrix compared to the old one.
    # This is done by taking the dot product of the permuted cost matrix
    #   with the identity matrix and then summing over the rows.
    # This is equivalent to picking the diagonal elements of the permuted
    #   cost matrix, and is achieved using torch.einsum.
    # In the end, the diagonal elements of the cost matrix are summed up.
    # The larger the sum, the closer the cost matrix is to the identity matrix,
    #   and the more progress has been made.
    eye_ = torch.eye(cost_mat.shape[0]).to(device)  # cost_mat is square
    optimality_old = torch.einsum("ij,ij->i", cost_mat[:, perm_old], eye_).sum()
    optimality_new = torch.einsum("ij,ij->i", cost_mat[:, perm_new], eye_).sum()

    progress = optimality_new.item() > optimality_old.item() + 1e-12
    return progress


class PermutationCoordinateDescent:
    r"""
    Implements the permutation coordinate descent algorithm.


    Args:
        model_a:
            The target model.

            Type: :class:`torch.nn.Module`.
        model_b:
            The model that will have its weights permuted.

            Type: :class:`torch.nn.Module`.
        input_data_b:
            The input data to the model_b.
            Used for tracing the model, so as long as the model can
            call its forward pass with this data, it can be anything.
            As this may be anything, this class cannot move it to any device.
            Therefore, it is the user's responsibility to ensure that
            :code:`input_data` is on the same device as :code:`model_b`.

            Type: Any.
        input_data_a:
            The input data to the model_a.
            The same comments apply as to :code:`input_data_b`.
            However, this argument is optional. If it is :code:`None`,
            then :code:`input_data_b` will be used for both models.
            Make sure that the device is correct in this case.

            Type: Any | None.
            Default: None.
        device_a:
            The device on which :code:`model_a` is located.
            Both :code:`model_a` and :code:`model_b` must
            either be :code:`None` or both be given.
            The purpose of this argument is to allow the user to
            use multiple GPUs if the models are so large that
            only one fits on a single GPU.

            Type: torch.device | str | None.
            Default: None.
        device_b:
            The device on which :code:`model_b` is located.

            Type: torch.device | str | None.
            Default: None.
        logging_level:
            The logging level to use.

            Can be any of the following:

            - :code:`logging.DEBUG` / :code:`'DEBUG'` / :code:`'debug'` / :code:`10`
            - :code:`logging.INFO` / :code:`'INFO'` / :code:`'info'` / :code:`20`
            - :code:`logging.WARNING` / :code:`'WARNING'`
              / :code:`'warning'` / :code:`30`
            - :code:`logging.WARN`  / :code:`'WARN'` / :code:`'warn'` / :code:`30`
            - :code:`logging.ERROR` / :code:`'ERROR'` / :code:`'error'` / :code:`40`
            - :code:`logging.CRITICAL` / :code:`'CRITICAL'`
              / :code:`'critical'` / :code:`50`
            - :code:`logging.FATAL` / :code:`'FATAL'` / :code:`'fatal'` / :code:`50`

            Type: int.
            Default: :code:`logging.ERROR`.
    """

    def __init__(
            self,
            model_a: nn.Module,
            model_b: nn.Module,
            input_data_b: Any,
            input_data_a: Any | None = None,
            device_a: torch.device | str | None = None,
            device_b: torch.device | str | None = None,
            logging_level: int | str = logging.ERROR,
    ) -> None:
        self.model_b = model_b
        self.device_a = device_a
        self.device_b = device_b
        self.logging_level = utils.parse_logging_level(logging_level)

        if self.logging_level <= logging.INFO:
            print("Initializing permutations...")

        self.pinit = PermutationInitializer(
            model_a, model_b, input_data_b, input_data_a
        )
        self.pinit.model_graph.enforce_identity()

        if self.logging_level <= logging.INFO:
            print("Done.")

    def rebasin(self, max_iterations: int = 100) -> None:
        """
        Run the permutation coordinate descent algorithm.

        Args:
            max_iterations:
                The maximum number of iterations to run the algorithm for.
                Likely to converge much faster than this.

                Default: 100.
        """
        self.calculate_permutations(max_iterations)
        self.apply_permutations()

    def calculate_permutations(self, max_iterations: int = 100) -> None:
        """
        Run the permutation coordinate descent algorithm to calculate the permutations.

        Args:
            max_iterations:
                The maximum number of iterations.
        """
        # Calculate the permutations
        if self.logging_level <= logging.INFO:
            print("Calculating permutations...")

        loop = tqdm(range(max_iterations), disable=self.logging_level > logging.INFO)
        for iteration in loop:
            progress = self._calculate_permutations_step(loop)
            if not progress:
                if self.logging_level <= logging.INFO:
                    print(f"Stopping early after {iteration + 1} steps.")
                break

    def _calculate_permutations_step(self, loop: tqdm[int]) -> bool:
        """
        Run one iteration of the permutation coordinate descent algorithm.

        Returns:
            True if the algorithm made progress, False otherwise.
        """
        progress = False

        for i in torch.randperm(len(self.pinit.model_graph.permutation_to_info)):
            # For mypy: give typehint
            perm_and_info: tuple[
                Permutation, list[PermutationInfo]
            ] = self.pinit.model_graph.permutation_to_info[i]

            perm, perm_infos = perm_and_info
            n = len(perm.perm_indices)
            cost_tensor = torch.zeros((n, n)).to(self.device_b)

            if self.logging_level == logging.DEBUG:
                loop.write("Calculating cost tensor...")
            for perm_info in perm_infos:
                # Copy so that the original parameter isn't moved to device_b.
                # If this were not done, then in one of the later steps,
                #   model_a would be almost completely on device_b.
                # This is a problem because we want the models to be able to live
                #   on different devices to allow for larger models
                #   by splitting their memory needs.
                # Also, we will apply some permutations, but don't want this
                #   to affect the original parameter, because we might do this
                #   again in another iteration.
                info = copy.deepcopy(perm_info)

                # Apply all permutations except the one we are currently working on.
                info.module.apply_permutations(except_axis=info.axis)
                w_a = info.parameter_a.to(self.device_b)
                w_b = info.parameter_b.to(self.device_b)

                # We want a square matrix as a cost tensor.
                # It should have shape (n, n).
                # To achieve this, we first move the axis of interest to the front.
                w_a = w_a.moveaxis(info.axis, 0) if len(w_a.shape) > 1 else w_a
                w_b = w_b.moveaxis(info.axis, 0) if len(w_b.shape) > 1 else w_b

                # Then, we reshape the tensor to (n, -1).
                # This means that all dimensions except the first one are flattened.
                # This way, multiplying w_a with w_b.T
                #   will result in a matrix of shape (n, n).
                # That matrix --- the cost_tensor --- encodes the similarity between
                #   w_a and w_b along the axis of interest.
                w_a = w_a.reshape(n, -1) if len(w_a.shape) > 1 else w_a
                w_b = w_b.reshape(n, -1).mT if len(w_b.shape) > 1 else w_b

                # We calculate the cost tensor by multiplying w_a with w_b.mT
                #   and adding it to the previous cost tensor.
                # We add the cost tensors because one permutation can apply to several
                #   weights and biases.
                # Adding the costs introduces a dependency between the permutations.
                # That dependency is desirable because it is important to not just
                #   bring the models close to each other layer by layer,
                #   but ideally to make the entire path through the model
                #   as similar as possible between the two.
                cost_tensor += w_a @ w_b

            if self.logging_level == logging.DEBUG:
                loop.write(f"Done. Cost matrix shape: {cost_tensor.shape}.")
                loop.write("Linear Sum Assignment...")
            # Calculate the ideal permutations of the rows and columns
            #   of the cost tensor in order to maximize its closeness to
            #   the identity matrix (see `calculate_progress`).
            # ri, ci: row-indices, column-indices
            ri, ci = linear_sum_assignment(
                cost_tensor.cpu().detach().numpy(), maximize=True
            )
            ri, ci = torch.from_numpy(ri), torch.from_numpy(ci)

            # Since we want to match features, we want to permute the columns only.
            # It should be the case that the rows naturally don't change.
            # If they do, that's a problem.
            assert torch.allclose(ri, torch.arange(n)), \
                "The rows of the cost tensor should not change."

            if self.logging_level == logging.DEBUG:
                loop.write("Done. \nProgress calculation...")
            # It is important to calculate the progress because
            #   the `_calculate_permutations_step`-method will be run several times.
            # If there were no interdependence between the permutations,
            #   this method would converge in one iteration.
            # However, there is interdependence,
            #   so we might need to run it several times.
            # If we calculate the progress and see that it is False,
            #   we can stop early.
            progress = progress or calculate_progress(
                cost_tensor,
                perm_old=perm.perm_indices,
                perm_new=ci,
                device=self.device_b
            )

            if self.logging_level == logging.DEBUG:
                loop.write(f"Done. Progress: {progress}\n")

            # Update the permutation.
            perm.perm_indices = ci

        return progress

    def apply_permutations(self) -> None:
        """
        Apply the calculated permutations to the model.
        """
        # Apply the permutations
        if self.logging_level == logging.DEBUG:
            print("Applying permutations...")

        self.pinit.model_graph.apply_permutations()
