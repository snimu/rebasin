from __future__ import annotations

from typing import Any

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import]
from torch import nn

from rebasin._wm_init_perms import PermutationInitializer


def calculate_progress(
        cost_mat: torch.Tensor, perm_old: torch.Tensor, perm_new: torch.Tensor
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

    Returns:
        True if the new permutation moves the cost matrix closer to the identity matrix
        than the old permutation does.
    """
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
    eye_ = torch.eye(cost_mat.shape[0])  # cost_mat is square
    optimality_old = torch.einsum("ij,ij->i", cost_mat[:, perm_old], eye_).sum()
    optimality_new = torch.einsum("ij,ij->i", cost_mat[:, perm_new], eye_).sum()

    progress = optimality_new.item() > optimality_old.item() + 1e-12
    return progress


class PermutationCoordinateDescent:
    """
    Implements the permutation coordinate descent algorithm.


    Args:
        model_a:
            The target model.
        model_b:
            The model that will have its weights permuted.
        input_data:
            One batch of input data to trace the model's layout.
    """

    def __init__(self, model_a: nn.Module, model_b: nn.Module, input_data: Any):
        self.model_b = model_b
        pinit = PermutationInitializer(model_a, model_b, input_data)
        self.permutations = pinit.permutations

    def calculate_permutations(self, max_iterations: int = 100) -> None:
        """
        Run the permutation coordinate descent algorithm to calculate the permutations.

        Args:
            max_iterations:
                The maximum number of iterations.
        """
        # Calculate the permutations
        for _i in range(max_iterations):
            progress = self._calculate_permutations_step()
            if not progress:
                break

    def _calculate_permutations_step(self) -> bool:
        """
        Run one iteration of the permutation coordinate descent algorithm.

        Returns:
            True if the algorithm made progress, False otherwise.
        """
        progress = False

        for i in torch.randperm(len(self.permutations)):
            permutation = self.permutations[i]
            n = len(permutation.perm_indices)
            cost_tensor = torch.zeros((n, n))

            for param_info in permutation.parameters:
                axis = param_info.axis
                w_a = param_info.param_a
                w_b = param_info.param_b
                # We want a square matrix as a cost tensor.
                # It should have shape (n, n).
                # To achieve this, we first move the axis of interest to the front.
                w_a = w_a.moveaxis(axis, 0) if len(w_a.shape) > 1 else w_a
                w_b = w_b.moveaxis(axis, 0) if len(w_b.shape) > 1 else w_b

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

            # It is important to calculate the progress because
            #   the `_calculate_permutations_step`-method will be run several times.
            # If there were no interdependence between the permutations,
            #   this method would converge in one iteration.
            # However, there is interdependence,
            #   so we might need to run it several times.
            # If we calculate the progress and see that it is False,
            #   we can stop early.
            progress = progress or calculate_progress(
                cost_tensor, perm_old=permutation.perm_indices, perm_new=ci
            )

            # Update the permutation.
            self.permutations[i].perm_indices = ci

        return progress

    def apply_permutations(self) -> None:
        """
        Apply the calculated permutations to the model.
        """
        # Apply the permutations
        for permutation in self.permutations:
            permutation.apply()
