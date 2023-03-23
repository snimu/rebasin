from __future__ import annotations

import torch


def progress(
        cost_mat: torch.Tensor, perm_old: torch.Tensor, perm_new: torch.Tensor
) -> bool:
    """
    Compute the progress of the permutation.

    Args:
        cost_mat:
            The cost matrix. A nxn matrix.
        perm_old:
            The old permutation. A vector of size n.
        perm_new:
            The new permutation. A vector of size n.

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

    prog = optimality_new.item() > optimality_old.item() + 1e-12
    return prog

