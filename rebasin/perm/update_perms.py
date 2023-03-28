from __future__ import annotations

import copy

import torch
from torch import nn

from rebasin.perm.structs import AppliesTo, ModuleInfo, Permutation


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


def apply_permutation(
        permutation: Permutation
) -> list[tuple[nn.Parameter, nn.Parameter] | tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply the permutation to the weights and biases of the modules.

    Args:
        permutation:
            The permutation holding the permutation info.

    Returns:
        A list of tuples. Each tuple contains the unpermuted parameter of model_a,
        and the corresponding permuted parameter of model_b.
    """
    outputs: list[
        tuple[torch.Tensor, torch.Tensor] | tuple[nn.Parameter, nn.Parameter]
        ] = []

    for module_info in permutation.modules:
        w_a, w_b, b_a, b_b = get_weights_and_biases(module_info)

        if module_info.applies_to in (AppliesTo.WEIGHT, AppliesTo.BOTH):
            w_b = permute_parameter(w_b, module_info.axis, permutation.perm_indices)
            outputs.append((w_a, w_b))
        if module_info.applies_to in (AppliesTo.BIAS, AppliesTo.BOTH):
            assert b_a is not None and b_b is not None
            b_b = permute_parameter(b_b, module_info.axis, permutation.perm_indices)
            outputs.append((b_a, b_b))

    return outputs


def apply_all_permutations(
        module_info: ModuleInfo,
        id_to_permutation: dict[int, list[Permutation]],
        except_axis: int | None = None
) -> list[tuple[nn.Parameter, nn.Parameter] | tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply all permutations belonging to module_b to its weights (and biases).

    Return the weights (and biases) of module_a and the permuted weights (and biases)
    of module_b.

    Args:
        module_info:
            The module info holding module_a (no permutation will be applied)
            and module_b (permutation will be applied).
        id_to_permutation:
            A dictionary mapping the id of a module to a list of permutations.
        except_axis:
            If not None, the permutation for this axis is not applied.

    Returns:
        A list of tuples. Each tuple contains the unpermuted parameter of model_a,
        and the corresponding permuted parameter of model_b.
    """
    outputs: list[
        tuple[torch.Tensor, torch.Tensor] | tuple[nn.Parameter, nn.Parameter]
        ] = []

    w_a, w_b, b_a, b_b = get_weights_and_biases(module_info)

    permutations = id_to_permutation[id(module_info.module_b)]
    permuted_bias = False

    for permutation in permutations:
        # Get perm_axis and applies_to for this permutation
        perm_axis = 0
        applies_to = AppliesTo.WEIGHT
        for m_info in permutation.modules:
            if m_info.module_b is module_info.module_b:
                perm_axis = m_info.axis
                applies_to = m_info.applies_to
                break

        # Skip this permutation if it is for the except_axis
        if except_axis is not None and perm_axis == except_axis:
            continue

        # Apply the permutation
        if applies_to in (AppliesTo.WEIGHT, AppliesTo.BOTH):
            w_b = permute_parameter(w_b, perm_axis, permutation.perm_indices)
        if applies_to in (AppliesTo.BIAS, AppliesTo.BOTH):
            assert b_b is not None
            b_b = permute_parameter(b_b, 0, permutation.perm_indices)
            permuted_bias = True

    outputs.append((w_a, w_b))

    if permuted_bias:
        assert b_a is not None and b_b is not None
        outputs.append((b_a, b_b))
    return outputs


def get_weights_and_biases(
        module_info: ModuleInfo
) -> tuple[
    nn.Parameter | torch.Tensor, nn.Parameter | torch.Tensor,
    nn.Parameter | torch.Tensor | None, nn.Parameter | torch.Tensor | None
]:
    """
    Get the weights and biases of module_a and module_b from module_info.

    Args:
        module_info:
            The module info holding module_a and module_b.

    Returns:
        A tuple containing the weights and biases of module_a and module_b.
        The biases are None if they do not exist.
    """
    w_a = copy.deepcopy(module_info.module_a.weight)
    w_b = copy.deepcopy(module_info.module_b.weight)
    b_a: torch.Tensor | nn.Parameter | None = None
    b_b: torch.Tensor | nn.Parameter | None = None

    a_has_bias = (
            hasattr(module_info.module_a, "bias")
            and module_info.module_a.bias is not None
    )
    b_has_bias = (
            hasattr(module_info.module_b, "bias")
            and module_info.module_b.bias is not None
    )
    assert a_has_bias == b_has_bias, \
        "Both modules must have a bias or neither must have a bias."

    no_module_weights = "Cannot handle weights that are Modules. " \
                        "They have to be Tensors or Parameters."
    no_module_bias = "Cannot handle biases that are Modules. " \
                     "They have to be Tensors or Parameters."

    assert isinstance(w_a, (torch.Tensor, nn.Parameter)), no_module_weights
    assert isinstance(w_b, (torch.Tensor, nn.Parameter)), no_module_weights

    if a_has_bias:
        b_a = copy.deepcopy(module_info.module_a.bias)  # type: ignore[assignment]
        b_b = copy.deepcopy(module_info.module_b.bias)  # type: ignore[assignment]
        assert isinstance(b_a, (torch.Tensor, nn.Parameter)), no_module_bias
        assert isinstance(b_b, (torch.Tensor, nn.Parameter)), no_module_bias

    return w_a, w_b, b_a, b_b


def permute_parameter(
        param: torch.Tensor | nn.Parameter, axis: int, perm_indices: torch.Tensor
) -> torch.Tensor | nn.Parameter:
    """
    Permute a parameter along a given axis.

    Args:
        param:
            The parameter to permute.
        axis:
            The axis along which to permute.
        perm_indices:
            The permutation indices.

    Returns:
        The permuted parameter.
    """
    param = param.moveaxis(axis, 0)
    param = param[perm_indices]
    param = param.moveaxis(0, axis)
    return param
