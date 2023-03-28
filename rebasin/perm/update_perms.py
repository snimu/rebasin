from __future__ import annotations

import copy
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import]
from torch import nn

from rebasin.perm.init_perms import PermutationInitializer
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
        permutation: Permutation, set_module_params: bool = False
) -> list[tuple[nn.Parameter, nn.Parameter] | tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply the permutation to the weights and biases of the modules.

    Args:
        permutation:
            The permutation holding the permutation info.
        set_module_params:
            If True, the weights and biases of the modules are set to the permuted
            values. If False, they are not.
            In both cases, the permuted values are returned.

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

        if set_module_params:
            set_weights_and_biases(module_info, w_b, b_b)

    return outputs


def set_weights_and_biases(
        module_info: ModuleInfo,
        weight: torch.Tensor | nn.Parameter,
        bias: torch.Tensor | nn.Parameter | None
) -> None:
    """
    Set the weight and bias of module_b of the module.

    Args:
        module_info:
            The module info holding module_b.
        weight:
            The permuted weight.
        bias:
            The permuted bias.
    """
    if isinstance(weight, torch.Tensor):
        weight = nn.Parameter(weight)
    if bias is not None and isinstance(bias, torch.Tensor):
        bias = nn.Parameter(bias)

    module_info.module_b.weight = weight
    if module_info.module_b.bias is not None and bias is not None:
        module_info.module_b.bias = bias


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

    @property
    def id_to_permutations(self) -> dict[int, list[Permutation]]:
        """
        Get a mapping from id of a module to the permutations that apply to it.

        Returns:
            A dictionary mapping the id of a module
            to the permutations that apply to it.
        """
        # Calculated dynamically because the permutations
        #   are changed in self.permutations, which doesn't automatically
        #   update the id_to_permutation dictionary.
        id_to_permutation: dict[int, list[Permutation]] = {}
        for permutation in self.permutations:
            for module_info in permutation.modules:
                if id(module_info.module_b) not in id_to_permutation:
                    id_to_permutation[id(module_info.module_b)] = []
                id_to_permutation[id(module_info.module_b)].append(permutation)
        return id_to_permutation

    def calculate_permutations(self, max_iterations: int = 100) -> None:
        """
        Run the permutation coordinate descent algorithm to calculate the permutations.

        Args:
            max_iterations:
                The maximum number of iterations.
            verbose:
                If True, print the current iteration number.
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
            tensors = []
            axes = []

            for module_info in permutation.modules:
                tens = apply_all_permutations(
                    module_info,
                    self.id_to_permutations,
                    except_axis=module_info.axis
                )
                tensors.extend(tens)
                axes.extend([module_info.axis] * len(tens))

            n = len(permutation.perm_indices)
            cost_tensor = torch.zeros((n, n))

            for axis, (w_a, w_b) in zip(axes, tensors, strict=True):
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
                cost_tensor, permutation.perm_indices, ci
            )

            # Update the permutation.
            self.permutations[i].perm_indices = ci

        return progress

    def apply_permutations(self) -> None:
        """
        Apply the calculated permutations to the model.
        """
        # Get the modules and set requires_grad to False.
        # This is necessary to manually set weights and biases.
        # Save the previous settings so that we can restore them later.
        modules_b = [m for m in self.model_b.modules() if hasattr(m, "weight")]
        grad_settings_weights = [m.weight.requires_grad for m in modules_b]
        grad_settings_biases = [
            m.bias.requires_grad
            if hasattr(m, "bias") and m.bias is not None
            else None
            for m in modules_b
        ]

        for module in modules_b:
            module.weight.requires_grad = False
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad = False

        # Apply the permutations
        for permutation in self.permutations:
            apply_permutation(permutation, set_module_params=True)

        # Reset the grad-settings
        for module, w_setting in zip(modules_b, grad_settings_weights, strict=True):
            module.weight.requires_grad = w_setting

        for module, b_setting in zip(modules_b, grad_settings_biases, strict=True):
            if hasattr(module, "bias") and module.bias is not None:
                assert b_setting is not None
                module.bias.requires_grad = b_setting
