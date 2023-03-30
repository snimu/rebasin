from __future__ import annotations

import copy
import sys
from typing import Any

import pytest
import torch
from torch import nn
from torchvision.models import resnet18  # type: ignore[import]

from rebasin import PermutationCoordinateDescent
from rebasin.weight_matching.permutation_coordinate_descent import (
    apply_all_permutations,
    apply_permutation,
    calculate_progress,
)
from rebasin.weight_matching.structs import (
    MODULE_AXES,
    AppliesTo,
    ModuleInfo,
    Permutation,
)
from tests.fixtures.models import MLP


def test_calculate_progress() -> None:
    """Test the calculation of the progress."""
    cost_mat = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    perm_old = torch.tensor([0, 1, 2])
    perm_new = torch.tensor([2, 0, 1])

    progress = calculate_progress(cost_mat, perm_old, perm_new)

    assert progress

    perm_old = torch.tensor([2, 0, 1])
    perm_new = torch.tensor([0, 1, 2])

    progress = calculate_progress(cost_mat, perm_old, perm_new)

    assert not progress


def test_apply_permutation() -> None:
    """Test the application of a permutation."""
    module_a = nn.Linear(3, 3)
    module_b = nn.Linear(3, 3)

    # Needed for manual setting of parameters
    for parameter in module_a.parameters():
        parameter.requires_grad = False
    for parameter in module_b.parameters():
        parameter.requires_grad = False

    w_a_old = copy.deepcopy(module_a.weight)
    b_a_old = copy.deepcopy(module_a.bias)
    w_b_old = copy.deepcopy(module_b.weight)
    b_b_old = copy.deepcopy(module_b.bias)

    module_info = ModuleInfo(
        module_a=module_a,
        module_b=module_b,
        applies_to=AppliesTo.BOTH,
        axis=0,
        axis_info=MODULE_AXES[nn.Linear]
    )

    permutation = Permutation(torch.tensor([2, 0, 1]), [module_info])
    params = apply_permutation(permutation)

    # Check types
    assert isinstance(params, list)
    assert all(isinstance(pair, tuple) for pair in params)
    assert all(
        isinstance(p, (torch.Tensor, nn.Parameter)) for pair in params for p in pair
    )

    # Check length: weight & bias!
    assert len(params) == 2

    # Check permutation
    assert torch.allclose(params[0][0], w_a_old)
    assert torch.allclose(params[1][0], b_a_old)
    assert torch.allclose(params[0][1], w_b_old[permutation.perm_indices])
    assert torch.allclose(params[1][1], b_b_old[permutation.perm_indices])


def test_apply_all_permutations() -> None:
    module_a = nn.Linear(3, 3)
    module_b = nn.Linear(3, 3)

    # Needed for manual setting of parameters
    for parameter in module_a.parameters():
        parameter.requires_grad = False
    for parameter in module_b.parameters():
        parameter.requires_grad = False

    w_a_old = copy.deepcopy(module_a.weight)
    b_a_old = copy.deepcopy(module_a.bias)
    w_b_old = copy.deepcopy(module_b.weight)
    b_b_old = copy.deepcopy(module_b.bias)

    module_info1 = ModuleInfo(
        module_a=module_a,
        module_b=module_b,
        applies_to=AppliesTo.BOTH,
        axis=0,
        axis_info=MODULE_AXES[nn.Linear]
    )

    module_info2 = ModuleInfo(
        module_a=module_a,
        module_b=module_b,
        applies_to=AppliesTo.WEIGHT,
        axis=1,
        axis_info=MODULE_AXES[nn.Linear]
    )

    perm0 = Permutation(torch.tensor([2, 0, 1]), [module_info1])
    perm1 = Permutation(torch.tensor([1, 2, 0]), [module_info2])

    id_to_perm = {id(module_b): [perm0, perm1]}

    params_full = apply_all_permutations(module_info1, id_to_perm)
    params_except = apply_all_permutations(module_info2, id_to_perm, except_axis=0)

    # Check types
    assert isinstance(params_full, list)
    assert all(isinstance(pair, tuple) for pair in params_full)
    assert all(
        isinstance(p, (torch.Tensor, nn.Parameter))
        for pair in params_full for p in pair
    )

    assert isinstance(params_except, list)
    assert all(isinstance(pair, tuple) for pair in params_except)
    assert all(
        isinstance(p, (torch.Tensor, nn.Parameter))
        for pair in params_except for p in pair
    )

    # Check length: w&b for params_full, w for params_except
    assert len(params_full) == 2
    assert len(params_except) == 1

    # Check permutation
    w_b_perm_full = w_b_old[perm0.perm_indices]
    w_b_perm_full = w_b_perm_full[:, perm1.perm_indices]

    assert torch.allclose(params_full[0][0], w_a_old)
    assert torch.allclose(params_full[1][0], b_a_old)
    assert torch.allclose(params_full[0][1], w_b_perm_full)
    assert torch.allclose(params_full[1][1], b_b_old[perm0.perm_indices])

    assert torch.allclose(params_except[0][0], w_a_old)
    assert torch.allclose(params_except[0][1], w_b_old[:, perm1.perm_indices])


class TestPermutationCoordinateDescent:

    @pytest.mark.skipif("--full-suite" not in sys.argv, reason="Slow test")
    def test_resnet18(self) -> None:
        model_a = resnet18()
        model_b = resnet18()
        self.common_tests(model_a, model_b, torch.randn(1, 3, 224, 224))

    def test_mlp(self) -> None:
        model_a = MLP(5)
        model_b = MLP(5)
        self.common_tests(model_a, model_b, torch.randn(5))

    @pytest.mark.xfail(
        reason="MultiheadAttention has multiple weights "
               "and I currently only look for parameters called 'weight'."
    )
    def test_multihead_attention(self) -> None:
        embed_dim = num_heads = 8

        model_a = nn.MultiheadAttention(embed_dim, num_heads)
        model_b = nn.MultiheadAttention(embed_dim, num_heads)

        query = torch.randn(embed_dim, num_heads, requires_grad=True)
        key = torch.randn(embed_dim, num_heads, requires_grad=True)
        value = torch.randn(embed_dim, num_heads, requires_grad=True)

        self.common_tests(model_a, model_b, (query, key, value))

    def common_tests(
            self,
            model_a: nn.Module,
            model_b: nn.Module,
            input_data: Any,
    ) -> None:
        pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
        pcd.calculate_permutations()
        model_b_old = copy.deepcopy(model_b)  # for comparison

        # Check that there are some permutations different from the identity
        is_arange_list: list[bool] = []
        for permutation in pcd.permutations:
            ci, n = permutation.perm_indices, len(permutation.perm_indices)
            is_arange_list.append(bool(torch.all(torch.eq(ci, torch.arange(n))).item()))

        assert not all(is_arange_list)

        # Check that:
        # - model_b was actually changed - I don't have to manually assing results
        # - model_b is closer to model_a than it was before the optimization
        pcd.apply_permutations()
        assert (
                self.model_distance(model_a, model_b)
                < self.model_distance(model_a, model_b_old)
        )

    @staticmethod
    def model_distance(model_a: nn.Module, model_b: nn.Module) -> float:
        """Calculate the distance between two models."""
        distance = 0.0
        for parameter_a, parameter_b in zip(
                model_a.parameters(), model_b.parameters(), strict=True
        ):
            distance += torch.norm(  # type: ignore[no-untyped-call]
                parameter_a - parameter_b
            ).item()
        return distance
