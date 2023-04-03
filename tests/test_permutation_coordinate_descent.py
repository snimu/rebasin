from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet18  # type: ignore[import]

from rebasin import PermutationCoordinateDescent
from rebasin.permutation_coordinate_descent import calculate_progress
from tests.fixtures.models import MLP, ModuleWithWeirdWeightAndBiasNames


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


class TestPermutationCoordinateDescent:
    def test_resnet18(self) -> None:
        # In a run of 500 tests, this test succeeded every time.
        model_a = resnet18()
        model_b = resnet18()
        self.common_tests(model_a, model_b, torch.randn(1, 3, 224, 224))

    def test_mlp(self) -> None:
        # Wide layers, or else some permutations will be the identity.
        # If every permutation is the identity, the test will fail.
        # Increasing layer size is better than increasing layer count.
        # At weights of size (25, 25), the test succeeds 500 out of 500 attempts.
        # For (5, 5) weights, the test fails fairly often.
        model_a = MLP(25, num_layers=5)
        model_b = MLP(25, num_layers=5)
        self.common_tests(model_a, model_b, torch.randn(25))

    def test_multihead_attention(self) -> None:
        embed_dim = num_heads = 8

        model_a = nn.MultiheadAttention(embed_dim, num_heads)
        model_b = nn.MultiheadAttention(embed_dim, num_heads)

        query = torch.randn(embed_dim, num_heads, requires_grad=True)
        key = torch.randn(embed_dim, num_heads, requires_grad=True)
        value = torch.randn(embed_dim, num_heads, requires_grad=True)

        self.common_tests(model_a, model_b, (query, key, value))

    def test_module_with_weird_weight_and_bias_names(self) -> None:
        model_a = ModuleWithWeirdWeightAndBiasNames()
        model_b = ModuleWithWeirdWeightAndBiasNames()

        self.common_tests(model_a, model_b, torch.randn(15))

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
                self.model_similarity(model_a, model_b)
                > self.model_similarity(model_a, model_b_old)
        )

    @staticmethod
    def model_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
        """Calculate the distance between two models."""
        total_dist = 0.0

        for parameter_a, parameter_b in zip(
                model_a.parameters(), model_b.parameters(), strict=True
        ):
            p_a, p_b = parameter_a.reshape(-1), parameter_b.reshape(-1)
            total_dist += float(p_a @ p_b)
        return abs(total_dist)
