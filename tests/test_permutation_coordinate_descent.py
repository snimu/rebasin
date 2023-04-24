from __future__ import annotations

import copy
import sys
from typing import Any

import pytest
import torch
from torch import nn
from torchvision.models import resnet18  # type: ignore[import]

from rebasin import PermutationCoordinateDescent
from rebasin.permutation_coordinate_descent import calculate_progress
from tests.fixtures.models import MLP, ModuleWithWeirdWeightAndBiasNames
from tests.fixtures.util import model_distance, model_similarity


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
    @pytest.mark.skipif("--full-suite" not in sys.argv, reason="Slow test")
    def test_resnet18(self) -> None:
        self.common_tests(resnet18)

    def test_mlp(self) -> None:
        in_features, num_layers = 50, 10
        self.common_tests(MLP, in_features, num_layers)

    def test_multihead_attention(self) -> None:
        embed_dim = num_heads = 32
        self.common_tests(nn.MultiheadAttention, embed_dim, num_heads)

    def test_module_with_weird_weight_and_bias_names(self) -> None:
        self.common_tests(ModuleWithWeirdWeightAndBiasNames)

    @staticmethod
    def common_tests(
            constructor: Any, *args: Any
    ) -> None:
        truecount = 0
        falsecount = 0

        for _ in range(100):
            model_a = constructor(*args)
            model_b = constructor(*args)

            model_b_old = copy.deepcopy(model_b)  # for comparison
            pcd = PermutationCoordinateDescent(model_a, model_b)
            pcd.calculate_permutations()

            assert pcd.permutations, "No permutations were found."

            # Check that there are some permutations different from the identity
            is_arange_list: list[bool] = []
            for permutation in pcd.permutations:
                ci, n = permutation.perm_indices, len(permutation.perm_indices)
                is_arange_list.append(
                    bool(torch.all(torch.eq(ci, torch.arange(n))).item())
                )

            assert not all(is_arange_list)

            # Check that:
            # - model_b was actually changed - I don't have to manually assing results
            # - model_b is closer to model_a than it was before the optimization
            pcd.apply_permutations()
            success = (
                    model_distance(model_a, model_b)
                    < model_distance(model_a, model_b_old)
            )
            if success:
                truecount += 1
            else:
                falsecount += 1

        assert truecount > falsecount


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
class TestPCDOnGPU:
    @staticmethod
    def test_mlp() -> None:
        device_b = torch.device("cuda")
        model_a, model_b = MLP(25), MLP(25).to(device_b)
        model_b_old = copy.deepcopy(model_b)

        pcd = PermutationCoordinateDescent(
            model_a, model_b, device_a="cpu", device_b=device_b
        )
        pcd.calculate_permutations()
        pcd.apply_permutations()

        assert (
                model_similarity(model_a, model_b)
                > model_similarity(model_a, model_b_old)
        )
