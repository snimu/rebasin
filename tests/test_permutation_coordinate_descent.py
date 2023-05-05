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
from tests.fixtures.models import MLP
from tests.fixtures.utils import allclose, model_distance, model_similarity


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
        self.common_tests(torch.randn(1, 3, 224, 224), resnet18, 1)

    def test_mlp(self) -> None:
        in_features, num_layers = 50, 5
        self.common_tests(torch.randn(50), MLP, 10, in_features, num_layers)

    # TODO: Fix this test
    @pytest.mark.xfail(reason="Currently has problem, fix later")
    def test_multihead_attention(self) -> None:
        embed_dim = num_heads = 32
        x = torch.randn(embed_dim, num_heads)
        self.common_tests((x, x, x), nn.MultiheadAttention, 10, embed_dim, num_heads)

    @staticmethod
    def common_tests(
            input_data: Any, constructor: Any, iters: int, *args: Any
    ) -> None:
        for _ in range(iters):
            model_a = constructor(*args)
            model_b = constructor(*args)

            model_b_old = copy.deepcopy(model_b)  # for comparison
            pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
            pcd.calculate_permutations()
            pcd.apply_permutations()

            assert (
                    model_distance(model_a, model_b)
                    < model_distance(model_a, model_b_old)
            )

    @staticmethod
    def test_enforce_identity() -> None:
        model_a = resnet18()
        model_b = resnet18()
        model_b_old = copy.deepcopy(model_b)

        x = torch.randn(1, 3, 224, 224)
        pcd = PermutationCoordinateDescent(model_a, model_b, x, enforce_identity=False)
        pcd.rebasin()

        assert not allclose(model_b(x), model_b_old(x))
        assert (
                model_distance(model_a, model_b)
                < model_distance(model_a, model_b_old)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
class TestPCDOnGPU:
    @staticmethod
    def test_mlp() -> None:
        device_b = torch.device("cuda")
        model_a, model_b = MLP(25), MLP(25).to(device_b)
        model_b_old = copy.deepcopy(model_b)

        pcd = PermutationCoordinateDescent(
            model_a, model_b, torch.randn(25), device_a="cpu", device_b=device_b
        )
        pcd.calculate_permutations()
        pcd.apply_permutations()

        assert (
                model_similarity(model_a, model_b)
                > model_similarity(model_a, model_b_old)
        )
