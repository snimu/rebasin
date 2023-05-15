from __future__ import annotations

import copy
import sys
from typing import Any

import pytest
import torch
from torch import nn
from torchvision.models import (  # type: ignore[import]
    ViT_B_16_Weights,
    resnet18,
    vit_b_16,
)

from rebasin import PermutationCoordinateDescent
from rebasin.permutation_coordinate_descent import calculate_progress
from tests.fixtures.models import MLP
from tests.fixtures.utils import (
    allclose,
    model_change_percent,
    model_distance,
    model_similarity,
)


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


def test_long_mlp() -> None:
    """Test that :class:`PermutationCoordinateDescent`
    can handle a linear network with 100 layers
    (each with a 10x10 weight matrix).

    Handle means that it can permute the weights s.t. the output doesn't change.
    """
    model_a, model_b = MLP(10, 100), MLP(10, 100)
    model_b_orig = copy.deepcopy(model_b)
    input_data = torch.randn(10)
    y_orig = model_b(input_data)

    pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
    pcd.calculate_permutations()
    pcd.apply_permutations()

    assert model_change_percent(model_b, model_b_orig) > 0.1

    y_new = model_b(input_data)
    assert torch.allclose(y_orig, y_new)


@pytest.mark.skipif(
    "--full-suite" not in sys.argv,
    reason="This test takes a long time to run. Run with --full-suite to include it.",
)
class TestPermutationCoordinateDescent:
    def test_resnet18(self) -> None:
        self.common_tests(torch.randn(1, 3, 224, 224), resnet18(), resnet18(), 1)

    def test_mlp(self) -> None:
        self.common_tests(torch.randn(50), MLP(50, 15), MLP(50, 15), 1)

    def test_vit_b_16(self) -> None:
        self.common_tests(
            torch.randn(1, 3, 224, 224),
            vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1),
            vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1),
            iters=1
        )

    @staticmethod
    def common_tests(
            input_data: Any, model_a: nn.Module, model_b: nn.Module, iters: int = 10
    ) -> None:
        for _ in range(iters):
            model_a = model_a.eval()
            model_b = model_b.eval()
            model_b_old = copy.deepcopy(model_b)  # for comparison

            y_orig = model_b(input_data)

            pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
            pcd.rebasin()

            assert (
                    model_distance(model_a, model_b)
                    < model_distance(model_a, model_b_old)
            )

            y_new = model_b(input_data)
            allclose(y_orig, y_new)

    @staticmethod
    def test_enforce_identity() -> None:
        model_a = resnet18()
        model_b = resnet18()
        model_b_old = copy.deepcopy(model_b)

        x = torch.randn(1, 3, 224, 224)
        pcd = PermutationCoordinateDescent(model_a, model_b, x)
        pcd.rebasin()

        assert allclose(model_b(x), model_b_old(x))
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
            model_a,
            model_b,
            input_data_b=torch.randn(25).to(device_b),
            input_data_a=torch.randn(25),
            device_a="cpu",
            device_b=device_b
        )
        pcd.calculate_permutations()
        pcd.apply_permutations()

        assert (
                model_similarity(model_a, model_b)
                > model_similarity(model_a, model_b_old)
        )
