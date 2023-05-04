"""Tests for the PermutationCoordinateDescent permutation initialization,
as defined in rebasin/initialization/_initializer.py: PermutationInitialization.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torchvision.models import resnet50  # type: ignore[import]

from rebasin._initializer import PermutationInitialization
from tests.fixtures.models import MLP
from tests.fixtures.utils import (
    allclose,
    model_change_percent,
    randomize_permutations,
    reset_bn_running_stats,
)


def test_on_mlp() -> None:
    """Test the :class:`PermutationInitialization` initializer on an MLP."""
    common_tests(MLP(25, 10), MLP(25, 10), torch.randn(25))


def test_on_resnet50() -> None:
    """Test the :class:`PermutationInitialization` initializer on ResNet50."""
    common_tests(resnet50(), resnet50(), torch.randn(1, 3, 224, 224))


def common_tests(model_a: nn.Module, model_b: nn.Module, x: torch.Tensor) -> None:
    model_b_orig = copy.deepcopy(model_b)
    reset_bn_running_stats(model_b)
    y_orig = model_b(x)

    initializer = PermutationInitialization(model_a, model_b, x)
    randomize_permutations(initializer)
    initializer.paths.apply_permutations()
    assert model_change_percent(model_b, model_b_orig) > 0.1

    reset_bn_running_stats(model_b)
    y_new = model_b(x)
    assert allclose(y_orig, y_new)
