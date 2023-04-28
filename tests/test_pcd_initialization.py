"""Tests for the PermutationCoordinateDescent permutation initialization,
as defined in rebasin/initialization/initializer.py: PermutationInitialization.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torchvision.models import resnet50  # type: ignore[import]

from rebasin.initialization.initializer import PermutationInitialization
from tests.fixtures.models import MLP
from tests.fixtures.util import allclose, model_change_percent, path_analysis


def randomize_permutations(initializer: PermutationInitialization) -> None:
    """Randomize the permutations in the initializer."""
    for _, perm_info in initializer.permutations.items():
        axis, mod_params = perm_info[0]
        permutation = mod_params.axis_to_permutation[axis]
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))


def reset_running_stats(model: nn.Module) -> None:
    """Reset the running statistics of all batch norm modules in the model."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()


def test_on_mlp() -> None:
    """Test the :class:`PermutationInitialization` initializer on an MLP."""

    model_a = MLP(25, 10)
    model_b = MLP(25, 10)
    model_b_orig = copy.deepcopy(model_b)
    x = torch.randn(25)
    y_orig = model_b(x)

    initializer = PermutationInitialization(model_a, model_b, x)

    assert len(initializer.paths.paths) == 1

    randomize_permutations(initializer)
    initializer.paths.apply_permutations()
    assert model_change_percent(model_b, model_b_orig) > 0.1

    y_new = model_b(x)
    assert allclose(y_orig, y_new)


def test_on_resnet50() -> None:
    """Test the :class:`PermutationInitialization` initializer on ResNet50."""

    model_a = resnet50()
    model_b = resnet50()
    model_b_orig = copy.deepcopy(model_b)
    x = torch.randn(1, 3, 224, 224)
    reset_running_stats(model_b)
    y_orig = model_b(x)

    initializer = PermutationInitialization(model_a, model_b, x)
    randomize_permutations(initializer)
    initializer.paths.apply_permutations()
    assert model_change_percent(model_b, model_b_orig) > 0.1

    reset_running_stats(model_b)
    y_new = model_b(x)
    assert allclose(y_orig, y_new)
