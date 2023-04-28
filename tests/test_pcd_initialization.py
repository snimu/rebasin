"""Tests for the PermutationCoordinateDescent permutation initialization,
as defined in rebasin/initialization/initializer.py: PermutationInitialization.
"""

from __future__ import annotations

import copy

import torch

from rebasin.initialization.initializer import PermutationInitialization
from tests.fixtures.models import MLP
from tests.fixtures.util import allclose, model_change_percent, path_analysis


def test_on_mlp() -> None:
    """Test the :class:`PermutationInitialization` initializer on an MLP."""

    model_a = MLP(25, 10)
    model_b = MLP(25, 10)
    model_b_orig = copy.deepcopy(model_b)
    x = torch.randn(25)
    y_orig = model_b(x)

    initializer = PermutationInitialization(model_a, model_b, x)

    assert len(initializer.paths.paths) == 1

    for _, perm_info in initializer.permutations.items():
        axis, mod_params = perm_info[0]
        permutation = mod_params.axis_to_permutation[axis]
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

    initializer.paths.apply_permutations()
    assert model_change_percent(model_b, model_b_orig) > 0.1

    y_new = model_b(x)
    assert allclose(y_orig, y_new)
