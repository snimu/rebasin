from __future__ import annotations

import copy
import sys

import pytest
import torch
from torchvision.models import (  # type: ignore[import]
    ViT_B_16_Weights,
    resnet18,
    vit_b_16,
)

from rebasin.permutation_initializer import PermutationInitializer
from tests.fixtures.models import MLP
from tests.fixtures.utils import (
    allclose,
    model_change_percent,
    tensor_diff_perc,
)


def test_mlp() -> None:
    model_a, model_b = MLP(15, 10), MLP(15, 10)
    model_b_orig = copy.deepcopy(model_b)
    input_data = torch.randn(1, 15)
    y_orig = model_b(input_data)

    pinit = PermutationInitializer(model_a, model_b, input_data)
    assert len(pinit.model_graph) == 1

    for permutation, _ in pinit.model_graph.permutation_to_info:
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

    pinit.model_graph.enforce_identity()
    pinit.model_graph.apply_permutations()
    assert model_change_percent(model_b, model_b_orig) > 0.1

    y_new = model_b(input_data)
    assert allclose(y_orig, y_new)


def test_mlp_control() -> None:
    model_a, model_b_orig = MLP(15, 10), MLP(15, 10)
    model_b1 = copy.deepcopy(model_b_orig)
    model_b2 = copy.deepcopy(model_b_orig)

    input_data = torch.randn(1, 15)

    pinit1 = PermutationInitializer(model_a, model_b1, input_data)
    pinit2 = PermutationInitializer(model_a, model_b2, input_data)

    for permutation, _ in pinit1.model_graph.permutation_to_info:
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

    pinit1.model_graph.enforce_identity()
    pinit1.model_graph.apply_permutations()

    for permutation, _ in pinit2.model_graph.permutation_to_info:
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

    pinit2.model_graph.input_permutation = None
    pinit2.model_graph.output_permutation = None
    pinit2.model_graph.apply_permutations()

    diff1 = tensor_diff_perc(model_b1(input_data), model_b_orig(input_data))
    diff2 = tensor_diff_perc(model_b2(input_data), model_b_orig(input_data))

    assert diff1 < diff2


def test_resnet18() -> None:
    model_a, model_b = resnet18(), resnet18()
    model_b_orig = copy.deepcopy(model_b)
    input_data = torch.randn(1, 3, 224, 224)

    pinit = PermutationInitializer(model_a, model_b, input_data)

    for permutation, _ in pinit.model_graph.permutation_to_info:
        del _
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

    pinit.model_graph.enforce_identity()
    pinit.model_graph.apply_permutations()
    assert model_change_percent(model_b, model_b_orig) > 0.1

    assert (
            pinit.model_graph[1].input_permutation
            is pinit.model_graph[1].output_permutation
            is pinit.model_graph[0].output_permutation
    )
    assert (
            pinit.model_graph[3].input_permutation
            is pinit.model_graph[3][0].input_permutation
            is pinit.model_graph[3][1].input_permutation
    )

    for module in model_b.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert torch.allclose(
                module.running_mean,  # type: ignore[arg-type]
                torch.zeros_like(module.running_mean)  # type: ignore[arg-type]
            )
            assert torch.allclose(
                module.running_var,  # type: ignore[arg-type]
                torch.ones_like(module.running_var)  # type: ignore[arg-type]
            )

    # nn.BatchNorm really plays a number on the enforce_identity() function,
    #   so for resnet18, I have to pick a much more relaxed metric
    #   than for other models.
    assert tensor_diff_perc(model_b(input_data), model_b_orig(input_data)) < 0.15
    assert tensor_diff_perc(model_b(input_data), model_a(input_data)) > 1.0


@pytest.mark.skipif("--full-suite" not in sys.argv, reason="Compute intense")
def test_vit_b_16() -> None:
    """Test how it works with modules that have no :class:`nn.BatchNorm2d`."""
    model_a = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model_b = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    model_b_orig = copy.deepcopy(model_b)
    input_data = torch.randn(1, 3, 224, 224)
    y_orig = model_b(input_data)

    pinit = PermutationInitializer(model_a, model_b, input_data)

    for permutation, _ in pinit.model_graph.permutation_to_info:
        del _
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

    pinit.model_graph.enforce_identity()
    pinit.model_graph.apply_permutations()
    assert model_change_percent(model_b, model_b_orig) > 0.1

    y_new = model_b(input_data)

    assert tensor_diff_perc(y_new, y_orig) < 1e-5
    assert allclose(y_new, y_orig)
    assert not allclose(model_a(input_data), model_b(input_data))


